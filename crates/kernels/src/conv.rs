//! 卷积（NHWC、四周两圈边框）：**抄自 calblas 0.1.0 的 `conv2d!`**（`src/kernels/vision.rs` 第 93–283 行，sha256 218ca5a4…），
//! 只加了两种激活，别的逐字不动（细账见仓库根的 VENDORED.md）：
//!
//! - `act = relu`：bias 之后 `max(y, 0)`，再加残差（若有）。TAESD 的「conv + ReLU」。
//! - `act = res_relu`：bias 之后不激活，加残差，**再** `max(y, 0)`。TAESD Block 末尾的 `relu(conv(x) + x)`。
//!
//! 为什么抄而不是在外面接一个 `binary!(relu)`：Block 末尾那一步融进卷积的尾处理，省掉一整张特征图的读写
//! （512² × 64 通道的 bf16 图是 34 MB，一次读写 ≈ 0.8 ms 量级，按 84 GB/s 估算，未实测）。
//!
//! 排布、权重、对齐、两核的约定全部同 calblas（见下面宏的文档与 calblas `vision.rs` 模块文档）：
//! 特征图 `[H + 4][W + 4][P]` bf16、边框是 0；权重 `[k·k][Cin][Cout]` bf16、bias f32 `[Cout]`；指针 64 B 对齐；`cores = 2`，调完要 `sync_cores()`，不能原地。
//! 数值：bf16 × bf16 在 TU 里 f32 累加 ⇒ + bias（f32）⇒ 舍成 bf16 ⇒ 激活（relu 精确）⇒（+ 残差，bf16 加法舍一次）⇒（res_relu：relu）。
//! 守着它的测试：`ref/ktest.py conv`（单 kernel 的随机输入 numpy 期望 + 上板比对），`ref/sim.py`（整步程序的接线）。

/// `y = act(conv(x) + b) (+ r)`：k = 1 / 3，s = 1 / 2；输入 [H·s + 4][W·s + 4][pin]，输出 [H + 4][W + 4][pout]（都从左上角的边框算起，地址已偏到通道窗口）。
/// `pin = (行距, 窗口偏移)`：指针给到 32 个通道对齐处（kernel 的指针参数要 64 B 对齐），窗口在像素内的偏移（0 / 16 …）是编译期常数；`pout` 同。
/// `act = silu | none | relu | res_relu`；末尾写 `, res = (行距, 偏移)` ⇒ 多一个残差参数（同形的图，act 之后加：Bottleneck 的 x + cv2(cv1(x))、SPPF 的 + x）。
/// （本仓加）`relu`：bias 之后 relu，再加残差（若有）；`res_relu`：bias 之后不激活、加残差、**再** relu ——
/// TAESD 的 Block 末尾 relu(conv(x) + x) 就是 `act = res_relu, res = …`。
#[macro_export]
macro_rules! conv2d {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, cin = $ci:expr, cout = $co:expr, k = $k:tt, s = $s:expr, pin = ($pi:expr, $xo:expr), pout = ($po:expr, $yo:expr), act = $act:ident $(, res = ($pr:expr, $ro:expr))? $(, weights = $wl:ident)?) => {
        #[npu::kernel(numerics = "ks1", cores = 2)]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, w: npu::Ptr<npu::Ddr, npu::bf16>, b: npu::Ptr<npu::Ddr, f32>, $(r: $crate::conv2d!(@res $pr),)? y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const B: usize = $crate::calblas::kernels::BORDER;
            const H: usize = $h;
            const W: usize = $w;
            const CI: usize = $ci;
            const CO: usize = $co;
            const K: usize = $k;
            const S: usize = $s;
            const PI: usize = $pi;
            const PO: usize = $po;
            const XO: usize = $xo;
            const YO: usize = $yo;
            // 输入的一行（含边框）；抽头 (0, 0) 在输入里的偏移（padding = K / 2）
            const WI: usize = W * S + 2 * B;
            const OFF: usize = B - K / 2;
            const CK: usize = $crate::calblas::kernels::Cols::chunk(CI);
            const KC: usize = CI / CK;
            const NC: usize = $crate::calblas::kernels::Cols(CK, CO).nc();
            const _: () = assert!(CI % 16 == 0 && CO % 16 == 0, "conv2d：通道数补到 16 的倍数");
            const TAIL: usize = W % 16;
            // 快路：步长 1、Cin 不分段、输入的 K 行（再多 16 个像素的余量）放得进 VB ⇒ 每个输出行把输入的 K 行读进 VB 一次，各抽头的 lhs 从 VB 取
            const FAST: bool = S == 1 && KC == 1 && (K * WI + 16) * CI * 2 <= 384 * 1024;
            // `weights = ci_major`：权重按 [Cin][k · k][Cout]（torch 的 [Cout, Cin · k · k] 转置一次就是它，框架在设备上转）；缺省 [k · k][Cin][Cout]
            const WCM: bool = $crate::conv2d!(@wl $($wl)?);
            let gw = $crate::calblas::rd!(w, [K * K * CI, CO]);
            let gb = $crate::calblas::rd!(b, [CO / NC, NC]);
            let (vo, vb, yh, tmp) = (vcache::<f32, _>([16, NC]), vcache::<f32, _>([1, NC]), vcache::<bf16, _>([16, NC]), vcache::<bf16, _>([16, NC]));
            $(let vr = vcache::<bf16, _>([16, NC]); const _: usize = $pr;)?
            let (h0, hn) = $crate::calblas::halves!(H);
            $crate::conv2d!(@taps $k, [x, gw, w, gb, y, lhs, acc, vo, vb, yh, tmp, h0, hn], $act, [$(r, vr, $pr, $ro)?]);
        }
    };
    (@taps 1, $l:tt, $act:ident, $res:tt) => { $crate::conv2d!(@go [(0, 0, r0, x0)], $l, $act, $res) };
    (@taps 3, $l:tt, $act:ident, $res:tt) => {
        $crate::conv2d!(@go [(0, 0, r0, x0) (0, 1, r1, x1) (0, 2, r2, x2) (1, 0, r3, x3) (1, 1, r4, x4) (1, 2, r5, x5) (2, 0, r6, x6) (2, 1, r7, x7) (2, 2, r8, x8)], $l, $act, $res)
    };
    (@go [$(($ky:expr, $kx:expr, $rh:ident, $vx:ident))*], [$x:ident, $gw:ident, $wp:ident, $gb:ident, $y:ident, $lhs:ident, $acc:ident, $vo:ident, $vb:ident, $yh:ident, $tmp:ident, $h0:ident, $hn:ident], $act:ident, [$($r:ident, $vr:ident, $pr:expr, $ro:expr)?]) => {{
        // 每个抽头一个输入视图：第 r 行 = 从抽头原点起的第 r·s 个像素（输出第 oy 行第 mb 块 ⇒ 第 oy · WI + mb · 16 行起 16 行）
        $(
            let $rh = mcache::<bf16, _>([CK, NC]);
            let $vx = $x.add(((($ky + OFF) * WI + $kx + OFF) * PI + XO) * 2).bind_strided::<npu::Read, 2>([H * WI, CI], [S * PI, 1]);
        )*
        // 输出（残差同形）：从内部左上角起 [H, ⌈W/16⌉, 16, P]
        const MB: usize = W.div_ceil(16);
        const WO: usize = W + 2 * B;
        let gy = $y.add(((B * WO + B) * PO + YO) * 2).bind_strided::<npu::Write, 4>([H, MB, 16, PO], [WO * PO, 16 * PO, PO, 1]);
        $(let gr = $r.add(((B * WO + B) * $pr + $ro) * 2).bind_strided::<npu::Read, 4>([H, MB, 16, $pr], [WO * $pr, 16 * $pr, $pr, 1]);)?
        if FAST {
            // 输入的 K 行（输出第 oy 行的各抽头都在里面：抽头 (ky, kx)、第 mb 块 ⇒ 第 ky · WI + kx + 16 · mb 行起 16 行）
            let xin = vcache::<bf16, _>([K * WI + 16, CI]);
            let gxin = $x.add(((OFF * WI + OFF) * PI + XO) * 2).bind_strided::<npu::Read, 2>([(H + K - 1) * WI + 16, CI], [PI, 1]);
            // 一行里每 4 块一组交错（余下的两块一组）：每个抽头的 rhs 进 GIB1 一次给 4 块用，4 个累加器轮流接串（TU 相邻两条没有依赖）
            // 快路自己的 lhs 与累加器（⛔ 和原路共用：docc 按一块 m.cache 的全部用处定矩阵模式，原路分段的 mma 会把它拉回向量模式）
            let fl = mcache::<bf16, _>([16, CI]);
            const G4: usize = W / 64;
            const G2: usize = (W / 16 - G4 * 4) / 2;
            for nb in 0..CO / NC {
                $($rh.load($crate::conv2d!(@wtap $gw, $wp, $ky * K + $kx, 0, nb * NC, CI, NC));)*
                $vb.load($gb.sub([nb, 0], [1, NC]));
                for i in 0..$hn {
                    let oy = $h0 + i;
                    xin.load(gxin.sub([oy * WI, 0], [K * WI + 16, CI]));
                    for g in 0..G4 {
                        // 每块一个累加器，都在这个循环体里填 0、接串、写回（docc 要一块累加器的用处都在它填 0 的那个块里，才走矩阵模式）
                        let (a0, a1) = (mcache::<f32, _>([16, NC]), mcache::<f32, _>([16, NC]));
                        let (a2, a3) = (mcache::<f32, _>([16, NC]), mcache::<f32, _>([16, NC]));
                        a0.fill(0.0f32);
                        a1.fill(0.0f32);
                        a2.fill(0.0f32);
                        a3.fill(0.0f32);
                        $(
                            fl.load_v(xin.rows($ky * WI + $kx + g * 64, 16));
                            a0.mma(fl, $rh);
                            fl.load_v(xin.rows($ky * WI + $kx + g * 64 + 16, 16));
                            a1.mma(fl, $rh);
                            fl.load_v(xin.rows($ky * WI + $kx + g * 64 + 32, 16));
                            a2.mma(fl, $rh);
                            fl.load_v(xin.rows($ky * WI + $kx + g * 64 + 48, 16));
                            a3.mma(fl, $rh);
                        )*
                        $crate::conv2d!(@tail a0, [$vo, $vb, $yh, $tmp, gy], oy, g * 4, nb, 16, $act, [$(gr, $vr, $pr)?]);
                        $crate::conv2d!(@tail a1, [$vo, $vb, $yh, $tmp, gy], oy, g * 4 + 1, nb, 16, $act, [$(gr, $vr, $pr)?]);
                        $crate::conv2d!(@tail a2, [$vo, $vb, $yh, $tmp, gy], oy, g * 4 + 2, nb, 16, $act, [$(gr, $vr, $pr)?]);
                        $crate::conv2d!(@tail a3, [$vo, $vb, $yh, $tmp, gy], oy, g * 4 + 3, nb, 16, $act, [$(gr, $vr, $pr)?]);
                    }
                    // 余下的整块两块一组，同样交错
                    for g in 0..G2 {
                        let p = G4 * 64 + g * 32;
                        let (a0, a1) = (mcache::<f32, _>([16, NC]), mcache::<f32, _>([16, NC]));
                        a0.fill(0.0f32);
                        a1.fill(0.0f32);
                        $(
                            fl.load_v(xin.rows($ky * WI + $kx + p, 16));
                            a0.mma(fl, $rh);
                            fl.load_v(xin.rows($ky * WI + $kx + p + 16, 16));
                            a1.mma(fl, $rh);
                        )*
                        $crate::conv2d!(@tail a0, [$vo, $vb, $yh, $tmp, gy], oy, G4 * 4 + g * 2, nb, 16, $act, [$(gr, $vr, $pr)?]);
                        $crate::conv2d!(@tail a1, [$vo, $vb, $yh, $tmp, gy], oy, G4 * 4 + g * 2 + 1, nb, 16, $act, [$(gr, $vr, $pr)?]);
                    }
                    // 凑不满一组的整块与尾块：逐块算，lhs 同样从 VB 取
                    for mb in G4 * 4 + G2 * 2..W / 16 {
                        let ab = mcache::<f32, _>([16, NC]);
                        $crate::conv2d!(@block vb, [$(($ky, $kx, $rh, $vx))*], [$gw, $wp, fl, ab, $vo, $vb, $yh, $tmp, gy, xin], oy, mb, nb, 16, $act, [$(gr, $vr, $pr)?]);
                    }
                    if TAIL > 0 {
                        let ab = mcache::<f32, _>([16, NC]);
                        $crate::conv2d!(@block vb, [$(($ky, $kx, $rh, $vx))*], [$gw, $wp, fl, ab, $vo, $vb, $yh, $tmp, gy, xin], oy, (W / 16), nb, TAIL, $act, [$(gr, $vr, $pr)?]);
                    }
                }
            }
        } else {
            let ($lhs, $acc) = (mcache::<bf16, _>([16, CK]), mcache::<f32, _>([16, NC]));
            for nb in 0..CO / NC {
                // 不分段：每个抽头的 rhs 每个 nb 块装一次、整块留着；分段的在块里逐段装（rhs 放不下整个 Cin）
                if KC == 1 {
                    $($rh.load($crate::conv2d!(@wtap $gw, $wp, $ky * K + $kx, 0, nb * NC, CI, NC));)*
                }
                $vb.load($gb.sub([nb, 0], [1, NC]));
                for i in 0..$hn {
                    let oy = $h0 + i;
                    for mb in 0..W / 16 {
                        $crate::conv2d!(@block ddr, [$(($ky, $kx, $rh, $vx))*], [$gw, $wp, $lhs, $acc, $vo, $vb, $yh, $tmp, gy, $x], oy, mb, nb, 16, $act, [$(gr, $vr, $pr)?]);
                    }
                    if TAIL > 0 {
                        $crate::conv2d!(@block ddr, [$(($ky, $kx, $rh, $vx))*], [$gw, $wp, $lhs, $acc, $vo, $vb, $yh, $tmp, gy, $x], oy, (W / 16), nb, TAIL, $act, [$(gr, $vr, $pr)?]);
                    }
                }
            }
        }
    }};
    // 第 oy 行第 mb 块（16 个像素，写回前 rows 个）；lhs 从 DDR 的抽头视图读（ddr）或从 VB 里输入的 K 行取（vb）
    (@block $from:ident, [$(($ky:expr, $kx:expr, $rh:ident, $vx:ident))*], [$gw:ident, $wp:ident, $lhs:ident, $acc:ident, $vo:ident, $vb:ident, $yh:ident, $tmp:ident, $gy:ident, $xin:ident], $oy:ident, $mb:expr, $nb:ident, $rows:expr, $act:ident, $res:tt) => {{
        $acc.fill(0.0f32);
        $(
            if KC == 1 {
                $crate::conv2d!(@lhs $from, $lhs, $vx, $xin, $oy, $mb, $ky, $kx);
                $acc.mma($lhs, $rh);
            } else {
                for kc in 0..KC {
                    $rh.load($crate::conv2d!(@wtap $gw, $wp, $ky * K + $kx, kc * CK, $nb * NC, CK, NC));
                    $lhs.load($vx.sub([$oy * WI + $mb * 16, kc * CK], [16, CK]));
                    $acc.mma($lhs, $rh);
                }
            }
        )*
        $crate::conv2d!(@tail $acc, [$vo, $vb, $yh, $tmp, $gy], $oy, $mb, $nb, $rows, $act, $res);
    }};
    (@lhs ddr, $lhs:ident, $vx:ident, $xin:ident, $oy:ident, $mb:expr, $ky:expr, $kx:expr) => { $lhs.load($vx.sub([$oy * WI + $mb * 16, 0], [16, CI])) };
    (@lhs vb, $lhs:ident, $vx:ident, $xin:ident, $oy:ident, $mb:expr, $ky:expr, $kx:expr) => { $lhs.load_v($xin.rows($ky * WI + $kx + $mb * 16, 16)) };
    // 一块的尾处理：+ bias（f32）⇒ bf16 ⇒ act ⇒ (+ 残差) ⇒ 写回前 rows 个像素
    (@tail $acc:ident, [$vo:ident, $vb:ident, $yh:ident, $tmp:ident, $gy:ident], $oy:ident, $mb:expr, $nb:ident, $rows:expr, $act:ident, [$($gr:ident, $vr:ident, $pr:expr)?]) => {{
        $vo.from_m($acc);
        $vo.add($vo, bcast($vb, &[0, 1]));
        $yh.cvt($vo, Cvt::default());
        $crate::conv2d!(@pre $act $yh, $tmp);
        $(
            $vr.rows(0, $rows).load($gr.sub([$oy, $mb, 0, $nb * NC], [1, 1, $rows, NC]));
            $yh.add($yh, $vr);
        )?
        $crate::conv2d!(@post $act $yh);
        if YO % 32 == 0 && NC % 32 == 0 {
            $yh.rows(0, $rows).store($gy.sub([$oy, $mb, 0, $nb * NC], [1, 1, $rows, NC]));
        } else {
            // 窗口（或第 nb 块）可能起在 64 B 行的后半段：dm2.vb2ddr 多行的目的要起在前半段 ⇒ 一行一写
            for r in 0..$rows {
                $yh.rows(r, 1).store($gy.sub([$oy, $mb, r, $nb * NC], [1, 1, 1, NC]));
            }
        }
    }};
    (@res $pr:expr) => { npu::Ptr<npu::Ddr, npu::bf16> };
    (@wl) => { false };
    (@wl ci_major) => { true };
    // 抽头 tap 的权重块：从第 r0 个输入通道、第 c0 个输出通道起 [rows, cols]（二维视图；ci_major 时抽头之间隔 Cout 个、通道之间隔 k · k · Cout 个）
    (@wtap $gw:ident, $wp:ident, $tap:expr, $r0:expr, $c0:expr, $rows:expr, $cols:expr) => {
        if WCM {
            $wp.add(($tap) * CO * 2).bind_strided::<npu::Read, 2>([CI, CO], [K * K * CO, 1]).sub([$r0, $c0], [$rows, $cols])
        } else {
            $gw.sub([($tap) * CI + $r0, $c0], [$rows, $cols])
        }
    };
    // ── 激活（本仓加的两种：relu、res_relu）──
    // @pre：bias 之后、残差之前；@post：残差之后。relu 在 bf16 上取 max(y, 0)（精确，不舍入；写法同 calblas binary!(relu)）
    (@pre silu $y:ident, $tmp:ident) => { $crate::calblas::logistic!(silu, $y, $y, $tmp) };
    (@pre none $y:ident, $tmp:ident) => {};
    (@pre relu $y:ident, $tmp:ident) => { $y.max($y, 0.0f32) };
    (@pre res_relu $y:ident, $tmp:ident) => {};
    (@post res_relu $y:ident) => { $y.max($y, 0.0f32) };
    (@post $act:ident $y:ident) => {};
}
