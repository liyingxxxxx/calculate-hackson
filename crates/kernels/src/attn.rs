//! 头维宽的注意力 `attn_wide!`（正式 VAE 中间块：单头、512 维、4096 个 token，DESIGN D14）。
//!
//! 为什么不直接用 calblas `attention!`：它的 rhs `[D, 128]`（k 块）与 `[128, D]`（v 块）在 D = 512 时是 128 KB，
//! TU 的操作数（GIB1）只有 64 KB，docc 报「gib1 放不下」（DESIGN F21）。这里把头维按 `DC = 128` 一段切开：
//!
//! ```text
//!   每头、每 16 行 q（两核各一半）：
//!   q 的 D / DC 段读进 VB；
//!   第一遍（每块 128 个 key）：s = Σ_段 q段 · k段ᵀ（TU 沿段接着累加，f32）⇒ bf16 ⇒ · 1/√D ⇒ 存进 VB 的整行分数；行最大 m
//!   第二遍（每块）：p = exp(s − m)（bf16，硬件近似）存回；l += 行和（f32）
//!   第三遍（每段 DC 个输出维）：acc [16, DC] = Σ_块 p 块 · v [块, 段]（TU）⇒ bf16 ⇒ ÷ l（bf16）⇒ 写回 o 的这一段
//! ```
//! 数值口径同 calblas `attention!`（分数与 p 舍 bf16、exp 近似、行和 f32、最后 bf16 里除）；不同的只有 q · kᵀ 沿段累加的次序
//! （TU 里 f32 接着加，和一次乘完只差加法次序）。分数整行存在 VB 里（spatial_attn! 的做法）⇒ q · kᵀ 只算一遍；
//! 一行 TK 个 key 的 bf16 要放得下（TK = 4096 时 128 KB），断言守着。
//!
//! 排布：q、k、v、o 都是稠密的 `[T, H · D]` bf16（token 主序，HF 的样子），第 h 头占第 h · D 起的 D 列；只做自注意力（TK = T）、不带掩码
//! （VAE 用不着；要掩码 / 交叉注意力的用 calblas `attention!`，头维 ≤ 256 它收得下）。
//!
//! **来源**：寻址（q / o 的 `[T·H, 16, D]` 视图、kᵀ 的列主序读）抄 calblas 0.1.0 `attention!`（`src/kernels/attention.rs`），
//! VB 里整行分数的写法抄 `spatial_attn!`（`src/kernels/vision.rs`），按段切头维是本仓加的（VENDORED.md）。
//! 守着它的测试：`ref/ktest.py attn_wide`（期望是 torch 的 softmax(q·kᵀ/√D)·v，f32 ⇒ 舍 bf16）、模拟器 `cvxsim.py` 的 `_k_attn_wide`。

/// `o = softmax(q · kᵀ / √D) · v`，每头独立：见模块文档。`rows` = token 数（16 的倍数、128 的倍数），`head` = 头维（128 的倍数）。
#[macro_export]
macro_rules! attn_wide {
    ($vis:vis $name:ident: rows = $t:expr, heads = $h:expr, head = $d:expr) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(q: npu::Ptr<npu::Ddr, npu::bf16>, k: npu::Ptr<npu::Ddr, npu::bf16>, v: npu::Ptr<npu::Ddr, npu::bf16>, o: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const T: usize = $t;
            const H: usize = $h;
            const D: usize = $d;
            // key 每 128 个一块；头维每 DC 个一段（rhs [DC, 128] / [128, DC] bf16 = 32 KB，GIB1 两个槽轮流用）
            const KB: usize = 128;
            const DC: usize = 128;
            const NDC: usize = D / DC;
            const NJ: usize = T / KB;
            const NEG: f32 = -3.0e38;
            const SCALE: f32 = $crate::calblas::HeadDim(D).inv_sqrt();
            const _: () = assert!(T % KB == 0 && D % DC == 0 && NJ * 16 * KB * 2 <= 256 * 1024,
                "attn_wide：token 数是 128 的倍数、头维是 128 的倍数、一行分数（bf16）≤ 256 KB");
            // q / o：[T · H, 16, D]（行距 [D, H · D, 1]）的第 qb · 16 · H + h 块 = 第 qb 块 16 行、第 h 头（同 calblas attention! 的 @at）
            let gq = q.bind_strided::<npu::Read, 3>([T * H, 16, D], [D, H * D, 1]);
            let go = o.bind_strided::<npu::Write, 3>([T * H, 16, D], [D, H * D, 1]);
            // kᵀ：[头, d, key]（d 行距 1）⇒ rhs [DC, 128 key]；v：[key, H · D] ⇒ rhs [128 key, DC]
            let gk = k.bind_strided::<npu::Read, 3>([H, D, T], [D, 1, H * D]);
            let gv = $crate::calblas::rd!(v, [T, H * D]);
            let (lq, rk, sacc) = (mcache::<bf16, _>([16, DC]), mcache::<bf16, _>([DC, KB]), mcache::<f32, _>([16, KB]));
            let (lp, rv, acc) = (mcache::<bf16, _>([16, KB]), mcache::<bf16, _>([KB, DC]), mcache::<f32, _>([16, DC]));
            // VB：q 的各段、整行分数 / p、一块的临时
            let (vq, sbuf) = (vcache::<bf16, _>([NDC * 16, DC]), vcache::<bf16, _>([NJ * 16, KB]));
            let (sf, sb, tf) = (vcache::<f32, _>([16, KB]), vcache::<bf16, _>([16, KB]), vcache::<f32, _>([16, KB]));
            let (m, top, mb) = (vcache::<f32, _>([16, 1]), vcache::<f32, _>([16, 1]), vcache::<bf16, _>([16, 1]));
            let (l, rs, lb) = (vcache::<f32, _>([16, 1]), vcache::<f32, _>([16, 1]), vcache::<bf16, _>([16, 1]));
            let (out, outb) = (vcache::<f32, _>([16, DC]), vcache::<bf16, _>([16, DC]));
            // 头 × q 块合成一层循环，两核各一半（同 calblas attention! 的 @loop [2]）
            const HALF: usize = $crate::calblas::Halves(H * (T / 16)).half();
            let mine = core_rank() * HALF;
            for i in 0..HALF {
                let j = mine + i;
                let (h, qb) = (j / (T / 16), j % (T / 16));
                for c in 0..NDC {
                    vq.rows(c * 16, 16).load(gq.sub([qb * 16 * H + h, 0, c * DC], [1, 16, DC]));
                }
                // 第一遍：分数（沿段累加）⇒ bf16 ⇒ 缩放 ⇒ 存进整行；行最大
                m.fill(NEG);
                for kb in 0..NJ {
                    sacc.fill(0.0f32);
                    for c in 0..NDC {
                        lq.load_v(vq.rows(c * 16, 16));
                        rk.load(gk.sub([h, c * DC, kb * KB], [1, DC, KB]));
                        sacc.mma(lq, rk);
                    }
                    sf.from_m(sacc);
                    sb.cvt(sf, Cvt::default());
                    sb.mul(sb, SCALE);
                    tf.cvt(sb, Cvt::default());
                    top.reduce(Combine::Max, tf, 1);
                    m.max(m, top);
                    sbuf.rows(kb * 16, 16).copy(sb);
                }
                // 第二遍：p = exp(s − m)，存回；行和
                mb.cvt(m, Cvt::default());
                l.fill(0.0f32);
                for kb in 0..NJ {
                    sb.copy(sbuf.rows(kb * 16, 16));
                    sb.sub(sb, bcast(mb, &[0, 1]));
                    sb.exp_approx(sb);
                    tf.cvt(sb, Cvt::default());
                    rs.reduce(Combine::Sum, tf, 1);
                    l.add(l, rs);
                    sbuf.rows(kb * 16, 16).copy(sb);
                }
                lb.cvt(l, Cvt::default());
                // 第三遍：每段输出维 acc = Σ_块 p · v ⇒ ÷ l ⇒ 写回
                for c in 0..NDC {
                    acc.fill(0.0f32);
                    for kb in 0..NJ {
                        lp.load_v(sbuf.rows(kb * 16, 16));
                        rv.load(gv.sub([kb * KB, h * D + c * DC], [KB, DC]));
                        acc.mma(lp, rv);
                    }
                    out.from_m(acc);
                    outb.cvt(out, Cvt::default());
                    outb.div(outb, bcast(lb, &[0, 1]));
                    outb.store(go.sub([qb * 16 * H + h, 0, c * DC], [1, 16, DC]));
                }
            }
        }
    };
}

/// 一块 K / V 给多块 q 用的注意力 `flash_attn!`（UNet 的自注意力 / 交叉注意力，DESIGN F27 / D21，新写）。
///
/// 为什么：calblas `attention!` 每 16 行 q、每块 128 个 key 走一串十来条 VU / TU 指令，一块约 3.5 µs、和头维无关
/// （真卡逐 kernel 计时：64² 的自注意力 4736 个 token 一次 625 ms，0.07 TFLOPS）；把 K 先转好也一样快 ⇒ 时间花在指令条数上。
/// 这里每次取 QT = 64 行 q：每块 K、V 从 DDR 读一次给 64 行用，VU 每条指令处理 64 × 128 个分数，指令条数约降到 1 / 4。
///
/// ```text
///   每头、每 64 行 q（两核各一半）：lq [64, D] 读进 TU
///   第一遍（每块 128 个 key）：s = bf16(q · Kᵀ块) · 1/√scale + 掩码（bf16）；m = max(m, 行最大)
///   第二遍（每块）：p = exp(s − m)（bf16，近似指令）；l += 行和（f32）；4 个 16 行的累加器 accᵢ += pᵢ · V块（TU 里各接各的串）
///   o = bf16(acc) / bf16(l)（4 个累加器先拼成 [64, D]，整块除）
/// ```
/// 数值口径和 calblas `attention!` 一样（分数与 p 舍 bf16、行和 f32、最后 bf16 里除），只是一次 64 行。
/// 试过的别的写法（2026-11-14 探针）：一遍走完（online softmax）要每块按行缩放 acc，f32 的 `exp_approx` 在 [16, 128] 上就把 VB 用光，
/// 头维 160 时 f32 acc 一行 640 B，按行乘标量不收（一行要 ≤ 512 B 或 512 B 的倍数）⇒ 回到两遍法、累加器留在 TU 里。
///
/// 代码量：docc 把 ≤ 64 轮的循环整个展开，64² 时两遍各 37 块约 350 KB。想把两遍合成一层 74 轮的循环（就不展开了），
/// docc 不收「运行时条件的 if 里有累加串」⇒ 照旧展开，UNet 那边把 Transformer 拆成三个单元好切段。
/// K 按原样 `[TK, H · D]` 读（同 calblas 的缺省：[D, 128] 块隔 H · D 取）：先用 `transpose2d!` 转好再读真卡上一样快，转置 kernel 还要 150–260 KB 代码。
///
/// 排布：q、o、k、v 都是 `[T, H · D]`（带边框的图当 token；交叉注意力的 k、v 是 `[TK, H · D]`），
/// mask 是 bf16 `[TK]`（0 或 −MAX，主机给：边框、补齐的 key 去掉）。D 是 32 的倍数（头维补零，`scale` 给真头维），T 是 64 的倍数。
/// 守着它的：模拟器 `cvxsim.py` 的 `_k_flash_attn`、UNet 整条对参照（`tools/unet.py sim / board`）。
///
/// 选项（DESIGN D23）：`qt = 64 | 128` ⇒ 一次取几行 q（缺省 64；256 时 VB 放不下）；`mask = none` ⇒ 不带掩码（token 是稠密的、没有补齐的 key 时；签名里没有 mask 参数）；
/// `q = prescaled` ⇒ 1/√scale 已经乘进 q 的投影权重（主机打包时），kernel 里不再乘。两样各省每块一条 VU 指令。
#[macro_export]
macro_rules! flash_attn {
    ($vis:vis $name:ident: rows = $t:expr, keys = $tk:expr, heads = $h:expr, head = $d:expr, scale = $sd:expr $(, mask = $mk:ident)? $(, q = $pre:ident)? $(, qt = $qt:tt)?) => {
        $crate::flash_attn!(@qt [$($qt)? 64] [$($mk)? keys] [$($pre)? scaled] $vis $name, $t, $tk, $h, $d, $sd);
    };
    // 一次取几行 q：64 ⇒ 4 个 16 行的累加器；128 ⇒ 8 个（累加器、给 TU 的 lhs 的名字在这里写好传下去：macro_rules 的卫生规则）
    (@qt [64 $($_q:tt)?] $($rest:tt)*) => {
        $crate::flash_attn!(@fn 64, [(a0, l0, 0) (a1, l1, 16) (a2, l2, 32) (a3, l3, 48)] $($rest)*);
    };
    (@qt [128 $($_q:tt)?] $($rest:tt)*) => {
        $crate::flash_attn!(@fn 128, [(a0, l0, 0) (a1, l1, 16) (a2, l2, 32) (a3, l3, 48) (a4, l4, 64) (a5, l5, 80) (a6, l6, 96) (a7, l7, 112)] $($rest)*);
    };
    (@fn $qt:tt, $accs:tt [keys $($_m:ident)?] $pre:tt $vis:vis $name:ident, $($a:tt)*) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(q: npu::Ptr<npu::Ddr, npu::bf16>, k: npu::Ptr<npu::Ddr, npu::bf16>, v: npu::Ptr<npu::Ddr, npu::bf16>, mask: npu::Ptr<npu::Ddr, npu::bf16>, o: npu::Ptr<npu::Ddr, npu::bf16>) {
            $crate::flash_attn!(@body $qt, $accs [keys mask] $pre q, k, v, o, $($a)*);
        }
    };
    (@fn $qt:tt, $accs:tt [none $($_m:ident)?] $pre:tt $vis:vis $name:ident, $($a:tt)*) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(q: npu::Ptr<npu::Ddr, npu::bf16>, k: npu::Ptr<npu::Ddr, npu::bf16>, v: npu::Ptr<npu::Ddr, npu::bf16>, o: npu::Ptr<npu::Ddr, npu::bf16>) {
            $crate::flash_attn!(@body $qt, $accs [none _nomask] $pre q, k, v, o, $($a)*);
        }
    };
    // 掩码：keys ⇒ 每块从 DDR 读一行、加上；none ⇒ 什么都不做
    // （局部变量名从调用处传进来：macro_rules 的卫生规则下，一个展开里 let 的名字另一个展开看不见）
    (@mdecl [keys $mask:ident], $gm:ident, $mk:ident) => {
        let $gm = $crate::calblas::rd!($mask, [NJ, KB]);
        let $mk = vcache::<bf16, _>([1, KB]);
    };
    (@mdecl [none $mask:ident], $gm:ident, $mk:ident) => {};
    (@madd [keys $mask:ident], $gm:ident, $mk:ident, $sb:ident, $kb:ident) => {
        $mk.load($gm.sub([$kb, 0], [1, KB]));
        $sb.add($sb, bcast($mk, &[0, 1]));
    };
    (@madd [none $mask:ident], $gm:ident, $mk:ident, $sb:ident, $kb:ident) => {};
    // 第一遍（只求行最大）：不带掩码、q 已经缩放过时直接在 f32 的分数上求最大，省掉 f32 ⇒ bf16 ⇒ f32 两次转换（D23）。
    // 行最大只是 exp 的稳定项：用没舍 bf16 的分数求，和第二遍舍过的分数最多差半个 bf16 ULP，p 最大到约 1.004，不影响归一。
    (@pass1 [none $($_m:ident)?] [prescaled $($_p:ident)?], [$sf:ident, $sb:ident, $tf:ident, $top:ident, $m:ident], $gm:ident, $mk:ident, $kb:ident) => {
        $top.reduce(Combine::Max, $sf, 1);
        $m.max($m, $top);
    };
    (@pass1 $mkk:tt $pre:tt, [$sf:ident, $sb:ident, $tf:ident, $top:ident, $m:ident], $gm:ident, $mk:ident, $kb:ident) => {
        $sb.cvt($sf, Cvt::default());
        $crate::flash_attn!(@scale $pre, $sb);
        $crate::flash_attn!(@madd $mkk, $gm, $mk, $sb, $kb);
        $tf.cvt($sb, Cvt::default());
        $top.reduce(Combine::Max, $tf, 1);
        $m.max($m, $top);
    };
    // 缩放：scaled ⇒ 乘 1/√scale；prescaled ⇒ 已经乘进 q
    (@scale [scaled $($_p:ident)?], $sb:ident) => { $sb.mul($sb, SCALE); };
    (@scale [prescaled $($_p:ident)?], $sb:ident) => {};
    (@body $qt:tt, [$(($acc:ident, $lhs:ident, $r:expr))*] $mk:tt $pre:tt $q:ident, $k:ident, $v:ident, $o:ident, $t:expr, $tk:expr, $h:expr, $d:expr, $sd:expr) => {{
        use npu::prelude::*;
        let (q, k, v, o) = ($q, $k, $v, $o);
        const T: usize = $t;
        const TK: usize = $tk;
        const H: usize = $h;
        const D: usize = $d;
        const QT: usize = $qt;
        const KB: usize = 128;
        const NJ: usize = TK / KB;
        const NQ: usize = T / QT;
        const NEG: f32 = -3.0e38;
        const SCALE: f32 = $crate::calblas::HeadDim($sd).inv_sqrt();
        const _: () = assert!(T % QT == 0 && TK % KB == 0 && D % 32 == 0,
            "flash_attn：q 行数是 QT 的倍数、key 行数是 128 的倍数、头维补到 32 的倍数");
        // q、o：[T · H, QT, D]（行距 [D, H · D, 1]）的第 qt · QT · H + h 块 = 第 qt 块 QT 行、第 h 头（同 calblas attention! 的 @at，16 换成 QT）
        let gq = q.bind_strided::<npu::Read, 3>([T * H, QT, D], [D, H * D, 1]);
        let go = o.bind_strided::<npu::Write, 3>([T * H, QT, D], [D, H * D, 1]);
        // Kᵀ：[头, d, key]（d 行距 1、key 行距 H · D）⇒ rhs [D, 128 key]（同 calblas attention! 的缺省）
        let gk = k.bind_strided::<npu::Read, 3>([H, D, TK], [D, 1, H * D]);
        let gv = $crate::calblas::rd!(v, [TK, H * D]);
        // 掩码每块一行，每块从 DDR 读一行（同 calblas attention! 的 mask = keys）
        $crate::flash_attn!(@mdecl $mk, gm, mk);
        let (lq, rk, sacc) = (mcache::<bf16, _>([QT, D]), mcache::<bf16, _>([D, KB]), mcache::<f32, _>([QT, KB]));
        let rv = mcache::<bf16, _>([KB, D]);
        $(let $lhs = mcache::<bf16, _>([16, KB]);)*
        let (sf, sb, tf) = (vcache::<f32, _>([QT, KB]), vcache::<bf16, _>([QT, KB]), vcache::<f32, _>([QT, KB]));
        let (m, top, mb) = (vcache::<f32, _>([QT, 1]), vcache::<f32, _>([QT, 1]), vcache::<bf16, _>([QT, 1]));
        let (l, rs, lb) = (vcache::<f32, _>([QT, 1]), vcache::<f32, _>([QT, 1]), vcache::<bf16, _>([QT, 1]));
        // 收尾：各累加器取出来拼成一整块 [QT, D]，再整块除以 l（⛔ 不对 [N, 1] 的向量按行切了再 bcast：DESIGN F28）
        let (out, oall, ob) = (vcache::<f32, _>([16, D]), vcache::<f32, _>([QT, D]), vcache::<bf16, _>([QT, D]));
        // 头 × q 块合成一层循环，两核各一半；key 块在里面（docc 展开）
        const HALF: usize = $crate::calblas::Halves(H * NQ).half();
        let mine = core_rank() * HALF;
        for i in 0..HALF {
            let j = mine + i;
            let (h, qt) = (j / NQ, j % NQ);
            lq.load(gq.sub([qt * QT * H + h, 0, 0], [1, QT, D]));
            // 第一遍：行最大
            m.fill(NEG);
            for kb in 0..NJ {
                rk.load(gk.sub([h, 0, kb * KB], [1, D, KB]));
                sacc.fill(0.0f32);
                sacc.mma(lq, rk);
                sf.from_m(sacc);
                $crate::flash_attn!(@pass1 $mk $pre, [sf, sb, tf, top, m], gm, mk, kb);
            }
            // 第二遍：p = exp(s − m)、行和、每 16 行一个累加器各接各的串（docc 要一块累加器的用处都在它填 0 的那个块里）
            mb.cvt(m, Cvt::default());
            l.fill(0.0f32);
            $(let $acc = mcache::<f32, _>([16, D]);)*
            $($acc.fill(0.0f32);)*
            for kb in 0..NJ {
                rk.load(gk.sub([h, 0, kb * KB], [1, D, KB]));
                rv.load(gv.sub([kb * KB, h * D], [KB, D]));
                sacc.fill(0.0f32);
                sacc.mma(lq, rk);
                sf.from_m(sacc);
                sb.cvt(sf, Cvt::default());
                $crate::flash_attn!(@scale $pre, sb);
                $crate::flash_attn!(@madd $mk, gm, mk, sb, kb);
                sb.sub(sb, bcast(mb, &[0, 1]));
                sb.exp_approx(sb);
                tf.cvt(sb, Cvt::default());
                rs.reduce(Combine::Sum, tf, 1);
                l.add(l, rs);
                $(
                    $lhs.load_v(sb.rows($r, 16));
                    $acc.mma($lhs, rv);
                )*
            }
            // o = bf16(acc) / bf16(l)：拼成 [QT, D] 再整块除、整块写回
            $(
                out.from_m($acc);
                oall.rows($r, 16).copy(out);
            )*
            ob.cvt(oall, Cvt::default());
            lb.cvt(l, Cvt::default());
            ob.div(ob, bcast(lb, &[0, 1]));
            ob.store(go.sub([qt * QT * H + h, 0, 0], [1, QT, D]));
        }
    }};
}
