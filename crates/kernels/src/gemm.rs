//! 多行的矩阵乘 `gemm_rows!`（新写，DESIGN D25）：`y = x · Wᵀ (+ bias)`，W 是 HF 原样 `[N][K]` bf16，同 calblas `linear!(w = hf)`
//! （UNet 的宽通道卷积 / 投影走这条路，D22），只是**一块权重读一次给 MG 块 16 行用**：
//!
//! ```text
//!   输出按 [MG · 16 行, 128 列] 一组（两核各一半的组）：MG 个累加器各 [16, 128] f32
//!   k 每 128 一步：rhs = W 的 [128 k, 128 n] 块读一次 ⇒ MG 块 lhs（各 [16, 128]，从 x 读）各乘一次、各接各的串
//!   收尾（每块）：同 linear! —— first 写部分和 p、mid 加进 p、last 加上 p 再 + bias ⇒ bf16 ⇒ 写 y
//! ```
//! 为什么：`linear!` 每块 16 行把整条 K 的权重从 DDR 读一遍（真卡 1.0–1.5 TFLOPS，UNet 一步 0.79 s），
//! 权重的搬运是 lhs 的 8 倍；MG 块共用一次 rhs，搬运约降到 1/3。数值和 `linear!` 逐位相同（同一个累加次序：k 从小到大，f32）。
//! 约束：R 是 16 · MG 的倍数，K、N 是 128 的倍数，K / 128 ≤ 64（k 步在体里展开），组数 (N / 128) · (R / 16 / MG) 是偶数（两核各一半）。
//! 守着它的：模拟器 `cvxsim.py` 的 `_k_gemm_rows`（同 linear!）、单 kernel 上板对 numpy、UNet 整条对参照。
#[macro_export]
macro_rules! gemm_rows {
    ($vis:vis $name:ident: rows = $r:expr, k = $k:expr, n = $n:expr, mg = $mg:tt $(, bias = $bt:ident)? $(, stage = $stage:ident)?) => {
        $crate::gemm_rows!(@mg [$mg] [$($stage)? full] [$($bt)? none] $vis $name, $r, $k, $n);
    };
    // 每组几块 16 行：累加器与 lhs 的名字在这里写好传下去（macro_rules 的卫生规则）
    (@mg [2] $($rest:tt)*) => { $crate::gemm_rows!(@fn 2, [(a0, l0, 0) (a1, l1, 1)] $($rest)*); };
    (@mg [3] $($rest:tt)*) => { $crate::gemm_rows!(@fn 3, [(a0, l0, 0) (a1, l1, 1) (a2, l2, 2)] $($rest)*); };
    (@mg [4] $($rest:tt)*) => { $crate::gemm_rows!(@fn 4, [(a0, l0, 0) (a1, l1, 1) (a2, l2, 2) (a3, l3, 3)] $($rest)*); };
    (@mg [6] $($rest:tt)*) => { $crate::gemm_rows!(@fn 6, [(a0, l0, 0) (a1, l1, 1) (a2, l2, 2) (a3, l3, 3) (a4, l4, 4) (a5, l5, 5)] $($rest)*); };
    (@mg [8] $($rest:tt)*) => {
        $crate::gemm_rows!(@fn 8, [(a0, l0, 0) (a1, l1, 1) (a2, l2, 2) (a3, l3, 3) (a4, l4, 4) (a5, l5, 5) (a6, l6, 6) (a7, l7, 7)] $($rest)*);
    };
    // 签名：只带这一段用得到的参数（同 linear! 的多行版）
    (@fn $mg:tt, $accs:tt [full $($_s:ident)?] [none $($_b:ident)?] $vis:vis $name:ident, $($a:tt)*) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, w: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            $crate::gemm_rows!(@body $mg, $accs, full, none, [x, w, _b, _p, y], $($a)*);
        }
    };
    (@fn $mg:tt, $accs:tt [full $($_s:ident)?] [f32 $($_b:ident)?] $vis:vis $name:ident, $($a:tt)*) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, w: npu::Ptr<npu::Ddr, npu::bf16>, b: npu::Ptr<npu::Ddr, f32>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            $crate::gemm_rows!(@body $mg, $accs, full, f32, [x, w, b, _p, y], $($a)*);
        }
    };
    (@fn $mg:tt, $accs:tt [first $($_s:ident)?] $bt:tt $vis:vis $name:ident, $($a:tt)*) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, w: npu::Ptr<npu::Ddr, npu::bf16>, p: npu::Ptr<npu::Ddr, f32>) {
            $crate::gemm_rows!(@body $mg, $accs, first, none, [x, w, _b, p, _y], $($a)*);
        }
    };
    (@fn $mg:tt, $accs:tt [mid $($_s:ident)?] $bt:tt $vis:vis $name:ident, $($a:tt)*) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, w: npu::Ptr<npu::Ddr, npu::bf16>, p: npu::Ptr<npu::Ddr, f32>) {
            $crate::gemm_rows!(@body $mg, $accs, mid, none, [x, w, _b, p, _y], $($a)*);
        }
    };
    (@fn $mg:tt, $accs:tt [last $($_s:ident)?] [none $($_b:ident)?] $vis:vis $name:ident, $($a:tt)*) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, w: npu::Ptr<npu::Ddr, npu::bf16>, p: npu::Ptr<npu::Ddr, f32>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            $crate::gemm_rows!(@body $mg, $accs, last, none, [x, w, _b, p, y], $($a)*);
        }
    };
    (@fn $mg:tt, $accs:tt [last $($_s:ident)?] [f32 $($_b:ident)?] $vis:vis $name:ident, $($a:tt)*) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(
            x: npu::Ptr<npu::Ddr, npu::bf16>,
            w: npu::Ptr<npu::Ddr, npu::bf16>,
            b: npu::Ptr<npu::Ddr, f32>,
            p: npu::Ptr<npu::Ddr, f32>,
            y: npu::Ptr<npu::Ddr, npu::bf16>,
        ) {
            $crate::gemm_rows!(@body $mg, $accs, last, f32, [x, w, b, p, y], $($a)*);
        }
    };
    (@body $mg:tt, [$(($acc:ident, $lhs:ident, $i:expr))*], $stage:ident, $bt:ident, [$x:ident, $w:ident, $b:ident, $p:ident, $y:ident], $r:expr, $k:expr, $n:expr) => {{
        use npu::prelude::*;
        const R: usize = $r;
        const K: usize = $k;
        const N: usize = $n;
        const M: usize = 16;
        const B: usize = 128;
        const KC: usize = 128;
        const MG: usize = $mg;
        const NG: usize = R / (M * MG);
        const _: () = assert!(R % (M * MG) == 0 && K % KC == 0 && N % B == 0 && K / KC <= 64 && ((N / B) * NG) % 2 == 0,
            "gemm_rows：行数是 16 · MG 的倍数、K / N 是 128 的倍数、K ≤ 8192、组数是偶数");
        let gx = $crate::calblas::rd!($x, [R, K]);
        // W：HF [N][K] ⇒ 第 nb 块 n、第 kc 段 k 的 rhs [128 k, 128 n]（k 连续，同 linear! 的多行版）
        let gw = $w.bind_strided::<npu::Read, 3>([N / B, K, B], [B * K, 1, K]);
        let rhs = mcache::<bf16, _>([KC, B]);
        $(let $lhs = mcache::<bf16, _>([M, KC]);)*
        let vo = vcache::<f32, _>([M, B]);
        $crate::gemm_rows!(@setup $stage $bt, [bf, yh, pv]);
        // 组 × n 块合成一层循环（n 块在外：相邻两组用同一块 W 的 n 列），两核各一半
        const HALF: usize = $crate::calblas::Halves((N / B) * NG).half();
        let mine = core_rank() * HALF;
        for i in 0..HALF {
            let j = mine + i;
            let (nb, g) = (j / NG, j % NG);
            $(let $acc = mcache::<f32, _>([M, B]);)*
            $($acc.fill(0.0f32);)*
            for kc in 0..K / KC {
                rhs.load(gw.sub([nb, kc * KC, 0], [1, KC, B]));
                $(
                    $lhs.load(gx.sub([(g * MG + $i) * M, kc * KC], [M, KC]));
                    $acc.mma($lhs, rhs);
                )*
            }
            $(
                vo.from_m($acc);
                $crate::gemm_rows!(@end $stage $bt, [vo, $b, $p, $y, nb, (g * MG + $i) * M * (N / B) + nb], [bf, yh, pv]);
            )*
        }
    }};
    // 各段要的临时块（同 linear! 的 @hf_setup）
    (@setup full none, [$bf:ident, $yh:ident, $pv:ident]) => { let $yh = vcache::<bf16, _>([M, B]); };
    (@setup full f32, [$bf:ident, $yh:ident, $pv:ident]) => { let ($bf, $yh) = (vcache::<f32, _>([1, B]), vcache::<bf16, _>([M, B])); };
    (@setup first $bt:ident, $v:tt) => {};
    (@setup mid $bt:ident, [$bf:ident, $yh:ident, $pv:ident]) => { let $pv = vcache::<f32, _>([M, B]); };
    (@setup last $bt:ident, [$bf:ident, $yh:ident, $pv:ident]) => {
        $crate::gemm_rows!(@setup full $bt, [$bf, $yh, $pv]);
        let $pv = vcache::<f32, _>([M, B]);
    };
    // 输出块的视图：[R·N/B, M, B]（行距 [B, N, 1]）的第 at 块（同 linear! 的 @hf_blk）
    (@blk $t:ident, $a:ident, $at:expr) => {
        $t.bind_strided::<npu::$a, 3>([R * N / B, M, B], [B, N, 1]).sub([$at, 0, 0], [1, M, B])
    };
    // 收尾（同 linear! 的 @hf_end）：first 写部分和、mid 加进部分和、last 加上部分和再走 full；full = + bias ⇒ bf16 ⇒ 写 y
    (@end first $bt:ident, [$vo:ident, $b:ident, $p:ident, $y:ident, $nb:ident, $at:expr], $v:tt) => {
        $vo.store($crate::gemm_rows!(@blk $p, Write, $at));
    };
    (@end mid $bt:ident, [$vo:ident, $b:ident, $p:ident, $y:ident, $nb:ident, $at:expr], [$bf:ident, $yh:ident, $pv:ident]) => {{
        $pv.load($crate::gemm_rows!(@blk $p, Read, $at));
        $vo.add($vo, $pv);
        $vo.store($crate::gemm_rows!(@blk $p, Write, $at));
    }};
    (@end last $bt:ident, [$vo:ident, $b:ident, $p:ident, $y:ident, $nb:ident, $at:expr], [$bf:ident, $yh:ident, $pv:ident]) => {{
        $pv.load($crate::gemm_rows!(@blk $p, Read, $at));
        $vo.add($vo, $pv);
        $crate::gemm_rows!(@end full $bt, [$vo, $b, $p, $y, $nb, $at], [$bf, $yh, $pv]);
    }};
    (@end full none, [$vo:ident, $b:ident, $p:ident, $y:ident, $nb:ident, $at:expr], [$bf:ident, $yh:ident, $pv:ident]) => {{
        $yh.cvt($vo, Cvt::default());
        $yh.store($crate::gemm_rows!(@blk $y, Write, $at));
    }};
    (@end full f32, [$vo:ident, $b:ident, $p:ident, $y:ident, $nb:ident, $at:expr], [$bf:ident, $yh:ident, $pv:ident]) => {{
        $bf.load($crate::calblas::rd!($b, [N / B, B]).sub([$nb, 0], [1, B]));
        $vo.add($vo, bcast($bf, &[0, 1]));
        $yh.cvt($vo, Cvt::default());
        $yh.store($crate::gemm_rows!(@blk $y, Write, $at));
    }};
}
