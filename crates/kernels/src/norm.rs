//! GroupNorm（NCHW）：**抄自 calblas 0.1.0 的 `group_norm!`**（`src/kernels/vision.rs` 第 523–600 行，sha256 218ca5a4…），
//! 只加了一个可选参数 `parts = N`（一组切几段进 VB），别的逐字不动（细账见仓库根的 VENDORED.md）。
//!
//! 为什么要它（DESIGN F22）：calblas 按「VB 放得下的最少段数」切。一组小的时候（64² × 512：4 段；128² × 512：16 段），
//! docc 0.4.1 会把「每核 16 组」的外层循环也整个展开，代码分别是 292 KB 与约 1.1 MB；后者一个 kernel 就超出 RISC-V 跳转的 ±1 MB，
//! 任何程序里都编不过。切成 64 段时（256²、512² 本来就是这样）外层不展开，代码约 112 KB。段数只改每次读进 VB 的块大小，
//! 均值 / 方差 / 归一的算法、数值口径、w 与 b 的展开格式（`[C · HW / CC][8]` f32）都和 calblas 那份一样。
//! 守着它的测试：`ref/ktest.py` 的 GroupNorm 用例（期望是 torch 的 `F.group_norm`）、模拟器 `cvxsim.py` 的 `_k_group_norm`。

#[macro_export]
macro_rules! group_norm {
    ($vis:vis $name:ident: hw = $hw:expr, c = $c:expr, groups = $g:expr, eps = $eps:expr, act = $act:ident $(, parts = $np:expr)?) => {
        #[npu::kernel(cores = 2, numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, w: npu::Ptr<npu::Ddr, f32>, b: npu::Ptr<npu::Ddr, f32>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const HW: usize = $hw;
            const C: usize = $c;
            const G: usize = $g;
            const CG: usize = C / G;
            const CC: usize = $crate::calblas::kernels::Cols::norm_cols(CG * HW, HW);
            const R: usize = CG * HW / CC;
            // 一组切几段进 VB（每段 bf16 一份、f32 两份，≤ 256 KB；段的行数仍是 8 的倍数）
            // （本仓加）`parts = N` 显式给段数；不给就是 calblas 原来的算法（VB 放得下的最少段数）
            const NH: usize = $crate::group_norm!(@parts R, CC $(, $np)?);
            const RP: usize = R / NH;
            const INV_N: f32 = 1.0 / (CG * HW) as f32;
            const _: () = assert!(C % G == 0 && HW % CC == 0 && RP % 8 == 0 && R % NH == 0, "group_norm：通道数整除组数，段数整除一组的行数，一段的行数要凑得成 8 的倍数");
            let gx = x.bind_strided::<npu::Read, 2>([C * HW / CC, CC], [CC, 1]);
            let gy = y.bind_strided::<npu::Write, 2>([C * HW / CC, CC], [CC, 1]);
            let (gw, gb) = ($crate::calblas::rd!(w, [C * HW / CC, 8]), $crate::calblas::rd!(b, [C * HW / CC, 8]));
            let (h, f, t) = (vcache::<bf16, _>([RP, CC]), vcache::<f32, _>([RP, CC]), vcache::<f32, _>([RP, CC]));
            let (s1, sp, s) = (vcache::<f32, _>([RP, 1]), vcache::<f32, _>([1, 1]), vcache::<f32, _>([1, 1]));
            let (w8, wv, bv) = (vcache::<f32, _>([RP, 8]), vcache::<f32, _>([RP, 1]), vcache::<f32, _>([RP, 1]));
            let (g0, gn) = $crate::calblas::halves!(G);
            for i in 0..gn {
                let r0 = (g0 + i) * R;
                // 第一遍：均值
                s.fill(0.0f32);
                for part in 0..NH {
                    h.load(gx.sub([r0 + part * RP, 0], [RP, CC]));
                    f.cvt(h, Cvt::default());
                    s1.reduce(Combine::Sum, f, 1);
                    sp.reduce(Combine::Sum, s1, 0);
                    s.add(s, sp);
                }
                s.mul(s, INV_N);
                // 第二遍：方差（两遍法：Σ (x − mean)²）
                let m = vcache::<f32, _>([1, 1]);
                m.copy(s);
                s.fill(0.0f32);
                for part in 0..NH {
                    h.load(gx.sub([r0 + part * RP, 0], [RP, CC]));
                    f.cvt(h, Cvt::default());
                    f.sub(f, bcast(m, &[0, 1]));
                    t.mul(f, f);
                    s1.reduce(Combine::Sum, t, 1);
                    sp.reduce(Combine::Sum, s1, 0);
                    s.add(s, sp);
                }
                s.mul(s, INV_N);
                s.add(s, $eps);
                s.rsqrt_approx(s);
                // 第三遍：归一、仿射、（silu）、写回
                for part in 0..NH {
                    let rp = r0 + part * RP;
                    h.load(gx.sub([rp, 0], [RP, CC]));
                    f.cvt(h, Cvt::default());
                    f.sub(f, bcast(m, &[0, 1]));
                    f.mul(f, bcast(s, &[0, 1]));
                    w8.load(gw.sub([rp, 0], [RP, 8]));
                    wv.reduce(Combine::Max, w8, 1);
                    w8.load(gb.sub([rp, 0], [RP, 8]));
                    bv.reduce(Combine::Max, w8, 1);
                    f.mul(f, bcast(wv, &[0, 1]));
                    f.add(f, bcast(bv, &[0, 1]));
                    h.cvt(f, Cvt::default());
                    $crate::group_norm!(@$act h);
                    h.store(gy.sub([rp, 0], [RP, CC]));
                }
            }
        }
    };
    (@silu $h:ident) => {{
        let tmp = vcache::<bf16, _>([RP, CC]);
        $crate::calblas::logistic!(silu, $h, $h, tmp);
    }};
    (@none $h:ident) => {};
    (@parts $r:ident, $cc:ident) => {
        $crate::calblas::kernels::Cols::norm_parts($r, $cc)
    };
    (@parts $r:ident, $cc:ident, $np:expr) => {
        $np
    };
}
