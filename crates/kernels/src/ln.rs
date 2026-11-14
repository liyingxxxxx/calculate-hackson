//! NHWC 原生的 LayerNorm：每个像素沿通道归一（UNet 的 Transformer 块里 norm1 / norm2 / norm3，DESIGN D18），直接在带边框的图上算，只写内部（新写）。
//!
//! ```text
//!   每颗核按行分（calblas halves!），每块 PX 个像素 [PX, C / 64, 64] 读进 VB、转 f32
//!   ⇒ 先沿段（第 1 维）、再沿最后一维（64 个）求和 ⇒ 每像素 mean ⇒ x − mean ⇒ 同样求 Σ(x − mean)² ⇒ var ⇒ rsqrt(var + eps)（近似指令）
//!   ⇒ · rstd · w + b（w、b 是 f32 [C]）⇒ bf16 ⇒ 写回输出图的内部（边框不写，仍是 0）
//! ```
//! 为什么不用 calblas `layer_norm!`：它按稠密的行读写整张图（带边框的图会把边框也归一成 b，不再是 0），而且 64² × 320 通道时 docc 不收
//! （探针 2026-11-14：「多行的块、最内一维有运行期的 valid，要逐行」）。数值口径同它：两遍法、f32 里算、出口舍一次 bf16。
//! 排布同 `gn_stats!`：x、y 是 `[H + 4][W + 4][P]` bf16 的图，前 C 个通道，C 是 64 的倍数。单核 kernel、按 `num_cores()` 分行 ⇒ 两核都调。
//! 守着它的：模拟器 `cvxsim.py` 的 `_k_ln_nhwc`（对 torch 的 LayerNorm）、UNet 的独立参照比对。

/// LayerNorm（每像素沿通道）：见模块文档。
#[macro_export]
macro_rules! ln_nhwc {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, c = $c:expr, p = $p:expr, eps = $eps:expr) => {
        #[npu::kernel(numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, wv: npu::Ptr<npu::Ddr, f32>, bv: npu::Ptr<npu::Ddr, f32>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const B: usize = $crate::calblas::kernels::BORDER;
            const H: usize = $h;
            const W: usize = $w;
            const C: usize = $c;
            const P: usize = $p;
            const CC: usize = $crate::GN_CC;
            const NCC: usize = C / CC;
            const WI: usize = W + 2 * B;
            const PX: usize = $crate::gn_px(W, C);
            const NCH: usize = W / PX;
            const INV_C: f32 = 1.0 / C as f32;
            const _: () = assert!(C % CC == 0 && C <= P, "ln_nhwc：通道数是 64 的倍数");
            let wf = $crate::calblas::stage!($crate::calblas::rd!(wv, [1, NCC, CC]), [1, NCC, CC]);
            let bf = $crate::calblas::stage!($crate::calblas::rd!(bv, [1, NCC, CC]), [1, NCC, CC]);
            let gx = x.add((B * WI + B) * P * 2).bind_strided::<npu::Read, 3>([H * WI, NCC, CC], [P, CC, 1]);
            let gy = y.add((B * WI + B) * P * 2).bind_strided::<npu::Write, 3>([H * WI, NCC, CC], [P, CC, 1]);
            let (hb, f, t) = (vcache::<bf16, _>([PX, NCC, CC]), vcache::<f32, _>([PX, NCC, CC]), vcache::<f32, _>([PX, NCC, CC]));
            // 先沿第 1 维（段）再沿最后一维（64 个）：docc 沿最后一维的归约要 8 的倍数，段数（320 通道是 5 段）不一定是
            let (s, m) = (vcache::<f32, _>([PX, 1, CC]), vcache::<f32, _>([PX, 1, 1]));
            let (h0, hn) = $crate::calblas::halves!(H);
            for j in 0..hn * NCH {
                let oy = h0 + j / NCH;
                let x0 = (j % NCH) * PX;
                hb.load(gx.sub([oy * WI + x0, 0, 0], [PX, NCC, CC]));
                f.cvt(hb, Cvt::default());
                s.reduce(Combine::Sum, f, 1);
                m.reduce(Combine::Sum, s, 2);
                m.mul(m, INV_C);
                f.sub(f, bcast(m, &[0, 1, 2]));
                t.mul(f, f);
                s.reduce(Combine::Sum, t, 1);
                m.reduce(Combine::Sum, s, 2);
                m.mul(m, INV_C);
                m.add(m, $eps);
                m.rsqrt_approx(m);
                f.mul(f, bcast(m, &[0, 1, 2]));
                f.mul(f, bcast(wf, &[0, 1, 2]));
                f.add(f, bcast(bf, &[0, 1, 2]));
                hb.cvt(f, Cvt::default());
                hb.store(gy.sub([oy * WI + x0, 0, 0], [PX, NCC, CC]));
            }
        }
    };
}
