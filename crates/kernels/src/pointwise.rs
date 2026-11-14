//! 逐元素的小 kernel（新写）。向量看成 `[R, C]`（C ≤ 256，取法同 calblas 的 `unary!`：[`calblas::Line`]、[`calblas::Rows`]），
//! 每块 T 行读进 VB、算完写回；块按核分（[`calblas::halves!`]），所以**两核都调**、调完 `sync_cores()`。
//! 块首在 512 B 行首上（C = 256 时），满足 calblas README「逐元素的块首 64 B 对齐」。

/// `y = tanh(x / s) · s`（n 个 bf16，进出 bf16，中间在 f32 里算：÷ s 写成乘编译期的 1 / s，tanh 用 VU 的硬件近似 `tanh_approx`）。
/// TAESD 解码器开头的「软夹」`tanh(x / 3) · 3`（diffusers `DecoderTiny.forward`）。0 映到 0：特征图的边框、补的通道照样是 0。
///
/// 数值：tanh 是近似指令（KS1 的 exp / rsqrt 离精确 ≤ 1 ULP，tanh 没标定过 ⇒ 上板时用 `ref/ktest.py tanh` 标定，写进已知限制）。
/// 不能原地（x、y 同址时两核的块不重叠，其实不冲突，但模板一律不原地，免得以后改成整行读时出错）。
#[macro_export]
macro_rules! tanh_scale {
    ($vis:vis $name:ident: n = $n:expr, scale = $s:expr) => {
        #[npu::kernel(numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const N: usize = $n;
            const C: usize = $crate::calblas::Line(N).cols(256);
            const R: usize = N / C;
            const T: usize = $crate::calblas::Rows(R).chunk();
            const S: f32 = $s;
            const INV: f32 = 1.0 / S;
            const _: () = assert!(C == 256 && R % T == 0, "tanh_scale：元素数要是 256 的倍数（块首对齐 512 B）");
            let (gx, gy) = ($crate::calblas::rd!(x, [R, C]), $crate::calblas::wr!(y, [R, C]));
            let (vx, vf) = (vcache::<bf16, _>([T, C]), vcache::<f32, _>([T, C]));
            let (b0, bn) = $crate::calblas::halves!(R / T);
            for i in 0..bn {
                let at = (b0 + i) * T;
                vx.load(gx.sub([at, 0], [T, C]));
                vf.cvt(vx, Cvt::default());
                vf.mul(vf, INV);
                vf.tanh_approx(vf);
                vf.mul(vf, S);
                vx.cvt(vf, Cvt::default());
                vx.store(gy.sub([at, 0], [T, C]));
            }
        }
    };
}

/// GEGLU 的后半：`y = h · gelu(g)`（n 个 bf16，进出 bf16，中间在 f32 里算；gelu 是 torch 的精确版 x · Φ(x)，
/// Φ 用 Abramowitz–Stegun 7.1.26（erf 误差 ≤ 1.5e-7），同 calblas `act!(gelu_erf)` 的算式（VENDORED.md），exp、倒数是近似指令）。
/// UNet 的前馈层（diffusers `GEGLU`：`hidden, gate = proj(x).chunk(2)`，返回 `hidden · gelu(gate)`）：两个 1×1 卷积分别出 h、g，再走这一个。
/// 为什么不把 gelu 并进卷积的尾处理：卷积在 64² × 1280 通道时 VB 已经快满，gelu 的 f32 临时块放不下（探针 2026-11-14）。
/// h、g、y 是同形的整块（带边框的图也行：边框两边都是 0，gelu(0) · 0 = 0，边框还是 0）。两核都调、调完 `sync_cores()`；不能原地。
#[macro_export]
macro_rules! geglu {
    ($vis:vis $name:ident: n = $n:expr) => {
        #[npu::kernel(numerics = "ks1")]
        $vis fn $name(h: npu::Ptr<npu::Ddr, npu::bf16>, g: npu::Ptr<npu::Ddr, npu::bf16>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const N: usize = $n;
            const C: usize = $crate::calblas::Line(N).cols(256);
            const R: usize = N / C;
            const T: usize = $crate::calblas::Rows(R).chunk();
            const _: () = assert!(C == 256 && R % T == 0, "geglu：元素数要是 256 的倍数（块首对齐 512 B）");
            let (gh, gg, gy) = ($crate::calblas::rd!(h, [R, C]), $crate::calblas::rd!(g, [R, C]), $crate::calblas::wr!(y, [R, C]));
            let (vh, vg) = (vcache::<bf16, _>([T, C]), vcache::<bf16, _>([T, C]));
            let (f, a, t, p, e) = (vcache::<f32, _>([T, C]), vcache::<f32, _>([T, C]), vcache::<f32, _>([T, C]), vcache::<f32, _>([T, C]), vcache::<f32, _>([T, C]));
            let (b0, bn) = $crate::calblas::halves!(R / T);
            for i in 0..bn {
                let at = (b0 + i) * T;
                vg.load(gg.sub([at, 0], [T, C]));
                f.cvt(vg, Cvt::default());
                a.mul(f, -1.0f32);
                a.max(a, f);
                a.mul(a, core::f32::consts::FRAC_1_SQRT_2);
                t.mul(a, 0.327_591_1f32);
                t.add(t, 1.0f32);
                t.recip_approx(t);
                p.mul(t, 1.061_405_4f32);
                p.add(p, -1.453_152f32);
                p.mul(p, t);
                p.add(p, 1.421_413_7f32);
                p.mul(p, t);
                p.add(p, -0.284_496_74f32);
                p.mul(p, t);
                p.add(p, 0.254_829_6f32);
                p.mul(p, t);
                e.mul(a, a);
                e.mul(e, -1.0f32);
                e.exp_approx(e);
                p.mul(p, e);
                p.mul(p, 0.5f32);
                a.mul(f, 1.0e30f32);
                a.max(a, 0.0f32);
                a.min(a, 1.0f32);
                e.mul(p, -2.0f32);
                e.add(e, 1.0f32);
                t.mul(a, e);
                t.add(t, p);
                f.mul(f, t);
                vh.load(gh.sub([at, 0], [T, C]));
                a.cvt(vh, Cvt::default());
                f.mul(f, a);
                vg.cvt(f, Cvt::default());
                vg.store(gy.sub([at, 0], [T, C]));
            }
        }
    };
}
