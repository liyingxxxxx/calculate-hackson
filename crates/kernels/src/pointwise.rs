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
