//! 时间步嵌入（新写；DESIGN D7）：diffusers `get_timestep_embedding(t, 2·half, flip_sin_to_cos = True, downscale_freq_shift = 0)`，
//! 即 `y = [cos(t · fᵢ) … , sin(t · fᵢ) …]`，fᵢ = exp(−ln(10000) · i / half)，i < half（SD1.5 的 UNet：half = 160 ⇒ 320 维）。
//!
//! 为什么在设备上算：ComfyUI 的采样器由 sigma 插出时间步，可能是小数（不是 0…999 的整数），打包时算好一张整数表不够用；
//! 去噪循环放进设备程序以后（D2），每一步的 t 从输入表里取，这里现算。
//!
//! 排法（docc 0.4.1 逼出来的）：VU 拿「VB 里的标量」（立即数也算）和向量做运算时，向量要在一行 512 B 以内或整行的倍数；
//! 320 个 f32 两样都不是 ⇒ 补到 N = ⌈2·half / 128⌉ · 128 个（SD1.5：384 = 3 整行），cos、sin 两半放在同一块里一起算：
//! `a = t · F + P`，`y = cos(a)`，主机打两张常数表（模型常数，和权重一样打进 @w，不算 CPU 承担的计算）：
//!   F = [f₀ … f_{half−1}, f₀ … f_{half−1}, 0 …]，P = [0 …（half 个）, −π/2 …（half 个）, 0 …] ⇒ 后一半 cos(t·f − π/2) = sin(t·f)。
//! 补的那几个输出是 cos(0) = 1，不用。
//!
//! 数值：t · F 在 f32 里乘（同 torch）；t 到 999 时角度到 999 弧度，直接取 cos 会把 f32 的相对误差放大成相位误差 ⇒
//! 先做 Cody–Waite 两段 2π 规约：k = round(a / 2π)，r = (a − k·C1) − k·C2（C1 = 6.28125 只有 8 位尾数，|k| < 2¹⁵ 时 k·C1 精确；
//! round 用魔数加减做：docc 0.4.1 没接 vec.round），r ∈ [−π, π] 再走 VU 的 `cos_approx`（硬件近似，上板用 `ref/ktest.py board temb` 标定），
//! 最后舍成 bf16。sin 那一半多一次 −π/2 的 f32 加法：t = 999 时相位多差 ≤ 3e-5 弧度，远小于 bf16 的 ULP；
//! t = 0 时 sin 那一半是 cos(−π/2 的 f32) ≈ −4.4e-8 而不是精确的 0（绝对差可忽略，后面接线性层）。
//!
//! 参数：`t`：32 B（8 份同一个 f32，主机写；VU 取单个标量要 32 B 一行，同 calblas `stage_scalars!`）；`fp`：F 与 P 两张 `[N]` f32（先 F 后 P）；
//! `y`：`[N]` bf16（前 2·half 个有用）。单核 kernel、不分行：整步程序里包在 `if core_rank() == 0` 里，后面跟 `sync_cores()`。
//! 守着它的测试：`ref/ktest.py temb`（t = 0、1、999、小数；期望是 diffusers 的 f32 参照舍 bf16）。

/// 补齐后的元素数（128 个 f32 = 512 B 一整行）。
#[must_use]
pub const fn temb_len(half: usize) -> usize {
    (2 * half).div_ceil(128) * 128
}

/// 见模块文档。
#[macro_export]
macro_rules! timestep_embed {
    ($vis:vis $name:ident: half = $half:expr) => {
        #[npu::kernel(numerics = "ks1")]
        $vis fn $name(t: npu::Ptr<npu::Ddr, f32>, fp: npu::Ptr<npu::Ddr, f32>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const N: usize = $crate::temb_len($half);
            const NR: usize = N / 128;
            const INV_2PI: f32 = 0.159_154_94;
            const C1: f32 = 6.281_25;
            const C2: f32 = 0.001_935_307_2; // 2π − C1
            const MAGIC: f32 = 12_582_912.0; // 1.5 · 2²³：(x + M) − M 就近舍到整数（|x| < 2²²）
            let t8 = vcache::<f32, _>([1, 8]);
            t8.load($crate::calblas::rd!(t, [1, 8]));
            let t1 = vcache::<f32, _>([1, 1]);
            t1.reduce(Combine::Max, t8, 1);
            let gfp = $crate::calblas::rd!(fp, [2 * NR, 128]);
            let (a, p, tb) = (vcache::<f32, _>([NR, 128]), vcache::<f32, _>([NR, 128]), vcache::<f32, _>([NR, 128]));
            a.load(gfp.sub([0, 0], [NR, 128]));
            p.load(gfp.sub([NR, 0], [NR, 128]));
            tb.broadcast(t1);
            a.mul(a, tb);
            a.add(a, p);
            let (k, r, q) = (vcache::<f32, _>([NR, 128]), vcache::<f32, _>([NR, 128]), vcache::<f32, _>([NR, 128]));
            k.mul(a, INV_2PI);
            k.add(k, MAGIC);
            k.sub(k, MAGIC);
            r.mul(k, C1);
            r.sub(a, r);
            q.mul(k, C2);
            r.sub(r, q);
            r.cos_approx(r);
            let yb = vcache::<bf16, _>([NR, 128]);
            yb.cvt(r, Cvt::default());
            yb.store($crate::calblas::wr!(y, [NR, 128]));
        }
    };
}
