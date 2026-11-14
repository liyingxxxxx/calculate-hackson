//! NHWC 原生的 GroupNorm（+ SiLU）：直接在卷积模板的带边框特征图上算，**不转排布**（新写，DESIGN D16）。
//!
//! 为什么要它：正式 VAE 原来走「`nhwc_to_nchw!` ⇒ calblas `group_norm!`（NCHW）⇒ `nchw_to_nhwc!`」三步（D13）。真卡计时（`tools/prof.py`）显示
//! 两次排布转换只有约 1.3 GB/s，512² × 256 通道一次就要 0.2 s，30 个 GN 的转换合计约 2.7 s，占整个解码器 5.2 s 的一半多（DESIGN F24）。
//! 这里把一个 GN 拆成两个 kernel、中间一条屏障，读两遍、写一遍：
//!
//! ```text
//!   gn_stats!：每颗核扫自己那一半的行（calblas halves!），每块 PX 个像素 [PX, C] 读进 VB、转 f32，沿像素（第 0 维）求和
//!              ⇒ 每通道的 Σx、Σx²（f32）⇒ st 的第 2·core、2·core + 1 行（st 是 [4][C] f32：两核各两行）
//!   ── sync_cores() ──
//!   gn_apply!：读 st、两核相加 ⇒ 每通道 Σx、Σx² ⇒ 乘组掩码 M [G, C] 再沿最后一维求和 ⇒ 每组的 mean、E[x²]（[G, 1]）
//!              ⇒ var = E[x²] − mean²（夹到 ≥ 0）⇒ rstd = rsqrt(var + eps)（近似指令）⇒ 乘 M 再沿第 0 维求和铺回每通道 ⇒ a = w · rstd、b' = b − mean · a
//!              ⇒ 每块 [PX, C]：f32(x) · a + b' ⇒ bf16 ⇒（silu）⇒ 写回输出图的内部（边框不写，仍是 0）
//! ```
//! 数值口径：出口同 calblas `group_norm!`（f32 里归一、仿射 ⇒ 舍 bf16 ⇒ SiLU 在 bf16 上），w、b 是 f32 `[C]`（不再按行展开）。
//! 不同的是方差：这里一遍求 Σx、Σx²（E[x²] − mean²），calblas 是两遍（Σ(x − mean)²）。f32 里 mean² 比 var 大很多时会损失精度，
//! VAE 的激活不是这种情形（模拟与 bf16 参照的比对守着；真卡上和 calblas 那一版比）。
//! 组内求和为什么用掩码：每组的通道数在 128 通道时只有 4，docc 0.4.1 沿最后一维归约要 8 的倍数、VB 里的转置也要列数是 8 的倍数
//! （探针 2026-11-14：「vec.transpose 只收 32 位元素、列数是 8 的倍数（[32, 4] f32）」）。M 的第 g 行在第 g 组的通道上是 1、其余是 0（主机给，f32），
//! 乘的是 0 / 1，和是精确的。
//! 排布：x、y 是 `[H + 4][W + 4][P]` bf16 的图，前 C 个通道；每组 C / G 个通道连着（同 torch 的 GroupNorm）。st 要 64 B 对齐、[4][C] f32；
//! w、b 是 f32 [C]，掩码 mk 是 f32 [G][C]。
//! 两个都是单核 kernel、按 `num_cores()` 分行 ⇒ **两核都调**；st 按两颗核排，所以程序要 `cores = 2`。
//! 守着它的：模拟器 `cvxsim.py` 的 `_k_gn_stats` / `_k_gn_apply`（对 torch 的 GroupNorm）、正式 VAE 的 6 段对参照。

/// GN 一块读几个像素：整除 W 的 2 的幂，`[PX, C]` 的 bf16 一份、f32 两份（10 B / 元素）放得进 192 KB（其余给掩码与每通道的量）；至多 W。
#[doc(hidden)]
#[must_use]
pub const fn gn_px(w: usize, c: usize) -> usize {
    let mut px = 1;
    while px * 2 <= w && w % (px * 2) == 0 && px * 2 * c * 10 <= 192 * 1024 {
        px *= 2;
    }
    px
}

/// GroupNorm 第一步：每颗核按通道累加自己那一半行的 Σx、Σx² ⇒ st 的第 2·core、2·core + 1 行。见模块文档。
#[macro_export]
macro_rules! gn_stats {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, c = $c:expr, p = $p:expr) => {
        #[npu::kernel(numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, st: npu::Ptr<npu::Ddr, f32>) {
            use npu::prelude::*;
            const B: usize = $crate::calblas::kernels::BORDER;
            const H: usize = $h;
            const W: usize = $w;
            const C: usize = $c;
            const P: usize = $p;
            const WI: usize = W + 2 * B;
            const PX: usize = $crate::gn_px(W, C);
            const NCH: usize = W / PX;
            const _: () = assert!(C % 32 == 0 && C <= P && PX >= 8, "gn_stats：通道数是 32 的倍数，一块至少 8 个像素");
            // 内部：第 oy 行第 x0 列起 PX 个像素 ⇒ 第 oy · WI + x0 行起 PX 行（行距 P）
            let gx = x.add((B * WI + B) * P * 2).bind_strided::<npu::Read, 2>([H * WI, P], [P, 1]);
            let gs = $crate::calblas::wr!(st, [4, C]);
            let (hb, f, t) = (vcache::<bf16, _>([PX, C]), vcache::<f32, _>([PX, C]), vcache::<f32, _>([PX, C]));
            let (s1, a1, a2) = (vcache::<f32, _>([1, C]), vcache::<f32, _>([1, C]), vcache::<f32, _>([1, C]));
            a1.fill(0.0f32);
            a2.fill(0.0f32);
            let (h0, hn) = $crate::calblas::halves!(H);
            for j in 0..hn * NCH {
                let oy = h0 + j / NCH;
                let x0 = (j % NCH) * PX;
                hb.load(gx.sub([oy * WI + x0, 0], [PX, C]));
                f.cvt(hb, Cvt::default());
                s1.reduce(Combine::Sum, f, 0);
                a1.add(a1, s1);
                t.mul(f, f);
                s1.reduce(Combine::Sum, t, 0);
                a2.add(a2, s1);
            }
            a1.store(gs.sub([2 * core_rank(), 0], [1, C]));
            a2.store(gs.sub([2 * core_rank() + 1, 0], [1, C]));
        }
    };
}

/// GroupNorm 第二步：两核的部分和 ⇒ 每组 mean / rstd ⇒ 每通道 a、b' ⇒ y = act(x · a + b')，只写内部。见模块文档。`act = silu | none`。
#[macro_export]
macro_rules! gn_apply {
    ($vis:vis $name:ident: h = $h:expr, w = $w:expr, c = $c:expr, p = $p:expr, groups = $g:expr, eps = $eps:expr, act = $act:ident) => {
        #[npu::kernel(numerics = "ks1")]
        $vis fn $name(x: npu::Ptr<npu::Ddr, npu::bf16>, st: npu::Ptr<npu::Ddr, f32>, wv: npu::Ptr<npu::Ddr, f32>, bv: npu::Ptr<npu::Ddr, f32>,
                      mk: npu::Ptr<npu::Ddr, f32>, y: npu::Ptr<npu::Ddr, npu::bf16>) {
            use npu::prelude::*;
            const B: usize = $crate::calblas::kernels::BORDER;
            const H: usize = $h;
            const W: usize = $w;
            const C: usize = $c;
            const P: usize = $p;
            const G: usize = $g;
            const CG: usize = C / G;
            const WI: usize = W + 2 * B;
            const PX: usize = $crate::gn_px(W, C);
            const NCH: usize = W / PX;
            const INV_N: f32 = 1.0 / (H * W * CG) as f32;
            const _: () = assert!(C % G == 0 && C % 32 == 0 && C <= P && PX >= 8, "gn_apply：通道数整除组数、是 32 的倍数");
            // 两核的部分和 ⇒ 每通道 Σx、Σx²
            let sv = $crate::calblas::stage!($crate::calblas::rd!(st, [4, C]), [4, C]);
            let (s1, s2) = (vcache::<f32, _>([1, C]), vcache::<f32, _>([1, C]));
            s1.add(sv.rows(0, 1), sv.rows(2, 1));
            s2.add(sv.rows(1, 1), sv.rows(3, 1));
            // 按组：M · s（每行一组）⇒ 沿最后一维（C 个，8 的倍数）求和 ⇒ [G, 1]
            let mf = $crate::calblas::stage!($crate::calblas::rd!(mk, [G, C]), [G, C]);
            let tg = vcache::<f32, _>([G, C]);
            let (m, v, m2) = (vcache::<f32, _>([G, 1]), vcache::<f32, _>([G, 1]), vcache::<f32, _>([G, 1]));
            tg.mul(mf, bcast(s1, &[0, 1]));
            m.reduce(Combine::Sum, tg, 1);
            tg.mul(mf, bcast(s2, &[0, 1]));
            v.reduce(Combine::Sum, tg, 1);
            m.mul(m, INV_N);
            v.mul(v, INV_N);
            m2.mul(m, m);
            v.sub(v, m2);
            v.max(v, 0.0f32);
            v.add(v, $eps);
            v.rsqrt_approx(v);
            // 每组的 rstd、mean 铺回每通道：M · bcast(·) ⇒ 沿第 0 维求和 ⇒ [1, C]
            let (a, mc) = (vcache::<f32, _>([1, C]), vcache::<f32, _>([1, C]));
            tg.mul(mf, bcast(v, &[0, 1]));
            a.reduce(Combine::Sum, tg, 0);
            tg.mul(mf, bcast(m, &[0, 1]));
            mc.reduce(Combine::Sum, tg, 0);
            // a = w · rstd，b' = b − mean · a
            let wf = $crate::calblas::stage!($crate::calblas::rd!(wv, [1, C]), [1, C]);
            let bf = $crate::calblas::stage!($crate::calblas::rd!(bv, [1, C]), [1, C]);
            a.mul(a, wf);
            mc.mul(mc, a);
            bf.sub(bf, mc);
            // 归一、仿射、（SiLU）、写回内部
            let gx = x.add((B * WI + B) * P * 2).bind_strided::<npu::Read, 2>([H * WI, P], [P, 1]);
            let gy = y.add((B * WI + B) * P * 2).bind_strided::<npu::Write, 2>([H * WI, P], [P, 1]);
            let (hb, f, tmp) = (vcache::<bf16, _>([PX, C]), vcache::<f32, _>([PX, C]), vcache::<bf16, _>([PX, C]));
            let (h0, hn) = $crate::calblas::halves!(H);
            for j in 0..hn * NCH {
                let oy = h0 + j / NCH;
                let x0 = (j % NCH) * PX;
                hb.load(gx.sub([oy * WI + x0, 0], [PX, C]));
                f.cvt(hb, Cvt::default());
                f.mul(f, bcast(a, &[0, 1]));
                f.add(f, bcast(bf, &[0, 1]));
                hb.cvt(f, Cvt::default());
                $crate::gn_apply!(@$act hb, tmp);
                hb.store(gy.sub([oy * WI + x0, 0], [PX, C]));
            }
        }
    };
    (@silu $h:ident, $tmp:ident) => { $crate::calblas::logistic!(silu, $h, $h, $tmp) };
    (@none $h:ident, $tmp:ident) => {};
}
