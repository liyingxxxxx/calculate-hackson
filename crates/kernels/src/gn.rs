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
//! 组内求和为什么用掩码：每组的通道数在 128 通道时只有 4（UNet 里还有 10、30、60），docc 0.4.1 沿最后一维归约要 8 的倍数、
//! VB 里的转置也要列数是 8 的倍数（探针 2026-11-14：「vec.transpose 只收 32 位元素、列数是 8 的倍数（[32, 4] f32）」）。
//! M 的第 g 行在第 g 组的通道上是 1、其余是 0（主机给，f32），乘的是 0 / 1，和是精确的。
//! 通道按 64 个一段（CC = 64，C 要是 64 的倍数）：数据块看成 `[PX, C / 64, 64]`，每通道的量是 `[1, C / 64, 64]`，最里一维 256 B ——
//! docc 要求「和 VB 里的标量做运算的向量在一行 512 B 以内或整行的倍数」（DESIGN F15；UNet 的 320 通道整行 1280 B 就不收），
//! 掩码也按段读（`[32, 64]`），宽到 2560 通道也放得进 VB。
//! 排布：x、y 是 `[H + 4][W + 4][P]` bf16 的图，前 C 个通道；每组 C / G 个通道连着（同 torch 的 GroupNorm）。st 要 64 B 对齐、[4][C] f32；
//! w、b 是 f32 [C]，掩码 mk 是 f32 [G][C]；C 是 64 的倍数（VAE 的 128 / 256 / 512、UNet 的 320–2560 都是）。
//! 两个都是单核 kernel、按 `num_cores()` 分行 ⇒ **两核都调**；st 按两颗核排，所以程序要 `cores = 2`。
//! 守着它的：模拟器 `cvxsim.py` 的 `_k_gn_stats` / `_k_gn_apply`（对 torch 的 GroupNorm）、正式 VAE 的 6 段对参照。

/// 通道一段几个（见模块文档）。
#[doc(hidden)]
pub const GN_CC: usize = 64;

/// GN / LN 一块读几个像素：整除 W 的 2 的幂，`[PX, C]` 的 bf16 一份、f32 两份（10 B / 元素）放得进 192 KB（其余给掩码与每通道的量）；至多 W。
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
            const CC: usize = $crate::GN_CC;
            const NCC: usize = C / CC;
            const WI: usize = W + 2 * B;
            const PX: usize = $crate::gn_px(W, C);
            const NCH: usize = W / PX;
            const _: () = assert!(C % CC == 0 && C <= P, "gn_stats：通道数是 64 的倍数");
            // 内部：第 oy 行第 x0 列起 PX 个像素 ⇒ 第 oy · WI + x0 行起 PX 行（行距 P），每行看成 [C / 64, 64]
            let gx = x.add((B * WI + B) * P * 2).bind_strided::<npu::Read, 3>([H * WI, NCC, CC], [P, CC, 1]);
            let gs = st.bind_strided::<npu::Write, 3>([4, NCC, CC], [C, CC, 1]);
            let (hb, f, t) = (vcache::<bf16, _>([PX, NCC, CC]), vcache::<f32, _>([PX, NCC, CC]), vcache::<f32, _>([PX, NCC, CC]));
            let (s1, a1, a2) = (vcache::<f32, _>([1, NCC, CC]), vcache::<f32, _>([1, NCC, CC]), vcache::<f32, _>([1, NCC, CC]));
            a1.fill(0.0f32);
            a2.fill(0.0f32);
            let (h0, hn) = $crate::calblas::halves!(H);
            for j in 0..hn * NCH {
                let oy = h0 + j / NCH;
                let x0 = (j % NCH) * PX;
                hb.load(gx.sub([oy * WI + x0, 0, 0], [PX, NCC, CC]));
                f.cvt(hb, Cvt::default());
                s1.reduce(Combine::Sum, f, 0);
                a1.add(a1, s1);
                t.mul(f, f);
                s1.reduce(Combine::Sum, t, 0);
                a2.add(a2, s1);
            }
            a1.store(gs.sub([2 * core_rank(), 0, 0], [1, NCC, CC]));
            a2.store(gs.sub([2 * core_rank() + 1, 0, 0], [1, NCC, CC]));
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
            const CC: usize = $crate::GN_CC;
            const NCC: usize = C / CC;
            const WI: usize = W + 2 * B;
            const PX: usize = $crate::gn_px(W, C);
            const NCH: usize = W / PX;
            const INV_N: f32 = 1.0 / (H * W * CG) as f32;
            const _: () = assert!(C % G == 0 && C % CC == 0 && C <= P, "gn_apply：通道数整除组数、是 64 的倍数");
            // 两核的部分和 ⇒ 每通道 Σx、Σx²（[NCC, 64]：一段一行）
            let sv = $crate::calblas::stage!($crate::calblas::rd!(st, [4 * NCC, CC]), [4 * NCC, CC]);
            let (s1, s2) = (vcache::<f32, _>([NCC, CC]), vcache::<f32, _>([NCC, CC]));
            s1.add(sv.rows(0, NCC), sv.rows(2 * NCC, NCC));
            s2.add(sv.rows(NCC, NCC), sv.rows(3 * NCC, NCC));
            // 按组：逐段 M[:, 段] · s[段] ⇒ 沿最后一维（64 个）求和 ⇒ 累加进 [G, 1]
            let gmk = mk.bind_strided::<npu::Read, 2>([G, C], [C, 1]);
            let (mc, tg) = (vcache::<f32, _>([G, CC]), vcache::<f32, _>([G, CC]));
            let (m, v, r, m2) = (vcache::<f32, _>([G, 1]), vcache::<f32, _>([G, 1]), vcache::<f32, _>([G, 1]), vcache::<f32, _>([G, 1]));
            m.fill(0.0f32);
            v.fill(0.0f32);
            for k in 0..NCC {
                mc.load(gmk.sub([0, k * CC], [G, CC]));
                tg.mul(mc, bcast(s1.rows(k, 1), &[0, 1]));
                r.reduce(Combine::Sum, tg, 1);
                m.add(m, r);
                tg.mul(mc, bcast(s2.rows(k, 1), &[0, 1]));
                r.reduce(Combine::Sum, tg, 1);
                v.add(v, r);
            }
            m.mul(m, INV_N);
            v.mul(v, INV_N);
            m2.mul(m, m);
            v.sub(v, m2);
            v.max(v, 0.0f32);
            v.add(v, $eps);
            v.rsqrt_approx(v);
            // 每组的 rstd、mean 铺回每通道：逐段 M[:, 段] · bcast(·) ⇒ 沿第 0 维求和 ⇒ [1, 64] ⇒ 第 k 行
            let (ar, mr, t1) = (vcache::<f32, _>([NCC, CC]), vcache::<f32, _>([NCC, CC]), vcache::<f32, _>([1, CC]));
            for k in 0..NCC {
                mc.load(gmk.sub([0, k * CC], [G, CC]));
                tg.mul(mc, bcast(v, &[0, 1]));
                t1.reduce(Combine::Sum, tg, 0);
                ar.rows(k, 1).copy(t1);
                tg.mul(mc, bcast(m, &[0, 1]));
                t1.reduce(Combine::Sum, tg, 0);
                mr.rows(k, 1).copy(t1);
            }
            // a = w · rstd，b' = b − mean · a（[1, NCC, 64]，和数据块同形好广播）
            let (a, bf, mf) = (vcache::<f32, _>([1, NCC, CC]), vcache::<f32, _>([1, NCC, CC]), vcache::<f32, _>([1, NCC, CC]));
            a.reshape(ar);
            mf.reshape(mr);
            let wf = $crate::calblas::stage!($crate::calblas::rd!(wv, [1, NCC, CC]), [1, NCC, CC]);
            bf.load($crate::calblas::rd!(bv, [1, NCC, CC]));
            a.mul(a, wf);
            mf.mul(mf, a);
            bf.sub(bf, mf);
            // 归一、仿射、（SiLU）、写回内部
            let gx = x.add((B * WI + B) * P * 2).bind_strided::<npu::Read, 3>([H * WI, NCC, CC], [P, CC, 1]);
            let gy = y.add((B * WI + B) * P * 2).bind_strided::<npu::Write, 3>([H * WI, NCC, CC], [P, CC, 1]);
            let (hb, f, tmp) = (vcache::<bf16, _>([PX, NCC, CC]), vcache::<f32, _>([PX, NCC, CC]), vcache::<bf16, _>([PX, NCC, CC]));
            let (h0, hn) = $crate::calblas::halves!(H);
            for j in 0..hn * NCH {
                let oy = h0 + j / NCH;
                let x0 = (j % NCH) * PX;
                hb.load(gx.sub([oy * WI + x0, 0, 0], [PX, NCC, CC]));
                f.cvt(hb, Cvt::default());
                f.mul(f, bcast(a, &[0, 1, 2]));
                f.add(f, bcast(bf, &[0, 1, 2]));
                hb.cvt(f, Cvt::default());
                $crate::gn_apply!(@$act hb, tmp);
                hb.store(gy.sub([oy * WI + x0, 0, 0], [PX, NCC, CC]));
            }
        }
    };
    (@silu $h:ident, $tmp:ident) => { $crate::calblas::logistic!(silu, $h, $h, $tmp) };
    (@none $h:ident, $tmp:ident) => {};
}
