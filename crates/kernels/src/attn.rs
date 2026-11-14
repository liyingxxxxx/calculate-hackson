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
