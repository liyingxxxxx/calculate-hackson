//! 本仓的 kernel 模板（宏）：calblas 0.1.0 里没有、SD1.5 / TAESD 要用的那几样。写法同 calblas：
//! `sd_kernels::conv2d!(pub 名字: 形状…)` 生成一个 `#[npu::kernel]`，特化（形状是编译期常量）放在各整步程序 crate 的 `kernels.rs` 里。
//!
//! | 宏 | 是什么 | 来源 |
//! | --- | --- | --- |
//! | [`conv2d!`] | NHWC 带边框的 1×1 / 3×3 卷积，`act = silu \| none \| relu \| res_relu` | 抄 calblas `conv2d!`，加 relu、res_relu（VENDORED.md） |
//! | [`tanh_scale!`] | `y = tanh(x / s) · s`，逐元素 | 新写 |
//! | [`timestep_embed!`] | 时间步嵌入 `[cos(t·f), sin(t·f)]`（f32 规约 + sin / cos 近似） | 新写（DESIGN D7） |
//! | [`attn_wide!`] | 头维宽（> 256）的自注意力，头维按 128 一段切（正式 VAE 中间块：单头 512 维） | 抄 calblas `attention!` / `spatial_attn!` 改（VENDORED.md，DESIGN D14） |
//!
//! 宏里经 `$crate::calblas::…` 用 calblas 的片段（`halves!`、`rd!`、`Cols`……），所以调用方只要依赖 `npu`（`#[npu::kernel]` 展开成 `::npu::…`）和本 crate。
#![no_std]

pub use calblas;

mod attn;
mod conv;
mod pointwise;
mod temb;
pub use temb::temb_len;
