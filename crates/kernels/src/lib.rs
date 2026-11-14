//! 本仓的 kernel 模板（宏）：calblas 0.1.0 里没有、SD1.5 / TAESD 要用的那几样。写法同 calblas：
//! `sd_kernels::conv2d!(pub 名字: 形状…)` 生成一个 `#[npu::kernel]`，特化（形状是编译期常量）放在各整步程序 crate 的 `kernels.rs` 里。
//!
//! | 宏 | 是什么 | 来源 |
//! | --- | --- | --- |
//! | [`conv2d!`] | NHWC 带边框的 1×1 / 3×3 卷积，`act = silu \| none \| relu \| res_relu`；宽图走「列块快路」 | 抄 calblas `conv2d!`，加 relu、res_relu、列块快路（VENDORED.md） |
//! | [`tanh_scale!`] | `y = tanh(x / s) · s`，逐元素 | 新写 |
//! | [`timestep_embed!`] | 时间步嵌入 `[cos(t·f), sin(t·f)]`（f32 规约 + sin / cos 近似） | 新写（DESIGN D7） |
//! | [`group_norm!`] | GroupNorm（NCHW，可带 SiLU），可选 `parts = N` 显式给段数（小组时控住 docc 展开的代码量，DESIGN F22） | 抄 calblas `group_norm!`，加 `parts`（VENDORED.md） |
//! | [`gn_stats!`] / [`gn_apply!`] | NHWC 原生的 GroupNorm（+ SiLU），直接在带边框的图上算、不转排布：统计一遍、归一一遍，中间一条屏障 | 新写（DESIGN D16） |
//! | [`ln_nhwc!`] | NHWC 原生的 LayerNorm（每像素沿通道），直接在带边框的图上算、只写内部 | 新写（DESIGN D18） |
//! | [`geglu!`] | GEGLU 的后半 `y = h · gelu(g)`（torch 精确 GELU，A&S 7.1.26），逐元素 | 算式抄 calblas `act!(gelu_erf)`（VENDORED.md） |
//! | [`copy_win!`] | 带边框图的内部拷进另一张图的通道窗口（UNet 跳连拼接） | 新写 |
//! | [`pack_int!`] / [`unpack_int!`] | 带边框的图的内部 ⇔ 稠密的 token（UNet 自注意力先压成稠密的再算，DESIGN D23） | 新写 |
//! | [`res_add!`] | 残差加到带边框图的内部（y = a + r，只写内部；宽通道卷积改走 `linear!` 之后用，DESIGN D22） | 新写 |
//! | [`bias_fold!`] | f32 bias + bf16 时间步投影 ⇒ f32（UNet 各 conv1 的 bias） | 新写 |
//! | [`attn_wide!`] | 头维宽（> 256）的自注意力，头维按 128 一段切（正式 VAE 中间块：单头 512 维） | 抄 calblas `attention!` / `spatial_attn!` 改（VENDORED.md，DESIGN D14） |
//! | [`flash_attn!`] | 注意力（mask = keys）：每块 K / V 给 64 行 q 用，数值口径同 calblas（UNet 的自 / 交叉注意力，DESIGN D21） | 寻址抄 calblas `attention!`，按 64 行分块是本仓写的（VENDORED.md） |
//!
//! 宏里经 `$crate::calblas::…` 用 calblas 的片段（`halves!`、`rd!`、`Cols`……），所以调用方只要依赖 `npu`（`#[npu::kernel]` 展开成 `::npu::…`）和本 crate。
#![no_std]

pub use calblas;

mod attn;
mod conv;
mod gn;
mod ln;
mod norm;
mod pointwise;
mod temb;
pub use conv::conv_tile;
pub use gn::{gn_px, GN_CC};
pub use temb::temb_len;
