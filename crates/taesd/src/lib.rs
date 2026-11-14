//! TAESD 解码器（512² 档位）的**整个程序**：`cargo build -p taesd --lib --target dopamine-unknown-none-cvx` 出的那一份 `.cvx`
//! 就是 decl + 全部 kernel + 入口，`tools/calbin.sh taesd --params <目录>` 交给 docc 出 calbin。
//!
//! ```text
//!   in_x（latent，NHWC 带边框，bf16）⇒ tanh(x/3)·3 ⇒ conv(4→64)+ReLU
//!   ⇒ 3 × Block @64² ⇒ 上采样 ⇒ conv ⇒ 3 × Block @128² ⇒ 上采样 ⇒ conv ⇒ 3 × Block @256² ⇒ 上采样 ⇒ conv ⇒ 1 × Block @512²
//!   ⇒ conv(64→3) ⇒ out_y（[0, 1] 的图，NHWC 带边框，bf16）
//!   Block(x) = ReLU(conv(ReLU(conv(ReLU(conv(x))))) + x)
//! ```
//! 全部 35 个卷积、3 次上采样、1 次 tanh 都在 NPU 上；主机只做打包（latent ⇒ bf16 NHWC）与解包（取 RGB、夹到 [0, 1]）。
//! diffusers 的 `x · 2 − 1`（[0, 1] ⇒ [−1, 1]）不在设备上做：ComfyUI 要的就是 [0, 1]，对 diffusers 比时主机换算。
#![no_std]

pub mod kernels;
pub mod layout;
mod step;
