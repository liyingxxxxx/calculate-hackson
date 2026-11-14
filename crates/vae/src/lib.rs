//! 正式 VAE 解码器（SD1.5 的 AutoencoderKL，512² 档位，B1b）的**整个程序**：`cargo build -p vae --lib --target dopamine-unknown-none-cvx`
//! 出的那一份 `.cvx` 就是 decl + 全部 kernel + 入口，`tools/calbin.sh vae --params <目录>` 交给 docc 出 calbin。
//!
//! ```text
//!   in_x（latent，VAE 原空间，NHWC 带边框，bf16）⇒ post_quant_conv（1×1）⇒ conv_in（4→512）@64²
//!   ⇒ 中间块：ResNet ⇒ 注意力（单头 512 维、4096 token）⇒ ResNet
//!   ⇒ up_blocks.0 @64²（3 个 ResNet，512）⇒ ×2 + 卷积 ⇒ up_blocks.1 @128²（3 个 ResNet，512）⇒ ×2 + 卷积
//!   ⇒ up_blocks.2 @256²（ResNet 512→256 带 shortcut，2 个 ResNet 256）⇒ ×2 + 卷积
//!   ⇒ up_blocks.3 @512²（ResNet 256→128 带 shortcut，2 个 ResNet 128）⇒ GroupNorm + SiLU ⇒ conv_out（128→3）⇒ out_y（[0, 1] 的图）
//!   ResNet(x) = conv2(SiLU(GN2(conv1(SiLU(GN1(x)))))) + shortcut(x)
//! ```
//! 全部 37 个卷积、30 个 GroupNorm（+ SiLU）、注意力、3 次上采样都在 NPU 上；主机只做打包（latent ⇒ bf16 NHWC）与解包（取 RGB）。
//! diffusers 的 `(x + 1) / 2`（[−1, 1] ⇒ [0, 1]）折进 conv_out 的权重（× 0.5）与 bias（× 0.5 + 0.5），设备直接出 ComfyUI 要的口径。
#![no_std]

pub mod kernels;
pub mod layout;
mod step;
