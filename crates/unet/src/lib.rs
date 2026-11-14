//! SD1.5 UNet（B3）的分段整步程序：每个 feature 一段（`s1` …），各是一个单芯粒两核的整步程序；`z*` 是量代码大小的探针。
//! 入口与偏移全由 `tools/unet_gen.py program` 生成（`src/step.rs`、`plan.json`），kernel 特化由 `tools/unet_gen.py kernels` 生成（`src/kernels.rs`）。
//! 一段 = 若干「单元」（ResNet、Transformer、上 / 下采样……）× CFG 两路；段间的张量经主机转手（只搬字节）。
#![no_std]
#![allow(unused_imports, dead_code)]

pub mod kernels;
mod step;
