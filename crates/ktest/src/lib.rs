//! 单 kernel 测试台：`tools/calbin.sh ktest --features <测试> --params <目录>` 出一个只调一个 kernel 的 calbin，
//! `ref/ktest.py` 出随机 / 边界输入与按 torch 语义另算的期望，上板读回逐元素比（同 calBLAS yolo26 `deploy/ktest.py` 的思路，代码自己写，
//! 而且走 `npu::decls!` + `#[npu::program]`，⛔ 拿脚本拼 MLIR 文本：calBLAS AGENTS.md §1）。
//!
//! kernel 特化**直接用 TAESD 那一份**（`#[path]` 引 `crates/taesd/src/kernels.rs`）：测的就是整步程序里跑的那个 kernel，不是另写的一份。
//! 整步程序在板上算错时，用它们逐个 kernel 二分。
//!
//! 每个测试的三块存储：`in_x`（输入图，之后 4 KiB 对齐接残差图）、`w`（权重，之后 4 KiB 对齐接 bias）、`out_y`（输出图）。
//! 字节数照 TAESD 的 layout 算，`ref/ktest.py` 读编出来的 `.cvx` 里的声明核对（不另写一份偏移）。
#![no_std]
#![allow(dead_code)]

#[path = "../../taesd/src/layout.rs"]
mod layout;
#[path = "../../taesd/src/kernels.rs"]
mod kernels;
/// 还没有整步程序用的新 kernel（以后的 UNet 用）：特化先放这里测。
mod extra;

#[cfg(any(
    all(feature = "conv_in", any(feature = "convr_64", feature = "conv_out", feature = "convr_512", feature = "tanh", feature = "up_256")),
    all(feature = "convr_64", any(feature = "conv_out", feature = "convr_512", feature = "tanh", feature = "up_256")),
    all(feature = "conv_out", any(feature = "convr_512", feature = "tanh", feature = "up_256")),
    all(feature = "convr_512", any(feature = "tanh", feature = "up_256")),
    all(feature = "tanh", feature = "up_256"),
    all(feature = "temb", any(feature = "conv_in", feature = "convr_64", feature = "conv_out", feature = "convr_512", feature = "tanh", feature = "up_256")),
))]
compile_error!("ktest：一次只开一个 feature（每个 feature 是一个整步程序）");

#[allow(unused_imports, reason = "各 feature 用到的常量不同")]
use layout::{align, img, C, CIN0, COUT, MARGIN_PX};

/// 输入图（带余量）之后、4 KiB 对齐处放残差图。
const fn r_off(res: usize, p: usize) -> u64 {
    align(img(res, p) + MARGIN_PX * p * 2, 4096) as u64
}

/// 一个卷积测试：程序只调一次 `$k`。`$xin` / `$wb` / `$yout` 是 decls! 的字面量（只收字面量），下面的断言核它们与形状对得上。
/// `kind = plain`：in_x = 输入图；`kind = residual`：in_x = 输入图 + 残差图（各 4 KiB 对齐）。
macro_rules! conv_test {
    ($feat:literal, $k:ident, $kind:ident, res = $res:expr, cin = $ci:expr, cout = $co:expr, in_x = $xin:literal, w = $wb:literal, out_y = $yout:literal) => {
        #[cfg(feature = $feat)]
        mod prog {
            use super::*;
            use npu::prelude::*;
            npu::decls! {
                in_x: input, bytes = $xin;
                w: weights, bytes = $wb;
                out_y: output, bytes = $yout;
            }
            const BO: u64 = align(9 * $ci * $co * 2, 4096) as u64;
            const RO: u64 = r_off($res, $ci);
            const _: () = assert!(BO as usize + align($co * 4, 4096) == $wb, "w 的字节数");
            const _: () = assert!(img($res, $co) == $yout, "out_y 的字节数");
            conv_test!(@prog $kind, $k, $res, $co, $xin);
        }
    };
    (@prog plain, $k:ident, $res:expr, $co:expr, $xin:literal) => {
        const _: () = assert!(RO as usize == $xin, "in_x 的字节数");
        #[npu::program(chiplets = 1, cores = 2)]
        pub fn step() {
            crate::kernels::$k(in_x().cast(), w().cast(), w().at(BO).cast(), out_y().cast());
            sync_cores();
        }
    };
    (@prog residual, $k:ident, $res:expr, $co:expr, $xin:literal) => {
        const _: () = assert!(RO as usize + align(img($res, $co), 4096) == $xin, "in_x 的字节数");
        #[npu::program(chiplets = 1, cores = 2)]
        pub fn step() {
            crate::kernels::$k(in_x().cast(), w().cast(), w().at(BO).cast(), in_x().at(RO).cast(), out_y().cast());
            sync_cores();
        }
    };
}

conv_test!("conv_in", td_conv_in, plain, res = 64, cin = CIN0, cout = C, in_x = 307200, w = 40960, out_y = 591872);
conv_test!("convr_64", td_convr_64, residual, res = 64, cin = C, cout = C, in_x = 1204224, w = 77824, out_y = 591872);
conv_test!("conv_out", td_conv_out, plain, res = 512, cin = C, cout = COUT, in_x = 34099200, w = 40960, out_y = 17040384);
conv_test!("convr_512", td_convr_512, residual, res = 512, cin = C, cout = C, in_x = 68182016, w = 77824, out_y = 34080768);

/// tanh(x / 3) · 3：整张带边框的 latent（同 TAESD 入口那一步）。
#[cfg(feature = "tanh")]
mod prog {
    use super::*;
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 295936;
        out_y: output, bytes = 295936;
    }
    const _: () = assert!(layout::IN_BYTES == 295936);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::kernels::td_tanh(in_x().cast(), out_y().cast());
        sync_cores();
    }
}

/// 最近邻 ×2（256² ⇒ 512²，64 通道），两核都调：验「单核 kernel 按 num_cores() 分行、两核各算一半」这个假设。
#[cfg(feature = "up_256")]
mod prog {
    use super::*;
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 8671232;
        out_y: output, bytes = 34080768;
    }
    const _: () = assert!(r_off(256, C) == 8671232 && img(512, C) == 34080768);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::kernels::td_up_256(in_x().cast(), out_y().cast());
        sync_cores();
    }
}

/// 时间步嵌入（SD1.5 UNet 的 320 维）：in_x = t（32 B，8 份同一个 f32），w = 常数表 F、P（各 [384] f32），out_y = [384] bf16（前 320 个有用）。
/// 单核 kernel，0 号核调。
#[cfg(feature = "temb")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        w: weights, bytes = 4096;
        out_y: output, bytes = 768;
    }
    const _: () = assert!(sd_kernels::temb_len(160) == 384 && 2 * 384 * 4 <= 4096 && 384 * 2 == 768);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        if core_rank() == 0 {
            crate::extra::kt_temb(in_x().cast(), w().cast(), out_y().cast());
        }
        sync_cores();
    }
}
