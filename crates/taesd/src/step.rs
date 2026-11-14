//! 入口：单芯粒两核（`cores = 2` 是 calBLAS AGENTS.md 说的常规路径）。
//!
//! 接线的规矩（calBLAS AGENTS.md、yolo26 上板踩出来的）：
//! - `cores = 2` 的 kernel（卷积）两核都调，调完 `sync_cores()` 才能读它的输出；输入、输出、残差三块互不重叠（不原地）。
//! - 按 `num_cores()` 分行的单核 kernel（`upsample2!`、`tanh_scale!`）**两核都调**：yolo26 在板上就是这么调 `upsample2!` 的。
//!   只让 0 号核调时，`num_cores()` 若返回 2，就只算了一半的行（冒烟程序 smoke 原来是这么写的，已改）。
//! - 偏移全部是编译期常量（`const` 项）：后端不收 kernel 外的 u64 运算，常量数组也只能用字面量下标取。
//!
//! 特征图在 @act 里的轮换（每个分辨率 a、b、c 三张，见 layout.rs）：Block(X, T, U) 写 T ⇐ relu(conv(X))、U ⇐ relu(conv(T))、
//! T ⇐ relu(conv(U) + X)，出口在 T；下一个 Block 的 X 就是这个 T，原来的 X 空出来当新的 T。
//! `ref/sim.py` 读编出来的 `.cvx` 照这条调用序列在 numpy 里重放（同一份偏移），查的就是这里的接线。
use crate::kernels::*;
use crate::layout::{self as L, ACT_BYTES, IN_BYTES, OUT_BYTES, W_BYTES, ZERO_BLOCKS};
use npu::prelude::*;

// decls! 的字节数只收字面量：下面的断言核它们与 layout.rs 算的一样（改了 layout 就照报错改这里）
npu::decls! {
    in_x: input, bytes = 295936;
    out_y: output, bytes = 17040384;
    w: weights, bytes = 2650112;
    act: workspace, bytes = 137199616;
}
const _: () = assert!(IN_BYTES == 295936, "in_x 的字节数");
const _: () = assert!(OUT_BYTES == 17040384, "out_y 的字节数");
const _: () = assert!(W_BYTES == 2650112, "w 的字节数");
const _: () = assert!(ACT_BYTES == 137199616, "act 的字节数");

/// 第 r 个分辨率的第 j 张特征图（@act 里的指针；偏移先落成 const 项）。
macro_rules! map {
    ($r:expr, $j:expr) => {{
        const AT: u64 = L::MAPS[$r][$j];
        act().at(AT)
    }};
}

/// 第 i 个卷积：x ⇒ y（可带残差 r），两核分算，调完屏障。
macro_rules! conv {
    ($k:ident, $i:expr, $x:expr => $y:expr) => {{
        const WO: u64 = L::W_OFF[$i];
        const BO: u64 = L::B_OFF[$i];
        $k($x.cast(), w().at(WO).cast(), w().at(BO).cast(), $y.cast());
        sync_cores();
    }};
    ($k:ident, $i:expr, $x:expr, res $r:expr => $y:expr) => {{
        const WO: u64 = L::W_OFF[$i];
        const BO: u64 = L::B_OFF[$i];
        $k($x.cast(), w().at(WO).cast(), w().at(BO).cast(), $r.cast(), $y.cast());
        sync_cores();
    }};
}

/// 一个 Block（第 i、i + 1、i + 2 个卷积），在分辨率 r 上，图 [X, T, U] 是 a / b / c 的下标；出口在 T。
macro_rules! block {
    ($k:ident, $kr:ident, $i:expr, $r:expr, [$x:expr, $t:expr, $u:expr]) => {{
        conv!($k, $i, map!($r, $x) => map!($r, $t));
        conv!($k, $i + 1, map!($r, $t) => map!($r, $u));
        conv!($kr, $i + 2, map!($r, $u), res map!($r, $x) => map!($r, $t));
    }};
}

/// 最近邻 ×2：分辨率 r 的图 j ⇒ 分辨率 r + 1 的图 a（两核都调，见模块文档）。
macro_rules! up {
    ($k:ident, $r:expr, $j:expr) => {{
        $k(map!($r, $j).cast(), map!($r + 1, 0).cast());
        sync_cores();
    }};
}

#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    // 0. @act 整块清零（两核各清一半）：特征图的边框与余量都靠它是 0
    const Z0: usize = ZERO_BLOCKS / 2;
    const AT1: u64 = (Z0 * 16384) as u64;
    if core_rank() == 0 {
        td_zero(act().cast(), Z0);
    } else {
        td_zero(act().at(AT1).cast(), ZERO_BLOCKS - Z0);
    }
    sync_cores();

    // 1. latent 软夹：tanh(x / 3) · 3（in_x ⇒ L0）
    td_tanh(in_x().cast(), act().at(L::L0).cast());
    sync_cores();

    // 2. 64²：conv(4→64) + ReLU ⇒ a；三个 Block：a ⇒ b ⇒ a ⇒ b
    conv!(td_conv_in, 0, act().at(L::L0) => map!(0, 0));
    block!(td_conv_64, td_convr_64, 1, 0, [0, 1, 2]);
    block!(td_conv_64, td_convr_64, 4, 0, [1, 0, 2]);
    block!(td_conv_64, td_convr_64, 7, 0, [0, 1, 2]);

    // 3. 128²：64².b ⇒ 上采样 ⇒ a ⇒ conv ⇒ b；三个 Block：b ⇒ a ⇒ b ⇒ a
    up!(td_up_64, 0, 1);
    conv!(td_convu_128, 10, map!(1, 0) => map!(1, 1));
    block!(td_conv_128, td_convr_128, 11, 1, [1, 0, 2]);
    block!(td_conv_128, td_convr_128, 14, 1, [0, 1, 2]);
    block!(td_conv_128, td_convr_128, 17, 1, [1, 0, 2]);

    // 4. 256²：128².a ⇒ 上采样 ⇒ a ⇒ conv ⇒ b；三个 Block：b ⇒ a ⇒ b ⇒ a
    up!(td_up_128, 1, 0);
    conv!(td_convu_256, 20, map!(2, 0) => map!(2, 1));
    block!(td_conv_256, td_convr_256, 21, 2, [1, 0, 2]);
    block!(td_conv_256, td_convr_256, 24, 2, [0, 1, 2]);
    block!(td_conv_256, td_convr_256, 27, 2, [1, 0, 2]);

    // 5. 512²：256².a ⇒ 上采样 ⇒ a ⇒ conv ⇒ b；一个 Block：b ⇒ a
    up!(td_up_256, 2, 0);
    conv!(td_convu_512, 30, map!(3, 0) => map!(3, 1));
    block!(td_conv_512, td_convr_512, 31, 3, [1, 0, 2]);

    // 6. 出口：conv(64→3，补到 32) ⇒ out_y（[0, 1] 的 RGB 在前 3 个通道）
    conv!(td_conv_out, 34, map!(3, 0) => out_y());
}
