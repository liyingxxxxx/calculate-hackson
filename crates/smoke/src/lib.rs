//! 冒烟程序：验的是「自己的工程经赛方 cargo 源依赖 calblas ⇒ 设备遍出 CalVX ⇒ docc 出 calbin」这条路本身，
//! 不验数值。形状取 TAESD 解码器开头那一段（64×64、64 通道的 3×3 卷积，接最近邻放大 2 倍），
//! 顺便摸清 calblas 视觉模板在整步程序里的接法——B1（VAE 解码上 NPU）会照这个写。
//!
//! 排布：特征图是 NHWC、四周补两圈边框（calblas `vision.rs` 的约定）：`[H + 4][W + 4][P]` 的 bf16，P = 通道行距。
//! 边框必须是 0（3×3 卷积的 padding = 1 读的就是它）：输入的边框由主机写，中间结果的边框由 workspace 初值保证。
//! ⚠ 这里没有清零中间图的边框——冒烟只看编不编得出；真程序在入口里用 `fill16k!` 清（见 crates/taesd 的 step.rs）。
#![no_std]
use npu::prelude::*;

/// 边框宽（像素），同 calblas 的 `kernels::BORDER`。
const B: u64 = 2;
/// 一张带边框的 h × w × c 图（bf16）占多少字节。
const fn img_bytes(h: u64, w: u64, c: u64) -> u64 {
    (h + 2 * B) * (w + 2 * B) * c * 2
}

calblas::conv2d!(pub smoke_conv3_64x64_64to64: h = 64, w = 64, cin = 64, cout = 64, k = 3, s = 1, pin = (64, 0), pout = (64, 0), act = none);
calblas::upsample2!(pub smoke_up2_64x64_c64: h = 64, w = 64, c = 64, pin = 64, pout = 64);

// decls! 的字节数只收字面量：下面的断言核它们与形状对得上。
npu::decls! {
    in_x: input, bytes = 591872;
    out_y: output, bytes = 2230272;
    w: weights, bytes = 77824;
    act: workspace, bytes = 1048576;
}

/// 权重块里的落点：卷积权重 `[抽头 9][Cin 64][Cout 64]` bf16，之后是 bias f32 `[64]`（4 KiB 对齐）。
const W_CONV: u64 = 0;
const W_BIAS: u64 = 73728;

const _: () = {
    assert!(img_bytes(64, 64, 64) == 591872, "in_x 的字节数");
    assert!(img_bytes(128, 128, 64) == 2230272, "out_y 的字节数");
    assert!(img_bytes(64, 64, 64) <= 1048576, "act 放得下中间图");
    assert!(W_BIAS == 9 * 64 * 64 * 2 && W_BIAS + 64 * 4 <= 77824, "权重块的落点");
};

#[npu::program(chiplets = 1, cores = 2)]
pub fn step() {
    let h = act().cast::<bf16>();
    // conv2d! 是两核分算的 kernel（cores = 2）：两核都调，调完必须 sync_cores() 才能读它的输出。
    smoke_conv3_64x64_64to64(in_x().cast(), w().at(W_CONV).cast(), w().at(W_BIAS).cast(), h);
    sync_cores();
    // upsample2! 是单核 kernel，但行按 num_cores() 分：两核都调（同 calblas yolo26 上板的那份）。
    // ⛔ 只让 0 号核调：num_cores() 是 2 时只算了一半的行（2026-11-14 改；ref/test_cvxsim.py 守着这一条）。
    smoke_up2_64x64_c64(h, out_y().cast());
    sync_cores();
}
