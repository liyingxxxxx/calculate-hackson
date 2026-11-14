//! TAESD 解码器（512² 档位）的存储布局：**设备端的整步与主机端的打包、模拟共用这一份**（主机经 `cargo run -p layout-dump` 读成 JSON）。
//! 只用 core 与 const fn，不依赖 npu：`crates/layout-dump` 用 `#[path]` 把本文件编进主机程序。
//!
//! 四个声明（`npu::decls!` 的字节数只收字面量，`step.rs` 里有断言核对和这里算的一样）：
//!
//! | 声明 | 内容 |
//! | --- | --- |
//! | `in_x`（input） | latent，NHWC 带两圈边框 `[64 + 4][64 + 4][32]` bf16：4 个真通道 + 28 个补的 0，边框是 0（主机写） |
//! | `out_y`（output） | 解码器最后一个卷积的输出 `[512 + 4][512 + 4][32]` bf16：前 3 个通道是 RGB，值域约 [0, 1]；边框与补的通道不用 |
//! | `w`（weights） | 35 个卷积，按 [`CONVS`] 的次序：权重 `[9][Cin][Cout]` bf16、bias `[Cout]` f32，各自 4 KiB 对齐 |
//! | `act`（workspace） | `L0`（tanh 之后的 latent）+ 每个分辨率三张 64 通道的特征图（a、b、c 轮换）；入口先整块清零 ⇒ 边框是 0 |
//!
//! 为什么每个分辨率各占三张、不复用一块大的：小图的边框落在大图用过的内部上就不是 0 了，卷积的 padding 会读到旧数据。
//! 分开放只多 ~1 MB（大头是 512² 那三张，各 34 MB），换来「清一次零、全程边框都对」。

/// 边框宽（像素），同 calblas `kernels::BORDER`。
pub const B: usize = 2;
/// latent 边长（512² 的图 ⇒ 64² 的 latent）。别的档位以后另出一份程序（形状是编译期常量）。
pub const LAT: usize = 64;
/// 特征图通道数（TAESD 的 decoder_block_out_channels 全是 64）。
pub const C: usize = 64;
/// 第一层的输入通道：latent 的 4 个补到 32（conv2d! 要 16 的倍数；通道行距 32 的倍数时 DDR ⇒ L1B 的整段搬才收）。
pub const CIN0: usize = 32;
/// 最后一层的输出通道：RGB 的 3 个补到 32（NC = 32 时写回能多行一搬，见 calblas `conv2d!` 的 @tail）。
pub const COUT: usize = 32;
/// 每张图之后留的余量（像素）：同 calblas `ktest.py` 的做法，卷积快路的末行多读不越界。
pub const MARGIN_PX: usize = 128;
/// 四个分辨率（latent 的 1、2、4、8 倍）。
pub const RES: [usize; 4] = [LAT, 2 * LAT, 4 * LAT, 8 * LAT];

/// 带边框的 h × h × p 图（bf16）的字节数。
#[must_use]
pub const fn img(h: usize, p: usize) -> usize {
    (h + 2 * B) * (h + 2 * B) * p * 2
}

/// 向上对齐到 `a` 字节。
#[must_use]
pub const fn align(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

pub const IN_BYTES: usize = img(LAT, CIN0);
pub const OUT_BYTES: usize = img(8 * LAT, COUT);

/// 一个卷积：safetensors 里的名字、真实与补齐后的通道数、有没有 bias、在哪个分辨率上算。
#[derive(Clone, Copy)]
pub struct Conv {
    /// 权重的键是 `{key}.weight`，bias 是 `{key}.bias`（`bias = false` 时没有，打包写全零）。
    pub key: &'static str,
    pub cin: usize,
    pub cout: usize,
    pub cin_pad: usize,
    pub cout_pad: usize,
    pub bias: bool,
    /// 输出图的边长。
    pub res: usize,
}

const fn conv(key: &'static str, cin: usize, cout: usize, cin_pad: usize, cout_pad: usize, bias: bool, res: usize) -> Conv {
    Conv { key, cin, cout, cin_pad, cout_pad, bias, res }
}
const fn c64(key: &'static str, bias: bool, res: usize) -> Conv {
    conv(key, C, C, C, C, bias, res)
}

/// 卷积的条数：1（入口）+ 3 × (3 个 Block × 3 + 上采样后 1) + 1 个 Block × 3 + 1（出口）。
pub const NCONV: usize = 35;

/// 全部卷积，按执行次序（diffusers `DecoderTiny.layers` 的下标：0 卷积、1 ReLU、2–4 Block、5 上采样、6 卷积（无 bias）……18 卷积）。
pub const CONVS: [Conv; NCONV] = [
    conv("decoder.layers.0", 4, C, CIN0, C, true, LAT),
    c64("decoder.layers.2.conv.0", true, LAT),
    c64("decoder.layers.2.conv.2", true, LAT),
    c64("decoder.layers.2.conv.4", true, LAT),
    c64("decoder.layers.3.conv.0", true, LAT),
    c64("decoder.layers.3.conv.2", true, LAT),
    c64("decoder.layers.3.conv.4", true, LAT),
    c64("decoder.layers.4.conv.0", true, LAT),
    c64("decoder.layers.4.conv.2", true, LAT),
    c64("decoder.layers.4.conv.4", true, LAT),
    c64("decoder.layers.6", false, 2 * LAT),
    c64("decoder.layers.7.conv.0", true, 2 * LAT),
    c64("decoder.layers.7.conv.2", true, 2 * LAT),
    c64("decoder.layers.7.conv.4", true, 2 * LAT),
    c64("decoder.layers.8.conv.0", true, 2 * LAT),
    c64("decoder.layers.8.conv.2", true, 2 * LAT),
    c64("decoder.layers.8.conv.4", true, 2 * LAT),
    c64("decoder.layers.9.conv.0", true, 2 * LAT),
    c64("decoder.layers.9.conv.2", true, 2 * LAT),
    c64("decoder.layers.9.conv.4", true, 2 * LAT),
    c64("decoder.layers.11", false, 4 * LAT),
    c64("decoder.layers.12.conv.0", true, 4 * LAT),
    c64("decoder.layers.12.conv.2", true, 4 * LAT),
    c64("decoder.layers.12.conv.4", true, 4 * LAT),
    c64("decoder.layers.13.conv.0", true, 4 * LAT),
    c64("decoder.layers.13.conv.2", true, 4 * LAT),
    c64("decoder.layers.13.conv.4", true, 4 * LAT),
    c64("decoder.layers.14.conv.0", true, 4 * LAT),
    c64("decoder.layers.14.conv.2", true, 4 * LAT),
    c64("decoder.layers.14.conv.4", true, 4 * LAT),
    c64("decoder.layers.16", false, 8 * LAT),
    c64("decoder.layers.17.conv.0", true, 8 * LAT),
    c64("decoder.layers.17.conv.2", true, 8 * LAT),
    c64("decoder.layers.17.conv.4", true, 8 * LAT),
    conv("decoder.layers.18", C, 3, C, COUT, true, 8 * LAT),
];

/// 第 i 个卷积权重 `[9][cin_pad][cout_pad]` bf16 的字节数。
#[must_use]
pub const fn w_len(i: usize) -> usize {
    9 * CONVS[i].cin_pad * CONVS[i].cout_pad * 2
}
/// 第 i 个卷积 bias `[cout_pad]` f32 的字节数。
#[must_use]
pub const fn b_len(i: usize) -> usize {
    CONVS[i].cout_pad * 4
}

/// @w 里每个卷积的（权重, bias）偏移：按 [`CONVS`] 的次序，权重、bias 交替铺，各自 4 KiB 对齐。
pub const W_OFF: [u64; NCONV] = offs(0);
pub const B_OFF: [u64; NCONV] = offs(1);
const fn offs(which: usize) -> [u64; NCONV] {
    let mut v = [0u64; NCONV];
    let (mut at, mut i) = (0usize, 0usize);
    while i < NCONV {
        if which == 0 {
            v[i] = at as u64;
        }
        at += align(w_len(i), 4096);
        if which == 1 {
            v[i] = at as u64;
        }
        at += align(b_len(i), 4096);
        i += 1;
    }
    v
}
/// @w 一共多少字节。
pub const W_BYTES: usize = B_OFF[NCONV - 1] as usize + align(b_len(NCONV - 1), 4096);

/// 一张 64 通道特征图（带边框 + 余量）在 @act 里占多少（4 KiB 对齐）。
#[must_use]
pub const fn map_len(res: usize) -> usize {
    align(img(res, C) + MARGIN_PX * C * 2, 4096)
}
/// tanh 之后的 latent（`[68][68][32]`）在 @act 的 0 处。
pub const L0: u64 = 0;
/// 第 r 个分辨率（[`RES`] 的下标）的第 j 张图（0 = a、1 = b、2 = c）。
pub const MAPS: [[u64; 3]; 4] = {
    let mut v = [[0u64; 3]; 4];
    let mut at = align(img(LAT, CIN0) + MARGIN_PX * CIN0 * 2, 4096);
    let mut r = 0;
    while r < 4 {
        let mut j = 0;
        while j < 3 {
            v[r][j] = at as u64;
            at += map_len(RES[r]);
            j += 1;
        }
        r += 1;
    }
    v
};
/// @act 一共多少字节：对齐到 16 KiB（入口用 calblas `fill16k!` 按 16 KiB 一块清零）。
pub const ACT_BYTES: usize = align(MAPS[3][2] as usize + map_len(RES[3]), 16384);
/// 清零的块数（每块 16 KiB）。
pub const ZERO_BLOCKS: usize = ACT_BYTES / 16384;

const _: () = {
    // 各卷积的分辨率与次序对得上：入口 + 3 段 ×（3 个 Block + 上采样后 1 个）+ 1 个 Block + 出口
    assert!(CONVS[0].res == LAT && CONVS[10].res == 2 * LAT && CONVS[20].res == 4 * LAT && CONVS[30].res == 8 * LAT);
    assert!(!CONVS[10].bias && !CONVS[20].bias && !CONVS[30].bias && CONVS[34].bias);
    assert!(CONVS[0].cin_pad % 32 == 0 && CONVS[34].cout_pad % 32 == 0);
    // 偏移都在 64 B 上（kernel 的指针参数要 64 B 对齐）
    assert!(W_OFF[1] % 4096 == 0 && MAPS[3][2] % 4096 == 0);
};
