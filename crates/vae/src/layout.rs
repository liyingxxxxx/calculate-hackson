//! 正式 VAE 解码器（SD1.5 的 AutoencoderKL，512² 档位，B1b）的存储布局：**设备端的整步与主机端的打包、模拟共用这一份**
//! （主机经 `cargo run -p layout-dump -- vae` 读成 JSON）。只用 core 与 const fn，不依赖 npu。
//!
//! **整个解码器切成 6 段程序**（DESIGN F22 / D15：docc 每核的代码不能超过约 1 MB，45 个 kernel 合起来约 2.7 MB）。
//! 每段是 `crates/vae` 的一个 feature（`s1` … `s6`），各出一个 calbin，主机按次序调 dort、把上一段的 out_y 原样当下一段的 in_x。
//! 段表见 [`STAGES`]；每段四个声明（`npu::decls!` 的字节数只收字面量，`step.rs` 里有断言核对和这里算的一样）：
//!
//! | 声明 | 内容 |
//! | --- | --- |
//! | `in_x`（input） | 这一段的输入：s1 是 latent（VAE 原空间，即 ComfyUI 的 LATENT）`[64 + 4][64 + 4][32]` bf16（4 个真通道 + 28 个补的 0）；其余段是上一段的输出图 |
//! | `out_y`（output） | 这一段的输出图（带边框，边框不写）；s6 是 conv_out 的输出 `[516][516][32]` bf16，前 3 个通道是 RGB，**已经是 [0, 1] 的口径**（(x + 1) / 2 折进了 conv_out） |
//! | `w`（weights） | 这一段的卷积（权重 `[k·k][Cin][Cout]` bf16、bias `[Cout]` f32）⇒ GroupNorm（w、b 按行展开的 f32）⇒（s2）注意力的 q、k、v（HF `[512][512]` bf16 + bias f32），各自 4 KiB 对齐 |
//! | `act`（workspace） | 这一段用到的带边框的图（入口整块清零 ⇒ 边框全程是 0）⇒ 稠密区（GroupNorm 的 NCHW 中转 N1、N2；s2 还有注意力的 TQ、Q、K、V、O：总是先写满再读，不清零） |
//!
//! 切点选在「下一段只经 GroupNorm 的排布转换、残差、1×1 卷积读这张图」的地方：它们只读内部，所以上一段输出的边框（没人写）不用清。
//! 特征图按「分辨率 × 通道」各占一块、互不复用：小图的边框落在大图用过的内部上就不是 0 了（同 TAESD 的 layout.rs）。
//! 每个形状有 A、B（ResNet 的进出轮换）、G（GroupNorm 的输出 = 卷积的输入）、H（conv1 的输出；有 shortcut 的 ResNet 里 GN2 读完它之后放 shortcut）；
//! U 放宽通道的上采样结果（有 shortcut 的那一段里之后放 GN1 的输出）。
//! GroupNorm 一律三步（DESIGN D13）：带边框的图 ⇒ `nhwc_to_nchw!` ⇒ N1 ⇒ `group_norm!` ⇒ N2 ⇒ `nchw_to_nhwc!` ⇒ G。
//! 注意力（DESIGN D14）：GN ⇒ N2（NCHW [512][4096]）⇒ `transpose2d!` ⇒ TQ [4096][512] ⇒ `linear!` ⇒ Q、K、V ⇒ `attn_wide!` ⇒ O
//! ⇒ `transpose2d!` ⇒ N1 ⇒ `nchw_to_nhwc!` ⇒ G ⇒ 1×1 卷积（proj_attn）带残差。

/// 边框宽（像素），同 calblas `kernels::BORDER`。
pub const B: usize = 2;
/// latent 边长（512² 的图 ⇒ 64² 的 latent）。
pub const LAT: usize = 64;
/// latent 的 4 个通道补到 32（conv2d! 要 16 的倍数；行距 32 的倍数时 DDR ⇒ L1B 才能整段搬）。
pub const CIN0: usize = 32;
/// RGB 的 3 个通道补到 32。
pub const COUT: usize = 32;
/// 每张图之后留的余量（像素）：卷积快路的末行多读、512 通道的上采样按第二个窗口（偏 256 通道）读写都不越界。
pub const MARGIN_PX: usize = 128;
/// GroupNorm 的组数、eps（SD1.5 VAE 的 config：norm_num_groups = 32，resnet_eps = 1e-6）。
pub const GROUPS: usize = 32;
/// 注意力：token 数（64² 的图）、维数（单头）。
pub const NT: usize = LAT * LAT;
pub const DA: usize = 512;

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

// ───────────────────────── 特征图 ─────────────────────────

/// 一张带边框的图：边长、通道数（= 像素行距）。
#[derive(Clone, Copy)]
pub struct Map {
    pub res: usize,
    pub ch: usize,
}

const fn map(res: usize, ch: usize) -> Map {
    Map { res, ch }
}

/// 全部带边框的图（名字 ⇒ 下标见下面的常量）。
pub const NMAP: usize = 21;
pub const MAPS: [Map; NMAP] = [
    map(LAT, CIN0), // L1：post_quant_conv 的输出
    map(64, 512),
    map(64, 512),
    map(64, 512),
    map(64, 512), // A64 B64 G64 H64
    map(128, 512),
    map(128, 512),
    map(128, 512),
    map(128, 512), // A128 B128 G128 H128
    map(256, 512),
    map(256, 512), // U256 X256（512 通道）
    map(256, 256),
    map(256, 256),
    map(256, 256),
    map(256, 256), // A256 B256 G256 H256
    map(512, 256),
    map(512, 256), // U512 X512（256 通道）
    map(512, 128),
    map(512, 128),
    map(512, 128),
    map(512, 128), // A512 B512 G512 H512
];
pub const L1: usize = 0;
pub const A64: usize = 1;
pub const B64: usize = 2;
pub const G64: usize = 3;
pub const H64: usize = 4;
pub const A128: usize = 5;
pub const B128: usize = 6;
pub const G128: usize = 7;
pub const H128: usize = 8;
pub const U256: usize = 9;
pub const X256: usize = 10;
pub const A256: usize = 11;
pub const B256: usize = 12;
pub const G256: usize = 13;
pub const H256: usize = 14;
pub const U512: usize = 15;
pub const X512: usize = 16;
pub const A512: usize = 17;
pub const B512: usize = 18;
pub const G512: usize = 19;
pub const H512: usize = 20;

/// 一张图在 @act 里占多少（带余量，4 KiB 对齐）。
#[must_use]
pub const fn map_len(i: usize) -> usize {
    align(img(MAPS[i].res, MAPS[i].ch) + MARGIN_PX * MAPS[i].ch * 2, 4096)
}

/// GroupNorm 的 NCHW 中转、注意力的 token 矩阵 [4096][512] bf16。
pub const TOK_BYTES: usize = NT * DA * 2;

// ───────────────────────── 卷积 ─────────────────────────

/// 一个卷积：safetensors 里的键（`{key}.weight` / `.bias`）、真实与补齐后的通道、核大小、输出图的边长；
/// `linear`：权重是 Linear 的 `[Cout][Cin]`（注意力的 proj_attn，设备上当 1×1 卷积算）；`fold`：把 (x + 1) / 2 折进去（conv_out）。
#[derive(Clone, Copy)]
pub struct Conv {
    pub key: &'static str,
    pub cin: usize,
    pub cout: usize,
    pub cin_pad: usize,
    pub cout_pad: usize,
    pub k: usize,
    pub res: usize,
    pub linear: bool,
    pub fold: bool,
}

const fn cv(key: &'static str, cin: usize, cout: usize, k: usize, res: usize) -> Conv {
    Conv { key, cin, cout, cin_pad: cin, cout_pad: cout, k, res, linear: false, fold: false }
}

/// 卷积的条数：post_quant 1 + conv_in 1 + 中间块 2 × 2 + proj_attn 1 + 4 段（3 个 ResNet × 2 + 上采样后 1 / shortcut 1）+ conv_out 1。
pub const NCONV: usize = 37;

/// 全部卷积，按执行次序（下标就是 step.rs 里 `conv!` 的第二个参数）。
pub const CONVS: [Conv; NCONV] = [
    Conv { key: "post_quant_conv", cin: 4, cout: 4, cin_pad: CIN0, cout_pad: CIN0, k: 1, res: 64, linear: false, fold: false },
    Conv { key: "decoder.conv_in", cin: 4, cout: 512, cin_pad: CIN0, cout_pad: 512, k: 3, res: 64, linear: false, fold: false },
    cv("decoder.mid_block.resnets.0.conv1", 512, 512, 3, 64),
    cv("decoder.mid_block.resnets.0.conv2", 512, 512, 3, 64),
    Conv { key: "decoder.mid_block.attentions.0.proj_attn", cin: 512, cout: 512, cin_pad: 512, cout_pad: 512, k: 1, res: 64, linear: true, fold: false },
    cv("decoder.mid_block.resnets.1.conv1", 512, 512, 3, 64),
    cv("decoder.mid_block.resnets.1.conv2", 512, 512, 3, 64),
    // up_blocks.0：64²、512 通道；上采样之后的卷积在 128² 上
    cv("decoder.up_blocks.0.resnets.0.conv1", 512, 512, 3, 64),
    cv("decoder.up_blocks.0.resnets.0.conv2", 512, 512, 3, 64),
    cv("decoder.up_blocks.0.resnets.1.conv1", 512, 512, 3, 64),
    cv("decoder.up_blocks.0.resnets.1.conv2", 512, 512, 3, 64),
    cv("decoder.up_blocks.0.resnets.2.conv1", 512, 512, 3, 64),
    cv("decoder.up_blocks.0.resnets.2.conv2", 512, 512, 3, 64),
    cv("decoder.up_blocks.0.upsamplers.0.conv", 512, 512, 3, 128),
    // up_blocks.1：128²、512 通道
    cv("decoder.up_blocks.1.resnets.0.conv1", 512, 512, 3, 128),
    cv("decoder.up_blocks.1.resnets.0.conv2", 512, 512, 3, 128),
    cv("decoder.up_blocks.1.resnets.1.conv1", 512, 512, 3, 128),
    cv("decoder.up_blocks.1.resnets.1.conv2", 512, 512, 3, 128),
    cv("decoder.up_blocks.1.resnets.2.conv1", 512, 512, 3, 128),
    cv("decoder.up_blocks.1.resnets.2.conv2", 512, 512, 3, 128),
    cv("decoder.up_blocks.1.upsamplers.0.conv", 512, 512, 3, 256),
    // up_blocks.2：256²、512 ⇒ 256 通道（第一个 ResNet 有 1×1 shortcut）
    cv("decoder.up_blocks.2.resnets.0.conv1", 512, 256, 3, 256),
    cv("decoder.up_blocks.2.resnets.0.conv2", 256, 256, 3, 256),
    cv("decoder.up_blocks.2.resnets.0.conv_shortcut", 512, 256, 1, 256),
    cv("decoder.up_blocks.2.resnets.1.conv1", 256, 256, 3, 256),
    cv("decoder.up_blocks.2.resnets.1.conv2", 256, 256, 3, 256),
    cv("decoder.up_blocks.2.resnets.2.conv1", 256, 256, 3, 256),
    cv("decoder.up_blocks.2.resnets.2.conv2", 256, 256, 3, 256),
    cv("decoder.up_blocks.2.upsamplers.0.conv", 256, 256, 3, 512),
    // up_blocks.3：512²、256 ⇒ 128 通道（第一个 ResNet 有 shortcut）；没有上采样
    cv("decoder.up_blocks.3.resnets.0.conv1", 256, 128, 3, 512),
    cv("decoder.up_blocks.3.resnets.0.conv2", 128, 128, 3, 512),
    cv("decoder.up_blocks.3.resnets.0.conv_shortcut", 256, 128, 1, 512),
    cv("decoder.up_blocks.3.resnets.1.conv1", 128, 128, 3, 512),
    cv("decoder.up_blocks.3.resnets.1.conv2", 128, 128, 3, 512),
    cv("decoder.up_blocks.3.resnets.2.conv1", 128, 128, 3, 512),
    cv("decoder.up_blocks.3.resnets.2.conv2", 128, 128, 3, 512),
    Conv { key: "decoder.conv_out", cin: 128, cout: 3, cin_pad: 128, cout_pad: COUT, k: 3, res: 512, linear: false, fold: true },
];

/// 第 i 个卷积权重 `[k·k][cin_pad][cout_pad]` bf16 的字节数。
#[must_use]
pub const fn w_len(i: usize) -> usize {
    CONVS[i].k * CONVS[i].k * CONVS[i].cin_pad * CONVS[i].cout_pad * 2
}
/// 第 i 个卷积 bias `[cout_pad]` f32 的字节数。
#[must_use]
pub const fn b_len(i: usize) -> usize {
    CONVS[i].cout_pad * 4
}

// ───────────────────────── GroupNorm ─────────────────────────

/// 一个 GroupNorm：键（`{key}.weight` / `.bias`）、通道数、图的边长、后面跟不跟 SiLU。
#[derive(Clone, Copy)]
pub struct Gn {
    pub key: &'static str,
    pub c: usize,
    pub res: usize,
    pub silu: bool,
}

const fn gn(key: &'static str, c: usize, res: usize) -> Gn {
    Gn { key, c, res, silu: true }
}

pub const NGN: usize = 30;
/// 全部 GroupNorm，按执行次序（下标就是 step.rs 里 `gn!` 的那个参数）。
pub const GNS: [Gn; NGN] = [
    gn("decoder.mid_block.resnets.0.norm1", 512, 64),
    gn("decoder.mid_block.resnets.0.norm2", 512, 64),
    Gn { key: "decoder.mid_block.attentions.0.group_norm", c: 512, res: 64, silu: false },
    gn("decoder.mid_block.resnets.1.norm1", 512, 64),
    gn("decoder.mid_block.resnets.1.norm2", 512, 64),
    gn("decoder.up_blocks.0.resnets.0.norm1", 512, 64),
    gn("decoder.up_blocks.0.resnets.0.norm2", 512, 64),
    gn("decoder.up_blocks.0.resnets.1.norm1", 512, 64),
    gn("decoder.up_blocks.0.resnets.1.norm2", 512, 64),
    gn("decoder.up_blocks.0.resnets.2.norm1", 512, 64),
    gn("decoder.up_blocks.0.resnets.2.norm2", 512, 64),
    gn("decoder.up_blocks.1.resnets.0.norm1", 512, 128),
    gn("decoder.up_blocks.1.resnets.0.norm2", 512, 128),
    gn("decoder.up_blocks.1.resnets.1.norm1", 512, 128),
    gn("decoder.up_blocks.1.resnets.1.norm2", 512, 128),
    gn("decoder.up_blocks.1.resnets.2.norm1", 512, 128),
    gn("decoder.up_blocks.1.resnets.2.norm2", 512, 128),
    gn("decoder.up_blocks.2.resnets.0.norm1", 512, 256),
    gn("decoder.up_blocks.2.resnets.0.norm2", 256, 256),
    gn("decoder.up_blocks.2.resnets.1.norm1", 256, 256),
    gn("decoder.up_blocks.2.resnets.1.norm2", 256, 256),
    gn("decoder.up_blocks.2.resnets.2.norm1", 256, 256),
    gn("decoder.up_blocks.2.resnets.2.norm2", 256, 256),
    gn("decoder.up_blocks.3.resnets.0.norm1", 256, 512),
    gn("decoder.up_blocks.3.resnets.0.norm2", 128, 512),
    gn("decoder.up_blocks.3.resnets.1.norm1", 128, 512),
    gn("decoder.up_blocks.3.resnets.1.norm2", 128, 512),
    gn("decoder.up_blocks.3.resnets.2.norm1", 128, 512),
    gn("decoder.up_blocks.3.resnets.2.norm2", 128, 512),
    gn("decoder.conv_norm_out", 128, 512),
];

/// calblas `group_norm!` 一行几个像素（`kernels::Cols::norm_cols` 的同一个算法：128 或更小，整除 HW，且一组的行数是 8 的倍数）。
/// 抄在这里是因为 layout-dump 是主机程序、不依赖 calblas；`step.rs` 里有断言核它和 calblas 的那份逐个相同。
#[must_use]
pub const fn norm_cols(n: usize, hw: usize) -> usize {
    let mut cc = if hw < 128 { hw } else { 128 };
    while cc > 16 && (hw % cc != 0 || (n / cc) % 8 != 0) {
        cc /= 2;
    }
    cc
}
/// 第 i 个 GroupNorm 一行几个像素。
#[must_use]
pub const fn gn_cols(i: usize) -> usize {
    let hw = GNS[i].res * GNS[i].res;
    norm_cols(GNS[i].c / GROUPS * hw, hw)
}
/// 第 i 个 GroupNorm 的 w（或 b）展开后的字节数：`[C · HW / CC][8]` f32（每行一个通道的值，写 8 份）。
#[must_use]
pub const fn gn_len(i: usize) -> usize {
    GNS[i].c * GNS[i].res * GNS[i].res / gn_cols(i) * 8 * 4
}

// ───────────────────────── 注意力的 q、k、v ─────────────────────────

/// HF 原样的 `[512][512]` bf16（`linear!` 的 `w = hf`）与 bias f32 `[512]`。
pub const QKV_KEYS: [&str; 3] = [
    "decoder.mid_block.attentions.0.query",
    "decoder.mid_block.attentions.0.key",
    "decoder.mid_block.attentions.0.value",
];
pub const QKV_W_LEN: usize = DA * DA * 2;
pub const QKV_B_LEN: usize = DA * 4;

// ───────────────────────── 段 ─────────────────────────

/// 一段程序：用哪些卷积（[c0, c1)）、哪些 GroupNorm（[g0, g1)）、有没有注意力、输入输出图的形状（边长、通道）、@act 里放哪些图。
#[derive(Clone, Copy)]
pub struct Stage {
    pub name: &'static str,
    pub c0: usize,
    pub c1: usize,
    pub g0: usize,
    pub g1: usize,
    pub attn: bool,
    pub inp: Map,
    pub out: Map,
    pub maps: &'static [usize],
}

pub const NSTAGE: usize = 6;
/// 6 段（DESIGN D15）。代码量是按探针估的（F22），每段 < 1 MB。
pub const STAGES: [Stage; NSTAGE] = [
    // s1：post_quant_conv、conv_in、中间块 ResNet 0（latent ⇒ 64² × 512）
    Stage { name: "s1", c0: 0, c1: 4, g0: 0, g1: 2, attn: false, inp: map(LAT, CIN0), out: map(64, 512), maps: &[L1, A64, G64, H64] },
    // s2：中间块的注意力（64² × 512 ⇒ 64² × 512）
    Stage { name: "s2", c0: 4, c1: 5, g0: 2, g1: 3, attn: true, inp: map(64, 512), out: map(64, 512), maps: &[G64] },
    // s3：中间块 ResNet 1、up_blocks.0（3 个 ResNet）、×2 + 卷积（64² × 512 ⇒ 128² × 512）
    Stage { name: "s3", c0: 5, c1: 14, g0: 3, g1: 11, attn: false, inp: map(64, 512), out: map(128, 512), maps: &[A64, B64, G64, H64, G128] },
    // s4：up_blocks.1（3 个 ResNet）、×2 + 卷积（128² × 512 ⇒ 256² × 512）
    Stage { name: "s4", c0: 14, c1: 21, g0: 11, g1: 17, attn: false, inp: map(128, 512), out: map(256, 512), maps: &[A128, B128, G128, H128, U256] },
    // s5：up_blocks.2（带 shortcut 的 ResNet 512 ⇒ 256、2 个 ResNet）、×2 + 卷积（256² × 512 ⇒ 512² × 256）
    Stage { name: "s5", c0: 21, c1: 29, g0: 17, g1: 23, attn: false, inp: map(256, 512), out: map(512, 256), maps: &[U256, A256, B256, G256, H256, U512] },
    // s6：up_blocks.3（带 shortcut 的 ResNet 256 ⇒ 128、2 个 ResNet）、GN + SiLU、conv_out（512² × 256 ⇒ 图）
    Stage { name: "s6", c0: 29, c1: 37, g0: 23, g1: 30, attn: false, inp: map(512, 256), out: map(512, COUT), maps: &[U512, A512, B512, G512, H512] },
];

/// 第 s 段的 in_x / out_y 字节数（带边框、不带余量：卷积快路读不出图的末尾，TAESD 的 in_x 也是这样，板上验过）。
#[must_use]
pub const fn in_bytes(s: usize) -> usize {
    img(STAGES[s].inp.res, STAGES[s].inp.ch)
}
#[must_use]
pub const fn out_bytes(s: usize) -> usize {
    img(STAGES[s].out.res, STAGES[s].out.ch)
}

/// 第 s 段 @act 里第 i 张图（全局下标，见 [`MAPS`]）的偏移；这一段没放这张图就编译不过。
#[must_use]
pub const fn map_off(s: usize, i: usize) -> u64 {
    let m = STAGES[s].maps;
    let (mut at, mut k) = (0usize, 0usize);
    while k < m.len() {
        if m[k] == i {
            return at as u64;
        }
        at += map_len(m[k]);
        k += 1;
    }
    panic!("这一段的 @act 里没有这张图");
}
/// 第 s 段带边框的图一共多少字节：对齐到 16 KiB（入口用 calblas `fill16k!` 按 16 KiB 一块清零，只清这一段）。
#[must_use]
pub const fn maps_bytes(s: usize) -> usize {
    let m = STAGES[s].maps;
    let (mut at, mut k) = (0usize, 0usize);
    while k < m.len() {
        at += map_len(m[k]);
        k += 1;
    }
    align(at, 16384)
}
#[must_use]
pub const fn zero_blocks(s: usize) -> usize {
    maps_bytes(s) / 16384
}
/// 第 s 段 GroupNorm 的 NCHW 中转要多大：这一段最大的那个 GN（C · HW · 2）。
#[must_use]
pub const fn nchw_bytes(s: usize) -> usize {
    let (mut n, mut j) = (0usize, STAGES[s].g0);
    while j < STAGES[s].g1 {
        let b = GNS[j].c * GNS[j].res * GNS[j].res * 2;
        if b > n {
            n = b;
        }
        j += 1;
    }
    n
}
/// 稠密区：N1、N2（GroupNorm 的中转）；注意力那一段再加 TQ、Q、K、V、O。
#[must_use]
pub const fn n1(s: usize) -> u64 {
    maps_bytes(s) as u64
}
#[must_use]
pub const fn n2(s: usize) -> u64 {
    n1(s) + nchw_bytes(s) as u64
}
/// 注意力的 token 矩阵：t = 0 TQ、1 Q、2 K、3 V、4 O。
#[must_use]
pub const fn tok(s: usize, t: usize) -> u64 {
    assert!(STAGES[s].attn, "只有注意力那一段有 token 矩阵");
    n2(s) + nchw_bytes(s) as u64 + (t * TOK_BYTES) as u64
}
/// 第 s 段 @act 一共多少字节。
#[must_use]
pub const fn act_bytes(s: usize) -> usize {
    n2(s) as usize + nchw_bytes(s) + if STAGES[s].attn { 5 * TOK_BYTES } else { 0 }
}

// ── 第 s 段的 @w：卷积 c0..c1（权重、bias）⇒ GroupNorm g0..g1（w、b）⇒（注意力那一段）q、k、v（权重、bias），各自 4 KiB 对齐、依次铺 ──

/// 第 s 段 @w 的条目数。
const fn n_entries(s: usize) -> usize {
    2 * (STAGES[s].c1 - STAGES[s].c0) + 2 * (STAGES[s].g1 - STAGES[s].g0) + if STAGES[s].attn { 6 } else { 0 }
}
/// 第 s 段第 e 个条目的字节数。
const fn entry_len(s: usize, e: usize) -> usize {
    let nc = 2 * (STAGES[s].c1 - STAGES[s].c0);
    let ng = 2 * (STAGES[s].g1 - STAGES[s].g0);
    if e < nc {
        let i = STAGES[s].c0 + e / 2;
        if e % 2 == 0 { w_len(i) } else { b_len(i) }
    } else if e < nc + ng {
        gn_len(STAGES[s].g0 + (e - nc) / 2)
    } else if (e - nc - ng) % 2 == 0 {
        QKV_W_LEN
    } else {
        QKV_B_LEN
    }
}
/// 第 s 段第 e 个条目在 @w 里的偏移。
const fn entry_off(s: usize, e: usize) -> u64 {
    let (mut at, mut k) = (0usize, 0usize);
    while k < e {
        at += align(entry_len(s, k), 4096);
        k += 1;
    }
    at as u64
}
/// 第 s 段里卷积 i（全局下标）的权重、bias 的偏移。
#[must_use]
pub const fn w_off(s: usize, i: usize) -> u64 {
    assert!(STAGES[s].c0 <= i && i < STAGES[s].c1, "这个卷积不在这一段");
    entry_off(s, 2 * (i - STAGES[s].c0))
}
#[must_use]
pub const fn b_off(s: usize, i: usize) -> u64 {
    assert!(STAGES[s].c0 <= i && i < STAGES[s].c1, "这个卷积不在这一段");
    entry_off(s, 2 * (i - STAGES[s].c0) + 1)
}
/// 第 s 段里 GroupNorm j（全局下标）的 w、b 的偏移。
#[must_use]
pub const fn gw_off(s: usize, j: usize) -> u64 {
    assert!(STAGES[s].g0 <= j && j < STAGES[s].g1, "这个 GroupNorm 不在这一段");
    entry_off(s, 2 * (STAGES[s].c1 - STAGES[s].c0) + 2 * (j - STAGES[s].g0))
}
#[must_use]
pub const fn gb_off(s: usize, j: usize) -> u64 {
    assert!(STAGES[s].g0 <= j && j < STAGES[s].g1, "这个 GroupNorm 不在这一段");
    entry_off(s, 2 * (STAGES[s].c1 - STAGES[s].c0) + 2 * (j - STAGES[s].g0) + 1)
}
/// 注意力那一段里 q（0）、k（1）、v（2）的权重、bias 的偏移。
#[must_use]
pub const fn qkv_w_off(s: usize, q: usize) -> u64 {
    assert!(STAGES[s].attn && q < 3, "只有注意力那一段有 q、k、v");
    entry_off(s, 2 * (STAGES[s].c1 - STAGES[s].c0) + 2 * (STAGES[s].g1 - STAGES[s].g0) + 2 * q)
}
#[must_use]
pub const fn qkv_b_off(s: usize, q: usize) -> u64 {
    assert!(STAGES[s].attn && q < 3, "只有注意力那一段有 q、k、v");
    entry_off(s, 2 * (STAGES[s].c1 - STAGES[s].c0) + 2 * (STAGES[s].g1 - STAGES[s].g0) + 2 * q + 1)
}
/// 第 s 段 @w 一共多少字节。
#[must_use]
pub const fn w_bytes(s: usize) -> usize {
    let n = n_entries(s);
    entry_off(s, n - 1) as usize + align(entry_len(s, n - 1), 4096)
}

const _: () = {
    assert!(NCONV == 37 && NGN == 30);
    // 各段的分辨率与次序：post_quant、conv_in 在 64²；每段上采样之后的卷积在下一个分辨率上；conv_out 在 512² 且折了 (x + 1) / 2
    assert!(CONVS[13].res == 128 && CONVS[20].res == 256 && CONVS[28].res == 512 && CONVS[36].fold && CONVS[4].linear);
    assert!(CONVS[23].k == 1 && CONVS[31].k == 1 && CONVS[0].k == 1);
    assert!(!GNS[2].silu && GNS[17].c == 512 && GNS[23].c == 256 && GNS[29].res == 512);
    // 段首尾相接：卷积、GroupNorm 都恰好分完；上一段的输出就是下一段的输入
    assert!(STAGES[0].c0 == 0 && STAGES[NSTAGE - 1].c1 == NCONV && STAGES[0].g0 == 0 && STAGES[NSTAGE - 1].g1 == NGN);
    let mut s = 1;
    while s < NSTAGE {
        assert!(STAGES[s].c0 == STAGES[s - 1].c1 && STAGES[s].g0 == STAGES[s - 1].g1);
        assert!(STAGES[s].inp.res == STAGES[s - 1].out.res && STAGES[s].inp.ch == STAGES[s - 1].out.ch);
        s += 1;
    }
    // 512 通道上采样的第二个窗口：偏 256 个通道读写，末尾多出的 512 B 落在余量里
    assert!(MARGIN_PX * 512 * 2 >= 512);
};
