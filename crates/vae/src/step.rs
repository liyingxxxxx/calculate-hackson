//! 6 段入口（DESIGN D15）：每段一个 feature（`s1` … `s6`），各是一个单芯粒两核的整步程序。接线的规矩同 TAESD（`crates/taesd/src/step.rs`）：
//! - `cores = 2` 的 kernel（卷积、GroupNorm、线性层、注意力）两核都调，调完 `sync_cores()`；输入、输出、残差互不重叠（不原地）；
//! - 按 `num_cores()` 分行 / 分块的单核 kernel（上采样、排布转换、转置）**两核都调**，调完也 `sync_cores()`；
//! - 偏移全部是编译期常量（`const` 项），按段从 layout.rs 取（`S` 是这一段的下标，宏在调用处取它）。
//!
//! 每个 ResNet 的进出在 A、B 两张图之间轮换（这一段的第一个 ResNet 从 in_x 读、最后一步写 out_y），
//! G 放 GroupNorm 的输出、H 放 conv1 的输出（有 shortcut 的那个，GN2 读完 H 之后它放 shortcut 的结果）。
//! `host/sd15npu/cvxsim.py` 读编出来的 `.cvx` 照这条调用序列在 numpy 里重放（同一份偏移），查的就是这里的接线。
#![allow(unused_macros, unused_imports)]
use crate::kernels::*;
use crate::layout::{self as L, NGN};
use npu::prelude::*;

#[cfg(any(
    all(feature = "s1", any(feature = "s2", feature = "s3", feature = "s4", feature = "s5", feature = "s6")),
    all(feature = "s2", any(feature = "s3", feature = "s4", feature = "s5", feature = "s6")),
    all(feature = "s3", any(feature = "s4", feature = "s5", feature = "s6")),
    all(feature = "s4", any(feature = "s5", feature = "s6")),
    all(feature = "s5", feature = "s6"),
))]
compile_error!("vae：一次只开一段（每个 feature 是一个整步程序）");

// layout.rs 抄的 norm_cols 和 calblas group_norm! 里用的那份逐个相同（GN 的 w、b 按这个展开）
const _: () = {
    let mut j = 0;
    while j < NGN {
        let hw = L::GNS[j].res * L::GNS[j].res;
        assert!(L::gn_cols(j) == calblas::kernels::Cols::norm_cols(L::GNS[j].c / L::GROUPS * hw, hw), "GN 的每行像素数和 calblas 不一致");
        j += 1;
    }
};

/// 第 i 张带边框的图（全局下标；这一段 @act 里的指针，偏移先落成 const 项）。
macro_rules! m {
    ($i:expr) => {{
        const AT: u64 = L::map_off(S, $i);
        act().at(AT)
    }};
}
/// 稠密区（N1、N2、注意力的 token 矩阵）。
macro_rules! d {
    ($off:expr) => {{
        const AT: u64 = $off;
        act().at(AT)
    }};
}

/// 入口第一步：这一段带边框的图整块清零（两核各清一半）：边框与余量都靠它是 0；稠密区总是先写满再读，不清。
macro_rules! zero {
    () => {{
        const Z: usize = L::zero_blocks(S);
        const Z0: usize = Z / 2;
        const AT1: u64 = (Z0 * 16384) as u64;
        if core_rank() == 0 {
            va_zero(act().cast(), Z0);
        } else {
            va_zero(act().at(AT1).cast(), Z - Z0);
        }
        sync_cores();
    }};
}

/// 第 i 个卷积：x ⇒ y（可带残差 r），两核分算，调完屏障。
macro_rules! conv {
    ($k:ident, $i:expr, $x:expr => $y:expr) => {{
        const WO: u64 = L::w_off(S, $i);
        const BO: u64 = L::b_off(S, $i);
        $k($x.cast(), w().at(WO).cast(), w().at(BO).cast(), $y.cast());
        sync_cores();
    }};
    ($k:ident, $i:expr, $x:expr, res $r:expr => $y:expr) => {{
        const WO: u64 = L::w_off(S, $i);
        const BO: u64 = L::b_off(S, $i);
        $k($x.cast(), w().at(WO).cast(), w().at(BO).cast(), $r.cast(), $y.cast());
        sync_cores();
    }};
}

/// 第 j 个 GroupNorm（DESIGN D13）：带边框的图 x ⇒ N1（NCHW）⇒ GN（+ SiLU）⇒ N2 ⇒ 带边框的图 g（只写内部，边框还是 0）。
/// `[tc, gn, th]` 是这个形状的三个 kernel。
macro_rules! gn {
    ([$tc:ident, $gk:ident, $th:ident], $j:expr, $x:expr => $g:expr) => {{
        const GW: u64 = L::gw_off(S, $j);
        const GB: u64 = L::gb_off(S, $j);
        $tc($x.cast(), d!(L::n1(S)).cast());
        sync_cores();
        $gk(d!(L::n1(S)).cast(), w().at(GW).cast(), w().at(GB).cast(), d!(L::n2(S)).cast());
        sync_cores();
        $th(d!(L::n2(S)).cast(), $g.cast());
        sync_cores();
    }};
}

/// 不带 shortcut 的 ResNet（卷积 c、c + 1，GroupNorm n、n + 1）：x ⇒ o；g、h 是临时。
macro_rules! resnet {
    ($gk:tt, $k:ident, $kr:ident, $c:expr, $n:expr, [$x:expr, $g:expr, $h:expr, $o:expr]) => {{
        gn!($gk, $n, $x => $g);
        conv!($k, $c, $g => $h);
        gn!($gk, $n + 1, $h => $g);
        conv!($kr, $c + 1, $g, res $x => $o);
    }};
}

/// 带 1×1 shortcut 的 ResNet（卷积 c = conv1、c + 1 = conv2、c + 2 = shortcut；GroupNorm n 在宽通道上、n + 1 在窄通道上）：
/// x（宽）⇒ GN1 ⇒ u（宽）⇒ conv1 ⇒ h ⇒ GN2 ⇒ g ⇒（h 空出来）shortcut(x) ⇒ h ⇒ conv2(g) + h ⇒ o。
macro_rules! resnet_sc {
    ($gw:tt, $gn:tt, $k1:ident, $ks:ident, $kr:ident, $c:expr, $n:expr, [$x:expr, $u:expr, $g:expr, $h:expr, $o:expr]) => {{
        gn!($gw, $n, $x => $u);
        conv!($k1, $c, $u => $h);
        gn!($gn, $n + 1, $h => $g);
        conv!($ks, $c + 2, $x => $h);
        conv!($kr, $c + 1, $g, res $h => $o);
    }};
}

/// 最近邻 ×2，512 通道：按两个 256 通道的窗口各调一次（第二个窗口指针偏 256 · 2 = 512 B）。
macro_rules! up512 {
    ($k:ident, $x:expr => $y:expr) => {{
        $k($x.cast(), $y.cast());
        sync_cores();
        $k($x.at(512).cast(), $y.at(512).cast());
        sync_cores();
    }};
}

/// 这一段的四个声明，字节数照 layout.rs 核（decls! 只收字面量：改了 layout 就照报错改这里）。
macro_rules! decls_checked {
    ($s:expr, $i:literal, $o:literal, $w:literal, $a:literal) => {
        npu::decls! {
            in_x: input, bytes = $i;
            out_y: output, bytes = $o;
            w: weights, bytes = $w;
            act: workspace, bytes = $a;
        }
        const _: () = assert!(L::in_bytes($s) == $i, "in_x 的字节数");
        const _: () = assert!(L::out_bytes($s) == $o, "out_y 的字节数");
        const _: () = assert!(L::w_bytes($s) == $w, "w 的字节数");
        const _: () = assert!(L::act_bytes($s) == $a, "act 的字节数");
    };
}

/// s1：latent ⇒ post_quant_conv（1×1）⇒ L1 ⇒ conv_in ⇒ A64 ⇒ 中间块 ResNet 0 ⇒ out_y（64² × 512）。
#[cfg(feature = "s1")]
mod prog {
    use super::*;
    const S: usize = 0;
    decls_checked!(S, 295936, 4734976, 11849728, 23298048);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        zero!();
        conv!(va_pqc, 0, in_x() => m!(L::L1));
        conv!(va_cin, 1, m!(L::L1) => m!(L::A64));
        resnet!([va_tc64, va_gn64, va_th64], va_c64, va_c64r, 2, 0, [m!(L::A64), m!(L::G64), m!(L::H64), out_y()]);
    }
}

/// s2：中间块的注意力（DESIGN D14）：x ⇒ GN ⇒ token ⇒ q、k、v ⇒ softmax(q·kᵀ/√512)·v ⇒ 回到带边框的图 ⇒ proj_attn + x ⇒ out_y。
#[cfg(feature = "s2")]
mod prog {
    use super::*;
    const S: usize = 1;
    decls_checked!(S, 4734976, 4734976, 3162112, 34226176);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        zero!();
        // GN（不带 SiLU）⇒ N2 [512][4096] ⇒ 转置成 token TQ [4096][512]
        const GW: u64 = L::gw_off(S, 2);
        const GB: u64 = L::gb_off(S, 2);
        va_tc64(in_x().cast(), d!(L::n1(S)).cast());
        sync_cores();
        va_gn64n(d!(L::n1(S)).cast(), w().at(GW).cast(), w().at(GB).cast(), d!(L::n2(S)).cast());
        sync_cores();
        va_tr_ct(d!(L::n2(S)).cast(), d!(L::tok(S, 0)).cast());
        sync_cores();
        // q、k、v：三个 linear!（HF 权重 [512][512] + f32 bias）
        const QW: u64 = L::qkv_w_off(S, 0);
        const QB: u64 = L::qkv_b_off(S, 0);
        const KW: u64 = L::qkv_w_off(S, 1);
        const KB: u64 = L::qkv_b_off(S, 1);
        const VW: u64 = L::qkv_w_off(S, 2);
        const VB: u64 = L::qkv_b_off(S, 2);
        va_qkv(d!(L::tok(S, 0)).cast(), w().at(QW).cast(), w().at(QB).cast(), d!(L::tok(S, 1)).cast());
        sync_cores();
        va_qkv(d!(L::tok(S, 0)).cast(), w().at(KW).cast(), w().at(KB).cast(), d!(L::tok(S, 2)).cast());
        sync_cores();
        va_qkv(d!(L::tok(S, 0)).cast(), w().at(VW).cast(), w().at(VB).cast(), d!(L::tok(S, 3)).cast());
        sync_cores();
        // O = softmax(q·kᵀ/√512)·v ⇒ 转回 NCHW ⇒ 带边框的图 G64 ⇒ proj_attn（1×1）+ 残差 ⇒ out_y
        va_attn(d!(L::tok(S, 1)).cast(), d!(L::tok(S, 2)).cast(), d!(L::tok(S, 3)).cast(), d!(L::tok(S, 4)).cast());
        sync_cores();
        va_tr_tc(d!(L::tok(S, 4)).cast(), d!(L::n1(S)).cast());
        sync_cores();
        va_th64(d!(L::n1(S)).cast(), m!(L::G64).cast());
        sync_cores();
        conv!(va_o64r, 4, m!(L::G64), res in_x() => out_y());
    }
}

/// s3：中间块 ResNet 1 ⇒ up_blocks.0 的 3 个 ResNet（64² × 512）⇒ ×2 ⇒ G128 ⇒ 卷积 ⇒ out_y（128² × 512）。
#[cfg(feature = "s3")]
mod prog {
    use super::*;
    const S: usize = 2;
    decls_checked!(S, 4734976, 17842176, 50892800, 45826048);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        zero!();
        resnet!([va_tc64, va_gn64, va_th64], va_c64, va_c64r, 5, 3, [in_x(), m!(L::G64), m!(L::H64), m!(L::B64)]);
        resnet!([va_tc64, va_gn64, va_th64], va_c64, va_c64r, 7, 5, [m!(L::B64), m!(L::G64), m!(L::H64), m!(L::A64)]);
        resnet!([va_tc64, va_gn64, va_th64], va_c64, va_c64r, 9, 7, [m!(L::A64), m!(L::G64), m!(L::H64), m!(L::B64)]);
        resnet!([va_tc64, va_gn64, va_th64], va_c64, va_c64r, 11, 9, [m!(L::B64), m!(L::G64), m!(L::H64), m!(L::A64)]);
        up512!(va_up64, m!(L::A64) => m!(L::G128));
        conv!(va_c128, 13, m!(L::G128) => out_y());
    }
}

/// s4：up_blocks.1 的 3 个 ResNet（128² × 512）⇒ ×2 ⇒ U256 ⇒ 卷积 ⇒ out_y（256² × 512）。
#[cfg(feature = "s4")]
mod prog {
    use super::*;
    const S: usize = 3;
    decls_checked!(S, 17842176, 69222400, 58224640, 174800896);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        zero!();
        resnet!([va_tc128, va_gn128, va_th128], va_c128, va_c128r, 14, 11, [in_x(), m!(L::G128), m!(L::H128), m!(L::B128)]);
        resnet!([va_tc128, va_gn128, va_th128], va_c128, va_c128r, 16, 13, [m!(L::B128), m!(L::G128), m!(L::H128), m!(L::A128)]);
        resnet!([va_tc128, va_gn128, va_th128], va_c128, va_c128r, 18, 15, [m!(L::A128), m!(L::G128), m!(L::H128), m!(L::B128)]);
        up512!(va_up128, m!(L::B128) => m!(L::U256));
        conv!(va_c256u, 20, m!(L::U256) => out_y());
    }
}

/// s5：up_blocks.2：带 shortcut 的 ResNet（512 ⇒ 256）⇒ 2 个 ResNet（256² × 256）⇒ ×2 ⇒ U512 ⇒ 卷积 ⇒ out_y（512² × 256）。
#[cfg(feature = "s5")]
mod prog {
    use super::*;
    const S: usize = 4;
    decls_checked!(S, 69222400, 136323072, 68452352, 478674944);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        zero!();
        resnet_sc!([va_tc256w, va_gn256w, va_th256w], [va_tc256, va_gn256, va_th256], va_c256a, va_s256, va_c256r, 21, 17,
            [in_x(), m!(L::U256), m!(L::G256), m!(L::H256), m!(L::A256)]);
        resnet!([va_tc256, va_gn256, va_th256], va_c256, va_c256r, 24, 19, [m!(L::A256), m!(L::G256), m!(L::H256), m!(L::B256)]);
        resnet!([va_tc256, va_gn256, va_th256], va_c256, va_c256r, 26, 21, [m!(L::B256), m!(L::G256), m!(L::H256), m!(L::A256)]);
        va_up256(m!(L::A256).cast(), m!(L::U512).cast());
        sync_cores();
        conv!(va_c512u, 28, m!(L::U512) => out_y());
    }
}

/// s6：up_blocks.3：带 shortcut 的 ResNet（256 ⇒ 128）⇒ 2 个 ResNet（512² × 128）⇒ GN + SiLU ⇒ conv_out ⇒ out_y（[0, 1] 的图）。
#[cfg(feature = "s6")]
mod prog {
    use super::*;
    const S: usize = 5;
    decls_checked!(S, 136323072, 17040384, 136454144, 677609472);
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        zero!();
        resnet_sc!([va_tc512w, va_gn512w, va_th512w], [va_tc512, va_gn512, va_th512], va_c512a, va_s512, va_c512r, 29, 23,
            [in_x(), m!(L::U512), m!(L::G512), m!(L::H512), m!(L::A512)]);
        resnet!([va_tc512, va_gn512, va_th512], va_c512, va_c512r, 32, 25, [m!(L::A512), m!(L::G512), m!(L::H512), m!(L::B512)]);
        resnet!([va_tc512, va_gn512, va_th512], va_c512, va_c512r, 34, 27, [m!(L::B512), m!(L::G512), m!(L::H512), m!(L::A512)]);
        gn!([va_tc512, va_gn512, va_th512], 29, m!(L::A512) => m!(L::G512));
        conv!(va_cout, 36, m!(L::G512) => out_y());
    }
}
