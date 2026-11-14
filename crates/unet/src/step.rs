//! 各段入口（**`tools/unet_gen.py program` 生成，不要手改**）：偏移都是字面量，和 `crates/unet/plan.json` 出自同一次计算。
//! 接线的规矩同 VAE（crates/vae/src/step.rs）：`cores = 2` 与按 num_cores() 分行的 kernel 两核都调、不分行的单核 kernel 只让 0 号核调，
//! 每次调用之后 `sync_cores()`；输入、输出互不重叠（不原地）。CFG 的两路先后跑同一串算子：权重同一份，段内临时图同一份，
//! 段间的张量（in_x、out_y）每路一份。
use crate::kernels::*;
use npu::prelude::*;

/// s1：time、conv_in、down0.res0
#[cfg(feature = "s1")]
mod s1 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 626688;
        out_y: output, bytes = 12533760;
        w: weights, bytes = 60153856;
        act: workspace, bytes = 10207232;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 311);
            un_zero(out_y().cast(), 382);
        } else {
            un_zero(act().at(5095424).cast(), 312);
            un_zero(out_y().at(6258688).cast(), 383);
        }
        sync_cores();
        // 时间步那一路（两路共用，算一次）
        if core_rank() == 0 {
            un_temb(in_x().at(0).cast(), w().at(0).cast(), act().at(0).cast());
        }
        sync_cores();
        un_l16_384_1280(act().at(0).cast(), w().at(4096).cast(), w().at(987136).cast(), act().at(12288).cast());
        sync_cores();
        if core_rank() == 0 {
            un_silu(act().at(12288).cast(), act().at(53248).cast());
        }
        sync_cores();
        un_l16_1280_1280(act().at(53248).cast(), w().at(995328).cast(), w().at(4272128).cast(), act().at(94208).cast());
        sync_cores();
        if core_rank() == 0 {
            un_silu(act().at(94208).cast(), act().at(135168).cast());
        }
        sync_cores();
        un_l16_1280_20224(act().at(135168).cast(), w().at(4280320).cast(), w().at(56053760).cast(), act().at(176128).cast());
        sync_cores();
        if core_rank() == 0 {
            un_bf(w().at(56135680).cast(), act().at(176128).cast(), out_y().at(0).cast());
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // conv_in
        un_c3_64_32_320(in_x().at(4096).cast(), w().at(56217600).cast(), w().at(56401920).cast(), out_y().at(81920).cast());
        sync_cores();
        // down0.res0
        un_gs_64_320(out_y().at(81920).cast(), act().at(823296).cast());
        sync_cores();
        un_ga_64_320_silu(out_y().at(81920).cast(), act().at(823296).cast(), w().at(56406016).cast(), w().at(56410112).cast(), w().at(56414208).cast(), act().at(864256).cast());
        sync_cores();
        un_c3_64_320_320(act().at(864256).cast(), w().at(56455168).cast(), out_y().at(0).cast(), act().at(3977216).cast());
        sync_cores();
        un_gs_64_320(act().at(3977216).cast(), act().at(823296).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(3977216).cast(), act().at(823296).cast(), w().at(58298368).cast(), w().at(58302464).cast(), w().at(56414208).cast(), act().at(7090176).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(7090176).cast(), w().at(58306560).cast(), w().at(60149760).cast(), out_y().at(81920).cast(), out_y().at(3194880).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // conv_in
        un_c3_64_32_320(in_x().at(315392).cast(), w().at(56217600).cast(), w().at(56401920).cast(), out_y().at(6307840).cast());
        sync_cores();
        // down0.res0
        un_gs_64_320(out_y().at(6307840).cast(), act().at(823296).cast());
        sync_cores();
        un_ga_64_320_silu(out_y().at(6307840).cast(), act().at(823296).cast(), w().at(56406016).cast(), w().at(56410112).cast(), w().at(56414208).cast(), act().at(864256).cast());
        sync_cores();
        un_c3_64_320_320(act().at(864256).cast(), w().at(56455168).cast(), out_y().at(0).cast(), act().at(3977216).cast());
        sync_cores();
        un_gs_64_320(act().at(3977216).cast(), act().at(823296).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(3977216).cast(), act().at(823296).cast(), w().at(58298368).cast(), w().at(58302464).cast(), w().at(56414208).cast(), act().at(7090176).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(7090176).cast(), w().at(58306560).cast(), w().at(60149760).cast(), out_y().at(6307840).cast(), out_y().at(9420800).cast());
        sync_cores();
    }
}

/// s2：down0.attn0
#[cfg(feature = "s2")]
mod s2 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 6619136;
        out_y: output, bytes = 6225920;
        w: weights, bytes = 6537216;
        act: workspace, bytes = 92454912;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2821);
            un_zero(out_y().cast(), 190);
        } else {
            un_zero(act().at(46219264).cast(), 2822);
            un_zero(out_y().at(3112960).cast(), 190);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down0.attn0
        un_gs_64_320(in_x().at(0).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(0).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(0).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down0.attn0
        un_gs_64_320(in_x().at(3309568).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(3309568).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(3309568).cast(), out_y().at(3112960).cast());
        sync_cores();
    }
}

/// s3：down0.res1
#[cfg(feature = "s3")]
mod s3 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 6307840;
        out_y: output, bytes = 6225920;
        w: weights, bytes = 3747840;
        act: workspace, bytes = 9388032;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 286);
            un_zero(out_y().cast(), 190);
        } else {
            un_zero(act().at(4685824).cast(), 287);
            un_zero(out_y().at(3112960).cast(), 190);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down0.res1
        un_gs_64_320(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_silu(in_x().at(81920).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_64_320_320(act().at(40960).cast(), w().at(49152).cast(), in_x().at(1280).cast(), act().at(3153920).cast());
        sync_cores();
        un_gs_64_320(act().at(3153920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(3153920).cast(), act().at(0).cast(), w().at(1892352).cast(), w().at(1896448).cast(), w().at(8192).cast(), act().at(6266880).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(6266880).cast(), w().at(1900544).cast(), w().at(3743744).cast(), in_x().at(81920).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down0.res1
        un_gs_64_320(in_x().at(3194880).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_silu(in_x().at(3194880).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_64_320_320(act().at(40960).cast(), w().at(49152).cast(), in_x().at(1280).cast(), act().at(3153920).cast());
        sync_cores();
        un_gs_64_320(act().at(3153920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(3153920).cast(), act().at(0).cast(), w().at(1892352).cast(), w().at(1896448).cast(), w().at(8192).cast(), act().at(6266880).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(6266880).cast(), w().at(1900544).cast(), w().at(3743744).cast(), in_x().at(3194880).cast(), out_y().at(3112960).cast());
        sync_cores();
    }
}

/// s4：down0.attn1、down0.down
#[cfg(feature = "s4")]
mod s4 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 6619136;
        out_y: output, bytes = 8192000;
        w: weights, bytes = 8384512;
        act: workspace, bytes = 92454912;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2821);
            un_zero(out_y().cast(), 250);
        } else {
            un_zero(act().at(46219264).cast(), 2822);
            un_zero(out_y().at(4096000).cast(), 250);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down0.attn1
        un_gs_64_320(in_x().at(0).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(0).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(0).cast(), out_y().at(0).cast());
        sync_cores();
        // down0.down
        un_c3_32_320_320_s2(out_y().at(0).cast(), w().at(6537216).cast(), w().at(8380416).cast(), out_y().at(3112960).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down0.attn1
        un_gs_64_320(in_x().at(3309568).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(3309568).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(3309568).cast(), out_y().at(4096000).cast());
        sync_cores();
        // down0.down
        un_c3_32_320_320_s2(out_y().at(4096000).cast(), w().at(6537216).cast(), w().at(8380416).cast(), out_y().at(7208960).cast());
        sync_cores();
    }
}

/// s5：down1.res0、down1.attn0、down1.res1、down1.attn1、down1.down
#[cfg(feature = "s5")]
mod s5 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 2441216;
        out_y: output, bytes = 9502720;
        w: weights, bytes = 73703424;
        act: workspace, bytes = 125288448;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 3823);
            un_zero(out_y().cast(), 290);
        } else {
            un_zero(act().at(62636032).cast(), 3824);
            un_zero(out_y().at(4751360).cast(), 290);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down1.res0
        un_gs_32_320(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_320_silu(in_x().at(81920).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_32_320_640(act().at(40960).cast(), w().at(49152).cast(), in_x().at(2560).cast(), act().at(1024000).cast());
        sync_cores();
        un_gs_32_640(act().at(1024000).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(1024000).cast(), act().at(0).cast(), w().at(3735552).cast(), w().at(3739648).cast(), w().at(3743744).cast(), act().at(2990080).cast());
        sync_cores();
        un_c1_32_320_640(in_x().at(81920).cast(), w().at(3825664).cast(), w().at(4235264).cast(), act().at(4956160).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(2990080).cast(), w().at(4239360).cast(), w().at(11612160).cast(), act().at(4956160).cast(), act().at(6922240).cast());
        sync_cores();
        // down1.attn0
        un_gs_32_640(act().at(6922240).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(6922240).cast(), act().at(0).cast(), w().at(11616256).cast(), w().at(11620352).cast(), w().at(3743744).cast(), act().at(8888320).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(8888320).cast(), w().at(11624448).cast(), w().at(12443648).cast(), act().at(10854400).cast());
        sync_cores();
        un_ln_32_640(act().at(10854400).cast(), w().at(12447744).cast(), w().at(12451840).cast(), act().at(12820480).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(12455936).cast(), act().at(14786560).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(13438976).cast(), act().at(17145856).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(14422016).cast(), act().at(19505152).cast());
        sync_cores();
        un_at_32_s(act().at(14786560).cast(), act().at(17145856).cast(), act().at(19505152).cast(), w().at(15405056).cast(), act().at(21864448).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(21864448).cast(), w().at(15409152).cast(), w().at(16392192).cast(), act().at(10854400).cast(), act().at(24223744).cast());
        sync_cores();
        un_ln_32_640(act().at(24223744).cast(), w().at(16396288).cast(), w().at(16400384).cast(), act().at(26189824).cast());
        sync_cores();
        un_lr_32_640_768(act().at(26189824).cast(), w().at(16404480).cast(), act().at(28155904).cast());
        sync_cores();
        un_kv_768(in_x().at(1064960).cast(), w().at(17387520).cast(), act().at(30515200).cast());
        sync_cores();
        un_kv_768(in_x().at(1064960).cast(), w().at(18567168).cast(), act().at(30711808).cast());
        sync_cores();
        un_at_32_x(act().at(28155904).cast(), act().at(30515200).cast(), act().at(30711808).cast(), w().at(19746816).cast(), act().at(30908416).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(30908416).cast(), w().at(19750912).cast(), w().at(20733952).cast(), act().at(24223744).cast(), act().at(33267712).cast());
        sync_cores();
        un_ln_32_640(act().at(33267712).cast(), w().at(20738048).cast(), w().at(20742144).cast(), act().at(35233792).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(35233792).cast(), w().at(20746240).cast(), w().at(24023040).cast(), act().at(37199872).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(35233792).cast(), w().at(24035328).cast(), w().at(27312128).cast(), act().at(45064192).cast());
        sync_cores();
        un_gg_32(act().at(37199872).cast(), act().at(45064192).cast(), act().at(52928512).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(52928512).cast(), w().at(27324416).cast(), w().at(30601216).cast(), act().at(33267712).cast(), act().at(60792832).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(60792832).cast(), w().at(30605312).cast(), w().at(31588352).cast(), act().at(6922240).cast(), out_y().at(0).cast());
        sync_cores();
        // down1.res1
        un_gs_32_640(out_y().at(0).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(out_y().at(0).cast(), act().at(0).cast(), w().at(31592448).cast(), w().at(31596544).cast(), w().at(3743744).cast(), act().at(63152128).cast());
        sync_cores();
        un_c3_32_640_640(act().at(63152128).cast(), w().at(31600640).cast(), in_x().at(5120).cast(), act().at(65118208).cast());
        sync_cores();
        un_gs_32_640(act().at(65118208).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(65118208).cast(), act().at(0).cast(), w().at(38973440).cast(), w().at(38977536).cast(), w().at(3743744).cast(), act().at(67084288).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(67084288).cast(), w().at(38981632).cast(), w().at(46354432).cast(), out_y().at(0).cast(), act().at(69050368).cast());
        sync_cores();
        // down1.attn1
        un_gs_32_640(act().at(69050368).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(69050368).cast(), act().at(0).cast(), w().at(46358528).cast(), w().at(46362624).cast(), w().at(3743744).cast(), act().at(71016448).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(71016448).cast(), w().at(46366720).cast(), w().at(47185920).cast(), act().at(72982528).cast());
        sync_cores();
        un_ln_32_640(act().at(72982528).cast(), w().at(47190016).cast(), w().at(47194112).cast(), act().at(74948608).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(47198208).cast(), act().at(76914688).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(48181248).cast(), act().at(79273984).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(49164288).cast(), act().at(81633280).cast());
        sync_cores();
        un_at_32_s(act().at(76914688).cast(), act().at(79273984).cast(), act().at(81633280).cast(), w().at(15405056).cast(), act().at(83992576).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(83992576).cast(), w().at(50147328).cast(), w().at(51130368).cast(), act().at(72982528).cast(), act().at(86351872).cast());
        sync_cores();
        un_ln_32_640(act().at(86351872).cast(), w().at(51134464).cast(), w().at(51138560).cast(), act().at(88317952).cast());
        sync_cores();
        un_lr_32_640_768(act().at(88317952).cast(), w().at(51142656).cast(), act().at(90284032).cast());
        sync_cores();
        un_kv_768(in_x().at(1064960).cast(), w().at(52125696).cast(), act().at(92643328).cast());
        sync_cores();
        un_kv_768(in_x().at(1064960).cast(), w().at(53305344).cast(), act().at(92839936).cast());
        sync_cores();
        un_at_32_x(act().at(90284032).cast(), act().at(92643328).cast(), act().at(92839936).cast(), w().at(19746816).cast(), act().at(93036544).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(93036544).cast(), w().at(54484992).cast(), w().at(55468032).cast(), act().at(86351872).cast(), act().at(95395840).cast());
        sync_cores();
        un_ln_32_640(act().at(95395840).cast(), w().at(55472128).cast(), w().at(55476224).cast(), act().at(97361920).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(97361920).cast(), w().at(55480320).cast(), w().at(58757120).cast(), act().at(99328000).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(97361920).cast(), w().at(58769408).cast(), w().at(62046208).cast(), act().at(107192320).cast());
        sync_cores();
        un_gg_32(act().at(99328000).cast(), act().at(107192320).cast(), act().at(115056640).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(115056640).cast(), w().at(62058496).cast(), w().at(65335296).cast(), act().at(95395840).cast(), act().at(122920960).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(122920960).cast(), w().at(65339392).cast(), w().at(66322432).cast(), act().at(69050368).cast(), out_y().at(1966080).cast());
        sync_cores();
        // down1.down
        un_c3_16_640_640_s2(out_y().at(1966080).cast(), w().at(66326528).cast(), w().at(73699328).cast(), out_y().at(3932160).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down1.res0
        un_gs_32_320(in_x().at(1261568).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_320_silu(in_x().at(1261568).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_32_320_640(act().at(40960).cast(), w().at(49152).cast(), in_x().at(2560).cast(), act().at(1024000).cast());
        sync_cores();
        un_gs_32_640(act().at(1024000).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(1024000).cast(), act().at(0).cast(), w().at(3735552).cast(), w().at(3739648).cast(), w().at(3743744).cast(), act().at(2990080).cast());
        sync_cores();
        un_c1_32_320_640(in_x().at(1261568).cast(), w().at(3825664).cast(), w().at(4235264).cast(), act().at(4956160).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(2990080).cast(), w().at(4239360).cast(), w().at(11612160).cast(), act().at(4956160).cast(), act().at(6922240).cast());
        sync_cores();
        // down1.attn0
        un_gs_32_640(act().at(6922240).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(6922240).cast(), act().at(0).cast(), w().at(11616256).cast(), w().at(11620352).cast(), w().at(3743744).cast(), act().at(8888320).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(8888320).cast(), w().at(11624448).cast(), w().at(12443648).cast(), act().at(10854400).cast());
        sync_cores();
        un_ln_32_640(act().at(10854400).cast(), w().at(12447744).cast(), w().at(12451840).cast(), act().at(12820480).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(12455936).cast(), act().at(14786560).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(13438976).cast(), act().at(17145856).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(14422016).cast(), act().at(19505152).cast());
        sync_cores();
        un_at_32_s(act().at(14786560).cast(), act().at(17145856).cast(), act().at(19505152).cast(), w().at(15405056).cast(), act().at(21864448).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(21864448).cast(), w().at(15409152).cast(), w().at(16392192).cast(), act().at(10854400).cast(), act().at(24223744).cast());
        sync_cores();
        un_ln_32_640(act().at(24223744).cast(), w().at(16396288).cast(), w().at(16400384).cast(), act().at(26189824).cast());
        sync_cores();
        un_lr_32_640_768(act().at(26189824).cast(), w().at(16404480).cast(), act().at(28155904).cast());
        sync_cores();
        un_kv_768(in_x().at(2244608).cast(), w().at(17387520).cast(), act().at(30515200).cast());
        sync_cores();
        un_kv_768(in_x().at(2244608).cast(), w().at(18567168).cast(), act().at(30711808).cast());
        sync_cores();
        un_at_32_x(act().at(28155904).cast(), act().at(30515200).cast(), act().at(30711808).cast(), w().at(19746816).cast(), act().at(30908416).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(30908416).cast(), w().at(19750912).cast(), w().at(20733952).cast(), act().at(24223744).cast(), act().at(33267712).cast());
        sync_cores();
        un_ln_32_640(act().at(33267712).cast(), w().at(20738048).cast(), w().at(20742144).cast(), act().at(35233792).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(35233792).cast(), w().at(20746240).cast(), w().at(24023040).cast(), act().at(37199872).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(35233792).cast(), w().at(24035328).cast(), w().at(27312128).cast(), act().at(45064192).cast());
        sync_cores();
        un_gg_32(act().at(37199872).cast(), act().at(45064192).cast(), act().at(52928512).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(52928512).cast(), w().at(27324416).cast(), w().at(30601216).cast(), act().at(33267712).cast(), act().at(60792832).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(60792832).cast(), w().at(30605312).cast(), w().at(31588352).cast(), act().at(6922240).cast(), out_y().at(4751360).cast());
        sync_cores();
        // down1.res1
        un_gs_32_640(out_y().at(4751360).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(out_y().at(4751360).cast(), act().at(0).cast(), w().at(31592448).cast(), w().at(31596544).cast(), w().at(3743744).cast(), act().at(63152128).cast());
        sync_cores();
        un_c3_32_640_640(act().at(63152128).cast(), w().at(31600640).cast(), in_x().at(5120).cast(), act().at(65118208).cast());
        sync_cores();
        un_gs_32_640(act().at(65118208).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(65118208).cast(), act().at(0).cast(), w().at(38973440).cast(), w().at(38977536).cast(), w().at(3743744).cast(), act().at(67084288).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(67084288).cast(), w().at(38981632).cast(), w().at(46354432).cast(), out_y().at(4751360).cast(), act().at(69050368).cast());
        sync_cores();
        // down1.attn1
        un_gs_32_640(act().at(69050368).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(69050368).cast(), act().at(0).cast(), w().at(46358528).cast(), w().at(46362624).cast(), w().at(3743744).cast(), act().at(71016448).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(71016448).cast(), w().at(46366720).cast(), w().at(47185920).cast(), act().at(72982528).cast());
        sync_cores();
        un_ln_32_640(act().at(72982528).cast(), w().at(47190016).cast(), w().at(47194112).cast(), act().at(74948608).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(47198208).cast(), act().at(76914688).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(48181248).cast(), act().at(79273984).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(49164288).cast(), act().at(81633280).cast());
        sync_cores();
        un_at_32_s(act().at(76914688).cast(), act().at(79273984).cast(), act().at(81633280).cast(), w().at(15405056).cast(), act().at(83992576).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(83992576).cast(), w().at(50147328).cast(), w().at(51130368).cast(), act().at(72982528).cast(), act().at(86351872).cast());
        sync_cores();
        un_ln_32_640(act().at(86351872).cast(), w().at(51134464).cast(), w().at(51138560).cast(), act().at(88317952).cast());
        sync_cores();
        un_lr_32_640_768(act().at(88317952).cast(), w().at(51142656).cast(), act().at(90284032).cast());
        sync_cores();
        un_kv_768(in_x().at(2244608).cast(), w().at(52125696).cast(), act().at(92643328).cast());
        sync_cores();
        un_kv_768(in_x().at(2244608).cast(), w().at(53305344).cast(), act().at(92839936).cast());
        sync_cores();
        un_at_32_x(act().at(90284032).cast(), act().at(92643328).cast(), act().at(92839936).cast(), w().at(19746816).cast(), act().at(93036544).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(93036544).cast(), w().at(54484992).cast(), w().at(55468032).cast(), act().at(86351872).cast(), act().at(95395840).cast());
        sync_cores();
        un_ln_32_640(act().at(95395840).cast(), w().at(55472128).cast(), w().at(55476224).cast(), act().at(97361920).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(97361920).cast(), w().at(55480320).cast(), w().at(58757120).cast(), act().at(99328000).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(97361920).cast(), w().at(58769408).cast(), w().at(62046208).cast(), act().at(107192320).cast());
        sync_cores();
        un_gg_32(act().at(99328000).cast(), act().at(107192320).cast(), act().at(115056640).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(115056640).cast(), w().at(62058496).cast(), w().at(65335296).cast(), act().at(95395840).cast(), act().at(122920960).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(122920960).cast(), w().at(65339392).cast(), w().at(66322432).cast(), act().at(69050368).cast(), out_y().at(6717440).cast());
        sync_cores();
        // down1.down
        un_c3_16_640_640_s2(out_y().at(6717440).cast(), w().at(66326528).cast(), w().at(73699328).cast(), out_y().at(8683520).cast());
        sync_cores();
    }
}

/// s6：down2.res0、down2.attn0
#[cfg(feature = "s6")]
mod s6 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 2113536;
        out_y: output, bytes = 3276800;
        w: weights, bytes = 115793920;
        act: workspace, bytes = 50675712;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 1546);
            un_zero(out_y().cast(), 100);
        } else {
            un_zero(act().at(25329664).cast(), 1547);
            un_zero(out_y().at(1638400).cast(), 100);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down2.res0
        un_gs_16_640(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_640_silu(in_x().at(81920).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_16_640_1280(act().at(40960).cast(), w().at(90112).cast(), in_x().at(7680).cast(), act().at(860160).cast());
        sync_cores();
        un_gs_16_1280(act().at(860160).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(860160).cast(), act().at(0).cast(), w().at(14835712).cast(), w().at(14843904).cast(), w().at(14852096).cast(), act().at(2498560).cast());
        sync_cores();
        un_c1_16_640_1280(in_x().at(81920).cast(), w().at(15015936).cast(), w().at(16654336).cast(), act().at(4136960).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(2498560).cast(), w().at(16662528).cast(), w().at(46153728).cast(), act().at(4136960).cast(), act().at(5775360).cast());
        sync_cores();
        // down2.attn0
        un_gs_16_1280(act().at(5775360).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(5775360).cast(), act().at(0).cast(), w().at(46161920).cast(), w().at(46170112).cast(), w().at(14852096).cast(), act().at(7413760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(7413760).cast(), w().at(46178304).cast(), w().at(49455104).cast(), act().at(9052160).cast());
        sync_cores();
        un_ln_16_1280(act().at(9052160).cast(), w().at(49463296).cast(), w().at(49471488).cast(), act().at(10690560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(10690560).cast(), w().at(49479680).cast(), w().at(52756480).cast(), act().at(12328960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(10690560).cast(), w().at(52764672).cast(), w().at(52756480).cast(), act().at(13967360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(10690560).cast(), w().at(56041472).cast(), w().at(52756480).cast(), act().at(15605760).cast());
        sync_cores();
        un_at_16_s(act().at(12328960).cast(), act().at(13967360).cast(), act().at(15605760).cast(), w().at(59318272).cast(), act().at(17244160).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(17244160).cast(), w().at(59322368).cast(), w().at(62599168).cast(), act().at(9052160).cast(), act().at(18882560).cast());
        sync_cores();
        un_ln_16_1280(act().at(18882560).cast(), w().at(62607360).cast(), w().at(62615552).cast(), act().at(20520960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(20520960).cast(), w().at(62623744).cast(), w().at(52756480).cast(), act().at(22159360).cast());
        sync_cores();
        un_kv_1280(in_x().at(901120).cast(), w().at(65900544).cast(), act().at(23797760).cast());
        sync_cores();
        un_kv_1280(in_x().at(901120).cast(), w().at(67866624).cast(), act().at(24125440).cast());
        sync_cores();
        un_at_16_x(act().at(22159360).cast(), act().at(23797760).cast(), act().at(24125440).cast(), w().at(69832704).cast(), act().at(24453120).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(24453120).cast(), w().at(69836800).cast(), w().at(73113600).cast(), act().at(18882560).cast(), act().at(26091520).cast());
        sync_cores();
        un_ln_16_1280(act().at(26091520).cast(), w().at(73121792).cast(), w().at(73129984).cast(), act().at(27729920).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(27729920).cast(), w().at(73138176).cast(), w().at(86245376).cast(), act().at(29368320).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(27729920).cast(), w().at(86265856).cast(), w().at(99373056).cast(), act().at(35921920).cast());
        sync_cores();
        un_gg_16(act().at(29368320).cast(), act().at(35921920).cast(), act().at(42475520).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(42475520).cast(), w().at(99393536).cast(), w().at(112500736).cast(), act().at(26091520).cast(), act().at(49029120).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(49029120).cast(), w().at(112508928).cast(), w().at(115785728).cast(), act().at(5775360).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down2.res0
        un_gs_16_640(in_x().at(1097728).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_640_silu(in_x().at(1097728).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_16_640_1280(act().at(40960).cast(), w().at(90112).cast(), in_x().at(7680).cast(), act().at(860160).cast());
        sync_cores();
        un_gs_16_1280(act().at(860160).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(860160).cast(), act().at(0).cast(), w().at(14835712).cast(), w().at(14843904).cast(), w().at(14852096).cast(), act().at(2498560).cast());
        sync_cores();
        un_c1_16_640_1280(in_x().at(1097728).cast(), w().at(15015936).cast(), w().at(16654336).cast(), act().at(4136960).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(2498560).cast(), w().at(16662528).cast(), w().at(46153728).cast(), act().at(4136960).cast(), act().at(5775360).cast());
        sync_cores();
        // down2.attn0
        un_gs_16_1280(act().at(5775360).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(5775360).cast(), act().at(0).cast(), w().at(46161920).cast(), w().at(46170112).cast(), w().at(14852096).cast(), act().at(7413760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(7413760).cast(), w().at(46178304).cast(), w().at(49455104).cast(), act().at(9052160).cast());
        sync_cores();
        un_ln_16_1280(act().at(9052160).cast(), w().at(49463296).cast(), w().at(49471488).cast(), act().at(10690560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(10690560).cast(), w().at(49479680).cast(), w().at(52756480).cast(), act().at(12328960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(10690560).cast(), w().at(52764672).cast(), w().at(52756480).cast(), act().at(13967360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(10690560).cast(), w().at(56041472).cast(), w().at(52756480).cast(), act().at(15605760).cast());
        sync_cores();
        un_at_16_s(act().at(12328960).cast(), act().at(13967360).cast(), act().at(15605760).cast(), w().at(59318272).cast(), act().at(17244160).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(17244160).cast(), w().at(59322368).cast(), w().at(62599168).cast(), act().at(9052160).cast(), act().at(18882560).cast());
        sync_cores();
        un_ln_16_1280(act().at(18882560).cast(), w().at(62607360).cast(), w().at(62615552).cast(), act().at(20520960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(20520960).cast(), w().at(62623744).cast(), w().at(52756480).cast(), act().at(22159360).cast());
        sync_cores();
        un_kv_1280(in_x().at(1916928).cast(), w().at(65900544).cast(), act().at(23797760).cast());
        sync_cores();
        un_kv_1280(in_x().at(1916928).cast(), w().at(67866624).cast(), act().at(24125440).cast());
        sync_cores();
        un_at_16_x(act().at(22159360).cast(), act().at(23797760).cast(), act().at(24125440).cast(), w().at(69832704).cast(), act().at(24453120).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(24453120).cast(), w().at(69836800).cast(), w().at(73113600).cast(), act().at(18882560).cast(), act().at(26091520).cast());
        sync_cores();
        un_ln_16_1280(act().at(26091520).cast(), w().at(73121792).cast(), w().at(73129984).cast(), act().at(27729920).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(27729920).cast(), w().at(73138176).cast(), w().at(86245376).cast(), act().at(29368320).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(27729920).cast(), w().at(86265856).cast(), w().at(99373056).cast(), act().at(35921920).cast());
        sync_cores();
        un_gg_16(act().at(29368320).cast(), act().at(35921920).cast(), act().at(42475520).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(42475520).cast(), w().at(99393536).cast(), w().at(112500736).cast(), act().at(26091520).cast(), act().at(49029120).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(49029120).cast(), w().at(112508928).cast(), w().at(115785728).cast(), act().at(5775360).cast(), out_y().at(1638400).cast());
        sync_cores();
    }
}

/// s7：down2.res1、down2.attn1
#[cfg(feature = "s7")]
mod s7 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 3751936;
        out_y: output, bytes = 3276800;
        w: weights, bytes = 128819200;
        act: workspace, bytes = 49856512;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 1521);
            un_zero(out_y().cast(), 100);
        } else {
            un_zero(act().at(24920064).cast(), 1522);
            un_zero(out_y().at(1638400).cast(), 100);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down2.res1
        un_gs_16_1280(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_silu(in_x().at(81920).cast(), act().at(0).cast(), w().at(0).cast(), w().at(8192).cast(), w().at(16384).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_16_1280_1280(act().at(40960).cast(), w().at(180224).cast(), in_x().at(12800).cast(), act().at(1679360).cast());
        sync_cores();
        un_gs_16_1280(act().at(1679360).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(1679360).cast(), act().at(0).cast(), w().at(29671424).cast(), w().at(29679616).cast(), w().at(16384).cast(), act().at(3317760).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(3317760).cast(), w().at(29687808).cast(), w().at(59179008).cast(), in_x().at(81920).cast(), act().at(4956160).cast());
        sync_cores();
        // down2.attn1
        un_gs_16_1280(act().at(4956160).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(4956160).cast(), act().at(0).cast(), w().at(59187200).cast(), w().at(59195392).cast(), w().at(16384).cast(), act().at(6594560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(6594560).cast(), w().at(59203584).cast(), w().at(62480384).cast(), act().at(8232960).cast());
        sync_cores();
        un_ln_16_1280(act().at(8232960).cast(), w().at(62488576).cast(), w().at(62496768).cast(), act().at(9871360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(9871360).cast(), w().at(62504960).cast(), w().at(65781760).cast(), act().at(11509760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(9871360).cast(), w().at(65789952).cast(), w().at(65781760).cast(), act().at(13148160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(9871360).cast(), w().at(69066752).cast(), w().at(65781760).cast(), act().at(14786560).cast());
        sync_cores();
        un_at_16_s(act().at(11509760).cast(), act().at(13148160).cast(), act().at(14786560).cast(), w().at(72343552).cast(), act().at(16424960).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(16424960).cast(), w().at(72347648).cast(), w().at(75624448).cast(), act().at(8232960).cast(), act().at(18063360).cast());
        sync_cores();
        un_ln_16_1280(act().at(18063360).cast(), w().at(75632640).cast(), w().at(75640832).cast(), act().at(19701760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(19701760).cast(), w().at(75649024).cast(), w().at(65781760).cast(), act().at(21340160).cast());
        sync_cores();
        un_kv_1280(in_x().at(1720320).cast(), w().at(78925824).cast(), act().at(22978560).cast());
        sync_cores();
        un_kv_1280(in_x().at(1720320).cast(), w().at(80891904).cast(), act().at(23306240).cast());
        sync_cores();
        un_at_16_x(act().at(21340160).cast(), act().at(22978560).cast(), act().at(23306240).cast(), w().at(82857984).cast(), act().at(23633920).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(23633920).cast(), w().at(82862080).cast(), w().at(86138880).cast(), act().at(18063360).cast(), act().at(25272320).cast());
        sync_cores();
        un_ln_16_1280(act().at(25272320).cast(), w().at(86147072).cast(), w().at(86155264).cast(), act().at(26910720).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(26910720).cast(), w().at(86163456).cast(), w().at(99270656).cast(), act().at(28549120).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(26910720).cast(), w().at(99291136).cast(), w().at(112398336).cast(), act().at(35102720).cast());
        sync_cores();
        un_gg_16(act().at(28549120).cast(), act().at(35102720).cast(), act().at(41656320).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(41656320).cast(), w().at(112418816).cast(), w().at(125526016).cast(), act().at(25272320).cast(), act().at(48209920).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(48209920).cast(), w().at(125534208).cast(), w().at(128811008).cast(), act().at(4956160).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down2.res1
        un_gs_16_1280(in_x().at(1916928).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_silu(in_x().at(1916928).cast(), act().at(0).cast(), w().at(0).cast(), w().at(8192).cast(), w().at(16384).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_16_1280_1280(act().at(40960).cast(), w().at(180224).cast(), in_x().at(12800).cast(), act().at(1679360).cast());
        sync_cores();
        un_gs_16_1280(act().at(1679360).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(1679360).cast(), act().at(0).cast(), w().at(29671424).cast(), w().at(29679616).cast(), w().at(16384).cast(), act().at(3317760).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(3317760).cast(), w().at(29687808).cast(), w().at(59179008).cast(), in_x().at(1916928).cast(), act().at(4956160).cast());
        sync_cores();
        // down2.attn1
        un_gs_16_1280(act().at(4956160).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(4956160).cast(), act().at(0).cast(), w().at(59187200).cast(), w().at(59195392).cast(), w().at(16384).cast(), act().at(6594560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(6594560).cast(), w().at(59203584).cast(), w().at(62480384).cast(), act().at(8232960).cast());
        sync_cores();
        un_ln_16_1280(act().at(8232960).cast(), w().at(62488576).cast(), w().at(62496768).cast(), act().at(9871360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(9871360).cast(), w().at(62504960).cast(), w().at(65781760).cast(), act().at(11509760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(9871360).cast(), w().at(65789952).cast(), w().at(65781760).cast(), act().at(13148160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(9871360).cast(), w().at(69066752).cast(), w().at(65781760).cast(), act().at(14786560).cast());
        sync_cores();
        un_at_16_s(act().at(11509760).cast(), act().at(13148160).cast(), act().at(14786560).cast(), w().at(72343552).cast(), act().at(16424960).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(16424960).cast(), w().at(72347648).cast(), w().at(75624448).cast(), act().at(8232960).cast(), act().at(18063360).cast());
        sync_cores();
        un_ln_16_1280(act().at(18063360).cast(), w().at(75632640).cast(), w().at(75640832).cast(), act().at(19701760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(19701760).cast(), w().at(75649024).cast(), w().at(65781760).cast(), act().at(21340160).cast());
        sync_cores();
        un_kv_1280(in_x().at(3555328).cast(), w().at(78925824).cast(), act().at(22978560).cast());
        sync_cores();
        un_kv_1280(in_x().at(3555328).cast(), w().at(80891904).cast(), act().at(23306240).cast());
        sync_cores();
        un_at_16_x(act().at(21340160).cast(), act().at(22978560).cast(), act().at(23306240).cast(), w().at(82857984).cast(), act().at(23633920).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(23633920).cast(), w().at(82862080).cast(), w().at(86138880).cast(), act().at(18063360).cast(), act().at(25272320).cast());
        sync_cores();
        un_ln_16_1280(act().at(25272320).cast(), w().at(86147072).cast(), w().at(86155264).cast(), act().at(26910720).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(26910720).cast(), w().at(86163456).cast(), w().at(99270656).cast(), act().at(28549120).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(26910720).cast(), w().at(99291136).cast(), w().at(112398336).cast(), act().at(35102720).cast());
        sync_cores();
        un_gg_16(act().at(28549120).cast(), act().at(35102720).cast(), act().at(41656320).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(41656320).cast(), w().at(112418816).cast(), w().at(125526016).cast(), act().at(25272320).cast(), act().at(48209920).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(48209920).cast(), w().at(125534208).cast(), w().at(128811008).cast(), act().at(4956160).cast(), out_y().at(1638400).cast());
        sync_cores();
    }
}

/// s8：down2.down、down3.res0、down3.res1、mid.res0、mid.attn、mid.res1、up0.res0、up0.res1、up0.res2
#[cfg(feature = "s8")]
mod s8 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 3751936;
        out_y: output, bytes = 1966080;
        w: weights, bytes = 620969984;
        act: workspace, bytes = 66568192;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2031);
            un_zero(out_y().cast(), 60);
        } else {
            un_zero(act().at(33275904).cast(), 2032);
            un_zero(out_y().at(983040).cast(), 60);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down2.down
        un_c3_8_1280_1280_s2(in_x().at(81920).cast(), w().at(0).cast(), w().at(29491200).cast(), act().at(0).cast());
        sync_cores();
        // down3.res0
        un_gs_8_1280(act().at(0).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(0).cast(), act().at(983040).cast(), w().at(29499392).cast(), w().at(29507584).cast(), w().at(29515776).cast(), act().at(1024000).cast());
        sync_cores();
        un_c3_8_1280_1280(act().at(1024000).cast(), w().at(29679616).cast(), in_x().at(17920).cast(), act().at(2007040).cast());
        sync_cores();
        un_gs_8_1280(act().at(2007040).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(2007040).cast(), act().at(983040).cast(), w().at(59170816).cast(), w().at(59179008).cast(), w().at(29515776).cast(), act().at(2990080).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(2990080).cast(), w().at(59187200).cast(), w().at(88678400).cast(), act().at(0).cast(), act().at(3973120).cast());
        sync_cores();
        // down3.res1
        un_gs_8_1280(act().at(3973120).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(3973120).cast(), act().at(983040).cast(), w().at(88686592).cast(), w().at(88694784).cast(), w().at(29515776).cast(), act().at(4956160).cast());
        sync_cores();
        un_c3_8_1280_1280(act().at(4956160).cast(), w().at(88702976).cast(), in_x().at(23040).cast(), act().at(5939200).cast());
        sync_cores();
        un_gs_8_1280(act().at(5939200).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(5939200).cast(), act().at(983040).cast(), w().at(118194176).cast(), w().at(118202368).cast(), w().at(29515776).cast(), act().at(6922240).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(6922240).cast(), w().at(118210560).cast(), w().at(147701760).cast(), act().at(3973120).cast(), act().at(7905280).cast());
        sync_cores();
        // mid.res0
        un_gs_8_1280(act().at(7905280).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(7905280).cast(), act().at(983040).cast(), w().at(147709952).cast(), w().at(147718144).cast(), w().at(29515776).cast(), act().at(8888320).cast());
        sync_cores();
        un_c3_8_1280_1280(act().at(8888320).cast(), w().at(147726336).cast(), in_x().at(28160).cast(), act().at(9871360).cast());
        sync_cores();
        un_gs_8_1280(act().at(9871360).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(9871360).cast(), act().at(983040).cast(), w().at(177217536).cast(), w().at(177225728).cast(), w().at(29515776).cast(), act().at(10854400).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(10854400).cast(), w().at(177233920).cast(), w().at(206725120).cast(), act().at(7905280).cast(), act().at(11837440).cast());
        sync_cores();
        // mid.attn
        un_gs_8_1280(act().at(11837440).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_none(act().at(11837440).cast(), act().at(983040).cast(), w().at(206733312).cast(), w().at(206741504).cast(), w().at(29515776).cast(), act().at(12820480).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(12820480).cast(), w().at(206749696).cast(), w().at(210026496).cast(), act().at(13803520).cast());
        sync_cores();
        un_ln_8_1280(act().at(13803520).cast(), w().at(210034688).cast(), w().at(210042880).cast(), act().at(14786560).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(14786560).cast(), w().at(210051072).cast(), w().at(213327872).cast(), act().at(15769600).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(14786560).cast(), w().at(213336064).cast(), w().at(213327872).cast(), act().at(16752640).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(14786560).cast(), w().at(216612864).cast(), w().at(213327872).cast(), act().at(17735680).cast());
        sync_cores();
        un_at_8_s(act().at(15769600).cast(), act().at(16752640).cast(), act().at(17735680).cast(), w().at(219889664).cast(), act().at(18718720).cast());
        sync_cores();
        un_c1_8_1280_1280_r(act().at(18718720).cast(), w().at(219893760).cast(), w().at(223170560).cast(), act().at(13803520).cast(), act().at(19701760).cast());
        sync_cores();
        un_ln_8_1280(act().at(19701760).cast(), w().at(223178752).cast(), w().at(223186944).cast(), act().at(20684800).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(20684800).cast(), w().at(223195136).cast(), w().at(213327872).cast(), act().at(21667840).cast());
        sync_cores();
        un_kv_1280(in_x().at(1720320).cast(), w().at(226471936).cast(), act().at(22650880).cast());
        sync_cores();
        un_kv_1280(in_x().at(1720320).cast(), w().at(228438016).cast(), act().at(22978560).cast());
        sync_cores();
        un_at_8_x(act().at(21667840).cast(), act().at(22650880).cast(), act().at(22978560).cast(), w().at(230404096).cast(), act().at(23306240).cast());
        sync_cores();
        un_c1_8_1280_1280_r(act().at(23306240).cast(), w().at(230408192).cast(), w().at(233684992).cast(), act().at(19701760).cast(), act().at(24289280).cast());
        sync_cores();
        un_ln_8_1280(act().at(24289280).cast(), w().at(233693184).cast(), w().at(233701376).cast(), act().at(25272320).cast());
        sync_cores();
        un_c1_8_1280_5120(act().at(25272320).cast(), w().at(233709568).cast(), w().at(246816768).cast(), act().at(26255360).cast());
        sync_cores();
        un_c1_8_1280_5120(act().at(25272320).cast(), w().at(246837248).cast(), w().at(259944448).cast(), act().at(30187520).cast());
        sync_cores();
        un_gg_8(act().at(26255360).cast(), act().at(30187520).cast(), act().at(34119680).cast());
        sync_cores();
        un_c1_8_5120_1280_r(act().at(34119680).cast(), w().at(259964928).cast(), w().at(273072128).cast(), act().at(24289280).cast(), act().at(38051840).cast());
        sync_cores();
        un_c1_8_1280_1280_r(act().at(38051840).cast(), w().at(273080320).cast(), w().at(276357120).cast(), act().at(11837440).cast(), act().at(39034880).cast());
        sync_cores();
        // mid.res1
        un_gs_8_1280(act().at(39034880).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(39034880).cast(), act().at(983040).cast(), w().at(276365312).cast(), w().at(276373504).cast(), w().at(29515776).cast(), act().at(40017920).cast());
        sync_cores();
        un_c3_8_1280_1280(act().at(40017920).cast(), w().at(276381696).cast(), in_x().at(33280).cast(), act().at(41000960).cast());
        sync_cores();
        un_gs_8_1280(act().at(41000960).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(41000960).cast(), act().at(983040).cast(), w().at(305872896).cast(), w().at(305881088).cast(), w().at(29515776).cast(), act().at(41984000).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(41984000).cast(), w().at(305889280).cast(), w().at(335380480).cast(), act().at(39034880).cast(), act().at(42967040).cast());
        sync_cores();
        // up0.res0
        un_cp_8_1280_2560(act().at(42967040).cast(), act().at(43950080).cast());
        sync_cores();
        un_cp_8_1280_2560(act().at(7905280).cast(), act().at(43952640).cast());
        sync_cores();
        un_gs_8_2560(act().at(43950080).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_2560_silu(act().at(43950080).cast(), act().at(983040).cast(), w().at(335388672).cast(), w().at(335400960).cast(), w().at(335413248).cast(), act().at(45916160).cast());
        sync_cores();
        un_c3_8_2560_1280(act().at(45916160).cast(), w().at(335740928).cast(), in_x().at(38400).cast(), act().at(47882240).cast());
        sync_cores();
        un_gs_8_1280(act().at(47882240).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(47882240).cast(), act().at(983040).cast(), w().at(394723328).cast(), w().at(394731520).cast(), w().at(29515776).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_8_2560_1280(act().at(43950080).cast(), w().at(394739712).cast(), w().at(401293312).cast(), act().at(49848320).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(48865280).cast(), w().at(401301504).cast(), w().at(430792704).cast(), act().at(49848320).cast(), act().at(50831360).cast());
        sync_cores();
        // up0.res1
        un_cp_8_1280_2560(act().at(50831360).cast(), act().at(51814400).cast());
        sync_cores();
        un_cp_8_1280_2560(act().at(3973120).cast(), act().at(51816960).cast());
        sync_cores();
        un_gs_8_2560(act().at(51814400).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_2560_silu(act().at(51814400).cast(), act().at(983040).cast(), w().at(430800896).cast(), w().at(430813184).cast(), w().at(335413248).cast(), act().at(53780480).cast());
        sync_cores();
        un_c3_8_2560_1280(act().at(53780480).cast(), w().at(430825472).cast(), in_x().at(43520).cast(), act().at(55746560).cast());
        sync_cores();
        un_gs_8_1280(act().at(55746560).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(55746560).cast(), act().at(983040).cast(), w().at(489807872).cast(), w().at(489816064).cast(), w().at(29515776).cast(), act().at(56729600).cast());
        sync_cores();
        un_c1_8_2560_1280(act().at(51814400).cast(), w().at(489824256).cast(), w().at(496377856).cast(), act().at(57712640).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(56729600).cast(), w().at(496386048).cast(), w().at(525877248).cast(), act().at(57712640).cast(), act().at(58695680).cast());
        sync_cores();
        // up0.res2
        un_cp_8_1280_2560(act().at(58695680).cast(), act().at(59678720).cast());
        sync_cores();
        un_cp_8_1280_2560(act().at(0).cast(), act().at(59681280).cast());
        sync_cores();
        un_gs_8_2560(act().at(59678720).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_2560_silu(act().at(59678720).cast(), act().at(983040).cast(), w().at(525885440).cast(), w().at(525897728).cast(), w().at(335413248).cast(), act().at(61644800).cast());
        sync_cores();
        un_c3_8_2560_1280(act().at(61644800).cast(), w().at(525910016).cast(), in_x().at(48640).cast(), act().at(63610880).cast());
        sync_cores();
        un_gs_8_1280(act().at(63610880).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(63610880).cast(), act().at(983040).cast(), w().at(584892416).cast(), w().at(584900608).cast(), w().at(29515776).cast(), act().at(64593920).cast());
        sync_cores();
        un_c1_8_2560_1280(act().at(59678720).cast(), w().at(584908800).cast(), w().at(591462400).cast(), act().at(65576960).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(64593920).cast(), w().at(591470592).cast(), w().at(620961792).cast(), act().at(65576960).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down2.down
        un_c3_8_1280_1280_s2(in_x().at(1916928).cast(), w().at(0).cast(), w().at(29491200).cast(), act().at(0).cast());
        sync_cores();
        // down3.res0
        un_gs_8_1280(act().at(0).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(0).cast(), act().at(983040).cast(), w().at(29499392).cast(), w().at(29507584).cast(), w().at(29515776).cast(), act().at(1024000).cast());
        sync_cores();
        un_c3_8_1280_1280(act().at(1024000).cast(), w().at(29679616).cast(), in_x().at(17920).cast(), act().at(2007040).cast());
        sync_cores();
        un_gs_8_1280(act().at(2007040).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(2007040).cast(), act().at(983040).cast(), w().at(59170816).cast(), w().at(59179008).cast(), w().at(29515776).cast(), act().at(2990080).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(2990080).cast(), w().at(59187200).cast(), w().at(88678400).cast(), act().at(0).cast(), act().at(3973120).cast());
        sync_cores();
        // down3.res1
        un_gs_8_1280(act().at(3973120).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(3973120).cast(), act().at(983040).cast(), w().at(88686592).cast(), w().at(88694784).cast(), w().at(29515776).cast(), act().at(4956160).cast());
        sync_cores();
        un_c3_8_1280_1280(act().at(4956160).cast(), w().at(88702976).cast(), in_x().at(23040).cast(), act().at(5939200).cast());
        sync_cores();
        un_gs_8_1280(act().at(5939200).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(5939200).cast(), act().at(983040).cast(), w().at(118194176).cast(), w().at(118202368).cast(), w().at(29515776).cast(), act().at(6922240).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(6922240).cast(), w().at(118210560).cast(), w().at(147701760).cast(), act().at(3973120).cast(), act().at(7905280).cast());
        sync_cores();
        // mid.res0
        un_gs_8_1280(act().at(7905280).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(7905280).cast(), act().at(983040).cast(), w().at(147709952).cast(), w().at(147718144).cast(), w().at(29515776).cast(), act().at(8888320).cast());
        sync_cores();
        un_c3_8_1280_1280(act().at(8888320).cast(), w().at(147726336).cast(), in_x().at(28160).cast(), act().at(9871360).cast());
        sync_cores();
        un_gs_8_1280(act().at(9871360).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(9871360).cast(), act().at(983040).cast(), w().at(177217536).cast(), w().at(177225728).cast(), w().at(29515776).cast(), act().at(10854400).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(10854400).cast(), w().at(177233920).cast(), w().at(206725120).cast(), act().at(7905280).cast(), act().at(11837440).cast());
        sync_cores();
        // mid.attn
        un_gs_8_1280(act().at(11837440).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_none(act().at(11837440).cast(), act().at(983040).cast(), w().at(206733312).cast(), w().at(206741504).cast(), w().at(29515776).cast(), act().at(12820480).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(12820480).cast(), w().at(206749696).cast(), w().at(210026496).cast(), act().at(13803520).cast());
        sync_cores();
        un_ln_8_1280(act().at(13803520).cast(), w().at(210034688).cast(), w().at(210042880).cast(), act().at(14786560).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(14786560).cast(), w().at(210051072).cast(), w().at(213327872).cast(), act().at(15769600).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(14786560).cast(), w().at(213336064).cast(), w().at(213327872).cast(), act().at(16752640).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(14786560).cast(), w().at(216612864).cast(), w().at(213327872).cast(), act().at(17735680).cast());
        sync_cores();
        un_at_8_s(act().at(15769600).cast(), act().at(16752640).cast(), act().at(17735680).cast(), w().at(219889664).cast(), act().at(18718720).cast());
        sync_cores();
        un_c1_8_1280_1280_r(act().at(18718720).cast(), w().at(219893760).cast(), w().at(223170560).cast(), act().at(13803520).cast(), act().at(19701760).cast());
        sync_cores();
        un_ln_8_1280(act().at(19701760).cast(), w().at(223178752).cast(), w().at(223186944).cast(), act().at(20684800).cast());
        sync_cores();
        un_c1_8_1280_1280(act().at(20684800).cast(), w().at(223195136).cast(), w().at(213327872).cast(), act().at(21667840).cast());
        sync_cores();
        un_kv_1280(in_x().at(3555328).cast(), w().at(226471936).cast(), act().at(22650880).cast());
        sync_cores();
        un_kv_1280(in_x().at(3555328).cast(), w().at(228438016).cast(), act().at(22978560).cast());
        sync_cores();
        un_at_8_x(act().at(21667840).cast(), act().at(22650880).cast(), act().at(22978560).cast(), w().at(230404096).cast(), act().at(23306240).cast());
        sync_cores();
        un_c1_8_1280_1280_r(act().at(23306240).cast(), w().at(230408192).cast(), w().at(233684992).cast(), act().at(19701760).cast(), act().at(24289280).cast());
        sync_cores();
        un_ln_8_1280(act().at(24289280).cast(), w().at(233693184).cast(), w().at(233701376).cast(), act().at(25272320).cast());
        sync_cores();
        un_c1_8_1280_5120(act().at(25272320).cast(), w().at(233709568).cast(), w().at(246816768).cast(), act().at(26255360).cast());
        sync_cores();
        un_c1_8_1280_5120(act().at(25272320).cast(), w().at(246837248).cast(), w().at(259944448).cast(), act().at(30187520).cast());
        sync_cores();
        un_gg_8(act().at(26255360).cast(), act().at(30187520).cast(), act().at(34119680).cast());
        sync_cores();
        un_c1_8_5120_1280_r(act().at(34119680).cast(), w().at(259964928).cast(), w().at(273072128).cast(), act().at(24289280).cast(), act().at(38051840).cast());
        sync_cores();
        un_c1_8_1280_1280_r(act().at(38051840).cast(), w().at(273080320).cast(), w().at(276357120).cast(), act().at(11837440).cast(), act().at(39034880).cast());
        sync_cores();
        // mid.res1
        un_gs_8_1280(act().at(39034880).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(39034880).cast(), act().at(983040).cast(), w().at(276365312).cast(), w().at(276373504).cast(), w().at(29515776).cast(), act().at(40017920).cast());
        sync_cores();
        un_c3_8_1280_1280(act().at(40017920).cast(), w().at(276381696).cast(), in_x().at(33280).cast(), act().at(41000960).cast());
        sync_cores();
        un_gs_8_1280(act().at(41000960).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(41000960).cast(), act().at(983040).cast(), w().at(305872896).cast(), w().at(305881088).cast(), w().at(29515776).cast(), act().at(41984000).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(41984000).cast(), w().at(305889280).cast(), w().at(335380480).cast(), act().at(39034880).cast(), act().at(42967040).cast());
        sync_cores();
        // up0.res0
        un_cp_8_1280_2560(act().at(42967040).cast(), act().at(43950080).cast());
        sync_cores();
        un_cp_8_1280_2560(act().at(7905280).cast(), act().at(43952640).cast());
        sync_cores();
        un_gs_8_2560(act().at(43950080).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_2560_silu(act().at(43950080).cast(), act().at(983040).cast(), w().at(335388672).cast(), w().at(335400960).cast(), w().at(335413248).cast(), act().at(45916160).cast());
        sync_cores();
        un_c3_8_2560_1280(act().at(45916160).cast(), w().at(335740928).cast(), in_x().at(38400).cast(), act().at(47882240).cast());
        sync_cores();
        un_gs_8_1280(act().at(47882240).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(47882240).cast(), act().at(983040).cast(), w().at(394723328).cast(), w().at(394731520).cast(), w().at(29515776).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_8_2560_1280(act().at(43950080).cast(), w().at(394739712).cast(), w().at(401293312).cast(), act().at(49848320).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(48865280).cast(), w().at(401301504).cast(), w().at(430792704).cast(), act().at(49848320).cast(), act().at(50831360).cast());
        sync_cores();
        // up0.res1
        un_cp_8_1280_2560(act().at(50831360).cast(), act().at(51814400).cast());
        sync_cores();
        un_cp_8_1280_2560(act().at(3973120).cast(), act().at(51816960).cast());
        sync_cores();
        un_gs_8_2560(act().at(51814400).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_2560_silu(act().at(51814400).cast(), act().at(983040).cast(), w().at(430800896).cast(), w().at(430813184).cast(), w().at(335413248).cast(), act().at(53780480).cast());
        sync_cores();
        un_c3_8_2560_1280(act().at(53780480).cast(), w().at(430825472).cast(), in_x().at(43520).cast(), act().at(55746560).cast());
        sync_cores();
        un_gs_8_1280(act().at(55746560).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(55746560).cast(), act().at(983040).cast(), w().at(489807872).cast(), w().at(489816064).cast(), w().at(29515776).cast(), act().at(56729600).cast());
        sync_cores();
        un_c1_8_2560_1280(act().at(51814400).cast(), w().at(489824256).cast(), w().at(496377856).cast(), act().at(57712640).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(56729600).cast(), w().at(496386048).cast(), w().at(525877248).cast(), act().at(57712640).cast(), act().at(58695680).cast());
        sync_cores();
        // up0.res2
        un_cp_8_1280_2560(act().at(58695680).cast(), act().at(59678720).cast());
        sync_cores();
        un_cp_8_1280_2560(act().at(0).cast(), act().at(59681280).cast());
        sync_cores();
        un_gs_8_2560(act().at(59678720).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_2560_silu(act().at(59678720).cast(), act().at(983040).cast(), w().at(525885440).cast(), w().at(525897728).cast(), w().at(335413248).cast(), act().at(61644800).cast());
        sync_cores();
        un_c3_8_2560_1280(act().at(61644800).cast(), w().at(525910016).cast(), in_x().at(48640).cast(), act().at(63610880).cast());
        sync_cores();
        un_gs_8_1280(act().at(63610880).cast(), act().at(983040).cast());
        sync_cores();
        un_ga_8_1280_silu(act().at(63610880).cast(), act().at(983040).cast(), w().at(584892416).cast(), w().at(584900608).cast(), w().at(29515776).cast(), act().at(64593920).cast());
        sync_cores();
        un_c1_8_2560_1280(act().at(59678720).cast(), w().at(584908800).cast(), w().at(591462400).cast(), act().at(65576960).cast());
        sync_cores();
        un_c3_8_1280_1280_r(act().at(64593920).cast(), w().at(591470592).cast(), w().at(620961792).cast(), act().at(65576960).cast(), out_y().at(983040).cast());
        sync_cores();
    }
}

/// s9：up0.up、up1.res0
#[cfg(feature = "s9")]
mod s9 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 5324800;
        out_y: output, bytes = 3276800;
        w: weights, bytes = 125075456;
        act: workspace, bytes = 14794752;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 451);
            un_zero(out_y().cast(), 100);
        } else {
            un_zero(act().at(7389184).cast(), 452);
            un_zero(out_y().at(1638400).cast(), 100);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up0.up
        un_up_8_1280_256(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(82432).cast(), act().at(512).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(82944).cast(), act().at(1024).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(83456).cast(), act().at(1536).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(83968).cast(), act().at(2048).cast());
        sync_cores();
        un_c3_16_1280_1280(act().at(0).cast(), w().at(0).cast(), w().at(29491200).cast(), act().at(1638400).cast());
        sync_cores();
        // up1.res0
        un_cp_16_1280_2560(act().at(1638400).cast(), act().at(3276800).cast());
        sync_cores();
        un_cp_16_1280_2560(in_x().at(1064960).cast(), act().at(3279360).cast());
        sync_cores();
        un_gs_16_2560(act().at(3276800).cast(), act().at(6553600).cast());
        sync_cores();
        un_ga_16_2560_silu(act().at(3276800).cast(), act().at(6553600).cast(), w().at(29499392).cast(), w().at(29511680).cast(), w().at(29523968).cast(), act().at(6594560).cast());
        sync_cores();
        un_c3_16_2560_1280(act().at(6594560).cast(), w().at(29851648).cast(), in_x().at(53760).cast(), act().at(9871360).cast());
        sync_cores();
        un_gs_16_1280(act().at(9871360).cast(), act().at(6553600).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(9871360).cast(), act().at(6553600).cast(), w().at(88834048).cast(), w().at(88842240).cast(), w().at(88850432).cast(), act().at(11509760).cast());
        sync_cores();
        un_c1_16_2560_1280(act().at(3276800).cast(), w().at(89014272).cast(), w().at(95567872).cast(), act().at(13148160).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(11509760).cast(), w().at(95576064).cast(), w().at(125067264).cast(), act().at(13148160).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up0.up
        un_up_8_1280_256(in_x().at(2703360).cast(), act().at(0).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(2703872).cast(), act().at(512).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(2704384).cast(), act().at(1024).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(2704896).cast(), act().at(1536).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(2705408).cast(), act().at(2048).cast());
        sync_cores();
        un_c3_16_1280_1280(act().at(0).cast(), w().at(0).cast(), w().at(29491200).cast(), act().at(1638400).cast());
        sync_cores();
        // up1.res0
        un_cp_16_1280_2560(act().at(1638400).cast(), act().at(3276800).cast());
        sync_cores();
        un_cp_16_1280_2560(in_x().at(3686400).cast(), act().at(3279360).cast());
        sync_cores();
        un_gs_16_2560(act().at(3276800).cast(), act().at(6553600).cast());
        sync_cores();
        un_ga_16_2560_silu(act().at(3276800).cast(), act().at(6553600).cast(), w().at(29499392).cast(), w().at(29511680).cast(), w().at(29523968).cast(), act().at(6594560).cast());
        sync_cores();
        un_c3_16_2560_1280(act().at(6594560).cast(), w().at(29851648).cast(), in_x().at(53760).cast(), act().at(9871360).cast());
        sync_cores();
        un_gs_16_1280(act().at(9871360).cast(), act().at(6553600).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(9871360).cast(), act().at(6553600).cast(), w().at(88834048).cast(), w().at(88842240).cast(), w().at(88850432).cast(), act().at(11509760).cast());
        sync_cores();
        un_c1_16_2560_1280(act().at(3276800).cast(), w().at(89014272).cast(), w().at(95567872).cast(), act().at(13148160).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(11509760).cast(), w().at(95576064).cast(), w().at(125067264).cast(), act().at(13148160).cast(), out_y().at(1638400).cast());
        sync_cores();
    }
}

/// s10：up1.attn0、up1.res1、up1.attn1
#[cfg(feature = "s10")]
mod s10 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 7028736;
        out_y: output, bytes = 3276800;
        w: weights, bytes = 234823680;
        act: workspace, bytes = 101302272;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 3091);
            un_zero(out_y().cast(), 100);
        } else {
            un_zero(act().at(50642944).cast(), 3092);
            un_zero(out_y().at(1638400).cast(), 100);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up1.attn0
        un_gs_16_1280(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_none(in_x().at(81920).cast(), act().at(0).cast(), w().at(0).cast(), w().at(8192).cast(), w().at(16384).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(40960).cast(), w().at(180224).cast(), w().at(3457024).cast(), act().at(1679360).cast());
        sync_cores();
        un_ln_16_1280(act().at(1679360).cast(), w().at(3465216).cast(), w().at(3473408).cast(), act().at(3317760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(3317760).cast(), w().at(3481600).cast(), w().at(6758400).cast(), act().at(4956160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(3317760).cast(), w().at(6766592).cast(), w().at(6758400).cast(), act().at(6594560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(3317760).cast(), w().at(10043392).cast(), w().at(6758400).cast(), act().at(8232960).cast());
        sync_cores();
        un_at_16_s(act().at(4956160).cast(), act().at(6594560).cast(), act().at(8232960).cast(), w().at(13320192).cast(), act().at(9871360).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(9871360).cast(), w().at(13324288).cast(), w().at(16601088).cast(), act().at(1679360).cast(), act().at(11509760).cast());
        sync_cores();
        un_ln_16_1280(act().at(11509760).cast(), w().at(16609280).cast(), w().at(16617472).cast(), act().at(13148160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(13148160).cast(), w().at(16625664).cast(), w().at(6758400).cast(), act().at(14786560).cast());
        sync_cores();
        un_kv_1280(in_x().at(1720320).cast(), w().at(19902464).cast(), act().at(16424960).cast());
        sync_cores();
        un_kv_1280(in_x().at(1720320).cast(), w().at(21868544).cast(), act().at(16752640).cast());
        sync_cores();
        un_at_16_x(act().at(14786560).cast(), act().at(16424960).cast(), act().at(16752640).cast(), w().at(23834624).cast(), act().at(17080320).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(17080320).cast(), w().at(23838720).cast(), w().at(27115520).cast(), act().at(11509760).cast(), act().at(18718720).cast());
        sync_cores();
        un_ln_16_1280(act().at(18718720).cast(), w().at(27123712).cast(), w().at(27131904).cast(), act().at(20357120).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(20357120).cast(), w().at(27140096).cast(), w().at(40247296).cast(), act().at(21995520).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(20357120).cast(), w().at(40267776).cast(), w().at(53374976).cast(), act().at(28549120).cast());
        sync_cores();
        un_gg_16(act().at(21995520).cast(), act().at(28549120).cast(), act().at(35102720).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(35102720).cast(), w().at(53395456).cast(), w().at(66502656).cast(), act().at(18718720).cast(), act().at(41656320).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(41656320).cast(), w().at(66510848).cast(), w().at(69787648).cast(), in_x().at(81920).cast(), act().at(43294720).cast());
        sync_cores();
        // up1.res1
        un_cp_16_1280_2560(act().at(43294720).cast(), act().at(44933120).cast());
        sync_cores();
        un_cp_16_1280_2560(in_x().at(1916928).cast(), act().at(44935680).cast());
        sync_cores();
        un_gs_16_2560(act().at(44933120).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_2560_silu(act().at(44933120).cast(), act().at(0).cast(), w().at(69795840).cast(), w().at(69808128).cast(), w().at(69820416).cast(), act().at(48209920).cast());
        sync_cores();
        un_c3_16_2560_1280(act().at(48209920).cast(), w().at(70148096).cast(), in_x().at(58880).cast(), act().at(51486720).cast());
        sync_cores();
        un_gs_16_1280(act().at(51486720).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(51486720).cast(), act().at(0).cast(), w().at(129130496).cast(), w().at(129138688).cast(), w().at(16384).cast(), act().at(53125120).cast());
        sync_cores();
        un_c1_16_2560_1280(act().at(44933120).cast(), w().at(129146880).cast(), w().at(135700480).cast(), act().at(54763520).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(53125120).cast(), w().at(135708672).cast(), w().at(165199872).cast(), act().at(54763520).cast(), act().at(56401920).cast());
        sync_cores();
        // up1.attn1
        un_gs_16_1280(act().at(56401920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(56401920).cast(), act().at(0).cast(), w().at(165208064).cast(), w().at(165216256).cast(), w().at(16384).cast(), act().at(58040320).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(58040320).cast(), w().at(165224448).cast(), w().at(168501248).cast(), act().at(59678720).cast());
        sync_cores();
        un_ln_16_1280(act().at(59678720).cast(), w().at(168509440).cast(), w().at(168517632).cast(), act().at(61317120).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(61317120).cast(), w().at(168525824).cast(), w().at(6758400).cast(), act().at(62955520).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(61317120).cast(), w().at(171802624).cast(), w().at(6758400).cast(), act().at(64593920).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(61317120).cast(), w().at(175079424).cast(), w().at(6758400).cast(), act().at(66232320).cast());
        sync_cores();
        un_at_16_s(act().at(62955520).cast(), act().at(64593920).cast(), act().at(66232320).cast(), w().at(13320192).cast(), act().at(67870720).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(67870720).cast(), w().at(178356224).cast(), w().at(181633024).cast(), act().at(59678720).cast(), act().at(69509120).cast());
        sync_cores();
        un_ln_16_1280(act().at(69509120).cast(), w().at(181641216).cast(), w().at(181649408).cast(), act().at(71147520).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(71147520).cast(), w().at(181657600).cast(), w().at(6758400).cast(), act().at(72785920).cast());
        sync_cores();
        un_kv_1280(in_x().at(1720320).cast(), w().at(184934400).cast(), act().at(74424320).cast());
        sync_cores();
        un_kv_1280(in_x().at(1720320).cast(), w().at(186900480).cast(), act().at(74752000).cast());
        sync_cores();
        un_at_16_x(act().at(72785920).cast(), act().at(74424320).cast(), act().at(74752000).cast(), w().at(23834624).cast(), act().at(75079680).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(75079680).cast(), w().at(188866560).cast(), w().at(192143360).cast(), act().at(69509120).cast(), act().at(76718080).cast());
        sync_cores();
        un_ln_16_1280(act().at(76718080).cast(), w().at(192151552).cast(), w().at(192159744).cast(), act().at(78356480).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(78356480).cast(), w().at(192167936).cast(), w().at(205275136).cast(), act().at(79994880).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(78356480).cast(), w().at(205295616).cast(), w().at(218402816).cast(), act().at(86548480).cast());
        sync_cores();
        un_gg_16(act().at(79994880).cast(), act().at(86548480).cast(), act().at(93102080).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(93102080).cast(), w().at(218423296).cast(), w().at(231530496).cast(), act().at(76718080).cast(), act().at(99655680).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(99655680).cast(), w().at(231538688).cast(), w().at(234815488).cast(), act().at(56401920).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up1.attn0
        un_gs_16_1280(in_x().at(3555328).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_none(in_x().at(3555328).cast(), act().at(0).cast(), w().at(0).cast(), w().at(8192).cast(), w().at(16384).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(40960).cast(), w().at(180224).cast(), w().at(3457024).cast(), act().at(1679360).cast());
        sync_cores();
        un_ln_16_1280(act().at(1679360).cast(), w().at(3465216).cast(), w().at(3473408).cast(), act().at(3317760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(3317760).cast(), w().at(3481600).cast(), w().at(6758400).cast(), act().at(4956160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(3317760).cast(), w().at(6766592).cast(), w().at(6758400).cast(), act().at(6594560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(3317760).cast(), w().at(10043392).cast(), w().at(6758400).cast(), act().at(8232960).cast());
        sync_cores();
        un_at_16_s(act().at(4956160).cast(), act().at(6594560).cast(), act().at(8232960).cast(), w().at(13320192).cast(), act().at(9871360).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(9871360).cast(), w().at(13324288).cast(), w().at(16601088).cast(), act().at(1679360).cast(), act().at(11509760).cast());
        sync_cores();
        un_ln_16_1280(act().at(11509760).cast(), w().at(16609280).cast(), w().at(16617472).cast(), act().at(13148160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(13148160).cast(), w().at(16625664).cast(), w().at(6758400).cast(), act().at(14786560).cast());
        sync_cores();
        un_kv_1280(in_x().at(5193728).cast(), w().at(19902464).cast(), act().at(16424960).cast());
        sync_cores();
        un_kv_1280(in_x().at(5193728).cast(), w().at(21868544).cast(), act().at(16752640).cast());
        sync_cores();
        un_at_16_x(act().at(14786560).cast(), act().at(16424960).cast(), act().at(16752640).cast(), w().at(23834624).cast(), act().at(17080320).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(17080320).cast(), w().at(23838720).cast(), w().at(27115520).cast(), act().at(11509760).cast(), act().at(18718720).cast());
        sync_cores();
        un_ln_16_1280(act().at(18718720).cast(), w().at(27123712).cast(), w().at(27131904).cast(), act().at(20357120).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(20357120).cast(), w().at(27140096).cast(), w().at(40247296).cast(), act().at(21995520).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(20357120).cast(), w().at(40267776).cast(), w().at(53374976).cast(), act().at(28549120).cast());
        sync_cores();
        un_gg_16(act().at(21995520).cast(), act().at(28549120).cast(), act().at(35102720).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(35102720).cast(), w().at(53395456).cast(), w().at(66502656).cast(), act().at(18718720).cast(), act().at(41656320).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(41656320).cast(), w().at(66510848).cast(), w().at(69787648).cast(), in_x().at(3555328).cast(), act().at(43294720).cast());
        sync_cores();
        // up1.res1
        un_cp_16_1280_2560(act().at(43294720).cast(), act().at(44933120).cast());
        sync_cores();
        un_cp_16_1280_2560(in_x().at(5390336).cast(), act().at(44935680).cast());
        sync_cores();
        un_gs_16_2560(act().at(44933120).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_2560_silu(act().at(44933120).cast(), act().at(0).cast(), w().at(69795840).cast(), w().at(69808128).cast(), w().at(69820416).cast(), act().at(48209920).cast());
        sync_cores();
        un_c3_16_2560_1280(act().at(48209920).cast(), w().at(70148096).cast(), in_x().at(58880).cast(), act().at(51486720).cast());
        sync_cores();
        un_gs_16_1280(act().at(51486720).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(51486720).cast(), act().at(0).cast(), w().at(129130496).cast(), w().at(129138688).cast(), w().at(16384).cast(), act().at(53125120).cast());
        sync_cores();
        un_c1_16_2560_1280(act().at(44933120).cast(), w().at(129146880).cast(), w().at(135700480).cast(), act().at(54763520).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(53125120).cast(), w().at(135708672).cast(), w().at(165199872).cast(), act().at(54763520).cast(), act().at(56401920).cast());
        sync_cores();
        // up1.attn1
        un_gs_16_1280(act().at(56401920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(56401920).cast(), act().at(0).cast(), w().at(165208064).cast(), w().at(165216256).cast(), w().at(16384).cast(), act().at(58040320).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(58040320).cast(), w().at(165224448).cast(), w().at(168501248).cast(), act().at(59678720).cast());
        sync_cores();
        un_ln_16_1280(act().at(59678720).cast(), w().at(168509440).cast(), w().at(168517632).cast(), act().at(61317120).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(61317120).cast(), w().at(168525824).cast(), w().at(6758400).cast(), act().at(62955520).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(61317120).cast(), w().at(171802624).cast(), w().at(6758400).cast(), act().at(64593920).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(61317120).cast(), w().at(175079424).cast(), w().at(6758400).cast(), act().at(66232320).cast());
        sync_cores();
        un_at_16_s(act().at(62955520).cast(), act().at(64593920).cast(), act().at(66232320).cast(), w().at(13320192).cast(), act().at(67870720).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(67870720).cast(), w().at(178356224).cast(), w().at(181633024).cast(), act().at(59678720).cast(), act().at(69509120).cast());
        sync_cores();
        un_ln_16_1280(act().at(69509120).cast(), w().at(181641216).cast(), w().at(181649408).cast(), act().at(71147520).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(71147520).cast(), w().at(181657600).cast(), w().at(6758400).cast(), act().at(72785920).cast());
        sync_cores();
        un_kv_1280(in_x().at(5193728).cast(), w().at(184934400).cast(), act().at(74424320).cast());
        sync_cores();
        un_kv_1280(in_x().at(5193728).cast(), w().at(186900480).cast(), act().at(74752000).cast());
        sync_cores();
        un_at_16_x(act().at(72785920).cast(), act().at(74424320).cast(), act().at(74752000).cast(), w().at(23834624).cast(), act().at(75079680).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(75079680).cast(), w().at(188866560).cast(), w().at(192143360).cast(), act().at(69509120).cast(), act().at(76718080).cast());
        sync_cores();
        un_ln_16_1280(act().at(76718080).cast(), w().at(192151552).cast(), w().at(192159744).cast(), act().at(78356480).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(78356480).cast(), w().at(192167936).cast(), w().at(205275136).cast(), act().at(79994880).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(78356480).cast(), w().at(205295616).cast(), w().at(218402816).cast(), act().at(86548480).cast());
        sync_cores();
        un_gg_16(act().at(79994880).cast(), act().at(86548480).cast(), act().at(93102080).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(93102080).cast(), w().at(218423296).cast(), w().at(231530496).cast(), act().at(76718080).cast(), act().at(99655680).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(99655680).cast(), w().at(231538688).cast(), w().at(234815488).cast(), act().at(56401920).cast(), out_y().at(1638400).cast());
        sync_cores();
    }
}

/// s11：up1.res2、up1.attn2、up1.up
#[cfg(feature = "s11")]
mod s11 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 5390336;
        out_y: output, bytes = 7864320;
        w: weights, bytes = 178233344;
        act: workspace, bytes = 60342272;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 1841);
            un_zero(out_y().cast(), 240);
        } else {
            un_zero(act().at(30162944).cast(), 1842);
            un_zero(out_y().at(3932160).cast(), 240);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up1.res2
        un_cp_16_1280_1920(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_cp_16_640_1920(in_x().at(1720320).cast(), act().at(2560).cast());
        sync_cores();
        un_gs_16_1920(act().at(0).cast(), act().at(2457600).cast());
        sync_cores();
        un_ga_16_1920_silu(act().at(0).cast(), act().at(2457600).cast(), w().at(0).cast(), w().at(8192).cast(), w().at(16384).cast(), act().at(2498560).cast());
        sync_cores();
        un_c3_16_1920_1280(act().at(2498560).cast(), w().at(262144).cast(), in_x().at(64000).cast(), act().at(4956160).cast());
        sync_cores();
        un_gs_16_1280(act().at(4956160).cast(), act().at(2457600).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(4956160).cast(), act().at(2457600).cast(), w().at(44498944).cast(), w().at(44507136).cast(), w().at(44515328).cast(), act().at(6594560).cast());
        sync_cores();
        un_c1_16_1920_1280(act().at(0).cast(), w().at(44679168).cast(), w().at(49594368).cast(), act().at(8232960).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(6594560).cast(), w().at(49602560).cast(), w().at(79093760).cast(), act().at(8232960).cast(), act().at(9871360).cast());
        sync_cores();
        // up1.attn2
        un_gs_16_1280(act().at(9871360).cast(), act().at(2457600).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(9871360).cast(), act().at(2457600).cast(), w().at(79101952).cast(), w().at(79110144).cast(), w().at(44515328).cast(), act().at(11509760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(11509760).cast(), w().at(79118336).cast(), w().at(82395136).cast(), act().at(13148160).cast());
        sync_cores();
        un_ln_16_1280(act().at(13148160).cast(), w().at(82403328).cast(), w().at(82411520).cast(), act().at(14786560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(14786560).cast(), w().at(82419712).cast(), w().at(85696512).cast(), act().at(16424960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(14786560).cast(), w().at(85704704).cast(), w().at(85696512).cast(), act().at(18063360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(14786560).cast(), w().at(88981504).cast(), w().at(85696512).cast(), act().at(19701760).cast());
        sync_cores();
        un_at_16_s(act().at(16424960).cast(), act().at(18063360).cast(), act().at(19701760).cast(), w().at(92258304).cast(), act().at(21340160).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(21340160).cast(), w().at(92262400).cast(), w().at(95539200).cast(), act().at(13148160).cast(), act().at(22978560).cast());
        sync_cores();
        un_ln_16_1280(act().at(22978560).cast(), w().at(95547392).cast(), w().at(95555584).cast(), act().at(24616960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(24616960).cast(), w().at(95563776).cast(), w().at(85696512).cast(), act().at(26255360).cast());
        sync_cores();
        un_kv_1280(in_x().at(2539520).cast(), w().at(98840576).cast(), act().at(27893760).cast());
        sync_cores();
        un_kv_1280(in_x().at(2539520).cast(), w().at(100806656).cast(), act().at(28221440).cast());
        sync_cores();
        un_at_16_x(act().at(26255360).cast(), act().at(27893760).cast(), act().at(28221440).cast(), w().at(102772736).cast(), act().at(28549120).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(28549120).cast(), w().at(102776832).cast(), w().at(106053632).cast(), act().at(22978560).cast(), act().at(30187520).cast());
        sync_cores();
        un_ln_16_1280(act().at(30187520).cast(), w().at(106061824).cast(), w().at(106070016).cast(), act().at(31825920).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(31825920).cast(), w().at(106078208).cast(), w().at(119185408).cast(), act().at(33464320).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(31825920).cast(), w().at(119205888).cast(), w().at(132313088).cast(), act().at(40017920).cast());
        sync_cores();
        un_gg_16(act().at(33464320).cast(), act().at(40017920).cast(), act().at(46571520).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(46571520).cast(), w().at(132333568).cast(), w().at(145440768).cast(), act().at(30187520).cast(), act().at(53125120).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(53125120).cast(), w().at(145448960).cast(), w().at(148725760).cast(), act().at(9871360).cast(), act().at(54763520).cast());
        sync_cores();
        // up1.up
        un_up_16_1280_256(act().at(54763520).cast(), act().at(56401920).cast());
        sync_cores();
        un_up_16_1280_256(act().at(54764032).cast(), act().at(56402432).cast());
        sync_cores();
        un_up_16_1280_256(act().at(54764544).cast(), act().at(56402944).cast());
        sync_cores();
        un_up_16_1280_256(act().at(54765056).cast(), act().at(56403456).cast());
        sync_cores();
        un_up_16_1280_256(act().at(54765568).cast(), act().at(56403968).cast());
        sync_cores();
        un_c3_32_1280_1280(act().at(56401920).cast(), w().at(148733952).cast(), w().at(178225152).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up1.res2
        un_cp_16_1280_1920(in_x().at(2736128).cast(), act().at(0).cast());
        sync_cores();
        un_cp_16_640_1920(in_x().at(4374528).cast(), act().at(2560).cast());
        sync_cores();
        un_gs_16_1920(act().at(0).cast(), act().at(2457600).cast());
        sync_cores();
        un_ga_16_1920_silu(act().at(0).cast(), act().at(2457600).cast(), w().at(0).cast(), w().at(8192).cast(), w().at(16384).cast(), act().at(2498560).cast());
        sync_cores();
        un_c3_16_1920_1280(act().at(2498560).cast(), w().at(262144).cast(), in_x().at(64000).cast(), act().at(4956160).cast());
        sync_cores();
        un_gs_16_1280(act().at(4956160).cast(), act().at(2457600).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(4956160).cast(), act().at(2457600).cast(), w().at(44498944).cast(), w().at(44507136).cast(), w().at(44515328).cast(), act().at(6594560).cast());
        sync_cores();
        un_c1_16_1920_1280(act().at(0).cast(), w().at(44679168).cast(), w().at(49594368).cast(), act().at(8232960).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(6594560).cast(), w().at(49602560).cast(), w().at(79093760).cast(), act().at(8232960).cast(), act().at(9871360).cast());
        sync_cores();
        // up1.attn2
        un_gs_16_1280(act().at(9871360).cast(), act().at(2457600).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(9871360).cast(), act().at(2457600).cast(), w().at(79101952).cast(), w().at(79110144).cast(), w().at(44515328).cast(), act().at(11509760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(11509760).cast(), w().at(79118336).cast(), w().at(82395136).cast(), act().at(13148160).cast());
        sync_cores();
        un_ln_16_1280(act().at(13148160).cast(), w().at(82403328).cast(), w().at(82411520).cast(), act().at(14786560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(14786560).cast(), w().at(82419712).cast(), w().at(85696512).cast(), act().at(16424960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(14786560).cast(), w().at(85704704).cast(), w().at(85696512).cast(), act().at(18063360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(14786560).cast(), w().at(88981504).cast(), w().at(85696512).cast(), act().at(19701760).cast());
        sync_cores();
        un_at_16_s(act().at(16424960).cast(), act().at(18063360).cast(), act().at(19701760).cast(), w().at(92258304).cast(), act().at(21340160).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(21340160).cast(), w().at(92262400).cast(), w().at(95539200).cast(), act().at(13148160).cast(), act().at(22978560).cast());
        sync_cores();
        un_ln_16_1280(act().at(22978560).cast(), w().at(95547392).cast(), w().at(95555584).cast(), act().at(24616960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(24616960).cast(), w().at(95563776).cast(), w().at(85696512).cast(), act().at(26255360).cast());
        sync_cores();
        un_kv_1280(in_x().at(5193728).cast(), w().at(98840576).cast(), act().at(27893760).cast());
        sync_cores();
        un_kv_1280(in_x().at(5193728).cast(), w().at(100806656).cast(), act().at(28221440).cast());
        sync_cores();
        un_at_16_x(act().at(26255360).cast(), act().at(27893760).cast(), act().at(28221440).cast(), w().at(102772736).cast(), act().at(28549120).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(28549120).cast(), w().at(102776832).cast(), w().at(106053632).cast(), act().at(22978560).cast(), act().at(30187520).cast());
        sync_cores();
        un_ln_16_1280(act().at(30187520).cast(), w().at(106061824).cast(), w().at(106070016).cast(), act().at(31825920).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(31825920).cast(), w().at(106078208).cast(), w().at(119185408).cast(), act().at(33464320).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(31825920).cast(), w().at(119205888).cast(), w().at(132313088).cast(), act().at(40017920).cast());
        sync_cores();
        un_gg_16(act().at(33464320).cast(), act().at(40017920).cast(), act().at(46571520).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(46571520).cast(), w().at(132333568).cast(), w().at(145440768).cast(), act().at(30187520).cast(), act().at(53125120).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(53125120).cast(), w().at(145448960).cast(), w().at(148725760).cast(), act().at(9871360).cast(), act().at(54763520).cast());
        sync_cores();
        // up1.up
        un_up_16_1280_256(act().at(54763520).cast(), act().at(56401920).cast());
        sync_cores();
        un_up_16_1280_256(act().at(54764032).cast(), act().at(56402432).cast());
        sync_cores();
        un_up_16_1280_256(act().at(54764544).cast(), act().at(56402944).cast());
        sync_cores();
        un_up_16_1280_256(act().at(54765056).cast(), act().at(56403456).cast());
        sync_cores();
        un_up_16_1280_256(act().at(54765568).cast(), act().at(56403968).cast());
        sync_cores();
        un_c3_32_1280_1280(act().at(56401920).cast(), w().at(148733952).cast(), w().at(178225152).cast(), out_y().at(3932160).cast());
        sync_cores();
    }
}

/// s12：up2.res0
#[cfg(feature = "s12")]
mod s12 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 11878400;
        out_y: output, bytes = 3932160;
        w: weights, bytes = 32309248;
        act: workspace, bytes = 17743872;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 541);
            un_zero(out_y().cast(), 120);
        } else {
            un_zero(act().at(8863744).cast(), 542);
            un_zero(out_y().at(1966080).cast(), 120);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up2.res0
        un_cp_32_1280_1920(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_cp_32_640_1920(in_x().at(4014080).cast(), act().at(2560).cast());
        sync_cores();
        un_gs_32_1920(act().at(0).cast(), act().at(5898240).cast());
        sync_cores();
        un_ga_32_1920_silu(act().at(0).cast(), act().at(5898240).cast(), w().at(0).cast(), w().at(8192).cast(), w().at(16384).cast(), act().at(5939200).cast());
        sync_cores();
        un_c3_32_1920_640(act().at(5939200).cast(), w().at(262144).cast(), in_x().at(69120).cast(), act().at(11837440).cast());
        sync_cores();
        un_gs_32_640(act().at(11837440).cast(), act().at(5898240).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(11837440).cast(), act().at(5898240).cast(), w().at(22380544).cast(), w().at(22384640).cast(), w().at(22388736).cast(), act().at(13803520).cast());
        sync_cores();
        un_c1_32_1920_640(act().at(0).cast(), w().at(22470656).cast(), w().at(24928256).cast(), act().at(15769600).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(13803520).cast(), w().at(24932352).cast(), w().at(32305152).cast(), act().at(15769600).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up2.res0
        un_cp_32_1280_1920(in_x().at(5980160).cast(), act().at(0).cast());
        sync_cores();
        un_cp_32_640_1920(in_x().at(9912320).cast(), act().at(2560).cast());
        sync_cores();
        un_gs_32_1920(act().at(0).cast(), act().at(5898240).cast());
        sync_cores();
        un_ga_32_1920_silu(act().at(0).cast(), act().at(5898240).cast(), w().at(0).cast(), w().at(8192).cast(), w().at(16384).cast(), act().at(5939200).cast());
        sync_cores();
        un_c3_32_1920_640(act().at(5939200).cast(), w().at(262144).cast(), in_x().at(69120).cast(), act().at(11837440).cast());
        sync_cores();
        un_gs_32_640(act().at(11837440).cast(), act().at(5898240).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(11837440).cast(), act().at(5898240).cast(), w().at(22380544).cast(), w().at(22384640).cast(), w().at(22388736).cast(), act().at(13803520).cast());
        sync_cores();
        un_c1_32_1920_640(act().at(0).cast(), w().at(22470656).cast(), w().at(24928256).cast(), act().at(15769600).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(13803520).cast(), w().at(24932352).cast(), w().at(32305152).cast(), act().at(15769600).cast(), out_y().at(1966080).cast());
        sync_cores();
    }
}

/// s13：up2.attn0、up2.res1、up2.attn1
#[cfg(feature = "s13")]
mod s13 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 8339456;
        out_y: output, bytes = 3932160;
        w: weights, bytes = 63979520;
        act: workspace, bytes = 126271488;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 3853);
            un_zero(out_y().cast(), 120);
        } else {
            un_zero(act().at(63127552).cast(), 3854);
            un_zero(out_y().at(1966080).cast(), 120);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up2.attn0
        un_gs_32_640(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(in_x().at(81920).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(40960).cast(), w().at(90112).cast(), w().at(909312).cast(), act().at(2007040).cast());
        sync_cores();
        un_ln_32_640(act().at(2007040).cast(), w().at(913408).cast(), w().at(917504).cast(), act().at(3973120).cast());
        sync_cores();
        un_lr_32_640_768(act().at(3973120).cast(), w().at(921600).cast(), act().at(5939200).cast());
        sync_cores();
        un_lr_32_640_768(act().at(3973120).cast(), w().at(1904640).cast(), act().at(8298496).cast());
        sync_cores();
        un_lr_32_640_768(act().at(3973120).cast(), w().at(2887680).cast(), act().at(10657792).cast());
        sync_cores();
        un_at_32_s(act().at(5939200).cast(), act().at(8298496).cast(), act().at(10657792).cast(), w().at(3870720).cast(), act().at(13017088).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(13017088).cast(), w().at(3874816).cast(), w().at(4857856).cast(), act().at(2007040).cast(), act().at(15376384).cast());
        sync_cores();
        un_ln_32_640(act().at(15376384).cast(), w().at(4861952).cast(), w().at(4866048).cast(), act().at(17342464).cast());
        sync_cores();
        un_lr_32_640_768(act().at(17342464).cast(), w().at(4870144).cast(), act().at(19308544).cast());
        sync_cores();
        un_kv_768(in_x().at(2048000).cast(), w().at(5853184).cast(), act().at(21667840).cast());
        sync_cores();
        un_kv_768(in_x().at(2048000).cast(), w().at(7032832).cast(), act().at(21864448).cast());
        sync_cores();
        un_at_32_x(act().at(19308544).cast(), act().at(21667840).cast(), act().at(21864448).cast(), w().at(8212480).cast(), act().at(22061056).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(22061056).cast(), w().at(8216576).cast(), w().at(9199616).cast(), act().at(15376384).cast(), act().at(24420352).cast());
        sync_cores();
        un_ln_32_640(act().at(24420352).cast(), w().at(9203712).cast(), w().at(9207808).cast(), act().at(26386432).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(26386432).cast(), w().at(9211904).cast(), w().at(12488704).cast(), act().at(28352512).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(26386432).cast(), w().at(12500992).cast(), w().at(15777792).cast(), act().at(36216832).cast());
        sync_cores();
        un_gg_32(act().at(28352512).cast(), act().at(36216832).cast(), act().at(44081152).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(44081152).cast(), w().at(15790080).cast(), w().at(19066880).cast(), act().at(24420352).cast(), act().at(51945472).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(51945472).cast(), w().at(19070976).cast(), w().at(20054016).cast(), in_x().at(81920).cast(), act().at(54304768).cast());
        sync_cores();
        // up2.res1
        un_cp_32_640_1280(act().at(54304768).cast(), act().at(56270848).cast());
        sync_cores();
        un_cp_32_640_1280(in_x().at(2244608).cast(), act().at(56272128).cast());
        sync_cores();
        un_gs_32_1280(act().at(56270848).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_1280_silu(act().at(56270848).cast(), act().at(0).cast(), w().at(20058112).cast(), w().at(20066304).cast(), w().at(20074496).cast(), act().at(60203008).cast());
        sync_cores();
        un_c3_32_1280_640(act().at(60203008).cast(), w().at(20238336).cast(), in_x().at(71680).cast(), act().at(64135168).cast());
        sync_cores();
        un_gs_32_640(act().at(64135168).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(64135168).cast(), act().at(0).cast(), w().at(34983936).cast(), w().at(34988032).cast(), w().at(8192).cast(), act().at(66101248).cast());
        sync_cores();
        un_c1_32_1280_640(act().at(56270848).cast(), w().at(34992128).cast(), w().at(36630528).cast(), act().at(68067328).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(66101248).cast(), w().at(36634624).cast(), w().at(44007424).cast(), act().at(68067328).cast(), act().at(70033408).cast());
        sync_cores();
        // up2.attn1
        un_gs_32_640(act().at(70033408).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(70033408).cast(), act().at(0).cast(), w().at(44011520).cast(), w().at(44015616).cast(), w().at(8192).cast(), act().at(71999488).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(71999488).cast(), w().at(44019712).cast(), w().at(44838912).cast(), act().at(73965568).cast());
        sync_cores();
        un_ln_32_640(act().at(73965568).cast(), w().at(44843008).cast(), w().at(44847104).cast(), act().at(75931648).cast());
        sync_cores();
        un_lr_32_640_768(act().at(75931648).cast(), w().at(44851200).cast(), act().at(77897728).cast());
        sync_cores();
        un_lr_32_640_768(act().at(75931648).cast(), w().at(45834240).cast(), act().at(80257024).cast());
        sync_cores();
        un_lr_32_640_768(act().at(75931648).cast(), w().at(46817280).cast(), act().at(82616320).cast());
        sync_cores();
        un_at_32_s(act().at(77897728).cast(), act().at(80257024).cast(), act().at(82616320).cast(), w().at(3870720).cast(), act().at(84975616).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(84975616).cast(), w().at(47800320).cast(), w().at(48783360).cast(), act().at(73965568).cast(), act().at(87334912).cast());
        sync_cores();
        un_ln_32_640(act().at(87334912).cast(), w().at(48787456).cast(), w().at(48791552).cast(), act().at(89300992).cast());
        sync_cores();
        un_lr_32_640_768(act().at(89300992).cast(), w().at(48795648).cast(), act().at(91267072).cast());
        sync_cores();
        un_kv_768(in_x().at(2048000).cast(), w().at(49778688).cast(), act().at(93626368).cast());
        sync_cores();
        un_kv_768(in_x().at(2048000).cast(), w().at(50958336).cast(), act().at(93822976).cast());
        sync_cores();
        un_at_32_x(act().at(91267072).cast(), act().at(93626368).cast(), act().at(93822976).cast(), w().at(8212480).cast(), act().at(94019584).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(94019584).cast(), w().at(52137984).cast(), w().at(53121024).cast(), act().at(87334912).cast(), act().at(96378880).cast());
        sync_cores();
        un_ln_32_640(act().at(96378880).cast(), w().at(53125120).cast(), w().at(53129216).cast(), act().at(98344960).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(98344960).cast(), w().at(53133312).cast(), w().at(56410112).cast(), act().at(100311040).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(98344960).cast(), w().at(56422400).cast(), w().at(59699200).cast(), act().at(108175360).cast());
        sync_cores();
        un_gg_32(act().at(100311040).cast(), act().at(108175360).cast(), act().at(116039680).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(116039680).cast(), w().at(59711488).cast(), w().at(62988288).cast(), act().at(96378880).cast(), act().at(123904000).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(123904000).cast(), w().at(62992384).cast(), w().at(63975424).cast(), act().at(70033408).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up2.attn0
        un_gs_32_640(in_x().at(4210688).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(in_x().at(4210688).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(40960).cast(), w().at(90112).cast(), w().at(909312).cast(), act().at(2007040).cast());
        sync_cores();
        un_ln_32_640(act().at(2007040).cast(), w().at(913408).cast(), w().at(917504).cast(), act().at(3973120).cast());
        sync_cores();
        un_lr_32_640_768(act().at(3973120).cast(), w().at(921600).cast(), act().at(5939200).cast());
        sync_cores();
        un_lr_32_640_768(act().at(3973120).cast(), w().at(1904640).cast(), act().at(8298496).cast());
        sync_cores();
        un_lr_32_640_768(act().at(3973120).cast(), w().at(2887680).cast(), act().at(10657792).cast());
        sync_cores();
        un_at_32_s(act().at(5939200).cast(), act().at(8298496).cast(), act().at(10657792).cast(), w().at(3870720).cast(), act().at(13017088).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(13017088).cast(), w().at(3874816).cast(), w().at(4857856).cast(), act().at(2007040).cast(), act().at(15376384).cast());
        sync_cores();
        un_ln_32_640(act().at(15376384).cast(), w().at(4861952).cast(), w().at(4866048).cast(), act().at(17342464).cast());
        sync_cores();
        un_lr_32_640_768(act().at(17342464).cast(), w().at(4870144).cast(), act().at(19308544).cast());
        sync_cores();
        un_kv_768(in_x().at(6176768).cast(), w().at(5853184).cast(), act().at(21667840).cast());
        sync_cores();
        un_kv_768(in_x().at(6176768).cast(), w().at(7032832).cast(), act().at(21864448).cast());
        sync_cores();
        un_at_32_x(act().at(19308544).cast(), act().at(21667840).cast(), act().at(21864448).cast(), w().at(8212480).cast(), act().at(22061056).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(22061056).cast(), w().at(8216576).cast(), w().at(9199616).cast(), act().at(15376384).cast(), act().at(24420352).cast());
        sync_cores();
        un_ln_32_640(act().at(24420352).cast(), w().at(9203712).cast(), w().at(9207808).cast(), act().at(26386432).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(26386432).cast(), w().at(9211904).cast(), w().at(12488704).cast(), act().at(28352512).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(26386432).cast(), w().at(12500992).cast(), w().at(15777792).cast(), act().at(36216832).cast());
        sync_cores();
        un_gg_32(act().at(28352512).cast(), act().at(36216832).cast(), act().at(44081152).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(44081152).cast(), w().at(15790080).cast(), w().at(19066880).cast(), act().at(24420352).cast(), act().at(51945472).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(51945472).cast(), w().at(19070976).cast(), w().at(20054016).cast(), in_x().at(4210688).cast(), act().at(54304768).cast());
        sync_cores();
        // up2.res1
        un_cp_32_640_1280(act().at(54304768).cast(), act().at(56270848).cast());
        sync_cores();
        un_cp_32_640_1280(in_x().at(6373376).cast(), act().at(56272128).cast());
        sync_cores();
        un_gs_32_1280(act().at(56270848).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_1280_silu(act().at(56270848).cast(), act().at(0).cast(), w().at(20058112).cast(), w().at(20066304).cast(), w().at(20074496).cast(), act().at(60203008).cast());
        sync_cores();
        un_c3_32_1280_640(act().at(60203008).cast(), w().at(20238336).cast(), in_x().at(71680).cast(), act().at(64135168).cast());
        sync_cores();
        un_gs_32_640(act().at(64135168).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(64135168).cast(), act().at(0).cast(), w().at(34983936).cast(), w().at(34988032).cast(), w().at(8192).cast(), act().at(66101248).cast());
        sync_cores();
        un_c1_32_1280_640(act().at(56270848).cast(), w().at(34992128).cast(), w().at(36630528).cast(), act().at(68067328).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(66101248).cast(), w().at(36634624).cast(), w().at(44007424).cast(), act().at(68067328).cast(), act().at(70033408).cast());
        sync_cores();
        // up2.attn1
        un_gs_32_640(act().at(70033408).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(70033408).cast(), act().at(0).cast(), w().at(44011520).cast(), w().at(44015616).cast(), w().at(8192).cast(), act().at(71999488).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(71999488).cast(), w().at(44019712).cast(), w().at(44838912).cast(), act().at(73965568).cast());
        sync_cores();
        un_ln_32_640(act().at(73965568).cast(), w().at(44843008).cast(), w().at(44847104).cast(), act().at(75931648).cast());
        sync_cores();
        un_lr_32_640_768(act().at(75931648).cast(), w().at(44851200).cast(), act().at(77897728).cast());
        sync_cores();
        un_lr_32_640_768(act().at(75931648).cast(), w().at(45834240).cast(), act().at(80257024).cast());
        sync_cores();
        un_lr_32_640_768(act().at(75931648).cast(), w().at(46817280).cast(), act().at(82616320).cast());
        sync_cores();
        un_at_32_s(act().at(77897728).cast(), act().at(80257024).cast(), act().at(82616320).cast(), w().at(3870720).cast(), act().at(84975616).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(84975616).cast(), w().at(47800320).cast(), w().at(48783360).cast(), act().at(73965568).cast(), act().at(87334912).cast());
        sync_cores();
        un_ln_32_640(act().at(87334912).cast(), w().at(48787456).cast(), w().at(48791552).cast(), act().at(89300992).cast());
        sync_cores();
        un_lr_32_640_768(act().at(89300992).cast(), w().at(48795648).cast(), act().at(91267072).cast());
        sync_cores();
        un_kv_768(in_x().at(6176768).cast(), w().at(49778688).cast(), act().at(93626368).cast());
        sync_cores();
        un_kv_768(in_x().at(6176768).cast(), w().at(50958336).cast(), act().at(93822976).cast());
        sync_cores();
        un_at_32_x(act().at(91267072).cast(), act().at(93626368).cast(), act().at(93822976).cast(), w().at(8212480).cast(), act().at(94019584).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(94019584).cast(), w().at(52137984).cast(), w().at(53121024).cast(), act().at(87334912).cast(), act().at(96378880).cast());
        sync_cores();
        un_ln_32_640(act().at(96378880).cast(), w().at(53125120).cast(), w().at(53129216).cast(), act().at(98344960).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(98344960).cast(), w().at(53133312).cast(), w().at(56410112).cast(), act().at(100311040).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(98344960).cast(), w().at(56422400).cast(), w().at(59699200).cast(), act().at(108175360).cast());
        sync_cores();
        un_gg_32(act().at(100311040).cast(), act().at(108175360).cast(), act().at(116039680).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(116039680).cast(), w().at(59711488).cast(), w().at(62988288).cast(), act().at(96378880).cast(), act().at(123904000).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(123904000).cast(), w().at(62992384).cast(), w().at(63975424).cast(), act().at(70033408).cast(), out_y().at(1966080).cast());
        sync_cores();
    }
}

/// s14：up2.res2、up2.attn2、up2.up
#[cfg(feature = "s14")]
mod s14 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 6373376;
        out_y: output, bytes = 12451840;
        w: weights, bytes = 47243264;
        act: workspace, bytes = 76267520;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2327);
            un_zero(out_y().cast(), 380);
        } else {
            un_zero(act().at(38125568).cast(), 2328);
            un_zero(out_y().at(6225920).cast(), 380);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up2.res2
        un_cp_32_640_960(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_cp_32_320_960(in_x().at(2048000).cast(), act().at(1280).cast());
        sync_cores();
        un_gs_32_960(act().at(0).cast(), act().at(2949120).cast());
        sync_cores();
        un_ga_32_960_silu(act().at(0).cast(), act().at(2949120).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(2990080).cast());
        sync_cores();
        un_c3_32_960_640(act().at(2990080).cast(), w().at(131072).cast(), in_x().at(74240).cast(), act().at(5939200).cast());
        sync_cores();
        un_gs_32_640(act().at(5939200).cast(), act().at(2949120).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(5939200).cast(), act().at(2949120).cast(), w().at(11190272).cast(), w().at(11194368).cast(), w().at(11198464).cast(), act().at(7905280).cast());
        sync_cores();
        un_c1_32_960_640(act().at(0).cast(), w().at(11280384).cast(), w().at(12509184).cast(), act().at(9871360).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(7905280).cast(), w().at(12513280).cast(), w().at(19886080).cast(), act().at(9871360).cast(), act().at(11837440).cast());
        sync_cores();
        // up2.attn2
        un_gs_32_640(act().at(11837440).cast(), act().at(2949120).cast());
        sync_cores();
        un_ga_32_640_none(act().at(11837440).cast(), act().at(2949120).cast(), w().at(19890176).cast(), w().at(19894272).cast(), w().at(11198464).cast(), act().at(13803520).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(13803520).cast(), w().at(19898368).cast(), w().at(20717568).cast(), act().at(15769600).cast());
        sync_cores();
        un_ln_32_640(act().at(15769600).cast(), w().at(20721664).cast(), w().at(20725760).cast(), act().at(17735680).cast());
        sync_cores();
        un_lr_32_640_768(act().at(17735680).cast(), w().at(20729856).cast(), act().at(19701760).cast());
        sync_cores();
        un_lr_32_640_768(act().at(17735680).cast(), w().at(21712896).cast(), act().at(22061056).cast());
        sync_cores();
        un_lr_32_640_768(act().at(17735680).cast(), w().at(22695936).cast(), act().at(24420352).cast());
        sync_cores();
        un_at_32_s(act().at(19701760).cast(), act().at(22061056).cast(), act().at(24420352).cast(), w().at(23678976).cast(), act().at(26779648).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(26779648).cast(), w().at(23683072).cast(), w().at(24666112).cast(), act().at(15769600).cast(), act().at(29138944).cast());
        sync_cores();
        un_ln_32_640(act().at(29138944).cast(), w().at(24670208).cast(), w().at(24674304).cast(), act().at(31105024).cast());
        sync_cores();
        un_lr_32_640_768(act().at(31105024).cast(), w().at(24678400).cast(), act().at(33071104).cast());
        sync_cores();
        un_kv_768(in_x().at(3031040).cast(), w().at(25661440).cast(), act().at(35430400).cast());
        sync_cores();
        un_kv_768(in_x().at(3031040).cast(), w().at(26841088).cast(), act().at(35627008).cast());
        sync_cores();
        un_at_32_x(act().at(33071104).cast(), act().at(35430400).cast(), act().at(35627008).cast(), w().at(28020736).cast(), act().at(35823616).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(35823616).cast(), w().at(28024832).cast(), w().at(29007872).cast(), act().at(29138944).cast(), act().at(38182912).cast());
        sync_cores();
        un_ln_32_640(act().at(38182912).cast(), w().at(29011968).cast(), w().at(29016064).cast(), act().at(40148992).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(40148992).cast(), w().at(29020160).cast(), w().at(32296960).cast(), act().at(42115072).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(40148992).cast(), w().at(32309248).cast(), w().at(35586048).cast(), act().at(49979392).cast());
        sync_cores();
        un_gg_32(act().at(42115072).cast(), act().at(49979392).cast(), act().at(57843712).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(57843712).cast(), w().at(35598336).cast(), w().at(38875136).cast(), act().at(38182912).cast(), act().at(65708032).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(65708032).cast(), w().at(38879232).cast(), w().at(39862272).cast(), act().at(11837440).cast(), act().at(68067328).cast());
        sync_cores();
        // up2.up
        un_up_32_640_256(act().at(68067328).cast(), act().at(70033408).cast());
        sync_cores();
        un_up_32_640_256(act().at(68067840).cast(), act().at(70033920).cast());
        sync_cores();
        un_up_32_640_128(act().at(68068352).cast(), act().at(70034432).cast());
        sync_cores();
        un_c3_64_640_640(act().at(70033408).cast(), w().at(39866368).cast(), w().at(47239168).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up2.res2
        un_cp_32_640_960(in_x().at(3227648).cast(), act().at(0).cast());
        sync_cores();
        un_cp_32_320_960(in_x().at(5193728).cast(), act().at(1280).cast());
        sync_cores();
        un_gs_32_960(act().at(0).cast(), act().at(2949120).cast());
        sync_cores();
        un_ga_32_960_silu(act().at(0).cast(), act().at(2949120).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(2990080).cast());
        sync_cores();
        un_c3_32_960_640(act().at(2990080).cast(), w().at(131072).cast(), in_x().at(74240).cast(), act().at(5939200).cast());
        sync_cores();
        un_gs_32_640(act().at(5939200).cast(), act().at(2949120).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(5939200).cast(), act().at(2949120).cast(), w().at(11190272).cast(), w().at(11194368).cast(), w().at(11198464).cast(), act().at(7905280).cast());
        sync_cores();
        un_c1_32_960_640(act().at(0).cast(), w().at(11280384).cast(), w().at(12509184).cast(), act().at(9871360).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(7905280).cast(), w().at(12513280).cast(), w().at(19886080).cast(), act().at(9871360).cast(), act().at(11837440).cast());
        sync_cores();
        // up2.attn2
        un_gs_32_640(act().at(11837440).cast(), act().at(2949120).cast());
        sync_cores();
        un_ga_32_640_none(act().at(11837440).cast(), act().at(2949120).cast(), w().at(19890176).cast(), w().at(19894272).cast(), w().at(11198464).cast(), act().at(13803520).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(13803520).cast(), w().at(19898368).cast(), w().at(20717568).cast(), act().at(15769600).cast());
        sync_cores();
        un_ln_32_640(act().at(15769600).cast(), w().at(20721664).cast(), w().at(20725760).cast(), act().at(17735680).cast());
        sync_cores();
        un_lr_32_640_768(act().at(17735680).cast(), w().at(20729856).cast(), act().at(19701760).cast());
        sync_cores();
        un_lr_32_640_768(act().at(17735680).cast(), w().at(21712896).cast(), act().at(22061056).cast());
        sync_cores();
        un_lr_32_640_768(act().at(17735680).cast(), w().at(22695936).cast(), act().at(24420352).cast());
        sync_cores();
        un_at_32_s(act().at(19701760).cast(), act().at(22061056).cast(), act().at(24420352).cast(), w().at(23678976).cast(), act().at(26779648).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(26779648).cast(), w().at(23683072).cast(), w().at(24666112).cast(), act().at(15769600).cast(), act().at(29138944).cast());
        sync_cores();
        un_ln_32_640(act().at(29138944).cast(), w().at(24670208).cast(), w().at(24674304).cast(), act().at(31105024).cast());
        sync_cores();
        un_lr_32_640_768(act().at(31105024).cast(), w().at(24678400).cast(), act().at(33071104).cast());
        sync_cores();
        un_kv_768(in_x().at(6176768).cast(), w().at(25661440).cast(), act().at(35430400).cast());
        sync_cores();
        un_kv_768(in_x().at(6176768).cast(), w().at(26841088).cast(), act().at(35627008).cast());
        sync_cores();
        un_at_32_x(act().at(33071104).cast(), act().at(35430400).cast(), act().at(35627008).cast(), w().at(28020736).cast(), act().at(35823616).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(35823616).cast(), w().at(28024832).cast(), w().at(29007872).cast(), act().at(29138944).cast(), act().at(38182912).cast());
        sync_cores();
        un_ln_32_640(act().at(38182912).cast(), w().at(29011968).cast(), w().at(29016064).cast(), act().at(40148992).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(40148992).cast(), w().at(29020160).cast(), w().at(32296960).cast(), act().at(42115072).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(40148992).cast(), w().at(32309248).cast(), w().at(35586048).cast(), act().at(49979392).cast());
        sync_cores();
        un_gg_32(act().at(42115072).cast(), act().at(49979392).cast(), act().at(57843712).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(57843712).cast(), w().at(35598336).cast(), w().at(38875136).cast(), act().at(38182912).cast(), act().at(65708032).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(65708032).cast(), w().at(38879232).cast(), w().at(39862272).cast(), act().at(11837440).cast(), act().at(68067328).cast());
        sync_cores();
        // up2.up
        un_up_32_640_256(act().at(68067328).cast(), act().at(70033408).cast());
        sync_cores();
        un_up_32_640_256(act().at(68067840).cast(), act().at(70033920).cast());
        sync_cores();
        un_up_32_640_128(act().at(68068352).cast(), act().at(70034432).cast());
        sync_cores();
        un_c3_64_640_640(act().at(70033408).cast(), w().at(39866368).cast(), w().at(47239168).cast(), out_y().at(6225920).cast());
        sync_cores();
    }
}

/// s15：up3.res0
#[cfg(feature = "s15")]
mod s15 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 18759680;
        out_y: output, bytes = 6225920;
        w: weights, bytes = 8175616;
        act: workspace, bytes = 28065792;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 856);
            un_zero(out_y().cast(), 190);
        } else {
            un_zero(act().at(14024704).cast(), 857);
            un_zero(out_y().at(3112960).cast(), 190);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up3.res0
        un_cp_64_640_960(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_cp_64_320_960(in_x().at(6307840).cast(), act().at(1280).cast());
        sync_cores();
        un_gs_64_960(act().at(0).cast(), act().at(9338880).cast());
        sync_cores();
        un_ga_64_960_silu(act().at(0).cast(), act().at(9338880).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(9379840).cast());
        sync_cores();
        un_c3_64_960_320(act().at(9379840).cast(), w().at(131072).cast(), in_x().at(76800).cast(), act().at(18718720).cast());
        sync_cores();
        un_gs_64_320(act().at(18718720).cast(), act().at(9338880).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(18718720).cast(), act().at(9338880).cast(), w().at(5660672).cast(), w().at(5664768).cast(), w().at(5668864).cast(), act().at(21831680).cast());
        sync_cores();
        un_c1_64_960_320(act().at(0).cast(), w().at(5709824).cast(), w().at(6324224).cast(), act().at(24944640).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(21831680).cast(), w().at(6328320).cast(), w().at(8171520).cast(), act().at(24944640).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.res0
        un_cp_64_640_960(in_x().at(9420800).cast(), act().at(0).cast());
        sync_cores();
        un_cp_64_320_960(in_x().at(15646720).cast(), act().at(1280).cast());
        sync_cores();
        un_gs_64_960(act().at(0).cast(), act().at(9338880).cast());
        sync_cores();
        un_ga_64_960_silu(act().at(0).cast(), act().at(9338880).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(9379840).cast());
        sync_cores();
        un_c3_64_960_320(act().at(9379840).cast(), w().at(131072).cast(), in_x().at(76800).cast(), act().at(18718720).cast());
        sync_cores();
        un_gs_64_320(act().at(18718720).cast(), act().at(9338880).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(18718720).cast(), act().at(9338880).cast(), w().at(5660672).cast(), w().at(5664768).cast(), w().at(5668864).cast(), act().at(21831680).cast());
        sync_cores();
        un_c1_64_960_320(act().at(0).cast(), w().at(5709824).cast(), w().at(6324224).cast(), act().at(24944640).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(21831680).cast(), w().at(6328320).cast(), w().at(8171520).cast(), act().at(24944640).cast(), out_y().at(3112960).cast());
        sync_cores();
    }
}

/// s16：up3.attn0
#[cfg(feature = "s16")]
mod s16 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 6619136;
        out_y: output, bytes = 6225920;
        w: weights, bytes = 6537216;
        act: workspace, bytes = 92454912;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2821);
            un_zero(out_y().cast(), 190);
        } else {
            un_zero(act().at(46219264).cast(), 2822);
            un_zero(out_y().at(3112960).cast(), 190);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up3.attn0
        un_gs_64_320(in_x().at(0).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(0).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(0).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.attn0
        un_gs_64_320(in_x().at(3309568).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(3309568).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(3309568).cast(), out_y().at(3112960).cast());
        sync_cores();
    }
}

/// s17：up3.res1
#[cfg(feature = "s17")]
mod s17 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 12533760;
        out_y: output, bytes = 6225920;
        w: weights, bytes = 6086656;
        act: workspace, bytes = 21839872;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 666);
            un_zero(out_y().cast(), 190);
        } else {
            un_zero(act().at(10911744).cast(), 667);
            un_zero(out_y().at(3112960).cast(), 190);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up3.res1
        un_cp_64_320_640(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_cp_64_320_640(in_x().at(3194880).cast(), act().at(640).cast());
        sync_cores();
        un_gs_64_640(act().at(0).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_640_silu(act().at(0).cast(), act().at(6225920).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(6266880).cast());
        sync_cores();
        un_c3_64_640_320(act().at(6266880).cast(), w().at(90112).cast(), in_x().at(78080).cast(), act().at(12492800).cast());
        sync_cores();
        un_gs_64_320(act().at(12492800).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(12492800).cast(), act().at(6225920).cast(), w().at(3776512).cast(), w().at(3780608).cast(), w().at(3784704).cast(), act().at(15605760).cast());
        sync_cores();
        un_c1_64_640_320(act().at(0).cast(), w().at(3825664).cast(), w().at(4235264).cast(), act().at(18718720).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(15605760).cast(), w().at(4239360).cast(), w().at(6082560).cast(), act().at(18718720).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.res1
        un_cp_64_320_640(in_x().at(6307840).cast(), act().at(0).cast());
        sync_cores();
        un_cp_64_320_640(in_x().at(9420800).cast(), act().at(640).cast());
        sync_cores();
        un_gs_64_640(act().at(0).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_640_silu(act().at(0).cast(), act().at(6225920).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(6266880).cast());
        sync_cores();
        un_c3_64_640_320(act().at(6266880).cast(), w().at(90112).cast(), in_x().at(78080).cast(), act().at(12492800).cast());
        sync_cores();
        un_gs_64_320(act().at(12492800).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(12492800).cast(), act().at(6225920).cast(), w().at(3776512).cast(), w().at(3780608).cast(), w().at(3784704).cast(), act().at(15605760).cast());
        sync_cores();
        un_c1_64_640_320(act().at(0).cast(), w().at(3825664).cast(), w().at(4235264).cast(), act().at(18718720).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(15605760).cast(), w().at(4239360).cast(), w().at(6082560).cast(), act().at(18718720).cast(), out_y().at(3112960).cast());
        sync_cores();
    }
}

/// s18：up3.attn1
#[cfg(feature = "s18")]
mod s18 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 6619136;
        out_y: output, bytes = 6225920;
        w: weights, bytes = 6537216;
        act: workspace, bytes = 92454912;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2821);
            un_zero(out_y().cast(), 190);
        } else {
            un_zero(act().at(46219264).cast(), 2822);
            un_zero(out_y().at(3112960).cast(), 190);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up3.attn1
        un_gs_64_320(in_x().at(0).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(0).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(0).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.attn1
        un_gs_64_320(in_x().at(3309568).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(3309568).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(3309568).cast(), out_y().at(3112960).cast());
        sync_cores();
    }
}

/// s19：up3.res2
#[cfg(feature = "s19")]
mod s19 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 12533760;
        out_y: output, bytes = 6225920;
        w: weights, bytes = 6086656;
        act: workspace, bytes = 21839872;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 666);
            un_zero(out_y().cast(), 190);
        } else {
            un_zero(act().at(10911744).cast(), 667);
            un_zero(out_y().at(3112960).cast(), 190);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up3.res2
        un_cp_64_320_640(in_x().at(81920).cast(), act().at(0).cast());
        sync_cores();
        un_cp_64_320_640(in_x().at(3194880).cast(), act().at(640).cast());
        sync_cores();
        un_gs_64_640(act().at(0).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_640_silu(act().at(0).cast(), act().at(6225920).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(6266880).cast());
        sync_cores();
        un_c3_64_640_320(act().at(6266880).cast(), w().at(90112).cast(), in_x().at(79360).cast(), act().at(12492800).cast());
        sync_cores();
        un_gs_64_320(act().at(12492800).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(12492800).cast(), act().at(6225920).cast(), w().at(3776512).cast(), w().at(3780608).cast(), w().at(3784704).cast(), act().at(15605760).cast());
        sync_cores();
        un_c1_64_640_320(act().at(0).cast(), w().at(3825664).cast(), w().at(4235264).cast(), act().at(18718720).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(15605760).cast(), w().at(4239360).cast(), w().at(6082560).cast(), act().at(18718720).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.res2
        un_cp_64_320_640(in_x().at(6307840).cast(), act().at(0).cast());
        sync_cores();
        un_cp_64_320_640(in_x().at(9420800).cast(), act().at(640).cast());
        sync_cores();
        un_gs_64_640(act().at(0).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_640_silu(act().at(0).cast(), act().at(6225920).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(6266880).cast());
        sync_cores();
        un_c3_64_640_320(act().at(6266880).cast(), w().at(90112).cast(), in_x().at(79360).cast(), act().at(12492800).cast());
        sync_cores();
        un_gs_64_320(act().at(12492800).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(12492800).cast(), act().at(6225920).cast(), w().at(3776512).cast(), w().at(3780608).cast(), w().at(3784704).cast(), act().at(15605760).cast());
        sync_cores();
        un_c1_64_640_320(act().at(0).cast(), w().at(3825664).cast(), w().at(4235264).cast(), act().at(18718720).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(15605760).cast(), w().at(4239360).cast(), w().at(6082560).cast(), act().at(18718720).cast(), out_y().at(3112960).cast());
        sync_cores();
    }
}

/// s20：up3.attn2、out
#[cfg(feature = "s20")]
mod s20 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 6619136;
        out_y: output, bytes = 622592;
        w: weights, bytes = 6733824;
        act: workspace, bytes = 98680832;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 3011);
            un_zero(out_y().cast(), 19);
        } else {
            un_zero(act().at(49332224).cast(), 3012);
            un_zero(out_y().at(311296).cast(), 19);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up3.attn2
        un_gs_64_320(in_x().at(0).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(0).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(3112960).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(0).cast(), act().at(92446720).cast());
        sync_cores();
        // out
        un_gs_64_320(act().at(92446720).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(92446720).cast(), act().at(0).cast(), w().at(6537216).cast(), w().at(6541312).cast(), w().at(8192).cast(), act().at(95559680).cast());
        sync_cores();
        un_c3_64_320_32(act().at(95559680).cast(), w().at(6545408).cast(), w().at(6729728).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.attn2
        un_gs_64_320(in_x().at(3309568).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_none(in_x().at(3309568).cast(), act().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), w().at(8192).cast(), act().at(40960).cast());
        sync_cores();
        un_c1_64_320_320(act().at(40960).cast(), w().at(49152).cast(), w().at(253952).cast(), act().at(3153920).cast());
        sync_cores();
        un_ln_64_320(act().at(3153920).cast(), w().at(258048).cast(), w().at(262144).cast(), act().at(6266880).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(266240).cast(), w().at(593920).cast(), act().at(9379840).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(598016).cast(), w().at(593920).cast(), act().at(14360576).cast());
        sync_cores();
        un_c1_64_320_512(act().at(6266880).cast(), w().at(925696).cast(), w().at(593920).cast(), act().at(19341312).cast());
        sync_cores();
        un_at_64_s(act().at(9379840).cast(), act().at(14360576).cast(), act().at(19341312).cast(), w().at(1253376).cast(), act().at(24322048).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(24322048).cast(), w().at(1265664).cast(), w().at(1593344).cast(), act().at(3153920).cast(), act().at(29302784).cast());
        sync_cores();
        un_ln_64_320(act().at(29302784).cast(), w().at(1597440).cast(), w().at(1601536).cast(), act().at(32415744).cast());
        sync_cores();
        un_c1_64_320_512(act().at(32415744).cast(), w().at(1605632).cast(), w().at(593920).cast(), act().at(35528704).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(1933312).cast(), act().at(40509440).cast());
        sync_cores();
        un_kv_512(in_x().at(6422528).cast(), w().at(2719744).cast(), act().at(40640512).cast());
        sync_cores();
        un_at_64_x(act().at(35528704).cast(), act().at(40509440).cast(), act().at(40640512).cast(), w().at(3506176).cast(), act().at(40771584).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(40771584).cast(), w().at(3510272).cast(), w().at(3837952).cast(), act().at(29302784).cast(), act().at(45752320).cast());
        sync_cores();
        un_ln_64_320(act().at(45752320).cast(), w().at(3842048).cast(), w().at(3846144).cast(), act().at(48865280).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(3850240).cast(), w().at(4669440).cast(), act().at(51978240).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(48865280).cast(), w().at(4677632).cast(), w().at(5496832).cast(), act().at(64430080).cast());
        sync_cores();
        un_gg_64(act().at(51978240).cast(), act().at(64430080).cast(), act().at(76881920).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(76881920).cast(), w().at(5505024).cast(), w().at(6324224).cast(), act().at(45752320).cast(), act().at(89333760).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(89333760).cast(), w().at(6328320).cast(), w().at(6533120).cast(), in_x().at(3309568).cast(), act().at(92446720).cast());
        sync_cores();
        // out
        un_gs_64_320(act().at(92446720).cast(), act().at(0).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(92446720).cast(), act().at(0).cast(), w().at(6537216).cast(), w().at(6541312).cast(), w().at(8192).cast(), act().at(95559680).cast());
        sync_cores();
        un_c3_64_320_32(act().at(95559680).cast(), w().at(6545408).cast(), w().at(6729728).cast(), out_y().at(311296).cast());
        sync_cores();
    }
}
