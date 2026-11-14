//! 各段入口（**`tools/unet_gen.py program` 生成，不要手改**）：偏移都是字面量，和 `crates/unet/plan.json` 出自同一次计算。
//! 接线的规矩同 VAE（crates/vae/src/step.rs）：`cores = 2` 与按 num_cores() 分行的 kernel 两核都调、不分行的单核 kernel 只让 0 号核调，
//! 每次调用之后 `sync_cores()`；输入、输出互不重叠（不原地）。CFG 的两路先后跑同一串算子：权重同一份，段内临时图同一份，
//! 段间的张量（in_x、out_y）每路一份。
use crate::kernels::*;
use npu::prelude::*;

/// s1：time、conv_in、down0.res0、down0.attn0、down0.attn0.cross
#[cfg(feature = "s1")]
mod s1 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 1019904;
        out_y: output, bytes = 18759680;
        w: weights, bytes = 63954944;
        act: workspace, bytes = 55918592;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 1706);
            un_zero(out_y().cast(), 572);
        } else {
            un_zero(act().at(27951104).cast(), 1707);
            un_zero(out_y().at(9371648).cast(), 573);
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
        // down0.attn0
        un_gs_64_320(out_y().at(3194880).cast(), act().at(823296).cast());
        sync_cores();
        un_ga_64_320_none(out_y().at(3194880).cast(), act().at(823296).cast(), w().at(60153856).cast(), w().at(60157952).cast(), w().at(56414208).cast(), act().at(10203136).cast());
        sync_cores();
        un_c1_64_320_320(act().at(10203136).cast(), w().at(60162048).cast(), w().at(60366848).cast(), act().at(13316096).cast());
        sync_cores();
        un_ln_64_320(act().at(13316096).cast(), w().at(60370944).cast(), w().at(60375040).cast(), act().at(16429056).cast());
        sync_cores();
        un_c1_64_320_512(act().at(16429056).cast(), w().at(60379136).cast(), w().at(60706816).cast(), act().at(19542016).cast());
        sync_cores();
        un_c1_64_320_512(act().at(16429056).cast(), w().at(60710912).cast(), w().at(60706816).cast(), act().at(24522752).cast());
        sync_cores();
        un_c1_64_320_512(act().at(16429056).cast(), w().at(61038592).cast(), w().at(60706816).cast(), act().at(29503488).cast());
        sync_cores();
        un_at_64_s(act().at(19542016).cast(), act().at(24522752).cast(), act().at(29503488).cast(), w().at(61366272).cast(), act().at(34484224).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(34484224).cast(), w().at(61378560).cast(), w().at(61706240).cast(), act().at(13316096).cast(), act().at(39464960).cast());
        sync_cores();
        // down0.attn0.cross
        un_ln_64_320(act().at(39464960).cast(), w().at(61710336).cast(), w().at(61714432).cast(), act().at(42577920).cast());
        sync_cores();
        un_c1_64_320_512(act().at(42577920).cast(), w().at(61718528).cast(), w().at(60706816).cast(), act().at(45690880).cast());
        sync_cores();
        un_kv_512(in_x().at(315392).cast(), w().at(62046208).cast(), act().at(50671616).cast());
        sync_cores();
        un_kv_512(in_x().at(315392).cast(), w().at(62832640).cast(), act().at(50802688).cast());
        sync_cores();
        un_at_64_x(act().at(45690880).cast(), act().at(50671616).cast(), act().at(50802688).cast(), w().at(63619072).cast(), act().at(50933760).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(50933760).cast(), w().at(63623168).cast(), w().at(63950848).cast(), act().at(39464960).cast(), out_y().at(6307840).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // conv_in
        un_c3_64_32_320(in_x().at(512000).cast(), w().at(56217600).cast(), w().at(56401920).cast(), out_y().at(9420800).cast());
        sync_cores();
        // down0.res0
        un_gs_64_320(out_y().at(9420800).cast(), act().at(823296).cast());
        sync_cores();
        un_ga_64_320_silu(out_y().at(9420800).cast(), act().at(823296).cast(), w().at(56406016).cast(), w().at(56410112).cast(), w().at(56414208).cast(), act().at(864256).cast());
        sync_cores();
        un_c3_64_320_320(act().at(864256).cast(), w().at(56455168).cast(), out_y().at(0).cast(), act().at(3977216).cast());
        sync_cores();
        un_gs_64_320(act().at(3977216).cast(), act().at(823296).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(3977216).cast(), act().at(823296).cast(), w().at(58298368).cast(), w().at(58302464).cast(), w().at(56414208).cast(), act().at(7090176).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(7090176).cast(), w().at(58306560).cast(), w().at(60149760).cast(), out_y().at(9420800).cast(), out_y().at(12533760).cast());
        sync_cores();
        // down0.attn0
        un_gs_64_320(out_y().at(12533760).cast(), act().at(823296).cast());
        sync_cores();
        un_ga_64_320_none(out_y().at(12533760).cast(), act().at(823296).cast(), w().at(60153856).cast(), w().at(60157952).cast(), w().at(56414208).cast(), act().at(10203136).cast());
        sync_cores();
        un_c1_64_320_320(act().at(10203136).cast(), w().at(60162048).cast(), w().at(60366848).cast(), act().at(13316096).cast());
        sync_cores();
        un_ln_64_320(act().at(13316096).cast(), w().at(60370944).cast(), w().at(60375040).cast(), act().at(16429056).cast());
        sync_cores();
        un_c1_64_320_512(act().at(16429056).cast(), w().at(60379136).cast(), w().at(60706816).cast(), act().at(19542016).cast());
        sync_cores();
        un_c1_64_320_512(act().at(16429056).cast(), w().at(60710912).cast(), w().at(60706816).cast(), act().at(24522752).cast());
        sync_cores();
        un_c1_64_320_512(act().at(16429056).cast(), w().at(61038592).cast(), w().at(60706816).cast(), act().at(29503488).cast());
        sync_cores();
        un_at_64_s(act().at(19542016).cast(), act().at(24522752).cast(), act().at(29503488).cast(), w().at(61366272).cast(), act().at(34484224).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(34484224).cast(), w().at(61378560).cast(), w().at(61706240).cast(), act().at(13316096).cast(), act().at(39464960).cast());
        sync_cores();
        // down0.attn0.cross
        un_ln_64_320(act().at(39464960).cast(), w().at(61710336).cast(), w().at(61714432).cast(), act().at(42577920).cast());
        sync_cores();
        un_c1_64_320_512(act().at(42577920).cast(), w().at(61718528).cast(), w().at(60706816).cast(), act().at(45690880).cast());
        sync_cores();
        un_kv_512(in_x().at(823296).cast(), w().at(62046208).cast(), act().at(50671616).cast());
        sync_cores();
        un_kv_512(in_x().at(823296).cast(), w().at(62832640).cast(), act().at(50802688).cast());
        sync_cores();
        un_at_64_x(act().at(45690880).cast(), act().at(50671616).cast(), act().at(50802688).cast(), w().at(63619072).cast(), act().at(50933760).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(50933760).cast(), w().at(63623168).cast(), w().at(63950848).cast(), act().at(39464960).cast(), out_y().at(15646720).cast());
        sync_cores();
    }
}

/// s2：down0.attn0.ff、down0.res1
#[cfg(feature = "s2")]
mod s2 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 12533760;
        out_y: output, bytes = 12451840;
        w: weights, bytes = 6443008;
        act: workspace, bytes = 52969472;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 1616);
            un_zero(out_y().cast(), 380);
        } else {
            un_zero(act().at(26476544).cast(), 1617);
            un_zero(out_y().at(6225920).cast(), 380);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down0.attn0.ff
        un_ln_64_320(in_x().at(81920).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(8192).cast(), w().at(827392).cast(), act().at(3112960).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(835584).cast(), w().at(1654784).cast(), act().at(15564800).cast());
        sync_cores();
        un_gg_64(act().at(3112960).cast(), act().at(15564800).cast(), act().at(28016640).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(28016640).cast(), w().at(1662976).cast(), w().at(2482176).cast(), in_x().at(81920).cast(), act().at(40468480).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(40468480).cast(), w().at(2486272).cast(), w().at(2691072).cast(), in_x().at(3194880).cast(), out_y().at(0).cast());
        sync_cores();
        // down0.res1
        un_gs_64_320(out_y().at(0).cast(), act().at(43581440).cast());
        sync_cores();
        un_ga_64_320_silu(out_y().at(0).cast(), act().at(43581440).cast(), w().at(2695168).cast(), w().at(2699264).cast(), w().at(2703360).cast(), act().at(43622400).cast());
        sync_cores();
        un_c3_64_320_320(act().at(43622400).cast(), w().at(2744320).cast(), in_x().at(1280).cast(), act().at(46735360).cast());
        sync_cores();
        un_gs_64_320(act().at(46735360).cast(), act().at(43581440).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(46735360).cast(), act().at(43581440).cast(), w().at(4587520).cast(), w().at(4591616).cast(), w().at(2703360).cast(), act().at(49848320).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(49848320).cast(), w().at(4595712).cast(), w().at(6438912).cast(), out_y().at(0).cast(), out_y().at(3112960).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down0.attn0.ff
        un_ln_64_320(in_x().at(6307840).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(8192).cast(), w().at(827392).cast(), act().at(3112960).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(835584).cast(), w().at(1654784).cast(), act().at(15564800).cast());
        sync_cores();
        un_gg_64(act().at(3112960).cast(), act().at(15564800).cast(), act().at(28016640).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(28016640).cast(), w().at(1662976).cast(), w().at(2482176).cast(), in_x().at(6307840).cast(), act().at(40468480).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(40468480).cast(), w().at(2486272).cast(), w().at(2691072).cast(), in_x().at(9420800).cast(), out_y().at(6225920).cast());
        sync_cores();
        // down0.res1
        un_gs_64_320(out_y().at(6225920).cast(), act().at(43581440).cast());
        sync_cores();
        un_ga_64_320_silu(out_y().at(6225920).cast(), act().at(43581440).cast(), w().at(2695168).cast(), w().at(2699264).cast(), w().at(2703360).cast(), act().at(43622400).cast());
        sync_cores();
        un_c3_64_320_320(act().at(43622400).cast(), w().at(2744320).cast(), in_x().at(1280).cast(), act().at(46735360).cast());
        sync_cores();
        un_gs_64_320(act().at(46735360).cast(), act().at(43581440).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(46735360).cast(), act().at(43581440).cast(), w().at(4587520).cast(), w().at(4591616).cast(), w().at(2703360).cast(), act().at(49848320).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(49848320).cast(), w().at(4595712).cast(), w().at(6438912).cast(), out_y().at(6225920).cast(), out_y().at(9338880).cast());
        sync_cores();
    }
}

/// s3：down0.attn1、down0.attn1.cross、down0.attn1.ff
#[cfg(feature = "s3")]
mod s3 {
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
        // down0.attn1.cross
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
        // down0.attn1.ff
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
        // down0.attn1.cross
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
        // down0.attn1.ff
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

/// s4：down0.down、down1.res0、down1.attn0、down1.attn0.cross、down1.attn0.ff、down1.res1、down1.attn1、down1.attn1.cross、down1.attn1.ff、down1.down
#[cfg(feature = "s4")]
mod s4 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 6701056;
        out_y: output, bytes = 11468800;
        w: weights, bytes = 75550720;
        act: workspace, bytes = 125288448;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 3823);
            un_zero(out_y().cast(), 350);
        } else {
            un_zero(act().at(62636032).cast(), 3824);
            un_zero(out_y().at(5734400).cast(), 350);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // down0.down
        un_c3_32_320_320_s2(in_x().at(81920).cast(), w().at(0).cast(), w().at(1843200).cast(), out_y().at(0).cast());
        sync_cores();
        // down1.res0
        un_gs_32_320(out_y().at(0).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_320_silu(out_y().at(0).cast(), act().at(0).cast(), w().at(1847296).cast(), w().at(1851392).cast(), w().at(1855488).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_32_320_640(act().at(40960).cast(), w().at(1896448).cast(), in_x().at(2560).cast(), act().at(1024000).cast());
        sync_cores();
        un_gs_32_640(act().at(1024000).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(1024000).cast(), act().at(0).cast(), w().at(5582848).cast(), w().at(5586944).cast(), w().at(5591040).cast(), act().at(2990080).cast());
        sync_cores();
        un_c1_32_320_640(out_y().at(0).cast(), w().at(5672960).cast(), w().at(6082560).cast(), act().at(4956160).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(2990080).cast(), w().at(6086656).cast(), w().at(13459456).cast(), act().at(4956160).cast(), act().at(6922240).cast());
        sync_cores();
        // down1.attn0
        un_gs_32_640(act().at(6922240).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(6922240).cast(), act().at(0).cast(), w().at(13463552).cast(), w().at(13467648).cast(), w().at(5591040).cast(), act().at(8888320).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(8888320).cast(), w().at(13471744).cast(), w().at(14290944).cast(), act().at(10854400).cast());
        sync_cores();
        un_ln_32_640(act().at(10854400).cast(), w().at(14295040).cast(), w().at(14299136).cast(), act().at(12820480).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(14303232).cast(), act().at(14786560).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(15286272).cast(), act().at(17145856).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(16269312).cast(), act().at(19505152).cast());
        sync_cores();
        un_at_32_s(act().at(14786560).cast(), act().at(17145856).cast(), act().at(19505152).cast(), w().at(17252352).cast(), act().at(21864448).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(21864448).cast(), w().at(17256448).cast(), w().at(18239488).cast(), act().at(10854400).cast(), act().at(24223744).cast());
        sync_cores();
        // down1.attn0.cross
        un_ln_32_640(act().at(24223744).cast(), w().at(18243584).cast(), w().at(18247680).cast(), act().at(26189824).cast());
        sync_cores();
        un_lr_32_640_768(act().at(26189824).cast(), w().at(18251776).cast(), act().at(28155904).cast());
        sync_cores();
        un_kv_768(in_x().at(3194880).cast(), w().at(19234816).cast(), act().at(30515200).cast());
        sync_cores();
        un_kv_768(in_x().at(3194880).cast(), w().at(20414464).cast(), act().at(30711808).cast());
        sync_cores();
        un_at_32_x(act().at(28155904).cast(), act().at(30515200).cast(), act().at(30711808).cast(), w().at(21594112).cast(), act().at(30908416).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(30908416).cast(), w().at(21598208).cast(), w().at(22581248).cast(), act().at(24223744).cast(), act().at(33267712).cast());
        sync_cores();
        // down1.attn0.ff
        un_ln_32_640(act().at(33267712).cast(), w().at(22585344).cast(), w().at(22589440).cast(), act().at(35233792).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(35233792).cast(), w().at(22593536).cast(), w().at(25870336).cast(), act().at(37199872).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(35233792).cast(), w().at(25882624).cast(), w().at(29159424).cast(), act().at(45064192).cast());
        sync_cores();
        un_gg_32(act().at(37199872).cast(), act().at(45064192).cast(), act().at(52928512).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(52928512).cast(), w().at(29171712).cast(), w().at(32448512).cast(), act().at(33267712).cast(), act().at(60792832).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(60792832).cast(), w().at(32452608).cast(), w().at(33435648).cast(), act().at(6922240).cast(), out_y().at(983040).cast());
        sync_cores();
        // down1.res1
        un_gs_32_640(out_y().at(983040).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(out_y().at(983040).cast(), act().at(0).cast(), w().at(33439744).cast(), w().at(33443840).cast(), w().at(5591040).cast(), act().at(63152128).cast());
        sync_cores();
        un_c3_32_640_640(act().at(63152128).cast(), w().at(33447936).cast(), in_x().at(5120).cast(), act().at(65118208).cast());
        sync_cores();
        un_gs_32_640(act().at(65118208).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(65118208).cast(), act().at(0).cast(), w().at(40820736).cast(), w().at(40824832).cast(), w().at(5591040).cast(), act().at(67084288).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(67084288).cast(), w().at(40828928).cast(), w().at(48201728).cast(), out_y().at(983040).cast(), act().at(69050368).cast());
        sync_cores();
        // down1.attn1
        un_gs_32_640(act().at(69050368).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(69050368).cast(), act().at(0).cast(), w().at(48205824).cast(), w().at(48209920).cast(), w().at(5591040).cast(), act().at(71016448).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(71016448).cast(), w().at(48214016).cast(), w().at(49033216).cast(), act().at(72982528).cast());
        sync_cores();
        un_ln_32_640(act().at(72982528).cast(), w().at(49037312).cast(), w().at(49041408).cast(), act().at(74948608).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(49045504).cast(), act().at(76914688).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(50028544).cast(), act().at(79273984).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(51011584).cast(), act().at(81633280).cast());
        sync_cores();
        un_at_32_s(act().at(76914688).cast(), act().at(79273984).cast(), act().at(81633280).cast(), w().at(17252352).cast(), act().at(83992576).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(83992576).cast(), w().at(51994624).cast(), w().at(52977664).cast(), act().at(72982528).cast(), act().at(86351872).cast());
        sync_cores();
        // down1.attn1.cross
        un_ln_32_640(act().at(86351872).cast(), w().at(52981760).cast(), w().at(52985856).cast(), act().at(88317952).cast());
        sync_cores();
        un_lr_32_640_768(act().at(88317952).cast(), w().at(52989952).cast(), act().at(90284032).cast());
        sync_cores();
        un_kv_768(in_x().at(3194880).cast(), w().at(53972992).cast(), act().at(92643328).cast());
        sync_cores();
        un_kv_768(in_x().at(3194880).cast(), w().at(55152640).cast(), act().at(92839936).cast());
        sync_cores();
        un_at_32_x(act().at(90284032).cast(), act().at(92643328).cast(), act().at(92839936).cast(), w().at(21594112).cast(), act().at(93036544).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(93036544).cast(), w().at(56332288).cast(), w().at(57315328).cast(), act().at(86351872).cast(), act().at(95395840).cast());
        sync_cores();
        // down1.attn1.ff
        un_ln_32_640(act().at(95395840).cast(), w().at(57319424).cast(), w().at(57323520).cast(), act().at(97361920).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(97361920).cast(), w().at(57327616).cast(), w().at(60604416).cast(), act().at(99328000).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(97361920).cast(), w().at(60616704).cast(), w().at(63893504).cast(), act().at(107192320).cast());
        sync_cores();
        un_gg_32(act().at(99328000).cast(), act().at(107192320).cast(), act().at(115056640).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(115056640).cast(), w().at(63905792).cast(), w().at(67182592).cast(), act().at(95395840).cast(), act().at(122920960).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(122920960).cast(), w().at(67186688).cast(), w().at(68169728).cast(), act().at(69050368).cast(), out_y().at(2949120).cast());
        sync_cores();
        // down1.down
        un_c3_16_640_640_s2(out_y().at(2949120).cast(), w().at(68173824).cast(), w().at(75546624).cast(), out_y().at(4915200).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // down0.down
        un_c3_32_320_320_s2(in_x().at(3391488).cast(), w().at(0).cast(), w().at(1843200).cast(), out_y().at(5734400).cast());
        sync_cores();
        // down1.res0
        un_gs_32_320(out_y().at(5734400).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_320_silu(out_y().at(5734400).cast(), act().at(0).cast(), w().at(1847296).cast(), w().at(1851392).cast(), w().at(1855488).cast(), act().at(40960).cast());
        sync_cores();
        un_c3_32_320_640(act().at(40960).cast(), w().at(1896448).cast(), in_x().at(2560).cast(), act().at(1024000).cast());
        sync_cores();
        un_gs_32_640(act().at(1024000).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(1024000).cast(), act().at(0).cast(), w().at(5582848).cast(), w().at(5586944).cast(), w().at(5591040).cast(), act().at(2990080).cast());
        sync_cores();
        un_c1_32_320_640(out_y().at(5734400).cast(), w().at(5672960).cast(), w().at(6082560).cast(), act().at(4956160).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(2990080).cast(), w().at(6086656).cast(), w().at(13459456).cast(), act().at(4956160).cast(), act().at(6922240).cast());
        sync_cores();
        // down1.attn0
        un_gs_32_640(act().at(6922240).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(6922240).cast(), act().at(0).cast(), w().at(13463552).cast(), w().at(13467648).cast(), w().at(5591040).cast(), act().at(8888320).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(8888320).cast(), w().at(13471744).cast(), w().at(14290944).cast(), act().at(10854400).cast());
        sync_cores();
        un_ln_32_640(act().at(10854400).cast(), w().at(14295040).cast(), w().at(14299136).cast(), act().at(12820480).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(14303232).cast(), act().at(14786560).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(15286272).cast(), act().at(17145856).cast());
        sync_cores();
        un_lr_32_640_768(act().at(12820480).cast(), w().at(16269312).cast(), act().at(19505152).cast());
        sync_cores();
        un_at_32_s(act().at(14786560).cast(), act().at(17145856).cast(), act().at(19505152).cast(), w().at(17252352).cast(), act().at(21864448).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(21864448).cast(), w().at(17256448).cast(), w().at(18239488).cast(), act().at(10854400).cast(), act().at(24223744).cast());
        sync_cores();
        // down1.attn0.cross
        un_ln_32_640(act().at(24223744).cast(), w().at(18243584).cast(), w().at(18247680).cast(), act().at(26189824).cast());
        sync_cores();
        un_lr_32_640_768(act().at(26189824).cast(), w().at(18251776).cast(), act().at(28155904).cast());
        sync_cores();
        un_kv_768(in_x().at(6504448).cast(), w().at(19234816).cast(), act().at(30515200).cast());
        sync_cores();
        un_kv_768(in_x().at(6504448).cast(), w().at(20414464).cast(), act().at(30711808).cast());
        sync_cores();
        un_at_32_x(act().at(28155904).cast(), act().at(30515200).cast(), act().at(30711808).cast(), w().at(21594112).cast(), act().at(30908416).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(30908416).cast(), w().at(21598208).cast(), w().at(22581248).cast(), act().at(24223744).cast(), act().at(33267712).cast());
        sync_cores();
        // down1.attn0.ff
        un_ln_32_640(act().at(33267712).cast(), w().at(22585344).cast(), w().at(22589440).cast(), act().at(35233792).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(35233792).cast(), w().at(22593536).cast(), w().at(25870336).cast(), act().at(37199872).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(35233792).cast(), w().at(25882624).cast(), w().at(29159424).cast(), act().at(45064192).cast());
        sync_cores();
        un_gg_32(act().at(37199872).cast(), act().at(45064192).cast(), act().at(52928512).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(52928512).cast(), w().at(29171712).cast(), w().at(32448512).cast(), act().at(33267712).cast(), act().at(60792832).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(60792832).cast(), w().at(32452608).cast(), w().at(33435648).cast(), act().at(6922240).cast(), out_y().at(6717440).cast());
        sync_cores();
        // down1.res1
        un_gs_32_640(out_y().at(6717440).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(out_y().at(6717440).cast(), act().at(0).cast(), w().at(33439744).cast(), w().at(33443840).cast(), w().at(5591040).cast(), act().at(63152128).cast());
        sync_cores();
        un_c3_32_640_640(act().at(63152128).cast(), w().at(33447936).cast(), in_x().at(5120).cast(), act().at(65118208).cast());
        sync_cores();
        un_gs_32_640(act().at(65118208).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(65118208).cast(), act().at(0).cast(), w().at(40820736).cast(), w().at(40824832).cast(), w().at(5591040).cast(), act().at(67084288).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(67084288).cast(), w().at(40828928).cast(), w().at(48201728).cast(), out_y().at(6717440).cast(), act().at(69050368).cast());
        sync_cores();
        // down1.attn1
        un_gs_32_640(act().at(69050368).cast(), act().at(0).cast());
        sync_cores();
        un_ga_32_640_none(act().at(69050368).cast(), act().at(0).cast(), w().at(48205824).cast(), w().at(48209920).cast(), w().at(5591040).cast(), act().at(71016448).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(71016448).cast(), w().at(48214016).cast(), w().at(49033216).cast(), act().at(72982528).cast());
        sync_cores();
        un_ln_32_640(act().at(72982528).cast(), w().at(49037312).cast(), w().at(49041408).cast(), act().at(74948608).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(49045504).cast(), act().at(76914688).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(50028544).cast(), act().at(79273984).cast());
        sync_cores();
        un_lr_32_640_768(act().at(74948608).cast(), w().at(51011584).cast(), act().at(81633280).cast());
        sync_cores();
        un_at_32_s(act().at(76914688).cast(), act().at(79273984).cast(), act().at(81633280).cast(), w().at(17252352).cast(), act().at(83992576).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(83992576).cast(), w().at(51994624).cast(), w().at(52977664).cast(), act().at(72982528).cast(), act().at(86351872).cast());
        sync_cores();
        // down1.attn1.cross
        un_ln_32_640(act().at(86351872).cast(), w().at(52981760).cast(), w().at(52985856).cast(), act().at(88317952).cast());
        sync_cores();
        un_lr_32_640_768(act().at(88317952).cast(), w().at(52989952).cast(), act().at(90284032).cast());
        sync_cores();
        un_kv_768(in_x().at(6504448).cast(), w().at(53972992).cast(), act().at(92643328).cast());
        sync_cores();
        un_kv_768(in_x().at(6504448).cast(), w().at(55152640).cast(), act().at(92839936).cast());
        sync_cores();
        un_at_32_x(act().at(90284032).cast(), act().at(92643328).cast(), act().at(92839936).cast(), w().at(21594112).cast(), act().at(93036544).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(93036544).cast(), w().at(56332288).cast(), w().at(57315328).cast(), act().at(86351872).cast(), act().at(95395840).cast());
        sync_cores();
        // down1.attn1.ff
        un_ln_32_640(act().at(95395840).cast(), w().at(57319424).cast(), w().at(57323520).cast(), act().at(97361920).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(97361920).cast(), w().at(57327616).cast(), w().at(60604416).cast(), act().at(99328000).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(97361920).cast(), w().at(60616704).cast(), w().at(63893504).cast(), act().at(107192320).cast());
        sync_cores();
        un_gg_32(act().at(99328000).cast(), act().at(107192320).cast(), act().at(115056640).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(115056640).cast(), w().at(63905792).cast(), w().at(67182592).cast(), act().at(95395840).cast(), act().at(122920960).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(122920960).cast(), w().at(67186688).cast(), w().at(68169728).cast(), act().at(69050368).cast(), out_y().at(8683520).cast());
        sync_cores();
        // down1.down
        un_c3_16_640_640_s2(out_y().at(8683520).cast(), w().at(68173824).cast(), w().at(75546624).cast(), out_y().at(10649600).cast());
        sync_cores();
    }
}

/// s5：down2.res0、down2.attn0、down2.attn0.cross、down2.attn0.ff
#[cfg(feature = "s5")]
mod s5 {
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
        // down2.attn0.cross
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
        // down2.attn0.ff
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
        // down2.attn0.cross
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
        // down2.attn0.ff
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

/// s6：down2.res1、down2.attn1、down2.attn1.cross、down2.attn1.ff
#[cfg(feature = "s6")]
mod s6 {
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
        // down2.attn1.cross
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
        // down2.attn1.ff
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
        // down2.attn1.cross
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
        // down2.attn1.ff
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

/// s7：down2.down、down3.res0、down3.res1、mid.res0、mid.attn、mid.attn.cross、mid.attn.ff、mid.res1、up0.res0、up0.res1、up0.res2
#[cfg(feature = "s7")]
mod s7 {
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
        // mid.attn.cross
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
        // mid.attn.ff
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
        // mid.attn.cross
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
        // mid.attn.ff
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

/// s8：up0.up、up1.res0、up1.attn0、up1.attn0.cross
#[cfg(feature = "s8")]
mod s8 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 5718016;
        out_y: output, bytes = 6553600;
        w: weights, bytes = 152035328;
        act: workspace, bytes = 33472512;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 1021);
            un_zero(out_y().cast(), 200);
        } else {
            un_zero(act().at(16728064).cast(), 1022);
            un_zero(out_y().at(3276800).cast(), 200);
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
        // up1.attn0
        un_gs_16_1280(out_y().at(0).cast(), act().at(6553600).cast());
        sync_cores();
        un_ga_16_1280_none(out_y().at(0).cast(), act().at(6553600).cast(), w().at(125075456).cast(), w().at(125083648).cast(), w().at(88850432).cast(), act().at(14786560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(14786560).cast(), w().at(125091840).cast(), w().at(128368640).cast(), act().at(16424960).cast());
        sync_cores();
        un_ln_16_1280(act().at(16424960).cast(), w().at(128376832).cast(), w().at(128385024).cast(), act().at(18063360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(18063360).cast(), w().at(128393216).cast(), w().at(131670016).cast(), act().at(19701760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(18063360).cast(), w().at(131678208).cast(), w().at(131670016).cast(), act().at(21340160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(18063360).cast(), w().at(134955008).cast(), w().at(131670016).cast(), act().at(22978560).cast());
        sync_cores();
        un_at_16_s(act().at(19701760).cast(), act().at(21340160).cast(), act().at(22978560).cast(), w().at(138231808).cast(), act().at(24616960).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(24616960).cast(), w().at(138235904).cast(), w().at(141512704).cast(), act().at(16424960).cast(), act().at(26255360).cast());
        sync_cores();
        // up1.attn0.cross
        un_ln_16_1280(act().at(26255360).cast(), w().at(141520896).cast(), w().at(141529088).cast(), act().at(27893760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(27893760).cast(), w().at(141537280).cast(), w().at(131670016).cast(), act().at(29532160).cast());
        sync_cores();
        un_kv_1280(in_x().at(2703360).cast(), w().at(144814080).cast(), act().at(31170560).cast());
        sync_cores();
        un_kv_1280(in_x().at(2703360).cast(), w().at(146780160).cast(), act().at(31498240).cast());
        sync_cores();
        un_at_16_x(act().at(29532160).cast(), act().at(31170560).cast(), act().at(31498240).cast(), w().at(148746240).cast(), act().at(31825920).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(31825920).cast(), w().at(148750336).cast(), w().at(152027136).cast(), act().at(26255360).cast(), out_y().at(1638400).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up0.up
        un_up_8_1280_256(in_x().at(2899968).cast(), act().at(0).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(2900480).cast(), act().at(512).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(2900992).cast(), act().at(1024).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(2901504).cast(), act().at(1536).cast());
        sync_cores();
        un_up_8_1280_256(in_x().at(2902016).cast(), act().at(2048).cast());
        sync_cores();
        un_c3_16_1280_1280(act().at(0).cast(), w().at(0).cast(), w().at(29491200).cast(), act().at(1638400).cast());
        sync_cores();
        // up1.res0
        un_cp_16_1280_2560(act().at(1638400).cast(), act().at(3276800).cast());
        sync_cores();
        un_cp_16_1280_2560(in_x().at(3883008).cast(), act().at(3279360).cast());
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
        un_c3_16_1280_1280_r(act().at(11509760).cast(), w().at(95576064).cast(), w().at(125067264).cast(), act().at(13148160).cast(), out_y().at(3276800).cast());
        sync_cores();
        // up1.attn0
        un_gs_16_1280(out_y().at(3276800).cast(), act().at(6553600).cast());
        sync_cores();
        un_ga_16_1280_none(out_y().at(3276800).cast(), act().at(6553600).cast(), w().at(125075456).cast(), w().at(125083648).cast(), w().at(88850432).cast(), act().at(14786560).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(14786560).cast(), w().at(125091840).cast(), w().at(128368640).cast(), act().at(16424960).cast());
        sync_cores();
        un_ln_16_1280(act().at(16424960).cast(), w().at(128376832).cast(), w().at(128385024).cast(), act().at(18063360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(18063360).cast(), w().at(128393216).cast(), w().at(131670016).cast(), act().at(19701760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(18063360).cast(), w().at(131678208).cast(), w().at(131670016).cast(), act().at(21340160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(18063360).cast(), w().at(134955008).cast(), w().at(131670016).cast(), act().at(22978560).cast());
        sync_cores();
        un_at_16_s(act().at(19701760).cast(), act().at(21340160).cast(), act().at(22978560).cast(), w().at(138231808).cast(), act().at(24616960).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(24616960).cast(), w().at(138235904).cast(), w().at(141512704).cast(), act().at(16424960).cast(), act().at(26255360).cast());
        sync_cores();
        // up1.attn0.cross
        un_ln_16_1280(act().at(26255360).cast(), w().at(141520896).cast(), w().at(141529088).cast(), act().at(27893760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(27893760).cast(), w().at(141537280).cast(), w().at(131670016).cast(), act().at(29532160).cast());
        sync_cores();
        un_kv_1280(in_x().at(5521408).cast(), w().at(144814080).cast(), act().at(31170560).cast());
        sync_cores();
        un_kv_1280(in_x().at(5521408).cast(), w().at(146780160).cast(), act().at(31498240).cast());
        sync_cores();
        un_at_16_x(act().at(29532160).cast(), act().at(31170560).cast(), act().at(31498240).cast(), w().at(148746240).cast(), act().at(31825920).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(31825920).cast(), w().at(148750336).cast(), w().at(152027136).cast(), act().at(26255360).cast(), out_y().at(4915200).cast());
        sync_cores();
    }
}

/// s9：up1.attn0.ff、up1.res1、up1.attn1、up1.attn1.cross、up1.attn1.ff
#[cfg(feature = "s9")]
mod s9 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 10305536;
        out_y: output, bytes = 3276800;
        w: weights, bytes = 207880192;
        act: workspace, bytes = 80986112;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2471);
            un_zero(out_y().cast(), 100);
        } else {
            un_zero(act().at(40484864).cast(), 2472);
            un_zero(out_y().at(1638400).cast(), 100);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up1.attn0.ff
        un_ln_16_1280(in_x().at(81920).cast(), w().at(0).cast(), w().at(8192).cast(), act().at(0).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(0).cast(), w().at(16384).cast(), w().at(13123584).cast(), act().at(1638400).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(0).cast(), w().at(13144064).cast(), w().at(26251264).cast(), act().at(8192000).cast());
        sync_cores();
        un_gg_16(act().at(1638400).cast(), act().at(8192000).cast(), act().at(14745600).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(14745600).cast(), w().at(26271744).cast(), w().at(39378944).cast(), in_x().at(81920).cast(), act().at(21299200).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(21299200).cast(), w().at(39387136).cast(), w().at(42663936).cast(), in_x().at(1720320).cast(), act().at(22937600).cast());
        sync_cores();
        // up1.res1
        un_cp_16_1280_2560(act().at(22937600).cast(), act().at(24576000).cast());
        sync_cores();
        un_cp_16_1280_2560(in_x().at(3358720).cast(), act().at(24578560).cast());
        sync_cores();
        un_gs_16_2560(act().at(24576000).cast(), act().at(27852800).cast());
        sync_cores();
        un_ga_16_2560_silu(act().at(24576000).cast(), act().at(27852800).cast(), w().at(42672128).cast(), w().at(42684416).cast(), w().at(42696704).cast(), act().at(27893760).cast());
        sync_cores();
        un_c3_16_2560_1280(act().at(27893760).cast(), w().at(43024384).cast(), in_x().at(58880).cast(), act().at(31170560).cast());
        sync_cores();
        un_gs_16_1280(act().at(31170560).cast(), act().at(27852800).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(31170560).cast(), act().at(27852800).cast(), w().at(102006784).cast(), w().at(102014976).cast(), w().at(102023168).cast(), act().at(32808960).cast());
        sync_cores();
        un_c1_16_2560_1280(act().at(24576000).cast(), w().at(102187008).cast(), w().at(108740608).cast(), act().at(34447360).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(32808960).cast(), w().at(108748800).cast(), w().at(138240000).cast(), act().at(34447360).cast(), act().at(36085760).cast());
        sync_cores();
        // up1.attn1
        un_gs_16_1280(act().at(36085760).cast(), act().at(27852800).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(36085760).cast(), act().at(27852800).cast(), w().at(138248192).cast(), w().at(138256384).cast(), w().at(102023168).cast(), act().at(37724160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(37724160).cast(), w().at(138264576).cast(), w().at(141541376).cast(), act().at(39362560).cast());
        sync_cores();
        un_ln_16_1280(act().at(39362560).cast(), w().at(141549568).cast(), w().at(141557760).cast(), act().at(41000960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(41000960).cast(), w().at(141565952).cast(), w().at(144842752).cast(), act().at(42639360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(41000960).cast(), w().at(144850944).cast(), w().at(144842752).cast(), act().at(44277760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(41000960).cast(), w().at(148127744).cast(), w().at(144842752).cast(), act().at(45916160).cast());
        sync_cores();
        un_at_16_s(act().at(42639360).cast(), act().at(44277760).cast(), act().at(45916160).cast(), w().at(151404544).cast(), act().at(47554560).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(47554560).cast(), w().at(151408640).cast(), w().at(154685440).cast(), act().at(39362560).cast(), act().at(49192960).cast());
        sync_cores();
        // up1.attn1.cross
        un_ln_16_1280(act().at(49192960).cast(), w().at(154693632).cast(), w().at(154701824).cast(), act().at(50831360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(50831360).cast(), w().at(154710016).cast(), w().at(144842752).cast(), act().at(52469760).cast());
        sync_cores();
        un_kv_1280(in_x().at(4997120).cast(), w().at(157986816).cast(), act().at(54108160).cast());
        sync_cores();
        un_kv_1280(in_x().at(4997120).cast(), w().at(159952896).cast(), act().at(54435840).cast());
        sync_cores();
        un_at_16_x(act().at(52469760).cast(), act().at(54108160).cast(), act().at(54435840).cast(), w().at(161918976).cast(), act().at(54763520).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(54763520).cast(), w().at(161923072).cast(), w().at(165199872).cast(), act().at(49192960).cast(), act().at(56401920).cast());
        sync_cores();
        // up1.attn1.ff
        un_ln_16_1280(act().at(56401920).cast(), w().at(165208064).cast(), w().at(165216256).cast(), act().at(58040320).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(58040320).cast(), w().at(165224448).cast(), w().at(178331648).cast(), act().at(59678720).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(58040320).cast(), w().at(178352128).cast(), w().at(191459328).cast(), act().at(66232320).cast());
        sync_cores();
        un_gg_16(act().at(59678720).cast(), act().at(66232320).cast(), act().at(72785920).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(72785920).cast(), w().at(191479808).cast(), w().at(204587008).cast(), act().at(56401920).cast(), act().at(79339520).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(79339520).cast(), w().at(204595200).cast(), w().at(207872000).cast(), act().at(36085760).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up1.attn0.ff
        un_ln_16_1280(in_x().at(5193728).cast(), w().at(0).cast(), w().at(8192).cast(), act().at(0).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(0).cast(), w().at(16384).cast(), w().at(13123584).cast(), act().at(1638400).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(0).cast(), w().at(13144064).cast(), w().at(26251264).cast(), act().at(8192000).cast());
        sync_cores();
        un_gg_16(act().at(1638400).cast(), act().at(8192000).cast(), act().at(14745600).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(14745600).cast(), w().at(26271744).cast(), w().at(39378944).cast(), in_x().at(5193728).cast(), act().at(21299200).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(21299200).cast(), w().at(39387136).cast(), w().at(42663936).cast(), in_x().at(6832128).cast(), act().at(22937600).cast());
        sync_cores();
        // up1.res1
        un_cp_16_1280_2560(act().at(22937600).cast(), act().at(24576000).cast());
        sync_cores();
        un_cp_16_1280_2560(in_x().at(8470528).cast(), act().at(24578560).cast());
        sync_cores();
        un_gs_16_2560(act().at(24576000).cast(), act().at(27852800).cast());
        sync_cores();
        un_ga_16_2560_silu(act().at(24576000).cast(), act().at(27852800).cast(), w().at(42672128).cast(), w().at(42684416).cast(), w().at(42696704).cast(), act().at(27893760).cast());
        sync_cores();
        un_c3_16_2560_1280(act().at(27893760).cast(), w().at(43024384).cast(), in_x().at(58880).cast(), act().at(31170560).cast());
        sync_cores();
        un_gs_16_1280(act().at(31170560).cast(), act().at(27852800).cast());
        sync_cores();
        un_ga_16_1280_silu(act().at(31170560).cast(), act().at(27852800).cast(), w().at(102006784).cast(), w().at(102014976).cast(), w().at(102023168).cast(), act().at(32808960).cast());
        sync_cores();
        un_c1_16_2560_1280(act().at(24576000).cast(), w().at(102187008).cast(), w().at(108740608).cast(), act().at(34447360).cast());
        sync_cores();
        un_c3_16_1280_1280_r(act().at(32808960).cast(), w().at(108748800).cast(), w().at(138240000).cast(), act().at(34447360).cast(), act().at(36085760).cast());
        sync_cores();
        // up1.attn1
        un_gs_16_1280(act().at(36085760).cast(), act().at(27852800).cast());
        sync_cores();
        un_ga_16_1280_none(act().at(36085760).cast(), act().at(27852800).cast(), w().at(138248192).cast(), w().at(138256384).cast(), w().at(102023168).cast(), act().at(37724160).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(37724160).cast(), w().at(138264576).cast(), w().at(141541376).cast(), act().at(39362560).cast());
        sync_cores();
        un_ln_16_1280(act().at(39362560).cast(), w().at(141549568).cast(), w().at(141557760).cast(), act().at(41000960).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(41000960).cast(), w().at(141565952).cast(), w().at(144842752).cast(), act().at(42639360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(41000960).cast(), w().at(144850944).cast(), w().at(144842752).cast(), act().at(44277760).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(41000960).cast(), w().at(148127744).cast(), w().at(144842752).cast(), act().at(45916160).cast());
        sync_cores();
        un_at_16_s(act().at(42639360).cast(), act().at(44277760).cast(), act().at(45916160).cast(), w().at(151404544).cast(), act().at(47554560).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(47554560).cast(), w().at(151408640).cast(), w().at(154685440).cast(), act().at(39362560).cast(), act().at(49192960).cast());
        sync_cores();
        // up1.attn1.cross
        un_ln_16_1280(act().at(49192960).cast(), w().at(154693632).cast(), w().at(154701824).cast(), act().at(50831360).cast());
        sync_cores();
        un_c1_16_1280_1280(act().at(50831360).cast(), w().at(154710016).cast(), w().at(144842752).cast(), act().at(52469760).cast());
        sync_cores();
        un_kv_1280(in_x().at(10108928).cast(), w().at(157986816).cast(), act().at(54108160).cast());
        sync_cores();
        un_kv_1280(in_x().at(10108928).cast(), w().at(159952896).cast(), act().at(54435840).cast());
        sync_cores();
        un_at_16_x(act().at(52469760).cast(), act().at(54108160).cast(), act().at(54435840).cast(), w().at(161918976).cast(), act().at(54763520).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(54763520).cast(), w().at(161923072).cast(), w().at(165199872).cast(), act().at(49192960).cast(), act().at(56401920).cast());
        sync_cores();
        // up1.attn1.ff
        un_ln_16_1280(act().at(56401920).cast(), w().at(165208064).cast(), w().at(165216256).cast(), act().at(58040320).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(58040320).cast(), w().at(165224448).cast(), w().at(178331648).cast(), act().at(59678720).cast());
        sync_cores();
        un_c1_16_1280_5120(act().at(58040320).cast(), w().at(178352128).cast(), w().at(191459328).cast(), act().at(66232320).cast());
        sync_cores();
        un_gg_16(act().at(59678720).cast(), act().at(66232320).cast(), act().at(72785920).cast());
        sync_cores();
        un_c1_16_5120_1280_r(act().at(72785920).cast(), w().at(191479808).cast(), w().at(204587008).cast(), act().at(56401920).cast(), act().at(79339520).cast());
        sync_cores();
        un_c1_16_1280_1280_r(act().at(79339520).cast(), w().at(204595200).cast(), w().at(207872000).cast(), act().at(36085760).cast(), out_y().at(1638400).cast());
        sync_cores();
    }
}

/// s10：up1.res2、up1.attn2、up1.attn2.cross、up1.attn2.ff、up1.up
#[cfg(feature = "s10")]
mod s10 {
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
        // up1.attn2.cross
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
        // up1.attn2.ff
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
        // up1.attn2.cross
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
        // up1.attn2.ff
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

/// s11：up2.res0、up2.attn0、up2.attn0.cross
#[cfg(feature = "s11")]
mod s11 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 12271616;
        out_y: output, bytes = 7864320;
        w: weights, bytes = 41431040;
        act: workspace, bytes = 42123264;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 1285);
            un_zero(out_y().cast(), 240);
        } else {
            un_zero(act().at(21053440).cast(), 1286);
            un_zero(out_y().at(3932160).cast(), 240);
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
        // up2.attn0
        un_gs_32_640(out_y().at(0).cast(), act().at(5898240).cast());
        sync_cores();
        un_ga_32_640_none(out_y().at(0).cast(), act().at(5898240).cast(), w().at(32309248).cast(), w().at(32313344).cast(), w().at(22388736).cast(), act().at(17735680).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(17735680).cast(), w().at(32317440).cast(), w().at(33136640).cast(), act().at(19701760).cast());
        sync_cores();
        un_ln_32_640(act().at(19701760).cast(), w().at(33140736).cast(), w().at(33144832).cast(), act().at(21667840).cast());
        sync_cores();
        un_lr_32_640_768(act().at(21667840).cast(), w().at(33148928).cast(), act().at(23633920).cast());
        sync_cores();
        un_lr_32_640_768(act().at(21667840).cast(), w().at(34131968).cast(), act().at(25993216).cast());
        sync_cores();
        un_lr_32_640_768(act().at(21667840).cast(), w().at(35115008).cast(), act().at(28352512).cast());
        sync_cores();
        un_at_32_s(act().at(23633920).cast(), act().at(25993216).cast(), act().at(28352512).cast(), w().at(36098048).cast(), act().at(30711808).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(30711808).cast(), w().at(36102144).cast(), w().at(37085184).cast(), act().at(19701760).cast(), act().at(33071104).cast());
        sync_cores();
        // up2.attn0.cross
        un_ln_32_640(act().at(33071104).cast(), w().at(37089280).cast(), w().at(37093376).cast(), act().at(35037184).cast());
        sync_cores();
        un_lr_32_640_768(act().at(35037184).cast(), w().at(37097472).cast(), act().at(37003264).cast());
        sync_cores();
        un_kv_768(in_x().at(5980160).cast(), w().at(38080512).cast(), act().at(39362560).cast());
        sync_cores();
        un_kv_768(in_x().at(5980160).cast(), w().at(39260160).cast(), act().at(39559168).cast());
        sync_cores();
        un_at_32_x(act().at(37003264).cast(), act().at(39362560).cast(), act().at(39559168).cast(), w().at(40439808).cast(), act().at(39755776).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(39755776).cast(), w().at(40443904).cast(), w().at(41426944).cast(), act().at(33071104).cast(), out_y().at(1966080).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up2.res0
        un_cp_32_1280_1920(in_x().at(6176768).cast(), act().at(0).cast());
        sync_cores();
        un_cp_32_640_1920(in_x().at(10108928).cast(), act().at(2560).cast());
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
        un_c3_32_640_640_r(act().at(13803520).cast(), w().at(24932352).cast(), w().at(32305152).cast(), act().at(15769600).cast(), out_y().at(3932160).cast());
        sync_cores();
        // up2.attn0
        un_gs_32_640(out_y().at(3932160).cast(), act().at(5898240).cast());
        sync_cores();
        un_ga_32_640_none(out_y().at(3932160).cast(), act().at(5898240).cast(), w().at(32309248).cast(), w().at(32313344).cast(), w().at(22388736).cast(), act().at(17735680).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(17735680).cast(), w().at(32317440).cast(), w().at(33136640).cast(), act().at(19701760).cast());
        sync_cores();
        un_ln_32_640(act().at(19701760).cast(), w().at(33140736).cast(), w().at(33144832).cast(), act().at(21667840).cast());
        sync_cores();
        un_lr_32_640_768(act().at(21667840).cast(), w().at(33148928).cast(), act().at(23633920).cast());
        sync_cores();
        un_lr_32_640_768(act().at(21667840).cast(), w().at(34131968).cast(), act().at(25993216).cast());
        sync_cores();
        un_lr_32_640_768(act().at(21667840).cast(), w().at(35115008).cast(), act().at(28352512).cast());
        sync_cores();
        un_at_32_s(act().at(23633920).cast(), act().at(25993216).cast(), act().at(28352512).cast(), w().at(36098048).cast(), act().at(30711808).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(30711808).cast(), w().at(36102144).cast(), w().at(37085184).cast(), act().at(19701760).cast(), act().at(33071104).cast());
        sync_cores();
        // up2.attn0.cross
        un_ln_32_640(act().at(33071104).cast(), w().at(37089280).cast(), w().at(37093376).cast(), act().at(35037184).cast());
        sync_cores();
        un_lr_32_640_768(act().at(35037184).cast(), w().at(37097472).cast(), act().at(37003264).cast());
        sync_cores();
        un_kv_768(in_x().at(12075008).cast(), w().at(38080512).cast(), act().at(39362560).cast());
        sync_cores();
        un_kv_768(in_x().at(12075008).cast(), w().at(39260160).cast(), act().at(39559168).cast());
        sync_cores();
        un_at_32_x(act().at(37003264).cast(), act().at(39362560).cast(), act().at(39559168).cast(), w().at(40439808).cast(), act().at(39755776).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(39755776).cast(), w().at(40443904).cast(), w().at(41426944).cast(), act().at(33071104).cast(), out_y().at(5898240).cast());
        sync_cores();
    }
}

/// s12：up2.attn0.ff、up2.res1、up2.attn1、up2.attn1.cross
#[cfg(feature = "s12")]
mod s12 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 12271616;
        out_y: output, bytes = 7864320;
        w: weights, bytes = 44011520;
        act: workspace, bytes = 68075520;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2077);
            un_zero(out_y().cast(), 240);
        } else {
            un_zero(act().at(34029568).cast(), 2078);
            un_zero(out_y().at(3932160).cast(), 240);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up2.attn0.ff
        un_ln_32_640(in_x().at(81920).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(0).cast(), w().at(8192).cast(), w().at(3284992).cast(), act().at(1966080).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(0).cast(), w().at(3297280).cast(), w().at(6574080).cast(), act().at(9830400).cast());
        sync_cores();
        un_gg_32(act().at(1966080).cast(), act().at(9830400).cast(), act().at(17694720).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(17694720).cast(), w().at(6586368).cast(), w().at(9863168).cast(), in_x().at(81920).cast(), act().at(25559040).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(25559040).cast(), w().at(9867264).cast(), w().at(10850304).cast(), in_x().at(2048000).cast(), act().at(27918336).cast());
        sync_cores();
        // up2.res1
        un_cp_32_640_1280(act().at(27918336).cast(), act().at(29884416).cast());
        sync_cores();
        un_cp_32_640_1280(in_x().at(4014080).cast(), act().at(29885696).cast());
        sync_cores();
        un_gs_32_1280(act().at(29884416).cast(), act().at(33816576).cast());
        sync_cores();
        un_ga_32_1280_silu(act().at(29884416).cast(), act().at(33816576).cast(), w().at(10854400).cast(), w().at(10862592).cast(), w().at(10870784).cast(), act().at(33857536).cast());
        sync_cores();
        un_c3_32_1280_640(act().at(33857536).cast(), w().at(11034624).cast(), in_x().at(71680).cast(), act().at(37789696).cast());
        sync_cores();
        un_gs_32_640(act().at(37789696).cast(), act().at(33816576).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(37789696).cast(), act().at(33816576).cast(), w().at(25780224).cast(), w().at(25784320).cast(), w().at(25788416).cast(), act().at(39755776).cast());
        sync_cores();
        un_c1_32_1280_640(act().at(29884416).cast(), w().at(25870336).cast(), w().at(27508736).cast(), act().at(41721856).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(39755776).cast(), w().at(27512832).cast(), w().at(34885632).cast(), act().at(41721856).cast(), out_y().at(0).cast());
        sync_cores();
        // up2.attn1
        un_gs_32_640(out_y().at(0).cast(), act().at(33816576).cast());
        sync_cores();
        un_ga_32_640_none(out_y().at(0).cast(), act().at(33816576).cast(), w().at(34889728).cast(), w().at(34893824).cast(), w().at(25788416).cast(), act().at(43687936).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(43687936).cast(), w().at(34897920).cast(), w().at(35717120).cast(), act().at(45654016).cast());
        sync_cores();
        un_ln_32_640(act().at(45654016).cast(), w().at(35721216).cast(), w().at(35725312).cast(), act().at(47620096).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(35729408).cast(), act().at(49586176).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(36712448).cast(), act().at(51945472).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(37695488).cast(), act().at(54304768).cast());
        sync_cores();
        un_at_32_s(act().at(49586176).cast(), act().at(51945472).cast(), act().at(54304768).cast(), w().at(38678528).cast(), act().at(56664064).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(56664064).cast(), w().at(38682624).cast(), w().at(39665664).cast(), act().at(45654016).cast(), act().at(59023360).cast());
        sync_cores();
        // up2.attn1.cross
        un_ln_32_640(act().at(59023360).cast(), w().at(39669760).cast(), w().at(39673856).cast(), act().at(60989440).cast());
        sync_cores();
        un_lr_32_640_768(act().at(60989440).cast(), w().at(39677952).cast(), act().at(62955520).cast());
        sync_cores();
        un_kv_768(in_x().at(5980160).cast(), w().at(40660992).cast(), act().at(65314816).cast());
        sync_cores();
        un_kv_768(in_x().at(5980160).cast(), w().at(41840640).cast(), act().at(65511424).cast());
        sync_cores();
        un_at_32_x(act().at(62955520).cast(), act().at(65314816).cast(), act().at(65511424).cast(), w().at(43020288).cast(), act().at(65708032).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(65708032).cast(), w().at(43024384).cast(), w().at(44007424).cast(), act().at(59023360).cast(), out_y().at(1966080).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up2.attn0.ff
        un_ln_32_640(in_x().at(6176768).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(0).cast(), w().at(8192).cast(), w().at(3284992).cast(), act().at(1966080).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(0).cast(), w().at(3297280).cast(), w().at(6574080).cast(), act().at(9830400).cast());
        sync_cores();
        un_gg_32(act().at(1966080).cast(), act().at(9830400).cast(), act().at(17694720).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(17694720).cast(), w().at(6586368).cast(), w().at(9863168).cast(), in_x().at(6176768).cast(), act().at(25559040).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(25559040).cast(), w().at(9867264).cast(), w().at(10850304).cast(), in_x().at(8142848).cast(), act().at(27918336).cast());
        sync_cores();
        // up2.res1
        un_cp_32_640_1280(act().at(27918336).cast(), act().at(29884416).cast());
        sync_cores();
        un_cp_32_640_1280(in_x().at(10108928).cast(), act().at(29885696).cast());
        sync_cores();
        un_gs_32_1280(act().at(29884416).cast(), act().at(33816576).cast());
        sync_cores();
        un_ga_32_1280_silu(act().at(29884416).cast(), act().at(33816576).cast(), w().at(10854400).cast(), w().at(10862592).cast(), w().at(10870784).cast(), act().at(33857536).cast());
        sync_cores();
        un_c3_32_1280_640(act().at(33857536).cast(), w().at(11034624).cast(), in_x().at(71680).cast(), act().at(37789696).cast());
        sync_cores();
        un_gs_32_640(act().at(37789696).cast(), act().at(33816576).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(37789696).cast(), act().at(33816576).cast(), w().at(25780224).cast(), w().at(25784320).cast(), w().at(25788416).cast(), act().at(39755776).cast());
        sync_cores();
        un_c1_32_1280_640(act().at(29884416).cast(), w().at(25870336).cast(), w().at(27508736).cast(), act().at(41721856).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(39755776).cast(), w().at(27512832).cast(), w().at(34885632).cast(), act().at(41721856).cast(), out_y().at(3932160).cast());
        sync_cores();
        // up2.attn1
        un_gs_32_640(out_y().at(3932160).cast(), act().at(33816576).cast());
        sync_cores();
        un_ga_32_640_none(out_y().at(3932160).cast(), act().at(33816576).cast(), w().at(34889728).cast(), w().at(34893824).cast(), w().at(25788416).cast(), act().at(43687936).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(43687936).cast(), w().at(34897920).cast(), w().at(35717120).cast(), act().at(45654016).cast());
        sync_cores();
        un_ln_32_640(act().at(45654016).cast(), w().at(35721216).cast(), w().at(35725312).cast(), act().at(47620096).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(35729408).cast(), act().at(49586176).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(36712448).cast(), act().at(51945472).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(37695488).cast(), act().at(54304768).cast());
        sync_cores();
        un_at_32_s(act().at(49586176).cast(), act().at(51945472).cast(), act().at(54304768).cast(), w().at(38678528).cast(), act().at(56664064).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(56664064).cast(), w().at(38682624).cast(), w().at(39665664).cast(), act().at(45654016).cast(), act().at(59023360).cast());
        sync_cores();
        // up2.attn1.cross
        un_ln_32_640(act().at(59023360).cast(), w().at(39669760).cast(), w().at(39673856).cast(), act().at(60989440).cast());
        sync_cores();
        un_lr_32_640_768(act().at(60989440).cast(), w().at(39677952).cast(), act().at(62955520).cast());
        sync_cores();
        un_kv_768(in_x().at(12075008).cast(), w().at(40660992).cast(), act().at(65314816).cast());
        sync_cores();
        un_kv_768(in_x().at(12075008).cast(), w().at(41840640).cast(), act().at(65511424).cast());
        sync_cores();
        un_at_32_x(act().at(62955520).cast(), act().at(65314816).cast(), act().at(65511424).cast(), w().at(43020288).cast(), act().at(65708032).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(65708032).cast(), w().at(43024384).cast(), w().at(44007424).cast(), act().at(59023360).cast(), out_y().at(5898240).cast());
        sync_cores();
    }
}

/// s13：up2.attn1.ff、up2.res2、up2.attn2、up2.attn2.cross、up2.attn2.ff、up2.up
#[cfg(feature = "s13")]
mod s13 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 10305536;
        out_y: output, bytes = 12451840;
        w: weights, bytes = 58097664;
        act: workspace, bytes = 106151936;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 3239);
            un_zero(out_y().cast(), 380);
        } else {
            un_zero(act().at(53067776).cast(), 3240);
            un_zero(out_y().at(6225920).cast(), 380);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up2.attn1.ff
        un_ln_32_640(in_x().at(81920).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(0).cast(), w().at(8192).cast(), w().at(3284992).cast(), act().at(1966080).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(0).cast(), w().at(3297280).cast(), w().at(6574080).cast(), act().at(9830400).cast());
        sync_cores();
        un_gg_32(act().at(1966080).cast(), act().at(9830400).cast(), act().at(17694720).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(17694720).cast(), w().at(6586368).cast(), w().at(9863168).cast(), in_x().at(81920).cast(), act().at(25559040).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(25559040).cast(), w().at(9867264).cast(), w().at(10850304).cast(), in_x().at(2048000).cast(), act().at(27918336).cast());
        sync_cores();
        // up2.res2
        un_cp_32_640_960(act().at(27918336).cast(), act().at(29884416).cast());
        sync_cores();
        un_cp_32_320_960(in_x().at(4014080).cast(), act().at(29885696).cast());
        sync_cores();
        un_gs_32_960(act().at(29884416).cast(), act().at(32833536).cast());
        sync_cores();
        un_ga_32_960_silu(act().at(29884416).cast(), act().at(32833536).cast(), w().at(10854400).cast(), w().at(10858496).cast(), w().at(10862592).cast(), act().at(32874496).cast());
        sync_cores();
        un_c3_32_960_640(act().at(32874496).cast(), w().at(10985472).cast(), in_x().at(74240).cast(), act().at(35823616).cast());
        sync_cores();
        un_gs_32_640(act().at(35823616).cast(), act().at(32833536).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(35823616).cast(), act().at(32833536).cast(), w().at(22044672).cast(), w().at(22048768).cast(), w().at(22052864).cast(), act().at(37789696).cast());
        sync_cores();
        un_c1_32_960_640(act().at(29884416).cast(), w().at(22134784).cast(), w().at(23363584).cast(), act().at(39755776).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(37789696).cast(), w().at(23367680).cast(), w().at(30740480).cast(), act().at(39755776).cast(), act().at(41721856).cast());
        sync_cores();
        // up2.attn2
        un_gs_32_640(act().at(41721856).cast(), act().at(32833536).cast());
        sync_cores();
        un_ga_32_640_none(act().at(41721856).cast(), act().at(32833536).cast(), w().at(30744576).cast(), w().at(30748672).cast(), w().at(22052864).cast(), act().at(43687936).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(43687936).cast(), w().at(30752768).cast(), w().at(31571968).cast(), act().at(45654016).cast());
        sync_cores();
        un_ln_32_640(act().at(45654016).cast(), w().at(31576064).cast(), w().at(31580160).cast(), act().at(47620096).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(31584256).cast(), act().at(49586176).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(32567296).cast(), act().at(51945472).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(33550336).cast(), act().at(54304768).cast());
        sync_cores();
        un_at_32_s(act().at(49586176).cast(), act().at(51945472).cast(), act().at(54304768).cast(), w().at(34533376).cast(), act().at(56664064).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(56664064).cast(), w().at(34537472).cast(), w().at(35520512).cast(), act().at(45654016).cast(), act().at(59023360).cast());
        sync_cores();
        // up2.attn2.cross
        un_ln_32_640(act().at(59023360).cast(), w().at(35524608).cast(), w().at(35528704).cast(), act().at(60989440).cast());
        sync_cores();
        un_lr_32_640_768(act().at(60989440).cast(), w().at(35532800).cast(), act().at(62955520).cast());
        sync_cores();
        un_kv_768(in_x().at(4997120).cast(), w().at(36515840).cast(), act().at(65314816).cast());
        sync_cores();
        un_kv_768(in_x().at(4997120).cast(), w().at(37695488).cast(), act().at(65511424).cast());
        sync_cores();
        un_at_32_x(act().at(62955520).cast(), act().at(65314816).cast(), act().at(65511424).cast(), w().at(38875136).cast(), act().at(65708032).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(65708032).cast(), w().at(38879232).cast(), w().at(39862272).cast(), act().at(59023360).cast(), act().at(68067328).cast());
        sync_cores();
        // up2.attn2.ff
        un_ln_32_640(act().at(68067328).cast(), w().at(39866368).cast(), w().at(39870464).cast(), act().at(70033408).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(70033408).cast(), w().at(39874560).cast(), w().at(43151360).cast(), act().at(71999488).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(70033408).cast(), w().at(43163648).cast(), w().at(46440448).cast(), act().at(79863808).cast());
        sync_cores();
        un_gg_32(act().at(71999488).cast(), act().at(79863808).cast(), act().at(87728128).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(87728128).cast(), w().at(46452736).cast(), w().at(49729536).cast(), act().at(68067328).cast(), act().at(95592448).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(95592448).cast(), w().at(49733632).cast(), w().at(50716672).cast(), act().at(41721856).cast(), act().at(97951744).cast());
        sync_cores();
        // up2.up
        un_up_32_640_256(act().at(97951744).cast(), act().at(99917824).cast());
        sync_cores();
        un_up_32_640_256(act().at(97952256).cast(), act().at(99918336).cast());
        sync_cores();
        un_up_32_640_128(act().at(97952768).cast(), act().at(99918848).cast());
        sync_cores();
        un_c3_64_640_640(act().at(99917824).cast(), w().at(50720768).cast(), w().at(58093568).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up2.attn1.ff
        un_ln_32_640(in_x().at(5193728).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(0).cast(), w().at(8192).cast(), w().at(3284992).cast(), act().at(1966080).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(0).cast(), w().at(3297280).cast(), w().at(6574080).cast(), act().at(9830400).cast());
        sync_cores();
        un_gg_32(act().at(1966080).cast(), act().at(9830400).cast(), act().at(17694720).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(17694720).cast(), w().at(6586368).cast(), w().at(9863168).cast(), in_x().at(5193728).cast(), act().at(25559040).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(25559040).cast(), w().at(9867264).cast(), w().at(10850304).cast(), in_x().at(7159808).cast(), act().at(27918336).cast());
        sync_cores();
        // up2.res2
        un_cp_32_640_960(act().at(27918336).cast(), act().at(29884416).cast());
        sync_cores();
        un_cp_32_320_960(in_x().at(9125888).cast(), act().at(29885696).cast());
        sync_cores();
        un_gs_32_960(act().at(29884416).cast(), act().at(32833536).cast());
        sync_cores();
        un_ga_32_960_silu(act().at(29884416).cast(), act().at(32833536).cast(), w().at(10854400).cast(), w().at(10858496).cast(), w().at(10862592).cast(), act().at(32874496).cast());
        sync_cores();
        un_c3_32_960_640(act().at(32874496).cast(), w().at(10985472).cast(), in_x().at(74240).cast(), act().at(35823616).cast());
        sync_cores();
        un_gs_32_640(act().at(35823616).cast(), act().at(32833536).cast());
        sync_cores();
        un_ga_32_640_silu(act().at(35823616).cast(), act().at(32833536).cast(), w().at(22044672).cast(), w().at(22048768).cast(), w().at(22052864).cast(), act().at(37789696).cast());
        sync_cores();
        un_c1_32_960_640(act().at(29884416).cast(), w().at(22134784).cast(), w().at(23363584).cast(), act().at(39755776).cast());
        sync_cores();
        un_c3_32_640_640_r(act().at(37789696).cast(), w().at(23367680).cast(), w().at(30740480).cast(), act().at(39755776).cast(), act().at(41721856).cast());
        sync_cores();
        // up2.attn2
        un_gs_32_640(act().at(41721856).cast(), act().at(32833536).cast());
        sync_cores();
        un_ga_32_640_none(act().at(41721856).cast(), act().at(32833536).cast(), w().at(30744576).cast(), w().at(30748672).cast(), w().at(22052864).cast(), act().at(43687936).cast());
        sync_cores();
        un_lr_32_640_640_b(act().at(43687936).cast(), w().at(30752768).cast(), w().at(31571968).cast(), act().at(45654016).cast());
        sync_cores();
        un_ln_32_640(act().at(45654016).cast(), w().at(31576064).cast(), w().at(31580160).cast(), act().at(47620096).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(31584256).cast(), act().at(49586176).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(32567296).cast(), act().at(51945472).cast());
        sync_cores();
        un_lr_32_640_768(act().at(47620096).cast(), w().at(33550336).cast(), act().at(54304768).cast());
        sync_cores();
        un_at_32_s(act().at(49586176).cast(), act().at(51945472).cast(), act().at(54304768).cast(), w().at(34533376).cast(), act().at(56664064).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(56664064).cast(), w().at(34537472).cast(), w().at(35520512).cast(), act().at(45654016).cast(), act().at(59023360).cast());
        sync_cores();
        // up2.attn2.cross
        un_ln_32_640(act().at(59023360).cast(), w().at(35524608).cast(), w().at(35528704).cast(), act().at(60989440).cast());
        sync_cores();
        un_lr_32_640_768(act().at(60989440).cast(), w().at(35532800).cast(), act().at(62955520).cast());
        sync_cores();
        un_kv_768(in_x().at(10108928).cast(), w().at(36515840).cast(), act().at(65314816).cast());
        sync_cores();
        un_kv_768(in_x().at(10108928).cast(), w().at(37695488).cast(), act().at(65511424).cast());
        sync_cores();
        un_at_32_x(act().at(62955520).cast(), act().at(65314816).cast(), act().at(65511424).cast(), w().at(38875136).cast(), act().at(65708032).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(65708032).cast(), w().at(38879232).cast(), w().at(39862272).cast(), act().at(59023360).cast(), act().at(68067328).cast());
        sync_cores();
        // up2.attn2.ff
        un_ln_32_640(act().at(68067328).cast(), w().at(39866368).cast(), w().at(39870464).cast(), act().at(70033408).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(70033408).cast(), w().at(39874560).cast(), w().at(43151360).cast(), act().at(71999488).cast());
        sync_cores();
        un_lr_32_640_2560_b(act().at(70033408).cast(), w().at(43163648).cast(), w().at(46440448).cast(), act().at(79863808).cast());
        sync_cores();
        un_gg_32(act().at(71999488).cast(), act().at(79863808).cast(), act().at(87728128).cast());
        sync_cores();
        un_c1_32_2560_640_r_o768_q640(act().at(87728128).cast(), w().at(46452736).cast(), w().at(49729536).cast(), act().at(68067328).cast(), act().at(95592448).cast());
        sync_cores();
        un_c1_32_768_640_r(act().at(95592448).cast(), w().at(49733632).cast(), w().at(50716672).cast(), act().at(41721856).cast(), act().at(97951744).cast());
        sync_cores();
        // up2.up
        un_up_32_640_256(act().at(97951744).cast(), act().at(99917824).cast());
        sync_cores();
        un_up_32_640_256(act().at(97952256).cast(), act().at(99918336).cast());
        sync_cores();
        un_up_32_640_128(act().at(97952768).cast(), act().at(99918848).cast());
        sync_cores();
        un_c3_64_640_640(act().at(99917824).cast(), w().at(50720768).cast(), w().at(58093568).cast(), out_y().at(6225920).cast());
        sync_cores();
    }
}

/// s14：up3.res0、up3.attn0、up3.attn0.cross
#[cfg(feature = "s14")]
mod s14 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 19152896;
        out_y: output, bytes = 12451840;
        w: weights, bytes = 11976704;
        act: workspace, bytes = 73777152;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2251);
            un_zero(out_y().cast(), 380);
        } else {
            un_zero(act().at(36880384).cast(), 2252);
            un_zero(out_y().at(6225920).cast(), 380);
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
        // up3.attn0
        un_gs_64_320(out_y().at(0).cast(), act().at(9338880).cast());
        sync_cores();
        un_ga_64_320_none(out_y().at(0).cast(), act().at(9338880).cast(), w().at(8175616).cast(), w().at(8179712).cast(), w().at(5668864).cast(), act().at(28057600).cast());
        sync_cores();
        un_c1_64_320_320(act().at(28057600).cast(), w().at(8183808).cast(), w().at(8388608).cast(), act().at(31170560).cast());
        sync_cores();
        un_ln_64_320(act().at(31170560).cast(), w().at(8392704).cast(), w().at(8396800).cast(), act().at(34283520).cast());
        sync_cores();
        un_c1_64_320_512(act().at(34283520).cast(), w().at(8400896).cast(), w().at(8728576).cast(), act().at(37396480).cast());
        sync_cores();
        un_c1_64_320_512(act().at(34283520).cast(), w().at(8732672).cast(), w().at(8728576).cast(), act().at(42377216).cast());
        sync_cores();
        un_c1_64_320_512(act().at(34283520).cast(), w().at(9060352).cast(), w().at(8728576).cast(), act().at(47357952).cast());
        sync_cores();
        un_at_64_s(act().at(37396480).cast(), act().at(42377216).cast(), act().at(47357952).cast(), w().at(9388032).cast(), act().at(52338688).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(52338688).cast(), w().at(9400320).cast(), w().at(9728000).cast(), act().at(31170560).cast(), act().at(57319424).cast());
        sync_cores();
        // up3.attn0.cross
        un_ln_64_320(act().at(57319424).cast(), w().at(9732096).cast(), w().at(9736192).cast(), act().at(60432384).cast());
        sync_cores();
        un_c1_64_320_512(act().at(60432384).cast(), w().at(9740288).cast(), w().at(8728576).cast(), act().at(63545344).cast());
        sync_cores();
        un_kv_512(in_x().at(9420800).cast(), w().at(10067968).cast(), act().at(68526080).cast());
        sync_cores();
        un_kv_512(in_x().at(9420800).cast(), w().at(10854400).cast(), act().at(68657152).cast());
        sync_cores();
        un_at_64_x(act().at(63545344).cast(), act().at(68526080).cast(), act().at(68657152).cast(), w().at(11640832).cast(), act().at(68788224).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(68788224).cast(), w().at(11644928).cast(), w().at(11972608).cast(), act().at(57319424).cast(), out_y().at(3112960).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.res0
        un_cp_64_640_960(in_x().at(9617408).cast(), act().at(0).cast());
        sync_cores();
        un_cp_64_320_960(in_x().at(15843328).cast(), act().at(1280).cast());
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
        un_c3_64_320_320_r(act().at(21831680).cast(), w().at(6328320).cast(), w().at(8171520).cast(), act().at(24944640).cast(), out_y().at(6225920).cast());
        sync_cores();
        // up3.attn0
        un_gs_64_320(out_y().at(6225920).cast(), act().at(9338880).cast());
        sync_cores();
        un_ga_64_320_none(out_y().at(6225920).cast(), act().at(9338880).cast(), w().at(8175616).cast(), w().at(8179712).cast(), w().at(5668864).cast(), act().at(28057600).cast());
        sync_cores();
        un_c1_64_320_320(act().at(28057600).cast(), w().at(8183808).cast(), w().at(8388608).cast(), act().at(31170560).cast());
        sync_cores();
        un_ln_64_320(act().at(31170560).cast(), w().at(8392704).cast(), w().at(8396800).cast(), act().at(34283520).cast());
        sync_cores();
        un_c1_64_320_512(act().at(34283520).cast(), w().at(8400896).cast(), w().at(8728576).cast(), act().at(37396480).cast());
        sync_cores();
        un_c1_64_320_512(act().at(34283520).cast(), w().at(8732672).cast(), w().at(8728576).cast(), act().at(42377216).cast());
        sync_cores();
        un_c1_64_320_512(act().at(34283520).cast(), w().at(9060352).cast(), w().at(8728576).cast(), act().at(47357952).cast());
        sync_cores();
        un_at_64_s(act().at(37396480).cast(), act().at(42377216).cast(), act().at(47357952).cast(), w().at(9388032).cast(), act().at(52338688).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(52338688).cast(), w().at(9400320).cast(), w().at(9728000).cast(), act().at(31170560).cast(), act().at(57319424).cast());
        sync_cores();
        // up3.attn0.cross
        un_ln_64_320(act().at(57319424).cast(), w().at(9732096).cast(), w().at(9736192).cast(), act().at(60432384).cast());
        sync_cores();
        un_c1_64_320_512(act().at(60432384).cast(), w().at(9740288).cast(), w().at(8728576).cast(), act().at(63545344).cast());
        sync_cores();
        un_kv_512(in_x().at(18956288).cast(), w().at(10067968).cast(), act().at(68526080).cast());
        sync_cores();
        un_kv_512(in_x().at(18956288).cast(), w().at(10854400).cast(), act().at(68657152).cast());
        sync_cores();
        un_at_64_x(act().at(63545344).cast(), act().at(68526080).cast(), act().at(68657152).cast(), w().at(11640832).cast(), act().at(68788224).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(68788224).cast(), w().at(11644928).cast(), w().at(11972608).cast(), act().at(57319424).cast(), out_y().at(9338880).cast());
        sync_cores();
    }
}

/// s15：up3.attn0.ff、up3.res1
#[cfg(feature = "s15")]
mod s15 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 18759680;
        out_y: output, bytes = 6225920;
        w: weights, bytes = 8781824;
        act: workspace, bytes = 68534272;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2091);
            un_zero(out_y().cast(), 190);
        } else {
            un_zero(act().at(34258944).cast(), 2092);
            un_zero(out_y().at(3112960).cast(), 190);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up3.attn0.ff
        un_ln_64_320(in_x().at(81920).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(8192).cast(), w().at(827392).cast(), act().at(3112960).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(835584).cast(), w().at(1654784).cast(), act().at(15564800).cast());
        sync_cores();
        un_gg_64(act().at(3112960).cast(), act().at(15564800).cast(), act().at(28016640).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(28016640).cast(), w().at(1662976).cast(), w().at(2482176).cast(), in_x().at(81920).cast(), act().at(40468480).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(40468480).cast(), w().at(2486272).cast(), w().at(2691072).cast(), in_x().at(3194880).cast(), act().at(43581440).cast());
        sync_cores();
        // up3.res1
        un_cp_64_320_640(act().at(43581440).cast(), act().at(46694400).cast());
        sync_cores();
        un_cp_64_320_640(in_x().at(6307840).cast(), act().at(46695040).cast());
        sync_cores();
        un_gs_64_640(act().at(46694400).cast(), act().at(52920320).cast());
        sync_cores();
        un_ga_64_640_silu(act().at(46694400).cast(), act().at(52920320).cast(), w().at(2695168).cast(), w().at(2699264).cast(), w().at(2703360).cast(), act().at(52961280).cast());
        sync_cores();
        un_c3_64_640_320(act().at(52961280).cast(), w().at(2785280).cast(), in_x().at(78080).cast(), act().at(59187200).cast());
        sync_cores();
        un_gs_64_320(act().at(59187200).cast(), act().at(52920320).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(59187200).cast(), act().at(52920320).cast(), w().at(6471680).cast(), w().at(6475776).cast(), w().at(6479872).cast(), act().at(62300160).cast());
        sync_cores();
        un_c1_64_640_320(act().at(46694400).cast(), w().at(6520832).cast(), w().at(6930432).cast(), act().at(65413120).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(62300160).cast(), w().at(6934528).cast(), w().at(8777728).cast(), act().at(65413120).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.attn0.ff
        un_ln_64_320(in_x().at(9420800).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(8192).cast(), w().at(827392).cast(), act().at(3112960).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(835584).cast(), w().at(1654784).cast(), act().at(15564800).cast());
        sync_cores();
        un_gg_64(act().at(3112960).cast(), act().at(15564800).cast(), act().at(28016640).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(28016640).cast(), w().at(1662976).cast(), w().at(2482176).cast(), in_x().at(9420800).cast(), act().at(40468480).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(40468480).cast(), w().at(2486272).cast(), w().at(2691072).cast(), in_x().at(12533760).cast(), act().at(43581440).cast());
        sync_cores();
        // up3.res1
        un_cp_64_320_640(act().at(43581440).cast(), act().at(46694400).cast());
        sync_cores();
        un_cp_64_320_640(in_x().at(15646720).cast(), act().at(46695040).cast());
        sync_cores();
        un_gs_64_640(act().at(46694400).cast(), act().at(52920320).cast());
        sync_cores();
        un_ga_64_640_silu(act().at(46694400).cast(), act().at(52920320).cast(), w().at(2695168).cast(), w().at(2699264).cast(), w().at(2703360).cast(), act().at(52961280).cast());
        sync_cores();
        un_c3_64_640_320(act().at(52961280).cast(), w().at(2785280).cast(), in_x().at(78080).cast(), act().at(59187200).cast());
        sync_cores();
        un_gs_64_320(act().at(59187200).cast(), act().at(52920320).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(59187200).cast(), act().at(52920320).cast(), w().at(6471680).cast(), w().at(6475776).cast(), w().at(6479872).cast(), act().at(62300160).cast());
        sync_cores();
        un_c1_64_640_320(act().at(46694400).cast(), w().at(6520832).cast(), w().at(6930432).cast(), act().at(65413120).cast());
        sync_cores();
        un_c3_64_320_320_r(act().at(62300160).cast(), w().at(6934528).cast(), w().at(8777728).cast(), act().at(65413120).cast(), out_y().at(3112960).cast());
        sync_cores();
    }
}

/// s16：up3.attn1、up3.attn1.cross、up3.attn1.ff
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
        // up3.attn1.cross
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
        // up3.attn1.ff
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
        // up3.attn1.cross
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
        // up3.attn1.ff
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

/// s17：up3.res2、up3.attn2、up3.attn2.cross
#[cfg(feature = "s17")]
mod s17 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 12926976;
        out_y: output, bytes = 12451840;
        w: weights, bytes = 9887744;
        act: workspace, bytes = 67551232;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 2061);
            un_zero(out_y().cast(), 380);
        } else {
            un_zero(act().at(33767424).cast(), 2062);
            un_zero(out_y().at(6225920).cast(), 380);
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
        // up3.attn2
        un_gs_64_320(out_y().at(0).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_320_none(out_y().at(0).cast(), act().at(6225920).cast(), w().at(6086656).cast(), w().at(6090752).cast(), w().at(3784704).cast(), act().at(21831680).cast());
        sync_cores();
        un_c1_64_320_320(act().at(21831680).cast(), w().at(6094848).cast(), w().at(6299648).cast(), act().at(24944640).cast());
        sync_cores();
        un_ln_64_320(act().at(24944640).cast(), w().at(6303744).cast(), w().at(6307840).cast(), act().at(28057600).cast());
        sync_cores();
        un_c1_64_320_512(act().at(28057600).cast(), w().at(6311936).cast(), w().at(6639616).cast(), act().at(31170560).cast());
        sync_cores();
        un_c1_64_320_512(act().at(28057600).cast(), w().at(6643712).cast(), w().at(6639616).cast(), act().at(36151296).cast());
        sync_cores();
        un_c1_64_320_512(act().at(28057600).cast(), w().at(6971392).cast(), w().at(6639616).cast(), act().at(41132032).cast());
        sync_cores();
        un_at_64_s(act().at(31170560).cast(), act().at(36151296).cast(), act().at(41132032).cast(), w().at(7299072).cast(), act().at(46112768).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(46112768).cast(), w().at(7311360).cast(), w().at(7639040).cast(), act().at(24944640).cast(), act().at(51093504).cast());
        sync_cores();
        // up3.attn2.cross
        un_ln_64_320(act().at(51093504).cast(), w().at(7643136).cast(), w().at(7647232).cast(), act().at(54206464).cast());
        sync_cores();
        un_c1_64_320_512(act().at(54206464).cast(), w().at(7651328).cast(), w().at(6639616).cast(), act().at(57319424).cast());
        sync_cores();
        un_kv_512(in_x().at(6307840).cast(), w().at(7979008).cast(), act().at(62300160).cast());
        sync_cores();
        un_kv_512(in_x().at(6307840).cast(), w().at(8765440).cast(), act().at(62431232).cast());
        sync_cores();
        un_at_64_x(act().at(57319424).cast(), act().at(62300160).cast(), act().at(62431232).cast(), w().at(9551872).cast(), act().at(62562304).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(62562304).cast(), w().at(9555968).cast(), w().at(9883648).cast(), act().at(51093504).cast(), out_y().at(3112960).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.res2
        un_cp_64_320_640(in_x().at(6504448).cast(), act().at(0).cast());
        sync_cores();
        un_cp_64_320_640(in_x().at(9617408).cast(), act().at(640).cast());
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
        un_c3_64_320_320_r(act().at(15605760).cast(), w().at(4239360).cast(), w().at(6082560).cast(), act().at(18718720).cast(), out_y().at(6225920).cast());
        sync_cores();
        // up3.attn2
        un_gs_64_320(out_y().at(6225920).cast(), act().at(6225920).cast());
        sync_cores();
        un_ga_64_320_none(out_y().at(6225920).cast(), act().at(6225920).cast(), w().at(6086656).cast(), w().at(6090752).cast(), w().at(3784704).cast(), act().at(21831680).cast());
        sync_cores();
        un_c1_64_320_320(act().at(21831680).cast(), w().at(6094848).cast(), w().at(6299648).cast(), act().at(24944640).cast());
        sync_cores();
        un_ln_64_320(act().at(24944640).cast(), w().at(6303744).cast(), w().at(6307840).cast(), act().at(28057600).cast());
        sync_cores();
        un_c1_64_320_512(act().at(28057600).cast(), w().at(6311936).cast(), w().at(6639616).cast(), act().at(31170560).cast());
        sync_cores();
        un_c1_64_320_512(act().at(28057600).cast(), w().at(6643712).cast(), w().at(6639616).cast(), act().at(36151296).cast());
        sync_cores();
        un_c1_64_320_512(act().at(28057600).cast(), w().at(6971392).cast(), w().at(6639616).cast(), act().at(41132032).cast());
        sync_cores();
        un_at_64_s(act().at(31170560).cast(), act().at(36151296).cast(), act().at(41132032).cast(), w().at(7299072).cast(), act().at(46112768).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(46112768).cast(), w().at(7311360).cast(), w().at(7639040).cast(), act().at(24944640).cast(), act().at(51093504).cast());
        sync_cores();
        // up3.attn2.cross
        un_ln_64_320(act().at(51093504).cast(), w().at(7643136).cast(), w().at(7647232).cast(), act().at(54206464).cast());
        sync_cores();
        un_c1_64_320_512(act().at(54206464).cast(), w().at(7651328).cast(), w().at(6639616).cast(), act().at(57319424).cast());
        sync_cores();
        un_kv_512(in_x().at(12730368).cast(), w().at(7979008).cast(), act().at(62300160).cast());
        sync_cores();
        un_kv_512(in_x().at(12730368).cast(), w().at(8765440).cast(), act().at(62431232).cast());
        sync_cores();
        un_at_64_x(act().at(57319424).cast(), act().at(62300160).cast(), act().at(62431232).cast(), w().at(9551872).cast(), act().at(62562304).cast());
        sync_cores();
        un_c1_64_512_320_r(act().at(62562304).cast(), w().at(9555968).cast(), w().at(9883648).cast(), act().at(51093504).cast(), out_y().at(9338880).cast());
        sync_cores();
    }
}

/// s18：up3.attn2.ff、out
#[cfg(feature = "s18")]
mod s18 {
    use super::*;
    npu::decls! {
        in_x: input, bytes = 12451840;
        out_y: output, bytes = 622592;
        w: weights, bytes = 2932736;
        act: workspace, bytes = 49856512;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0
        if core_rank() == 0 {
            un_zero(act().cast(), 1521);
            un_zero(out_y().cast(), 19);
        } else {
            un_zero(act().at(24920064).cast(), 1522);
            un_zero(out_y().at(311296).cast(), 19);
        }
        sync_cores();
        // ── 第 0 路（有提示词）──
        // up3.attn2.ff
        un_ln_64_320(in_x().at(0).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(8192).cast(), w().at(827392).cast(), act().at(3112960).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(835584).cast(), w().at(1654784).cast(), act().at(15564800).cast());
        sync_cores();
        un_gg_64(act().at(3112960).cast(), act().at(15564800).cast(), act().at(28016640).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(28016640).cast(), w().at(1662976).cast(), w().at(2482176).cast(), in_x().at(0).cast(), act().at(40468480).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(40468480).cast(), w().at(2486272).cast(), w().at(2691072).cast(), in_x().at(3112960).cast(), act().at(43581440).cast());
        sync_cores();
        // out
        un_gs_64_320(act().at(43581440).cast(), act().at(46694400).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(43581440).cast(), act().at(46694400).cast(), w().at(2695168).cast(), w().at(2699264).cast(), w().at(2703360).cast(), act().at(46735360).cast());
        sync_cores();
        un_c3_64_320_32(act().at(46735360).cast(), w().at(2744320).cast(), w().at(2928640).cast(), out_y().at(0).cast());
        sync_cores();
        // ── 第 1 路（空提示词）──
        // up3.attn2.ff
        un_ln_64_320(in_x().at(6225920).cast(), w().at(0).cast(), w().at(4096).cast(), act().at(0).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(8192).cast(), w().at(827392).cast(), act().at(3112960).cast());
        sync_cores();
        un_c1_64_320_1280(act().at(0).cast(), w().at(835584).cast(), w().at(1654784).cast(), act().at(15564800).cast());
        sync_cores();
        un_gg_64(act().at(3112960).cast(), act().at(15564800).cast(), act().at(28016640).cast());
        sync_cores();
        un_c1_64_1280_320_r(act().at(28016640).cast(), w().at(1662976).cast(), w().at(2482176).cast(), in_x().at(6225920).cast(), act().at(40468480).cast());
        sync_cores();
        un_c1_64_320_320_r(act().at(40468480).cast(), w().at(2486272).cast(), w().at(2691072).cast(), in_x().at(9338880).cast(), act().at(43581440).cast());
        sync_cores();
        // out
        un_gs_64_320(act().at(43581440).cast(), act().at(46694400).cast());
        sync_cores();
        un_ga_64_320_silu(act().at(43581440).cast(), act().at(46694400).cast(), w().at(2695168).cast(), w().at(2699264).cast(), w().at(2703360).cast(), act().at(46735360).cast());
        sync_cores();
        un_c3_64_320_32(act().at(46735360).cast(), w().at(2744320).cast(), w().at(2928640).cast(), out_y().at(311296).cast());
        sync_cores();
    }
}
