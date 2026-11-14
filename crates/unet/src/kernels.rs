//! SD1.5 UNet 的 kernel 特化（**`tools/unet_gen.py kernels` 生成，不要手改**）。形状是编译期常量：每个形状一份，名字带 `un_` 前缀。
//! 模拟器（`host/sd15npu/cvxsim.py`）按这里的宏参数认 kernel 的语义（字面量）；calbin 里只有这一段调到的 kernel 进 ELF。
//! 卷积全是 `act = none`（SiLU 在 GroupNorm 那一步做）；投影（q / k / v、to_out、前馈）当 1×1 卷积；注意力头维补零（40 ⇒ 64、80 ⇒ 96），
//! `scale` 给真头维；交叉注意力的 key 是上下文的 128 行（77 行有用，补的用掩码去掉）。

// ── conv2d（51 个）──
sd_kernels::conv2d!(pub un_c3_64_32_320: h = 64, w = 64, cin = 32, cout = 320, k = 3, s = 1, pin = (32, 0), pout = (320, 0), act = none);
sd_kernels::conv2d!(pub un_c3_64_320_320: h = 64, w = 64, cin = 320, cout = 320, k = 3, s = 1, pin = (320, 0), pout = (320, 0), act = none);
sd_kernels::conv2d!(pub un_c3_64_320_320_r: h = 64, w = 64, cin = 320, cout = 320, k = 3, s = 1, pin = (320, 0), pout = (320, 0), act = none, res = (320, 0));
sd_kernels::conv2d!(pub un_c1_64_320_320: h = 64, w = 64, cin = 320, cout = 320, k = 1, s = 1, pin = (320, 0), pout = (320, 0), act = none);
sd_kernels::conv2d!(pub un_c1_64_320_512: h = 64, w = 64, cin = 320, cout = 512, k = 1, s = 1, pin = (320, 0), pout = (512, 0), act = none);
sd_kernels::conv2d!(pub un_c1_64_512_320_r: h = 64, w = 64, cin = 512, cout = 320, k = 1, s = 1, pin = (512, 0), pout = (320, 0), act = none, res = (320, 0));
sd_kernels::conv2d!(pub un_c1_64_320_1280: h = 64, w = 64, cin = 320, cout = 1280, k = 1, s = 1, pin = (320, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c1_64_1280_320_r: h = 64, w = 64, cin = 1280, cout = 320, k = 1, s = 1, pin = (1280, 0), pout = (320, 0), act = none, res = (320, 0));
sd_kernels::conv2d!(pub un_c1_64_320_320_r: h = 64, w = 64, cin = 320, cout = 320, k = 1, s = 1, pin = (320, 0), pout = (320, 0), act = none, res = (320, 0));
sd_kernels::conv2d!(pub un_c3_32_320_320_s2: h = 32, w = 32, cin = 320, cout = 320, k = 3, s = 2, pin = (320, 0), pout = (320, 0), act = none);
sd_kernels::conv2d!(pub un_c3_32_320_640: h = 32, w = 32, cin = 320, cout = 640, k = 3, s = 1, pin = (320, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c1_32_320_640: h = 32, w = 32, cin = 320, cout = 640, k = 1, s = 1, pin = (320, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c3_32_640_640_r: h = 32, w = 32, cin = 640, cout = 640, k = 3, s = 1, pin = (640, 0), pout = (640, 0), act = none, res = (640, 0));
sd_kernels::conv2d!(pub un_c1_32_768_640_r: h = 32, w = 32, cin = 768, cout = 640, k = 1, s = 1, pin = (768, 0), pout = (640, 0), act = none, res = (640, 0));
sd_kernels::conv2d!(pub un_c1_32_2560_640_r_o768_q640: h = 32, w = 32, cin = 2560, cout = 640, k = 1, s = 1, pin = (2560, 0), pout = (768, 0), act = none, res = (640, 0));
sd_kernels::conv2d!(pub un_c3_32_640_640: h = 32, w = 32, cin = 640, cout = 640, k = 3, s = 1, pin = (640, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c3_16_640_640_s2: h = 16, w = 16, cin = 640, cout = 640, k = 3, s = 2, pin = (640, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c3_16_640_1280: h = 16, w = 16, cin = 640, cout = 1280, k = 3, s = 1, pin = (640, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c1_16_640_1280: h = 16, w = 16, cin = 640, cout = 1280, k = 1, s = 1, pin = (640, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c3_16_1280_1280_r: h = 16, w = 16, cin = 1280, cout = 1280, k = 3, s = 1, pin = (1280, 0), pout = (1280, 0), act = none, res = (1280, 0));
sd_kernels::conv2d!(pub un_c1_16_1280_1280: h = 16, w = 16, cin = 1280, cout = 1280, k = 1, s = 1, pin = (1280, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c1_16_1280_1280_r: h = 16, w = 16, cin = 1280, cout = 1280, k = 1, s = 1, pin = (1280, 0), pout = (1280, 0), act = none, res = (1280, 0));
sd_kernels::conv2d!(pub un_c1_16_1280_5120: h = 16, w = 16, cin = 1280, cout = 5120, k = 1, s = 1, pin = (1280, 0), pout = (5120, 0), act = none);
sd_kernels::conv2d!(pub un_c1_16_5120_1280_r: h = 16, w = 16, cin = 5120, cout = 1280, k = 1, s = 1, pin = (5120, 0), pout = (1280, 0), act = none, res = (1280, 0));
sd_kernels::conv2d!(pub un_c3_16_1280_1280: h = 16, w = 16, cin = 1280, cout = 1280, k = 3, s = 1, pin = (1280, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c3_8_1280_1280_s2: h = 8, w = 8, cin = 1280, cout = 1280, k = 3, s = 2, pin = (1280, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c3_8_1280_1280: h = 8, w = 8, cin = 1280, cout = 1280, k = 3, s = 1, pin = (1280, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c3_8_1280_1280_r: h = 8, w = 8, cin = 1280, cout = 1280, k = 3, s = 1, pin = (1280, 0), pout = (1280, 0), act = none, res = (1280, 0));
sd_kernels::conv2d!(pub un_c1_8_1280_1280: h = 8, w = 8, cin = 1280, cout = 1280, k = 1, s = 1, pin = (1280, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c1_8_1280_1280_r: h = 8, w = 8, cin = 1280, cout = 1280, k = 1, s = 1, pin = (1280, 0), pout = (1280, 0), act = none, res = (1280, 0));
sd_kernels::conv2d!(pub un_c1_8_1280_5120: h = 8, w = 8, cin = 1280, cout = 5120, k = 1, s = 1, pin = (1280, 0), pout = (5120, 0), act = none);
sd_kernels::conv2d!(pub un_c1_8_5120_1280_r: h = 8, w = 8, cin = 5120, cout = 1280, k = 1, s = 1, pin = (5120, 0), pout = (1280, 0), act = none, res = (1280, 0));
sd_kernels::conv2d!(pub un_c3_8_2560_1280: h = 8, w = 8, cin = 2560, cout = 1280, k = 3, s = 1, pin = (2560, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c1_8_2560_1280: h = 8, w = 8, cin = 2560, cout = 1280, k = 1, s = 1, pin = (2560, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c3_16_2560_1280: h = 16, w = 16, cin = 2560, cout = 1280, k = 3, s = 1, pin = (2560, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c1_16_2560_1280: h = 16, w = 16, cin = 2560, cout = 1280, k = 1, s = 1, pin = (2560, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c3_16_1920_1280: h = 16, w = 16, cin = 1920, cout = 1280, k = 3, s = 1, pin = (1920, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c1_16_1920_1280: h = 16, w = 16, cin = 1920, cout = 1280, k = 1, s = 1, pin = (1920, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c3_32_1280_1280: h = 32, w = 32, cin = 1280, cout = 1280, k = 3, s = 1, pin = (1280, 0), pout = (1280, 0), act = none);
sd_kernels::conv2d!(pub un_c3_32_1920_640: h = 32, w = 32, cin = 1920, cout = 640, k = 3, s = 1, pin = (1920, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c1_32_1920_640: h = 32, w = 32, cin = 1920, cout = 640, k = 1, s = 1, pin = (1920, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c3_32_1280_640: h = 32, w = 32, cin = 1280, cout = 640, k = 3, s = 1, pin = (1280, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c1_32_1280_640: h = 32, w = 32, cin = 1280, cout = 640, k = 1, s = 1, pin = (1280, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c3_32_960_640: h = 32, w = 32, cin = 960, cout = 640, k = 3, s = 1, pin = (960, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c1_32_960_640: h = 32, w = 32, cin = 960, cout = 640, k = 1, s = 1, pin = (960, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c3_64_640_640: h = 64, w = 64, cin = 640, cout = 640, k = 3, s = 1, pin = (640, 0), pout = (640, 0), act = none);
sd_kernels::conv2d!(pub un_c3_64_960_320: h = 64, w = 64, cin = 960, cout = 320, k = 3, s = 1, pin = (960, 0), pout = (320, 0), act = none);
sd_kernels::conv2d!(pub un_c1_64_960_320: h = 64, w = 64, cin = 960, cout = 320, k = 1, s = 1, pin = (960, 0), pout = (320, 0), act = none);
sd_kernels::conv2d!(pub un_c3_64_640_320: h = 64, w = 64, cin = 640, cout = 320, k = 3, s = 1, pin = (640, 0), pout = (320, 0), act = none);
sd_kernels::conv2d!(pub un_c1_64_640_320: h = 64, w = 64, cin = 640, cout = 320, k = 1, s = 1, pin = (640, 0), pout = (320, 0), act = none);
sd_kernels::conv2d!(pub un_c3_64_320_32: h = 64, w = 64, cin = 320, cout = 32, k = 3, s = 1, pin = (320, 0), pout = (32, 0), act = none);

// ── gn_stats（14 个）──
sd_kernels::gn_stats!(pub un_gs_64_320: h = 64, w = 64, c = 320, p = 320);
sd_kernels::gn_stats!(pub un_gs_32_320: h = 32, w = 32, c = 320, p = 320);
sd_kernels::gn_stats!(pub un_gs_32_640: h = 32, w = 32, c = 640, p = 640);
sd_kernels::gn_stats!(pub un_gs_16_640: h = 16, w = 16, c = 640, p = 640);
sd_kernels::gn_stats!(pub un_gs_16_1280: h = 16, w = 16, c = 1280, p = 1280);
sd_kernels::gn_stats!(pub un_gs_8_1280: h = 8, w = 8, c = 1280, p = 1280);
sd_kernels::gn_stats!(pub un_gs_8_2560: h = 8, w = 8, c = 2560, p = 2560);
sd_kernels::gn_stats!(pub un_gs_16_2560: h = 16, w = 16, c = 2560, p = 2560);
sd_kernels::gn_stats!(pub un_gs_16_1920: h = 16, w = 16, c = 1920, p = 1920);
sd_kernels::gn_stats!(pub un_gs_32_1920: h = 32, w = 32, c = 1920, p = 1920);
sd_kernels::gn_stats!(pub un_gs_32_1280: h = 32, w = 32, c = 1280, p = 1280);
sd_kernels::gn_stats!(pub un_gs_32_960: h = 32, w = 32, c = 960, p = 960);
sd_kernels::gn_stats!(pub un_gs_64_960: h = 64, w = 64, c = 960, p = 960);
sd_kernels::gn_stats!(pub un_gs_64_640: h = 64, w = 64, c = 640, p = 640);

// ── gn_apply（18 个）──
sd_kernels::gn_apply!(pub un_ga_64_320_silu: h = 64, w = 64, c = 320, p = 320, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_64_320_none: h = 64, w = 64, c = 320, p = 320, groups = 32, eps = 1e-06f32, act = none);
sd_kernels::gn_apply!(pub un_ga_32_320_silu: h = 32, w = 32, c = 320, p = 320, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_32_640_silu: h = 32, w = 32, c = 640, p = 640, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_32_640_none: h = 32, w = 32, c = 640, p = 640, groups = 32, eps = 1e-06f32, act = none);
sd_kernels::gn_apply!(pub un_ga_16_640_silu: h = 16, w = 16, c = 640, p = 640, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_16_1280_silu: h = 16, w = 16, c = 1280, p = 1280, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_16_1280_none: h = 16, w = 16, c = 1280, p = 1280, groups = 32, eps = 1e-06f32, act = none);
sd_kernels::gn_apply!(pub un_ga_8_1280_silu: h = 8, w = 8, c = 1280, p = 1280, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_8_1280_none: h = 8, w = 8, c = 1280, p = 1280, groups = 32, eps = 1e-06f32, act = none);
sd_kernels::gn_apply!(pub un_ga_8_2560_silu: h = 8, w = 8, c = 2560, p = 2560, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_16_2560_silu: h = 16, w = 16, c = 2560, p = 2560, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_16_1920_silu: h = 16, w = 16, c = 1920, p = 1920, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_32_1920_silu: h = 32, w = 32, c = 1920, p = 1920, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_32_1280_silu: h = 32, w = 32, c = 1280, p = 1280, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_32_960_silu: h = 32, w = 32, c = 960, p = 960, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_64_960_silu: h = 64, w = 64, c = 960, p = 960, groups = 32, eps = 1e-05f32, act = silu);
sd_kernels::gn_apply!(pub un_ga_64_640_silu: h = 64, w = 64, c = 640, p = 640, groups = 32, eps = 1e-05f32, act = silu);

// ── ln_nhwc（4 个）──
sd_kernels::ln_nhwc!(pub un_ln_64_320: h = 64, w = 64, c = 320, p = 320, eps = 1e-05f32);
sd_kernels::ln_nhwc!(pub un_ln_32_640: h = 32, w = 32, c = 640, p = 640, eps = 1e-05f32);
sd_kernels::ln_nhwc!(pub un_ln_16_1280: h = 16, w = 16, c = 1280, p = 1280, eps = 1e-05f32);
sd_kernels::ln_nhwc!(pub un_ln_8_1280: h = 8, w = 8, c = 1280, p = 1280, eps = 1e-05f32);

// ── attention（8 个）──
calblas::attention!(pub un_at_64_s: rows = 4736, heads = 8, head = 64, scale = 40, cores = 2, mask = keys);
calblas::attention!(pub un_at_64_x: rows = 4736, keys = 128, heads = 8, head = 64, scale = 40, cores = 2, mask = keys);
calblas::attention!(pub un_at_32_s: rows = 1408, heads = 8, head = 96, scale = 80, cores = 2, mask = keys);
calblas::attention!(pub un_at_32_x: rows = 1408, keys = 128, heads = 8, head = 96, scale = 80, cores = 2, mask = keys);
calblas::attention!(pub un_at_16_s: rows = 512, heads = 8, head = 160, scale = 160, cores = 2, mask = keys);
calblas::attention!(pub un_at_16_x: rows = 512, keys = 128, heads = 8, head = 160, scale = 160, cores = 2, mask = keys);
calblas::attention!(pub un_at_8_s: rows = 256, heads = 8, head = 160, scale = 160, cores = 2, mask = keys);
calblas::attention!(pub un_at_8_x: rows = 256, keys = 128, heads = 8, head = 160, scale = 160, cores = 2, mask = keys);

// ── linear（9 个）──
calblas::linear!(pub un_l16_384_1280: rows = 16, k = 384, n = 1280, w = hf, cores = 2, bias = f32);
calblas::linear!(pub un_l16_1280_1280: rows = 16, k = 1280, n = 1280, w = hf, cores = 2, bias = f32);
calblas::linear!(pub un_l16_1280_20224: rows = 16, k = 1280, n = 20224, w = hf, cores = 2, bias = f32);
calblas::linear!(pub un_kv_512: rows = 128, k = 768, n = 512, w = hf, cores = 2);
calblas::linear!(pub un_lr_32_640_640_b: rows = 1408, k = 640, n = 640, w = hf, cores = 2, bias = f32);
calblas::linear!(pub un_lr_32_640_768: rows = 1408, k = 640, n = 768, w = hf, cores = 2);
calblas::linear!(pub un_kv_768: rows = 128, k = 768, n = 768, w = hf, cores = 2);
calblas::linear!(pub un_lr_32_640_2560_b: rows = 1408, k = 640, n = 2560, w = hf, cores = 2, bias = f32);
calblas::linear!(pub un_kv_1280: rows = 128, k = 768, n = 1280, w = hf, cores = 2);

// ── geglu（4 个）──
sd_kernels::geglu!(pub un_gg_64: n = 5918720);
sd_kernels::geglu!(pub un_gg_32: n = 3317760);
sd_kernels::geglu!(pub un_gg_16: n = 2048000);
sd_kernels::geglu!(pub un_gg_8: n = 737280);

// ── upsample2（4 个）──
calblas::upsample2!(pub un_up_8_1280_256: h = 8, w = 8, c = 256, pin = 1280, pout = 1280);
calblas::upsample2!(pub un_up_16_1280_256: h = 16, w = 16, c = 256, pin = 1280, pout = 1280);
calblas::upsample2!(pub un_up_32_640_256: h = 32, w = 32, c = 256, pin = 640, pout = 640);
calblas::upsample2!(pub un_up_32_640_128: h = 32, w = 32, c = 128, pin = 640, pout = 640);

// ── copy_win（12 个）──
sd_kernels::copy_win!(pub un_cp_8_1280_2560: h = 8, w = 8, c = 1280, pin = 1280, pout = 2560);
sd_kernels::copy_win!(pub un_cp_16_1280_2560: h = 16, w = 16, c = 1280, pin = 1280, pout = 2560);
sd_kernels::copy_win!(pub un_cp_16_1280_1920: h = 16, w = 16, c = 1280, pin = 1280, pout = 1920);
sd_kernels::copy_win!(pub un_cp_16_640_1920: h = 16, w = 16, c = 640, pin = 640, pout = 1920);
sd_kernels::copy_win!(pub un_cp_32_1280_1920: h = 32, w = 32, c = 1280, pin = 1280, pout = 1920);
sd_kernels::copy_win!(pub un_cp_32_640_1920: h = 32, w = 32, c = 640, pin = 640, pout = 1920);
sd_kernels::copy_win!(pub un_cp_32_640_1280: h = 32, w = 32, c = 640, pin = 640, pout = 1280);
sd_kernels::copy_win!(pub un_cp_32_640_960: h = 32, w = 32, c = 640, pin = 640, pout = 960);
sd_kernels::copy_win!(pub un_cp_32_320_960: h = 32, w = 32, c = 320, pin = 320, pout = 960);
sd_kernels::copy_win!(pub un_cp_64_640_960: h = 64, w = 64, c = 640, pin = 640, pout = 960);
sd_kernels::copy_win!(pub un_cp_64_320_960: h = 64, w = 64, c = 320, pin = 320, pout = 960);
sd_kernels::copy_win!(pub un_cp_64_320_640: h = 64, w = 64, c = 320, pin = 320, pout = 640);

// ── timestep_embed（1 个）──
sd_kernels::timestep_embed!(pub un_temb: half = 160);

// ── unary（1 个）──
calblas::unary!(pub un_silu: silu, n = 20480);

// ── bias_fold（1 个）──
sd_kernels::bias_fold!(pub un_bf: n = 20224);

// ── fill16k（1 个）──
calblas::fill16k!(pub un_zero: value = 0.0f32);
