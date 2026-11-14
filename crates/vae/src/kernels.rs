//! kernel 特化（形状是编译期常量：每个形状一份）。卷积 18、NHWC GroupNorm 6 对（统计 + 归一）、注意力前的 GroupNorm 1（+ 排布转换 2）、转置 2、线性层 1、注意力 1、上采样 3、清零 1。
//! 名字带 `va_` 前缀（calbin 里各 kernel 同在一个命名空间）。`host/sd15npu/cvxsim.py` 按这里的宏参数认 kernel 的语义，
//! 所以宏参数一律写字面量（模拟器用正则读；`crates/ktest` 用 `#[path]` 引这一份，测的就是整步程序里跑的那个 kernel）。
//!
//! 卷积全是 `act = none`：VAE 的 SiLU 在 GroupNorm 那一步做（`group_norm!` 的 `act = silu`），卷积之后不激活；
//! ResNet 的 conv2、注意力的 proj_attn 带残差（`res = …`，act 之后加）。

// ── 卷积（sd_kernels::conv2d!，抄自 calblas，见 VENDORED.md）──
// 64²：post_quant_conv（1×1，4 ⇒ 4，补到 32）、conv_in（4 ⇒ 512）、512 ⇒ 512、带残差的、proj_attn（1×1 带残差）
sd_kernels::conv2d!(pub va_pqc: h = 64, w = 64, cin = 32, cout = 32, k = 1, s = 1, pin = (32, 0), pout = (32, 0), act = none);
sd_kernels::conv2d!(pub va_cin: h = 64, w = 64, cin = 32, cout = 512, k = 3, s = 1, pin = (32, 0), pout = (512, 0), act = none);
sd_kernels::conv2d!(pub va_c64: h = 64, w = 64, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none);
sd_kernels::conv2d!(pub va_c64r: h = 64, w = 64, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none, res = (512, 0));
sd_kernels::conv2d!(pub va_o64r: h = 64, w = 64, cin = 512, cout = 512, k = 1, s = 1, pin = (512, 0), pout = (512, 0), act = none, res = (512, 0));
// 128²：512 ⇒ 512（也是 up_blocks.0 上采样之后的那个卷积）、带残差的
sd_kernels::conv2d!(pub va_c128: h = 128, w = 128, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none);
sd_kernels::conv2d!(pub va_c128r: h = 128, w = 128, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none, res = (512, 0));
// 256²：上采样之后 512 ⇒ 512、ResNet 0 的 conv1 512 ⇒ 256 与 1×1 shortcut、256 ⇒ 256、带残差的
sd_kernels::conv2d!(pub va_c256u: h = 256, w = 256, cin = 512, cout = 512, k = 3, s = 1, pin = (512, 0), pout = (512, 0), act = none);
sd_kernels::conv2d!(pub va_c256a: h = 256, w = 256, cin = 512, cout = 256, k = 3, s = 1, pin = (512, 0), pout = (256, 0), act = none);
sd_kernels::conv2d!(pub va_s256: h = 256, w = 256, cin = 512, cout = 256, k = 1, s = 1, pin = (512, 0), pout = (256, 0), act = none);
sd_kernels::conv2d!(pub va_c256: h = 256, w = 256, cin = 256, cout = 256, k = 3, s = 1, pin = (256, 0), pout = (256, 0), act = none);
sd_kernels::conv2d!(pub va_c256r: h = 256, w = 256, cin = 256, cout = 256, k = 3, s = 1, pin = (256, 0), pout = (256, 0), act = none, res = (256, 0));
// 512²：上采样之后 256 ⇒ 256、ResNet 0 的 conv1 256 ⇒ 128 与 1×1 shortcut、128 ⇒ 128、带残差的、conv_out（128 ⇒ 3，补到 32）
sd_kernels::conv2d!(pub va_c512u: h = 512, w = 512, cin = 256, cout = 256, k = 3, s = 1, pin = (256, 0), pout = (256, 0), act = none);
sd_kernels::conv2d!(pub va_c512a: h = 512, w = 512, cin = 256, cout = 128, k = 3, s = 1, pin = (256, 0), pout = (128, 0), act = none);
sd_kernels::conv2d!(pub va_s512: h = 512, w = 512, cin = 256, cout = 128, k = 1, s = 1, pin = (256, 0), pout = (128, 0), act = none);
sd_kernels::conv2d!(pub va_c512: h = 512, w = 512, cin = 128, cout = 128, k = 3, s = 1, pin = (128, 0), pout = (128, 0), act = none);
sd_kernels::conv2d!(pub va_c512r: h = 512, w = 512, cin = 128, cout = 128, k = 3, s = 1, pin = (128, 0), pout = (128, 0), act = none, res = (128, 0));
sd_kernels::conv2d!(pub va_cout: h = 512, w = 512, cin = 128, cout = 32, k = 3, s = 1, pin = (128, 0), pout = (32, 0), act = none);

// ── GroupNorm + SiLU：NHWC 原生（本仓 gn_stats! / gn_apply!，DESIGN D16），直接在带边框的图上算，不转排布 ──
// 每个形状一对：统计（每核的 Σx、Σx²）⇒ 屏障 ⇒ 归一（只写内部）。两个都是单核 kernel、按 num_cores() 分行 ⇒ 两核都调
sd_kernels::gn_stats!(pub va_gs64: h = 64, w = 64, c = 512, p = 512);
sd_kernels::gn_apply!(pub va_ga64: h = 64, w = 64, c = 512, p = 512, groups = 32, eps = 1e-6f32, act = silu);
sd_kernels::gn_stats!(pub va_gs128: h = 128, w = 128, c = 512, p = 512);
sd_kernels::gn_apply!(pub va_ga128: h = 128, w = 128, c = 512, p = 512, groups = 32, eps = 1e-6f32, act = silu);
sd_kernels::gn_stats!(pub va_gs256w: h = 256, w = 256, c = 512, p = 512);
sd_kernels::gn_apply!(pub va_ga256w: h = 256, w = 256, c = 512, p = 512, groups = 32, eps = 1e-6f32, act = silu);
sd_kernels::gn_stats!(pub va_gs256: h = 256, w = 256, c = 256, p = 256);
sd_kernels::gn_apply!(pub va_ga256: h = 256, w = 256, c = 256, p = 256, groups = 32, eps = 1e-6f32, act = silu);
sd_kernels::gn_stats!(pub va_gs512w: h = 512, w = 512, c = 256, p = 256);
sd_kernels::gn_apply!(pub va_ga512w: h = 512, w = 512, c = 256, p = 256, groups = 32, eps = 1e-6f32, act = silu);
sd_kernels::gn_stats!(pub va_gs512: h = 512, w = 512, c = 128, p = 128);
sd_kernels::gn_apply!(pub va_ga512: h = 512, w = 512, c = 128, p = 128, groups = 32, eps = 1e-6f32, act = silu);

// ── 注意力前面那个 GroupNorm（不带 SiLU，后面要转成 token）：还走 D13 的「排布转换 + calblas group_norm!（NCHW）」──
calblas::group_norm!(pub va_gn64n: hw = 4096, c = 512, groups = 32, eps = 1e-6f32, act = none);
calblas::nhwc_to_nchw!(pub va_tc64: h = 64, w = 64, c = 512, p = 512);
calblas::nchw_to_nhwc!(pub va_th64: h = 64, w = 64, c = 512, p = 512);

// ── 注意力（DESIGN D14）：NCHW [512][4096] ⇔ token [4096][512]、q / k / v 的线性层、头维 512 的注意力 ──
calblas::transpose2d!(pub va_tr_ct: rows = 512, cols = 4096);
calblas::transpose2d!(pub va_tr_tc: rows = 4096, cols = 512);
calblas::linear!(pub va_qkv: rows = 4096, k = 512, n = 512, w = hf, cores = 2, bias = f32);
sd_kernels::attn_wide!(pub va_attn: rows = 4096, heads = 1, head = 512);

// ── 最近邻 ×2：512 通道的按两个 256 通道的窗口各调一次（upsample2! 一次至多 256 通道，DESIGN F21）；两核都调 ──
calblas::upsample2!(pub va_up64: h = 64, w = 64, c = 256, pin = 512, pout = 512);
calblas::upsample2!(pub va_up128: h = 128, w = 128, c = 256, pin = 512, pout = 512);
calblas::upsample2!(pub va_up256: h = 256, w = 256, c = 256, pin = 256, pout = 256);

// ── 带边框的图整块清零（每块 16 KiB）：边框靠它是 0 ──
calblas::fill16k!(pub va_zero: value = 0.0f32);
