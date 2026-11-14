//! kernel 特化（形状是编译期常量：每个形状一份）。卷积 13 份、上采样 3 份、tanh 1 份、清零 1 份。
//! 名字带 `td_` 前缀（calbin 里各 kernel 同在一个命名空间）；`ref/sim.py` 按这里的宏参数认 kernel 的语义，改了这里要能被它的正则读到。
use crate::layout::{C, CIN0, COUT, IN_BYTES, LAT};

// 入口：tanh(x / 3) · 3，整张带边框的 latent（边框、补的通道是 0，tanh(0) = 0 照样是 0）
sd_kernels::tanh_scale!(pub td_tanh: n = IN_BYTES / 2, scale = 3.0f32);

// conv(4→64) + ReLU：latent 补到 32 通道
sd_kernels::conv2d!(pub td_conv_in: h = LAT, w = LAT, cin = CIN0, cout = C, k = 3, s = 1, pin = (CIN0, 0), pout = (C, 0), act = relu);

// Block 的前两个卷积：conv + ReLU
sd_kernels::conv2d!(pub td_conv_64: h = 64, w = 64, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = relu);
sd_kernels::conv2d!(pub td_conv_128: h = 128, w = 128, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = relu);
sd_kernels::conv2d!(pub td_conv_256: h = 256, w = 256, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = relu);
sd_kernels::conv2d!(pub td_conv_512: h = 512, w = 512, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = relu);

// Block 的第三个卷积：relu(conv + x)（残差之后再 ReLU）
sd_kernels::conv2d!(pub td_convr_64: h = 64, w = 64, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = res_relu, res = (C, 0));
sd_kernels::conv2d!(pub td_convr_128: h = 128, w = 128, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = res_relu, res = (C, 0));
sd_kernels::conv2d!(pub td_convr_256: h = 256, w = 256, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = res_relu, res = (C, 0));
sd_kernels::conv2d!(pub td_convr_512: h = 512, w = 512, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = res_relu, res = (C, 0));

// 上采样之后的卷积（没有 bias ⇒ 打包给全零 bias，不激活）
sd_kernels::conv2d!(pub td_convu_128: h = 128, w = 128, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = none);
sd_kernels::conv2d!(pub td_convu_256: h = 256, w = 256, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = none);
sd_kernels::conv2d!(pub td_convu_512: h = 512, w = 512, cin = C, cout = C, k = 3, s = 1, pin = (C, 0), pout = (C, 0), act = none);

// 出口：conv(64→3)，补到 32 个输出通道
sd_kernels::conv2d!(pub td_conv_out: h = 8 * LAT, w = 8 * LAT, cin = C, cout = COUT, k = 3, s = 1, pin = (C, 0), pout = (COUT, 0), act = none);

// 最近邻 ×2（通道数写字面量：宏里有个同名的 const C，传 C 进去会自引用）（单核 kernel，行按 num_cores() 分 ⇒ 两核都调，同 calblas yolo26 上板的那份）
calblas::upsample2!(pub td_up_64: h = 64, w = 64, c = 64, pin = 64, pout = 64);
calblas::upsample2!(pub td_up_128: h = 128, w = 128, c = 64, pin = 64, pout = 64);
calblas::upsample2!(pub td_up_256: h = 256, w = 256, c = 64, pin = 64, pout = 64);

// @act 整块清零（每块 16 KiB）：特征图的边框靠它是 0
calblas::fill16k!(pub td_zero: value = 0.0f32);
