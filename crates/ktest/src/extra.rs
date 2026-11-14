//! 还没进整步程序的新 kernel 的特化（`ref/ktest.py` 的模拟器从这里和 crates/taesd/src/kernels.rs 一起读宏参数）。

// SD1.5 UNet 的时间步嵌入：320 维（half = 160）
sd_kernels::timestep_embed!(pub kt_temb: half = 160);
