//! 真卡上逐个 kernel 计时（**`tools/prof.py --model unet gen` 生成，不要手改**）：每个 feature 一个程序，把UNet的一个 kernel 就地连调
//! 4 遍，输入输出各 64 B、权重全零、@act 不清零（只量时间、不看数）。`base` 只有屏障，扣固定开销用。
#![no_std]
#![allow(dead_code)]

#[path = "../../unet/src/kernels.rs"]
mod vae_kernels;

const N_FEATURES: usize = cfg!(feature = "base") as usize + cfg!(feature = "un_c3_64_32_320") as usize + cfg!(feature = "un_c3_64_320_320") as usize + cfg!(feature = "un_c3_64_320_320_r") as usize + cfg!(feature = "un_c1_64_320_320") as usize + cfg!(feature = "un_c1_64_320_512") as usize + cfg!(feature = "un_c1_64_512_320_r") as usize + cfg!(feature = "un_c1_64_320_1280") as usize + cfg!(feature = "un_c1_64_1280_320_r") as usize + cfg!(feature = "un_c1_64_320_320_r") as usize + cfg!(feature = "un_c3_32_320_320_s2") as usize + cfg!(feature = "un_c3_32_320_640") as usize + cfg!(feature = "un_c1_32_320_640") as usize + cfg!(feature = "un_c3_32_640_640_r") as usize + cfg!(feature = "un_c1_32_768_640_r") as usize + cfg!(feature = "un_c1_32_2560_640_r_o768_q640") as usize + cfg!(feature = "un_c3_32_640_640") as usize + cfg!(feature = "un_c3_16_640_640_s2") as usize + cfg!(feature = "un_c3_16_640_1280") as usize + cfg!(feature = "un_c1_16_640_1280") as usize + cfg!(feature = "un_c3_16_1280_1280_r") as usize + cfg!(feature = "un_c1_16_1280_1280") as usize + cfg!(feature = "un_c1_16_1280_1280_r") as usize + cfg!(feature = "un_c1_16_1280_5120") as usize + cfg!(feature = "un_c1_16_5120_1280_r") as usize + cfg!(feature = "un_c3_16_1280_1280") as usize + cfg!(feature = "un_c3_8_1280_1280_s2") as usize + cfg!(feature = "un_c3_8_1280_1280") as usize + cfg!(feature = "un_c3_8_1280_1280_r") as usize + cfg!(feature = "un_c1_8_1280_1280") as usize + cfg!(feature = "un_c1_8_1280_1280_r") as usize + cfg!(feature = "un_c1_8_1280_5120") as usize + cfg!(feature = "un_c1_8_5120_1280_r") as usize + cfg!(feature = "un_c3_8_2560_1280") as usize + cfg!(feature = "un_c1_8_2560_1280") as usize + cfg!(feature = "un_c3_16_2560_1280") as usize + cfg!(feature = "un_c1_16_2560_1280") as usize + cfg!(feature = "un_c3_16_1920_1280") as usize + cfg!(feature = "un_c1_16_1920_1280") as usize + cfg!(feature = "un_c3_32_1280_1280") as usize + cfg!(feature = "un_c3_32_1920_640") as usize + cfg!(feature = "un_c1_32_1920_640") as usize + cfg!(feature = "un_c3_32_1280_640") as usize + cfg!(feature = "un_c1_32_1280_640") as usize + cfg!(feature = "un_c3_32_960_640") as usize + cfg!(feature = "un_c1_32_960_640") as usize + cfg!(feature = "un_c3_64_640_640") as usize + cfg!(feature = "un_c3_64_960_320") as usize + cfg!(feature = "un_c1_64_960_320") as usize + cfg!(feature = "un_c3_64_640_320") as usize + cfg!(feature = "un_c1_64_640_320") as usize + cfg!(feature = "un_c3_64_320_32") as usize + cfg!(feature = "un_gs_64_320") as usize + cfg!(feature = "un_gs_32_320") as usize + cfg!(feature = "un_gs_32_640") as usize + cfg!(feature = "un_gs_16_640") as usize + cfg!(feature = "un_gs_16_1280") as usize + cfg!(feature = "un_gs_8_1280") as usize + cfg!(feature = "un_gs_8_2560") as usize + cfg!(feature = "un_gs_16_2560") as usize + cfg!(feature = "un_gs_16_1920") as usize + cfg!(feature = "un_gs_32_1920") as usize + cfg!(feature = "un_gs_32_1280") as usize + cfg!(feature = "un_gs_32_960") as usize + cfg!(feature = "un_gs_64_960") as usize + cfg!(feature = "un_gs_64_640") as usize + cfg!(feature = "un_ga_64_320_silu") as usize + cfg!(feature = "un_ga_64_320_none") as usize + cfg!(feature = "un_ga_32_320_silu") as usize + cfg!(feature = "un_ga_32_640_silu") as usize + cfg!(feature = "un_ga_32_640_none") as usize + cfg!(feature = "un_ga_16_640_silu") as usize + cfg!(feature = "un_ga_16_1280_silu") as usize + cfg!(feature = "un_ga_16_1280_none") as usize + cfg!(feature = "un_ga_8_1280_silu") as usize + cfg!(feature = "un_ga_8_1280_none") as usize + cfg!(feature = "un_ga_8_2560_silu") as usize + cfg!(feature = "un_ga_16_2560_silu") as usize + cfg!(feature = "un_ga_16_1920_silu") as usize + cfg!(feature = "un_ga_32_1920_silu") as usize + cfg!(feature = "un_ga_32_1280_silu") as usize + cfg!(feature = "un_ga_32_960_silu") as usize + cfg!(feature = "un_ga_64_960_silu") as usize + cfg!(feature = "un_ga_64_640_silu") as usize + cfg!(feature = "un_ln_64_320") as usize + cfg!(feature = "un_ln_32_640") as usize + cfg!(feature = "un_ln_16_1280") as usize + cfg!(feature = "un_ln_8_1280") as usize + cfg!(feature = "un_at_64_s") as usize + cfg!(feature = "un_at_64_x") as usize + cfg!(feature = "un_at_32_s") as usize + cfg!(feature = "un_at_32_x") as usize + cfg!(feature = "un_at_16_s") as usize + cfg!(feature = "un_at_16_x") as usize + cfg!(feature = "un_at_8_s") as usize + cfg!(feature = "un_at_8_x") as usize + cfg!(feature = "un_l16_384_1280") as usize + cfg!(feature = "un_l16_1280_1280") as usize + cfg!(feature = "un_l16_1280_20224") as usize + cfg!(feature = "un_kv_512") as usize + cfg!(feature = "un_lr_32_640_640_b") as usize + cfg!(feature = "un_lr_32_640_768") as usize + cfg!(feature = "un_kv_768") as usize + cfg!(feature = "un_lr_32_640_2560_b") as usize + cfg!(feature = "un_kv_1280") as usize + cfg!(feature = "un_gg_64") as usize + cfg!(feature = "un_gg_32") as usize + cfg!(feature = "un_gg_16") as usize + cfg!(feature = "un_gg_8") as usize + cfg!(feature = "un_up_8_1280_256") as usize + cfg!(feature = "un_up_16_1280_256") as usize + cfg!(feature = "un_up_32_640_256") as usize + cfg!(feature = "un_up_32_640_128") as usize + cfg!(feature = "un_cp_8_1280_2560") as usize + cfg!(feature = "un_cp_16_1280_2560") as usize + cfg!(feature = "un_cp_16_1280_1920") as usize + cfg!(feature = "un_cp_16_640_1920") as usize + cfg!(feature = "un_cp_32_1280_1920") as usize + cfg!(feature = "un_cp_32_640_1920") as usize + cfg!(feature = "un_cp_32_640_1280") as usize + cfg!(feature = "un_cp_32_640_960") as usize + cfg!(feature = "un_cp_32_320_960") as usize + cfg!(feature = "un_cp_64_640_960") as usize + cfg!(feature = "un_cp_64_320_960") as usize + cfg!(feature = "un_cp_64_320_640") as usize + cfg!(feature = "un_temb") as usize + cfg!(feature = "un_silu") as usize + cfg!(feature = "un_bf") as usize;
const _: () = assert!(N_FEATURES <= 1, "prof：一次只开一个 feature");

#[cfg(feature = "base")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
    }
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        sync_cores();
        sync_cores();
        sync_cores();
        sync_cores();
    }
}

/// un_c3_64_32_320：conv2d {'h': 64, 'w': 64, 'cin': 32, 'cout': 320, 'k': 3, 's': 1, 'pin': (32, 0), 'pout': (320, 0), 'act': 'none'}
#[cfg(feature = "un_c3_64_32_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 188416;
        act: workspace, bytes = 3268608;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 307200;
    const W_W: u64 = 0;
    const W_B: u64 = 184320;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_64_32_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_32_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_32_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_32_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_64_320_320：conv2d {'h': 64, 'w': 64, 'cin': 320, 'cout': 320, 'k': 3, 's': 1, 'pin': (320, 0), 'pout': (320, 0), 'act': 'none'}
#[cfg(feature = "un_c3_64_320_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1847296;
        act: workspace, bytes = 6004736;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3043328;
    const W_W: u64 = 0;
    const W_B: u64 = 1843200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_64_320_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_64_320_320_r：conv2d {'h': 64, 'w': 64, 'cin': 320, 'cout': 320, 'k': 3, 's': 1, 'pin': (320, 0), 'pout': (320, 0), 'act': 'none', 'res': (320, 0)}
#[cfg(feature = "un_c3_64_320_320_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1847296;
        act: workspace, bytes = 8966144;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3043328;
    const A_R: u64 = 6004736;
    const W_W: u64 = 0;
    const W_B: u64 = 1843200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_64_320_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_64_320_320：conv2d {'h': 64, 'w': 64, 'cin': 320, 'cout': 320, 'k': 1, 's': 1, 'pin': (320, 0), 'pout': (320, 0), 'act': 'none'}
#[cfg(feature = "un_c1_64_320_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 208896;
        act: workspace, bytes = 6004736;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3043328;
    const W_W: u64 = 0;
    const W_B: u64 = 204800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_64_320_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_64_320_512：conv2d {'h': 64, 'w': 64, 'cin': 320, 'cout': 512, 'k': 1, 's': 1, 'pin': (320, 0), 'pout': (512, 0), 'act': 'none'}
#[cfg(feature = "un_c1_64_320_512")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 331776;
        act: workspace, bytes = 7778304;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3043328;
    const W_W: u64 = 0;
    const W_B: u64 = 327680;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_64_320_512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_64_512_320_r：conv2d {'h': 64, 'w': 64, 'cin': 512, 'cout': 320, 'k': 1, 's': 1, 'pin': (512, 0), 'pout': (320, 0), 'act': 'none', 'res': (320, 0)}
#[cfg(feature = "un_c1_64_512_320_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 331776;
        act: workspace, bytes = 10788864;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4866048;
    const A_R: u64 = 7827456;
    const W_W: u64 = 0;
    const W_B: u64 = 327680;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_64_512_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_512_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_512_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_512_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_64_320_1280：conv2d {'h': 64, 'w': 64, 'cin': 320, 'cout': 1280, 'k': 1, 's': 1, 'pin': (320, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c1_64_320_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 827392;
        act: workspace, bytes = 14880768;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3043328;
    const W_W: u64 = 0;
    const W_B: u64 = 819200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_64_320_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_64_1280_320_r：conv2d {'h': 64, 'w': 64, 'cin': 1280, 'cout': 320, 'k': 1, 's': 1, 'pin': (1280, 0), 'pout': (320, 0), 'act': 'none', 'res': (320, 0)}
#[cfg(feature = "un_c1_64_1280_320_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 823296;
        act: workspace, bytes = 18087936;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 12165120;
    const A_R: u64 = 15126528;
    const W_W: u64 = 0;
    const W_B: u64 = 819200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_64_1280_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_1280_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_1280_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_1280_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_64_320_320_r：conv2d {'h': 64, 'w': 64, 'cin': 320, 'cout': 320, 'k': 1, 's': 1, 'pin': (320, 0), 'pout': (320, 0), 'act': 'none', 'res': (320, 0)}
#[cfg(feature = "un_c1_64_320_320_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 208896;
        act: workspace, bytes = 8966144;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3043328;
    const A_R: u64 = 6004736;
    const W_W: u64 = 0;
    const W_B: u64 = 204800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_64_320_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_320_320_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_32_320_320_s2：conv2d {'h': 32, 'w': 32, 'cin': 320, 'cout': 320, 'k': 3, 's': 2, 'pin': (320, 0), 'pout': (320, 0), 'act': 'none'}
#[cfg(feature = "un_c3_32_320_320_s2")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1847296;
        act: workspace, bytes = 3874816;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3043328;
    const W_W: u64 = 0;
    const W_B: u64 = 1843200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_32_320_320_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_320_320_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_320_320_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_320_320_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_32_320_640：conv2d {'h': 32, 'w': 32, 'cin': 320, 'cout': 640, 'k': 3, 's': 1, 'pin': (320, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c3_32_320_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3690496;
        act: workspace, bytes = 2572288;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 913408;
    const W_W: u64 = 0;
    const W_B: u64 = 3686400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_32_320_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_320_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_320_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_320_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_32_320_640：conv2d {'h': 32, 'w': 32, 'cin': 320, 'cout': 640, 'k': 1, 's': 1, 'pin': (320, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c1_32_320_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 413696;
        act: workspace, bytes = 2572288;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 913408;
    const W_W: u64 = 0;
    const W_B: u64 = 409600;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_32_320_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_320_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_320_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_320_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_32_640_640_r：conv2d {'h': 32, 'w': 32, 'cin': 640, 'cout': 640, 'k': 3, 's': 1, 'pin': (640, 0), 'pout': (640, 0), 'act': 'none', 'res': (640, 0)}
#[cfg(feature = "un_c3_32_640_640_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 7376896;
        act: workspace, bytes = 5140480;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1822720;
    const A_R: u64 = 3481600;
    const W_W: u64 = 0;
    const W_B: u64 = 7372800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_32_640_640_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_640_640_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_640_640_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_640_640_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_32_768_640_r：conv2d {'h': 32, 'w': 32, 'cin': 768, 'cout': 640, 'k': 1, 's': 1, 'pin': (768, 0), 'pout': (640, 0), 'act': 'none', 'res': (640, 0)}
#[cfg(feature = "un_c1_32_768_640_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 987136;
        act: workspace, bytes = 5505024;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2187264;
    const A_R: u64 = 3846144;
    const W_W: u64 = 0;
    const W_B: u64 = 983040;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_32_768_640_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_768_640_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_768_640_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_768_640_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_32_2560_640_r_o768_q640：conv2d {'h': 32, 'w': 32, 'cin': 2560, 'cout': 640, 'k': 1, 's': 1, 'pin': (2560, 0), 'pout': (768, 0), 'act': 'none', 'res': (640, 0)}
#[cfg(feature = "un_c1_32_2560_640_r_o768_q640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3280896;
        act: workspace, bytes = 10940416;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 7290880;
    const A_R: u64 = 9281536;
    const W_W: u64 = 0;
    const W_B: u64 = 3276800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_32_2560_640_r_o768_q640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_2560_640_r_o768_q640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_2560_640_r_o768_q640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_2560_640_r_o768_q640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_32_640_640：conv2d {'h': 32, 'w': 32, 'cin': 640, 'cout': 640, 'k': 3, 's': 1, 'pin': (640, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c3_32_640_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 7376896;
        act: workspace, bytes = 3481600;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1822720;
    const W_W: u64 = 0;
    const W_B: u64 = 7372800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_32_640_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_640_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_640_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_640_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_16_640_640_s2：conv2d {'h': 16, 'w': 16, 'cin': 640, 'cout': 640, 'k': 3, 's': 2, 'pin': (640, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c3_16_640_640_s2")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 7376896;
        act: workspace, bytes = 2334720;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1822720;
    const W_W: u64 = 0;
    const W_B: u64 = 7372800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_16_640_640_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_640_640_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_640_640_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_640_640_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_16_640_1280：conv2d {'h': 16, 'w': 16, 'cin': 640, 'cout': 1280, 'k': 3, 's': 1, 'pin': (640, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c3_16_640_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 14753792;
        act: workspace, bytes = 1699840;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 675840;
    const W_W: u64 = 0;
    const W_B: u64 = 14745600;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_16_640_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_640_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_640_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_640_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_16_640_1280：conv2d {'h': 16, 'w': 16, 'cin': 640, 'cout': 1280, 'k': 1, 's': 1, 'pin': (640, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c1_16_640_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1646592;
        act: workspace, bytes = 1699840;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 675840;
    const W_W: u64 = 0;
    const W_B: u64 = 1638400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_16_640_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_640_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_640_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_640_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_16_1280_1280_r：conv2d {'h': 16, 'w': 16, 'cin': 1280, 'cout': 1280, 'k': 3, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none', 'res': (1280, 0)}
#[cfg(feature = "un_c3_16_1280_1280_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 29499392;
        act: workspace, bytes = 3399680;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1351680;
    const A_R: u64 = 2375680;
    const W_W: u64 = 0;
    const W_B: u64 = 29491200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_16_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_16_1280_1280：conv2d {'h': 16, 'w': 16, 'cin': 1280, 'cout': 1280, 'k': 1, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c1_16_1280_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3284992;
        act: workspace, bytes = 2375680;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1351680;
    const W_W: u64 = 0;
    const W_B: u64 = 3276800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_16_1280_1280_r：conv2d {'h': 16, 'w': 16, 'cin': 1280, 'cout': 1280, 'k': 1, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none', 'res': (1280, 0)}
#[cfg(feature = "un_c1_16_1280_1280_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3284992;
        act: workspace, bytes = 3399680;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1351680;
    const A_R: u64 = 2375680;
    const W_W: u64 = 0;
    const W_B: u64 = 3276800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_16_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_16_1280_5120：conv2d {'h': 16, 'w': 16, 'cin': 1280, 'cout': 5120, 'k': 1, 's': 1, 'pin': (1280, 0), 'pout': (5120, 0), 'act': 'none'}
#[cfg(feature = "un_c1_16_1280_5120")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 13127680;
        act: workspace, bytes = 5447680;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1351680;
    const W_W: u64 = 0;
    const W_B: u64 = 13107200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_16_1280_5120(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_5120(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_5120(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1280_5120(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_16_5120_1280_r：conv2d {'h': 16, 'w': 16, 'cin': 5120, 'cout': 1280, 'k': 1, 's': 1, 'pin': (5120, 0), 'pout': (1280, 0), 'act': 'none', 'res': (1280, 0)}
#[cfg(feature = "un_c1_16_5120_1280_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 13115392;
        act: workspace, bytes = 7454720;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 5406720;
    const A_R: u64 = 6430720;
    const W_W: u64 = 0;
    const W_B: u64 = 13107200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_16_5120_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_5120_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_5120_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_5120_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_16_1280_1280：conv2d {'h': 16, 'w': 16, 'cin': 1280, 'cout': 1280, 'k': 3, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c3_16_1280_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 29499392;
        act: workspace, bytes = 2375680;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1351680;
    const W_W: u64 = 0;
    const W_B: u64 = 29491200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_8_1280_1280_s2：conv2d {'h': 8, 'w': 8, 'cin': 1280, 'cout': 1280, 'k': 3, 's': 2, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c3_8_1280_1280_s2")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 29499392;
        act: workspace, bytes = 1720320;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1351680;
    const W_W: u64 = 0;
    const W_B: u64 = 29491200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_8_1280_1280_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280_s2(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_8_1280_1280：conv2d {'h': 8, 'w': 8, 'cin': 1280, 'cout': 1280, 'k': 3, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c3_8_1280_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 29499392;
        act: workspace, bytes = 1064960;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 696320;
    const W_W: u64 = 0;
    const W_B: u64 = 29491200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_8_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_8_1280_1280_r：conv2d {'h': 8, 'w': 8, 'cin': 1280, 'cout': 1280, 'k': 3, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none', 'res': (1280, 0)}
#[cfg(feature = "un_c3_8_1280_1280_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 29499392;
        act: workspace, bytes = 1433600;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 696320;
    const A_R: u64 = 1064960;
    const W_W: u64 = 0;
    const W_B: u64 = 29491200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_8_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_8_1280_1280：conv2d {'h': 8, 'w': 8, 'cin': 1280, 'cout': 1280, 'k': 1, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c1_8_1280_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3284992;
        act: workspace, bytes = 1064960;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 696320;
    const W_W: u64 = 0;
    const W_B: u64 = 3276800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_8_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_8_1280_1280_r：conv2d {'h': 8, 'w': 8, 'cin': 1280, 'cout': 1280, 'k': 1, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none', 'res': (1280, 0)}
#[cfg(feature = "un_c1_8_1280_1280_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3284992;
        act: workspace, bytes = 1433600;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 696320;
    const A_R: u64 = 1064960;
    const W_W: u64 = 0;
    const W_B: u64 = 3276800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_8_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_8_1280_5120：conv2d {'h': 8, 'w': 8, 'cin': 1280, 'cout': 5120, 'k': 1, 's': 1, 'pin': (1280, 0), 'pout': (5120, 0), 'act': 'none'}
#[cfg(feature = "un_c1_8_1280_5120")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 13127680;
        act: workspace, bytes = 2170880;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 696320;
    const W_W: u64 = 0;
    const W_B: u64 = 13107200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_8_1280_5120(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_5120(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_5120(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_1280_5120(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_8_5120_1280_r：conv2d {'h': 8, 'w': 8, 'cin': 5120, 'cout': 1280, 'k': 1, 's': 1, 'pin': (5120, 0), 'pout': (1280, 0), 'act': 'none', 'res': (1280, 0)}
#[cfg(feature = "un_c1_8_5120_1280_r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 13115392;
        act: workspace, bytes = 3522560;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2785280;
    const A_R: u64 = 3153920;
    const W_W: u64 = 0;
    const W_B: u64 = 13107200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_8_5120_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_5120_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_5120_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_5120_1280_r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_8_2560_1280：conv2d {'h': 8, 'w': 8, 'cin': 2560, 'cout': 1280, 'k': 3, 's': 1, 'pin': (2560, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c3_8_2560_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 58990592;
        act: workspace, bytes = 1761280;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1392640;
    const W_W: u64 = 0;
    const W_B: u64 = 58982400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_8_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_8_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_8_2560_1280：conv2d {'h': 8, 'w': 8, 'cin': 2560, 'cout': 1280, 'k': 1, 's': 1, 'pin': (2560, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c1_8_2560_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 6561792;
        act: workspace, bytes = 1761280;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1392640;
    const W_W: u64 = 0;
    const W_B: u64 = 6553600;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_8_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_8_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_16_2560_1280：conv2d {'h': 16, 'w': 16, 'cin': 2560, 'cout': 1280, 'k': 3, 's': 1, 'pin': (2560, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c3_16_2560_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 58990592;
        act: workspace, bytes = 3727360;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2703360;
    const W_W: u64 = 0;
    const W_B: u64 = 58982400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_16_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_16_2560_1280：conv2d {'h': 16, 'w': 16, 'cin': 2560, 'cout': 1280, 'k': 1, 's': 1, 'pin': (2560, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c1_16_2560_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 6561792;
        act: workspace, bytes = 3727360;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2703360;
    const W_W: u64 = 0;
    const W_B: u64 = 6553600;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_16_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_2560_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_16_1920_1280：conv2d {'h': 16, 'w': 16, 'cin': 1920, 'cout': 1280, 'k': 3, 's': 1, 'pin': (1920, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c3_16_1920_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 44244992;
        act: workspace, bytes = 3051520;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2027520;
    const W_W: u64 = 0;
    const W_B: u64 = 44236800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_16_1920_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1920_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1920_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_16_1920_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_16_1920_1280：conv2d {'h': 16, 'w': 16, 'cin': 1920, 'cout': 1280, 'k': 1, 's': 1, 'pin': (1920, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c1_16_1920_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4923392;
        act: workspace, bytes = 3051520;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2027520;
    const W_W: u64 = 0;
    const W_B: u64 = 4915200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_16_1920_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1920_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1920_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_16_1920_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_32_1280_1280：conv2d {'h': 32, 'w': 32, 'cin': 1280, 'cout': 1280, 'k': 3, 's': 1, 'pin': (1280, 0), 'pout': (1280, 0), 'act': 'none'}
#[cfg(feature = "un_c3_32_1280_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 29499392;
        act: workspace, bytes = 6963200;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3645440;
    const W_W: u64 = 0;
    const W_B: u64 = 29491200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_32_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_32_1920_640：conv2d {'h': 32, 'w': 32, 'cin': 1920, 'cout': 640, 'k': 3, 's': 1, 'pin': (1920, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c3_32_1920_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 22122496;
        act: workspace, bytes = 7127040;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 5468160;
    const W_W: u64 = 0;
    const W_B: u64 = 22118400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_32_1920_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1920_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1920_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1920_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_32_1920_640：conv2d {'h': 32, 'w': 32, 'cin': 1920, 'cout': 640, 'k': 1, 's': 1, 'pin': (1920, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c1_32_1920_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 2461696;
        act: workspace, bytes = 7127040;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 5468160;
    const W_W: u64 = 0;
    const W_B: u64 = 2457600;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_32_1920_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_1920_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_1920_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_1920_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_32_1280_640：conv2d {'h': 32, 'w': 32, 'cin': 1280, 'cout': 640, 'k': 3, 's': 1, 'pin': (1280, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c3_32_1280_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 14749696;
        act: workspace, bytes = 5304320;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3645440;
    const W_W: u64 = 0;
    const W_B: u64 = 14745600;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_32_1280_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1280_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1280_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_1280_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_32_1280_640：conv2d {'h': 32, 'w': 32, 'cin': 1280, 'cout': 640, 'k': 1, 's': 1, 'pin': (1280, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c1_32_1280_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1642496;
        act: workspace, bytes = 5304320;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3645440;
    const W_W: u64 = 0;
    const W_B: u64 = 1638400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_32_1280_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_1280_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_1280_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_1280_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_32_960_640：conv2d {'h': 32, 'w': 32, 'cin': 960, 'cout': 640, 'k': 3, 's': 1, 'pin': (960, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c3_32_960_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 11063296;
        act: workspace, bytes = 4395008;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2736128;
    const W_W: u64 = 0;
    const W_B: u64 = 11059200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_32_960_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_960_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_960_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_32_960_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_32_960_640：conv2d {'h': 32, 'w': 32, 'cin': 960, 'cout': 640, 'k': 1, 's': 1, 'pin': (960, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c1_32_960_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1232896;
        act: workspace, bytes = 4395008;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2736128;
    const W_W: u64 = 0;
    const W_B: u64 = 1228800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_32_960_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_960_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_960_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_32_960_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_64_640_640：conv2d {'h': 64, 'w': 64, 'cin': 640, 'cout': 640, 'k': 3, 's': 1, 'pin': (640, 0), 'pout': (640, 0), 'act': 'none'}
#[cfg(feature = "un_c3_64_640_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 7376896;
        act: workspace, bytes = 12001280;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 6082560;
    const W_W: u64 = 0;
    const W_B: u64 = 7372800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_64_640_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_640_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_640_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_640_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_64_960_320：conv2d {'h': 64, 'w': 64, 'cin': 960, 'cout': 320, 'k': 3, 's': 1, 'pin': (960, 0), 'pout': (320, 0), 'act': 'none'}
#[cfg(feature = "un_c3_64_960_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 5533696;
        act: workspace, bytes = 12087296;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 9125888;
    const W_W: u64 = 0;
    const W_B: u64 = 5529600;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_64_960_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_960_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_960_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_960_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_64_960_320：conv2d {'h': 64, 'w': 64, 'cin': 960, 'cout': 320, 'k': 1, 's': 1, 'pin': (960, 0), 'pout': (320, 0), 'act': 'none'}
#[cfg(feature = "un_c1_64_960_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 618496;
        act: workspace, bytes = 12087296;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 9125888;
    const W_W: u64 = 0;
    const W_B: u64 = 614400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_64_960_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_960_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_960_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_960_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_64_640_320：conv2d {'h': 64, 'w': 64, 'cin': 640, 'cout': 320, 'k': 3, 's': 1, 'pin': (640, 0), 'pout': (320, 0), 'act': 'none'}
#[cfg(feature = "un_c3_64_640_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3690496;
        act: workspace, bytes = 9043968;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 6082560;
    const W_W: u64 = 0;
    const W_B: u64 = 3686400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_64_640_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_640_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_640_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_640_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c1_64_640_320：conv2d {'h': 64, 'w': 64, 'cin': 640, 'cout': 320, 'k': 1, 's': 1, 'pin': (640, 0), 'pout': (320, 0), 'act': 'none'}
#[cfg(feature = "un_c1_64_640_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 413696;
        act: workspace, bytes = 9043968;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 6082560;
    const W_W: u64 = 0;
    const W_B: u64 = 409600;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c1_64_640_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_640_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_640_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c1_64_640_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_c3_64_320_32：conv2d {'h': 64, 'w': 64, 'cin': 320, 'cout': 32, 'k': 3, 's': 1, 'pin': (320, 0), 'pout': (32, 0), 'act': 'none'}
#[cfg(feature = "un_c3_64_320_32")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 188416;
        act: workspace, bytes = 3342336;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3043328;
    const W_W: u64 = 0;
    const W_B: u64 = 184320;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_c3_64_320_32(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_32(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_32(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_c3_64_320_32(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_gs_64_320：gn_stats {'h': 64, 'w': 64, 'c': 320, 'p': 320}
#[cfg(feature = "un_gs_64_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 2969600;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 2961408;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_64_320(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_320(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_320(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_320(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_32_320：gn_stats {'h': 32, 'w': 32, 'c': 320, 'p': 320}
#[cfg(feature = "un_gs_32_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 839680;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 831488;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_32_320(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_320(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_320(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_320(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_32_640：gn_stats {'h': 32, 'w': 32, 'c': 640, 'p': 640}
#[cfg(feature = "un_gs_32_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 1671168;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 1658880;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_32_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_16_640：gn_stats {'h': 16, 'w': 16, 'c': 640, 'p': 640}
#[cfg(feature = "un_gs_16_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 524288;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 512000;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_16_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_16_1280：gn_stats {'h': 16, 'w': 16, 'c': 1280, 'p': 1280}
#[cfg(feature = "un_gs_16_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 1044480;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 1024000;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_16_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_8_1280：gn_stats {'h': 8, 'w': 8, 'c': 1280, 'p': 1280}
#[cfg(feature = "un_gs_8_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 389120;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 368640;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_8_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_8_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_8_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_8_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_8_2560：gn_stats {'h': 8, 'w': 8, 'c': 2560, 'p': 2560}
#[cfg(feature = "un_gs_8_2560")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 778240;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 737280;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_8_2560(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_8_2560(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_8_2560(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_8_2560(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_16_2560：gn_stats {'h': 16, 'w': 16, 'c': 2560, 'p': 2560}
#[cfg(feature = "un_gs_16_2560")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 2088960;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 2048000;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_16_2560(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_2560(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_2560(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_2560(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_16_1920：gn_stats {'h': 16, 'w': 16, 'c': 1920, 'p': 1920}
#[cfg(feature = "un_gs_16_1920")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 1568768;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 1536000;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_16_1920(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_1920(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_1920(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_16_1920(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_32_1920：gn_stats {'h': 32, 'w': 32, 'c': 1920, 'p': 1920}
#[cfg(feature = "un_gs_32_1920")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 5009408;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 4976640;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_32_1920(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_1920(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_1920(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_1920(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_32_1280：gn_stats {'h': 32, 'w': 32, 'c': 1280, 'p': 1280}
#[cfg(feature = "un_gs_32_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 3338240;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 3317760;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_32_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_1280(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_32_960：gn_stats {'h': 32, 'w': 32, 'c': 960, 'p': 960}
#[cfg(feature = "un_gs_32_960")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 2506752;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 2490368;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_32_960(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_960(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_960(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_32_960(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_64_960：gn_stats {'h': 64, 'w': 64, 'c': 960, 'p': 960}
#[cfg(feature = "un_gs_64_960")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 8896512;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 8880128;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_64_960(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_960(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_960(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_960(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_gs_64_640：gn_stats {'h': 64, 'w': 64, 'c': 640, 'p': 640}
#[cfg(feature = "un_gs_64_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 5931008;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 5918720;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gs_64_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::un_gs_64_640(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// un_ga_64_320_silu：gn_apply {'h': 64, 'w': 64, 'c': 320, 'p': 320, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_64_320_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 49152;
        act: workspace, bytes = 5931008;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 2961408;
    const A_Y: u64 = 2969600;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_64_320_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_320_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_320_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_320_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_64_320_none：gn_apply {'h': 64, 'w': 64, 'c': 320, 'p': 320, 'groups': 32, 'eps': 1e-06, 'act': 'none'}
#[cfg(feature = "un_ga_64_320_none")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 49152;
        act: workspace, bytes = 5931008;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 2961408;
    const A_Y: u64 = 2969600;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_64_320_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_320_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_320_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_320_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_32_320_silu：gn_apply {'h': 32, 'w': 32, 'c': 320, 'p': 320, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_32_320_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 49152;
        act: workspace, bytes = 1671168;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 831488;
    const A_Y: u64 = 839680;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_32_320_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_320_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_320_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_320_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_32_640_silu：gn_apply {'h': 32, 'w': 32, 'c': 640, 'p': 640, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_32_640_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 90112;
        act: workspace, bytes = 3330048;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 1658880;
    const A_Y: u64 = 1671168;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_32_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_32_640_none：gn_apply {'h': 32, 'w': 32, 'c': 640, 'p': 640, 'groups': 32, 'eps': 1e-06, 'act': 'none'}
#[cfg(feature = "un_ga_32_640_none")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 90112;
        act: workspace, bytes = 3330048;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 1658880;
    const A_Y: u64 = 1671168;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_32_640_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_640_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_640_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_640_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_16_640_silu：gn_apply {'h': 16, 'w': 16, 'c': 640, 'p': 640, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_16_640_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 90112;
        act: workspace, bytes = 1036288;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 512000;
    const A_Y: u64 = 524288;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_16_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_16_1280_silu：gn_apply {'h': 16, 'w': 16, 'c': 1280, 'p': 1280, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_16_1280_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 180224;
        act: workspace, bytes = 2068480;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 1024000;
    const A_Y: u64 = 1044480;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    const W_M: u64 = 16384;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_16_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_16_1280_none：gn_apply {'h': 16, 'w': 16, 'c': 1280, 'p': 1280, 'groups': 32, 'eps': 1e-06, 'act': 'none'}
#[cfg(feature = "un_ga_16_1280_none")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 180224;
        act: workspace, bytes = 2068480;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 1024000;
    const A_Y: u64 = 1044480;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    const W_M: u64 = 16384;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_16_1280_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1280_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1280_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1280_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_8_1280_silu：gn_apply {'h': 8, 'w': 8, 'c': 1280, 'p': 1280, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_8_1280_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 180224;
        act: workspace, bytes = 757760;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 368640;
    const A_Y: u64 = 389120;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    const W_M: u64 = 16384;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_8_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_8_1280_none：gn_apply {'h': 8, 'w': 8, 'c': 1280, 'p': 1280, 'groups': 32, 'eps': 1e-06, 'act': 'none'}
#[cfg(feature = "un_ga_8_1280_none")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 180224;
        act: workspace, bytes = 757760;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 368640;
    const A_Y: u64 = 389120;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    const W_M: u64 = 16384;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_8_1280_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_1280_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_1280_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_1280_none(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_8_2560_silu：gn_apply {'h': 8, 'w': 8, 'c': 2560, 'p': 2560, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_8_2560_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 352256;
        act: workspace, bytes = 1515520;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 737280;
    const A_Y: u64 = 778240;
    const W_W: u64 = 0;
    const W_B: u64 = 12288;
    const W_M: u64 = 24576;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_8_2560_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_2560_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_2560_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_8_2560_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_16_2560_silu：gn_apply {'h': 16, 'w': 16, 'c': 2560, 'p': 2560, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_16_2560_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 352256;
        act: workspace, bytes = 4136960;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 2048000;
    const A_Y: u64 = 2088960;
    const W_W: u64 = 0;
    const W_B: u64 = 12288;
    const W_M: u64 = 24576;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_16_2560_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_2560_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_2560_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_2560_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_16_1920_silu：gn_apply {'h': 16, 'w': 16, 'c': 1920, 'p': 1920, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_16_1920_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 262144;
        act: workspace, bytes = 3104768;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 1536000;
    const A_Y: u64 = 1568768;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    const W_M: u64 = 16384;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_16_1920_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1920_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1920_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_16_1920_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_32_1920_silu：gn_apply {'h': 32, 'w': 32, 'c': 1920, 'p': 1920, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_32_1920_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 262144;
        act: workspace, bytes = 9986048;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 4976640;
    const A_Y: u64 = 5009408;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    const W_M: u64 = 16384;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_32_1920_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_1920_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_1920_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_1920_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_32_1280_silu：gn_apply {'h': 32, 'w': 32, 'c': 1280, 'p': 1280, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_32_1280_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 180224;
        act: workspace, bytes = 6656000;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 3317760;
    const A_Y: u64 = 3338240;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    const W_M: u64 = 16384;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_32_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_1280_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_32_960_silu：gn_apply {'h': 32, 'w': 32, 'c': 960, 'p': 960, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_32_960_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 131072;
        act: workspace, bytes = 4997120;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 2490368;
    const A_Y: u64 = 2506752;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_32_960_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_960_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_960_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_32_960_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_64_960_silu：gn_apply {'h': 64, 'w': 64, 'c': 960, 'p': 960, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_64_960_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 131072;
        act: workspace, bytes = 17776640;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 8880128;
    const A_Y: u64 = 8896512;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_64_960_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_960_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_960_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_960_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ga_64_640_silu：gn_apply {'h': 64, 'w': 64, 'c': 640, 'p': 640, 'groups': 32, 'eps': 1e-05, 'act': 'silu'}
#[cfg(feature = "un_ga_64_640_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 90112;
        act: workspace, bytes = 11849728;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 5918720;
    const A_Y: u64 = 5931008;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ga_64_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ga_64_640_silu(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ln_64_320：ln_nhwc {'h': 64, 'w': 64, 'c': 320, 'p': 320, 'eps': 1e-05}
#[cfg(feature = "un_ln_64_320")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 8192;
        act: workspace, bytes = 5922816;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2961408;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ln_64_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_64_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_64_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_64_320(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ln_32_640：ln_nhwc {'h': 32, 'w': 32, 'c': 640, 'p': 640, 'eps': 1e-05}
#[cfg(feature = "un_ln_32_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 8192;
        act: workspace, bytes = 3317760;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1658880;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ln_32_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_32_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_32_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_32_640(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ln_16_1280：ln_nhwc {'h': 16, 'w': 16, 'c': 1280, 'p': 1280, 'eps': 1e-05}
#[cfg(feature = "un_ln_16_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 16384;
        act: workspace, bytes = 2048000;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1024000;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ln_16_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_16_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_16_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_16_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_ln_8_1280：ln_nhwc {'h': 8, 'w': 8, 'c': 1280, 'p': 1280, 'eps': 1e-05}
#[cfg(feature = "un_ln_8_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 16384;
        act: workspace, bytes = 737280;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 368640;
    const W_W: u64 = 0;
    const W_B: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_ln_8_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_8_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_8_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_ln_8_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_at_64_s：attention {'rows': 4736, 'heads': 8, 'head': 64, 'scale': 40, 'cores': 2, 'mask': 'keys'}
#[cfg(feature = "un_at_64_s")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 12288;
        act: workspace, bytes = 19398656;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 4849664;
    const A_V: u64 = 9699328;
    const A_O: u64 = 14548992;
    const W_M: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_at_64_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_64_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_64_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_64_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// un_at_64_x：attention {'rows': 4736, 'keys': 128, 'heads': 8, 'head': 64, 'scale': 40, 'cores': 2, 'mask': 'keys'}
#[cfg(feature = "un_at_64_x")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4096;
        act: workspace, bytes = 9961472;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 4849664;
    const A_V: u64 = 4980736;
    const A_O: u64 = 5111808;
    const W_M: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_at_64_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_64_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_64_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_64_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// un_at_32_s：attention {'rows': 1408, 'heads': 8, 'head': 96, 'scale': 80, 'cores': 2, 'mask': 'keys'}
#[cfg(feature = "un_at_32_s")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4096;
        act: workspace, bytes = 8650752;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 2162688;
    const A_V: u64 = 4325376;
    const A_O: u64 = 6488064;
    const W_M: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_at_32_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_32_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_32_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_32_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// un_at_32_x：attention {'rows': 1408, 'keys': 128, 'heads': 8, 'head': 96, 'scale': 80, 'cores': 2, 'mask': 'keys'}
#[cfg(feature = "un_at_32_x")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4096;
        act: workspace, bytes = 4718592;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 2162688;
    const A_V: u64 = 2359296;
    const A_O: u64 = 2555904;
    const W_M: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_at_32_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_32_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_32_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_32_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// un_at_16_s：attention {'rows': 512, 'heads': 8, 'head': 160, 'scale': 160, 'cores': 2, 'mask': 'keys'}
#[cfg(feature = "un_at_16_s")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4096;
        act: workspace, bytes = 5242880;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 1310720;
    const A_V: u64 = 2621440;
    const A_O: u64 = 3932160;
    const W_M: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_at_16_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_16_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_16_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_16_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// un_at_16_x：attention {'rows': 512, 'keys': 128, 'heads': 8, 'head': 160, 'scale': 160, 'cores': 2, 'mask': 'keys'}
#[cfg(feature = "un_at_16_x")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4096;
        act: workspace, bytes = 3276800;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 1310720;
    const A_V: u64 = 1638400;
    const A_O: u64 = 1966080;
    const W_M: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_at_16_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_16_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_16_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_16_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// un_at_8_s：attention {'rows': 256, 'heads': 8, 'head': 160, 'scale': 160, 'cores': 2, 'mask': 'keys'}
#[cfg(feature = "un_at_8_s")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4096;
        act: workspace, bytes = 2621440;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 655360;
    const A_V: u64 = 1310720;
    const A_O: u64 = 1966080;
    const W_M: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_at_8_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_8_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_8_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_8_s(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// un_at_8_x：attention {'rows': 256, 'keys': 128, 'heads': 8, 'head': 160, 'scale': 160, 'cores': 2, 'mask': 'keys'}
#[cfg(feature = "un_at_8_x")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4096;
        act: workspace, bytes = 1966080;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 655360;
    const A_V: u64 = 983040;
    const A_O: u64 = 1310720;
    const W_M: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_at_8_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_8_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_8_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::un_at_8_x(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), w().at(W_M).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// un_l16_384_1280：linear {'rows': 16, 'k': 384, 'n': 1280, 'w': 'hf', 'cores': 2, 'bias': 'f32'}
#[cfg(feature = "un_l16_384_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 991232;
        act: workspace, bytes = 53248;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 12288;
    const W_W: u64 = 0;
    const W_B: u64 = 983040;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_l16_384_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_384_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_384_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_384_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_l16_1280_1280：linear {'rows': 16, 'k': 1280, 'n': 1280, 'w': 'hf', 'cores': 2, 'bias': 'f32'}
#[cfg(feature = "un_l16_1280_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3284992;
        act: workspace, bytes = 81920;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 40960;
    const W_W: u64 = 0;
    const W_B: u64 = 3276800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_l16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_1280_1280(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_l16_1280_20224：linear {'rows': 16, 'k': 1280, 'n': 20224, 'w': 'hf', 'cores': 2, 'bias': 'f32'}
#[cfg(feature = "un_l16_1280_20224")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 51855360;
        act: workspace, bytes = 688128;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 40960;
    const W_W: u64 = 0;
    const W_B: u64 = 51773440;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_l16_1280_20224(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_1280_20224(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_1280_20224(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_l16_1280_20224(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_kv_512：linear {'rows': 128, 'k': 768, 'n': 512, 'w': 'hf', 'cores': 2}
#[cfg(feature = "un_kv_512")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 786432;
        act: workspace, bytes = 327680;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 196608;
    const W_W: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_kv_512(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_512(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_512(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_512(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_lr_32_640_640_b：linear {'rows': 1408, 'k': 640, 'n': 640, 'w': 'hf', 'cores': 2, 'bias': 'f32'}
#[cfg(feature = "un_lr_32_640_640_b")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 823296;
        act: workspace, bytes = 3604480;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1802240;
    const W_W: u64 = 0;
    const W_B: u64 = 819200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_lr_32_640_640_b(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_640_b(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_640_b(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_640_b(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_lr_32_640_768：linear {'rows': 1408, 'k': 640, 'n': 768, 'w': 'hf', 'cores': 2}
#[cfg(feature = "un_lr_32_640_768")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 983040;
        act: workspace, bytes = 3964928;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1802240;
    const W_W: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_lr_32_640_768(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_768(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_768(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_768(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_kv_768：linear {'rows': 128, 'k': 768, 'n': 768, 'w': 'hf', 'cores': 2}
#[cfg(feature = "un_kv_768")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1179648;
        act: workspace, bytes = 393216;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 196608;
    const W_W: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_kv_768(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_768(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_768(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_768(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_lr_32_640_2560_b：linear {'rows': 1408, 'k': 640, 'n': 2560, 'w': 'hf', 'cores': 2, 'bias': 'f32'}
#[cfg(feature = "un_lr_32_640_2560_b")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 3289088;
        act: workspace, bytes = 9011200;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1802240;
    const W_W: u64 = 0;
    const W_B: u64 = 3276800;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_lr_32_640_2560_b(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_2560_b(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_2560_b(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_lr_32_640_2560_b(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_kv_1280：linear {'rows': 128, 'k': 768, 'n': 1280, 'w': 'hf', 'cores': 2}
#[cfg(feature = "un_kv_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1966080;
        act: workspace, bytes = 524288;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 196608;
    const W_W: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_kv_1280(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_1280(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_1280(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_kv_1280(act().at(A_X).cast(), w().at(W_W).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_gg_64：geglu {'n': 5918720}
#[cfg(feature = "un_gg_64")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 35512320;
    }
    const A_H: u64 = 0;
    const A_G: u64 = 11837440;
    const A_Y: u64 = 23674880;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gg_64(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_64(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_64(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_64(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_gg_32：geglu {'n': 3317760}
#[cfg(feature = "un_gg_32")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 19906560;
    }
    const A_H: u64 = 0;
    const A_G: u64 = 6635520;
    const A_Y: u64 = 13271040;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gg_32(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_32(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_32(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_32(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_gg_16：geglu {'n': 2048000}
#[cfg(feature = "un_gg_16")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 12288000;
    }
    const A_H: u64 = 0;
    const A_G: u64 = 4096000;
    const A_Y: u64 = 8192000;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gg_16(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_16(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_16(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_16(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_gg_8：geglu {'n': 737280}
#[cfg(feature = "un_gg_8")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 4423680;
    }
    const A_H: u64 = 0;
    const A_G: u64 = 1474560;
    const A_Y: u64 = 2949120;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_gg_8(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_8(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_8(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_gg_8(act().at(A_H).cast(), act().at(A_G).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_up_8_1280_256：upsample2 {'h': 8, 'w': 8, 'c': 256, 'pin': 1280, 'pout': 1280}
#[cfg(feature = "un_up_8_1280_256")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 2048000;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 696320;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_up_8_1280_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_8_1280_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_8_1280_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_8_1280_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_up_16_1280_256：upsample2 {'h': 16, 'w': 16, 'c': 256, 'pin': 1280, 'pout': 1280}
#[cfg(feature = "un_up_16_1280_256")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 4997120;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1351680;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_up_16_1280_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_16_1280_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_16_1280_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_16_1280_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_up_32_640_256：upsample2 {'h': 32, 'w': 32, 'c': 256, 'pin': 640, 'pout': 640}
#[cfg(feature = "un_up_32_640_256")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 7905280;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1822720;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_up_32_640_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_32_640_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_32_640_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_32_640_256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_up_32_640_128：upsample2 {'h': 32, 'w': 32, 'c': 128, 'pin': 640, 'pout': 640}
#[cfg(feature = "un_up_32_640_128")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 7905280;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1822720;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_up_32_640_128(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_32_640_128(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_32_640_128(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_up_32_640_128(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_8_1280_2560：copy_win {'h': 8, 'w': 8, 'c': 1280, 'pin': 1280, 'pout': 2560}
#[cfg(feature = "un_cp_8_1280_2560")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 1105920;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 368640;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_8_1280_2560(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_8_1280_2560(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_8_1280_2560(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_8_1280_2560(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_16_1280_2560：copy_win {'h': 16, 'w': 16, 'c': 1280, 'pin': 1280, 'pout': 2560}
#[cfg(feature = "un_cp_16_1280_2560")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 3072000;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1024000;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_16_1280_2560(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_1280_2560(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_1280_2560(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_1280_2560(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_16_1280_1920：copy_win {'h': 16, 'w': 16, 'c': 1280, 'pin': 1280, 'pout': 1920}
#[cfg(feature = "un_cp_16_1280_1920")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 2560000;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1024000;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_16_1280_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_1280_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_1280_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_1280_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_16_640_1920：copy_win {'h': 16, 'w': 16, 'c': 640, 'pin': 640, 'pout': 1920}
#[cfg(feature = "un_cp_16_640_1920")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 2048000;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 512000;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_16_640_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_640_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_640_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_16_640_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_32_1280_1920：copy_win {'h': 32, 'w': 32, 'c': 1280, 'pin': 1280, 'pout': 1920}
#[cfg(feature = "un_cp_32_1280_1920")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 8294400;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 3317760;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_32_1280_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_1280_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_1280_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_1280_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_32_640_1920：copy_win {'h': 32, 'w': 32, 'c': 640, 'pin': 640, 'pout': 1920}
#[cfg(feature = "un_cp_32_640_1920")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 6635520;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1658880;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_32_640_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_1920(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_32_640_1280：copy_win {'h': 32, 'w': 32, 'c': 640, 'pin': 640, 'pout': 1280}
#[cfg(feature = "un_cp_32_640_1280")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 4976640;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1658880;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_32_640_1280(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_1280(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_1280(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_1280(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_32_640_960：copy_win {'h': 32, 'w': 32, 'c': 640, 'pin': 640, 'pout': 960}
#[cfg(feature = "un_cp_32_640_960")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 4149248;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 1658880;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_32_640_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_640_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_32_320_960：copy_win {'h': 32, 'w': 32, 'c': 320, 'pin': 320, 'pout': 960}
#[cfg(feature = "un_cp_32_320_960")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 3321856;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 831488;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_32_320_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_320_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_320_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_32_320_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_64_640_960：copy_win {'h': 64, 'w': 64, 'c': 640, 'pin': 640, 'pout': 960}
#[cfg(feature = "un_cp_64_640_960")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 14798848;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 5918720;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_64_640_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_640_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_640_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_640_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_64_320_960：copy_win {'h': 64, 'w': 64, 'c': 320, 'pin': 320, 'pout': 960}
#[cfg(feature = "un_cp_64_320_960")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 11841536;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2961408;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_64_320_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_320_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_320_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_320_960(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_cp_64_320_640：copy_win {'h': 64, 'w': 64, 'c': 320, 'pin': 320, 'pout': 640}
#[cfg(feature = "un_cp_64_320_640")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 8880128;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 2961408;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_cp_64_320_640(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_320_640(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_320_640(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_cp_64_320_640(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_temb：timestep_embed {'half': 160}
#[cfg(feature = "un_temb")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4096;
        act: workspace, bytes = 8192;
    }
    const A_T: u64 = 0;
    const A_Y: u64 = 4096;
    const W_FP: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_temb(act().at(A_T).cast(), w().at(W_FP).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_temb(act().at(A_T).cast(), w().at(W_FP).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_temb(act().at(A_T).cast(), w().at(W_FP).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_temb(act().at(A_T).cast(), w().at(W_FP).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_silu：unary {'n': 20480, 'f': 'silu'}
#[cfg(feature = "un_silu")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 81920;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 40960;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_silu(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_silu(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_silu(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_silu(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// un_bf：bias_fold {'n': 20224}
#[cfg(feature = "un_bf")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 81920;
        act: workspace, bytes = 122880;
    }
    const A_B: u64 = 0;
    const A_Y: u64 = 40960;
    const W_A: u64 = 0;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::un_bf(w().at(W_A).cast(), act().at(A_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_bf(w().at(W_A).cast(), act().at(A_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_bf(w().at(W_A).cast(), act().at(A_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::un_bf(w().at(W_A).cast(), act().at(A_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}
