//! 真卡上逐个 kernel 计时（**`tools/prof.py gen` 生成，不要手改**）：每个 feature 一个程序，把正式 VAE 的一个 kernel 就地连调
//! 4 遍，输入输出各 64 B、权重全零、@act 不清零（只量时间、不看数）。`base` 只有屏障，扣固定开销用。
#![no_std]
#![allow(dead_code)]

#[path = "../../vae/src/kernels.rs"]
mod vae_kernels;

const N_FEATURES: usize = cfg!(feature = "base") as usize + cfg!(feature = "va_pqc") as usize + cfg!(feature = "va_cin") as usize + cfg!(feature = "va_c64") as usize + cfg!(feature = "va_c64r") as usize + cfg!(feature = "va_o64r") as usize + cfg!(feature = "va_c128") as usize + cfg!(feature = "va_c128r") as usize + cfg!(feature = "va_c256u") as usize + cfg!(feature = "va_c256a") as usize + cfg!(feature = "va_s256") as usize + cfg!(feature = "va_c256") as usize + cfg!(feature = "va_c256r") as usize + cfg!(feature = "va_c512u") as usize + cfg!(feature = "va_c512a") as usize + cfg!(feature = "va_s512") as usize + cfg!(feature = "va_c512") as usize + cfg!(feature = "va_c512r") as usize + cfg!(feature = "va_cout") as usize + cfg!(feature = "va_gs64") as usize + cfg!(feature = "va_ga64") as usize + cfg!(feature = "va_gs128") as usize + cfg!(feature = "va_ga128") as usize + cfg!(feature = "va_gs256w") as usize + cfg!(feature = "va_ga256w") as usize + cfg!(feature = "va_gs256") as usize + cfg!(feature = "va_ga256") as usize + cfg!(feature = "va_gs512w") as usize + cfg!(feature = "va_ga512w") as usize + cfg!(feature = "va_gs512") as usize + cfg!(feature = "va_ga512") as usize + cfg!(feature = "va_gn64n") as usize + cfg!(feature = "va_tc64") as usize + cfg!(feature = "va_th64") as usize + cfg!(feature = "va_tr_ct") as usize + cfg!(feature = "va_tr_tc") as usize + cfg!(feature = "va_qkv") as usize + cfg!(feature = "va_attn") as usize + cfg!(feature = "va_up64") as usize + cfg!(feature = "va_up128") as usize + cfg!(feature = "va_up256") as usize;
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

/// va_pqc：conv2d {'h': 64, 'w': 64, 'cin': 32, 'cout': 32, 'k': 1, 's': 1, 'pin': (32, 0), 'pout': (32, 0), 'act': 'none'}
#[cfg(feature = "va_pqc")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 8192;
        act: workspace, bytes = 606208;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 307200;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_pqc(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_pqc(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_pqc(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_pqc(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_cin：conv2d {'h': 64, 'w': 64, 'cin': 32, 'cout': 512, 'k': 3, 's': 1, 'pin': (32, 0), 'pout': (512, 0), 'act': 'none'}
#[cfg(feature = "va_cin")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 299008;
        act: workspace, bytes = 5042176;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 307200;
    const W_W: u64 = 0;
    const W_B: u64 = 294912;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_cin(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_cin(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_cin(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_cin(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c64：conv2d {'h': 64, 'w': 64, 'cin': 512, 'cout': 512, 'k': 3, 's': 1, 'pin': (512, 0), 'pout': (512, 0), 'act': 'none'}
#[cfg(feature = "va_c64")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4722688;
        act: workspace, bytes = 9601024;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4866048;
    const W_W: u64 = 0;
    const W_B: u64 = 4718592;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c64(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c64(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c64(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c64(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c64r：conv2d {'h': 64, 'w': 64, 'cin': 512, 'cout': 512, 'k': 3, 's': 1, 'pin': (512, 0), 'pout': (512, 0), 'act': 'none', 'res': (512, 0)}
#[cfg(feature = "va_c64r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4722688;
        act: workspace, bytes = 14336000;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4866048;
    const A_R: u64 = 9601024;
    const W_W: u64 = 0;
    const W_B: u64 = 4718592;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c64r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c64r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c64r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c64r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_o64r：conv2d {'h': 64, 'w': 64, 'cin': 512, 'cout': 512, 'k': 1, 's': 1, 'pin': (512, 0), 'pout': (512, 0), 'act': 'none', 'res': (512, 0)}
#[cfg(feature = "va_o64r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 528384;
        act: workspace, bytes = 14336000;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4866048;
    const A_R: u64 = 9601024;
    const W_W: u64 = 0;
    const W_B: u64 = 524288;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_o64r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_o64r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_o64r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_o64r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c128：conv2d {'h': 128, 'w': 128, 'cin': 512, 'cout': 512, 'k': 3, 's': 1, 'pin': (512, 0), 'pout': (512, 0), 'act': 'none'}
#[cfg(feature = "va_c128")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4722688;
        act: workspace, bytes = 35815424;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 17973248;
    const W_W: u64 = 0;
    const W_B: u64 = 4718592;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c128(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c128(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c128(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c128(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c128r：conv2d {'h': 128, 'w': 128, 'cin': 512, 'cout': 512, 'k': 3, 's': 1, 'pin': (512, 0), 'pout': (512, 0), 'act': 'none', 'res': (512, 0)}
#[cfg(feature = "va_c128r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4722688;
        act: workspace, bytes = 53657600;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 17973248;
    const A_R: u64 = 35815424;
    const W_W: u64 = 0;
    const W_B: u64 = 4718592;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c128r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c128r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c128r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c128r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c256u：conv2d {'h': 256, 'w': 256, 'cin': 512, 'cout': 512, 'k': 3, 's': 1, 'pin': (512, 0), 'pout': (512, 0), 'act': 'none'}
#[cfg(feature = "va_c256u")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 4722688;
        act: workspace, bytes = 138575872;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 69353472;
    const W_W: u64 = 0;
    const W_B: u64 = 4718592;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c256u(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256u(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256u(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256u(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c256a：conv2d {'h': 256, 'w': 256, 'cin': 512, 'cout': 256, 'k': 3, 's': 1, 'pin': (512, 0), 'pout': (256, 0), 'act': 'none'}
#[cfg(feature = "va_c256a")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 2363392;
        act: workspace, bytes = 103964672;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 69353472;
    const W_W: u64 = 0;
    const W_B: u64 = 2359296;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c256a(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256a(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256a(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256a(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_s256：conv2d {'h': 256, 'w': 256, 'cin': 512, 'cout': 256, 'k': 1, 's': 1, 'pin': (512, 0), 'pout': (256, 0), 'act': 'none'}
#[cfg(feature = "va_s256")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 266240;
        act: workspace, bytes = 103964672;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 69353472;
    const W_W: u64 = 0;
    const W_B: u64 = 262144;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_s256(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_s256(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_s256(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_s256(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c256：conv2d {'h': 256, 'w': 256, 'cin': 256, 'cout': 256, 'k': 3, 's': 1, 'pin': (256, 0), 'pout': (256, 0), 'act': 'none'}
#[cfg(feature = "va_c256")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1183744;
        act: workspace, bytes = 69287936;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 34676736;
    const W_W: u64 = 0;
    const W_B: u64 = 1179648;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c256(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c256r：conv2d {'h': 256, 'w': 256, 'cin': 256, 'cout': 256, 'k': 3, 's': 1, 'pin': (256, 0), 'pout': (256, 0), 'act': 'none', 'res': (256, 0)}
#[cfg(feature = "va_c256r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1183744;
        act: workspace, bytes = 103899136;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 34676736;
    const A_R: u64 = 69287936;
    const W_W: u64 = 0;
    const W_B: u64 = 1179648;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c256r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c256r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c512u：conv2d {'h': 512, 'w': 512, 'cin': 256, 'cout': 256, 'k': 3, 's': 1, 'pin': (256, 0), 'pout': (256, 0), 'act': 'none'}
#[cfg(feature = "va_c512u")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1183744;
        act: workspace, bytes = 272711680;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 136388608;
    const W_W: u64 = 0;
    const W_B: u64 = 1179648;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c512u(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512u(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512u(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512u(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c512a：conv2d {'h': 512, 'w': 512, 'cin': 256, 'cout': 128, 'k': 3, 's': 1, 'pin': (256, 0), 'pout': (128, 0), 'act': 'none'}
#[cfg(feature = "va_c512a")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 593920;
        act: workspace, bytes = 204550144;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 136388608;
    const W_W: u64 = 0;
    const W_B: u64 = 589824;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c512a(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512a(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512a(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512a(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_s512：conv2d {'h': 512, 'w': 512, 'cin': 256, 'cout': 128, 'k': 1, 's': 1, 'pin': (256, 0), 'pout': (128, 0), 'act': 'none'}
#[cfg(feature = "va_s512")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 69632;
        act: workspace, bytes = 204550144;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 136388608;
    const W_W: u64 = 0;
    const W_B: u64 = 65536;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_s512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_s512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_s512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_s512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c512：conv2d {'h': 512, 'w': 512, 'cin': 128, 'cout': 128, 'k': 3, 's': 1, 'pin': (128, 0), 'pout': (128, 0), 'act': 'none'}
#[cfg(feature = "va_c512")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 299008;
        act: workspace, bytes = 136355840;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 68194304;
    const W_W: u64 = 0;
    const W_B: u64 = 294912;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_c512r：conv2d {'h': 512, 'w': 512, 'cin': 128, 'cout': 128, 'k': 3, 's': 1, 'pin': (128, 0), 'pout': (128, 0), 'act': 'none', 'res': (128, 0)}
#[cfg(feature = "va_c512r")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 299008;
        act: workspace, bytes = 204517376;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 68194304;
    const A_R: u64 = 136355840;
    const W_W: u64 = 0;
    const W_B: u64 = 294912;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_c512r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_c512r(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_R).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_cout：conv2d {'h': 512, 'w': 512, 'cin': 128, 'cout': 32, 'k': 3, 's': 1, 'pin': (128, 0), 'pout': (32, 0), 'act': 'none'}
#[cfg(feature = "va_cout")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 77824;
        act: workspace, bytes = 85237760;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 68194304;
    const W_W: u64 = 0;
    const W_B: u64 = 73728;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_cout(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_cout(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_cout(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_cout(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_gs64：gn_stats {'h': 64, 'w': 64, 'c': 512, 'p': 512}
#[cfg(feature = "va_gs64")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 4743168;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 4734976;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_gs64(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs64(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs64(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs64(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// va_ga64：gn_apply {'h': 64, 'w': 64, 'c': 512, 'p': 512, 'groups': 32, 'eps': 1e-06, 'act': 'silu'}
#[cfg(feature = "va_ga64")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 73728;
        act: workspace, bytes = 9478144;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 4734976;
    const A_Y: u64 = 4743168;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_ga64(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga64(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga64(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga64(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_gs128：gn_stats {'h': 128, 'w': 128, 'c': 512, 'p': 512}
#[cfg(feature = "va_gs128")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 17850368;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 17842176;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_gs128(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs128(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs128(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs128(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// va_ga128：gn_apply {'h': 128, 'w': 128, 'c': 512, 'p': 512, 'groups': 32, 'eps': 1e-06, 'act': 'silu'}
#[cfg(feature = "va_ga128")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 73728;
        act: workspace, bytes = 35692544;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 17842176;
    const A_Y: u64 = 17850368;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_ga128(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga128(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga128(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga128(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_gs256w：gn_stats {'h': 256, 'w': 256, 'c': 512, 'p': 512}
#[cfg(feature = "va_gs256w")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 69230592;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 69222400;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_gs256w(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs256w(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs256w(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs256w(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// va_ga256w：gn_apply {'h': 256, 'w': 256, 'c': 512, 'p': 512, 'groups': 32, 'eps': 1e-06, 'act': 'silu'}
#[cfg(feature = "va_ga256w")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 73728;
        act: workspace, bytes = 138452992;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 69222400;
    const A_Y: u64 = 69230592;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_ga256w(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga256w(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga256w(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga256w(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_gs256：gn_stats {'h': 256, 'w': 256, 'c': 256, 'p': 256}
#[cfg(feature = "va_gs256")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 34615296;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 34611200;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_gs256(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs256(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs256(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs256(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// va_ga256：gn_apply {'h': 256, 'w': 256, 'c': 256, 'p': 256, 'groups': 32, 'eps': 1e-06, 'act': 'silu'}
#[cfg(feature = "va_ga256")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 40960;
        act: workspace, bytes = 69226496;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 34611200;
    const A_Y: u64 = 34615296;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_ga256(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga256(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga256(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga256(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_gs512w：gn_stats {'h': 512, 'w': 512, 'c': 256, 'p': 256}
#[cfg(feature = "va_gs512w")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 136327168;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 136323072;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_gs512w(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs512w(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs512w(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs512w(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// va_ga512w：gn_apply {'h': 512, 'w': 512, 'c': 256, 'p': 256, 'groups': 32, 'eps': 1e-06, 'act': 'silu'}
#[cfg(feature = "va_ga512w")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 40960;
        act: workspace, bytes = 272650240;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 136323072;
    const A_Y: u64 = 136327168;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_ga512w(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga512w(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga512w(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga512w(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_gs512：gn_stats {'h': 512, 'w': 512, 'c': 128, 'p': 128}
#[cfg(feature = "va_gs512")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 68165632;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 68161536;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_gs512(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs512(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs512(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
        crate::vae_kernels::va_gs512(act().at(A_X).cast(), act().at(A_ST).cast());
        sync_cores();
    }
}

/// va_ga512：gn_apply {'h': 512, 'w': 512, 'c': 128, 'p': 128, 'groups': 32, 'eps': 1e-06, 'act': 'silu'}
#[cfg(feature = "va_ga512")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 24576;
        act: workspace, bytes = 136327168;
    }
    const A_X: u64 = 0;
    const A_ST: u64 = 68161536;
    const A_Y: u64 = 68165632;
    const W_W: u64 = 0;
    const W_B: u64 = 4096;
    const W_M: u64 = 8192;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_ga512(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga512(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga512(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_ga512(act().at(A_X).cast(), act().at(A_ST).cast(), w().at(W_W).cast(), w().at(W_B).cast(), w().at(W_M).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_gn64n：group_norm {'hw': 4096, 'c': 512, 'groups': 32, 'eps': 1e-06, 'act': 'none'}
#[cfg(feature = "va_gn64n")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 1048576;
        act: workspace, bytes = 8388608;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4194304;
    const W_W: u64 = 0;
    const W_B: u64 = 524288;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_gn64n(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_gn64n(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_gn64n(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_gn64n(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_tc64：nhwc_to_nchw {'h': 64, 'w': 64, 'c': 512, 'p': 512}
#[cfg(feature = "va_tc64")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 8929280;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4734976;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_tc64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tc64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tc64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tc64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_th64：nchw_to_nhwc {'h': 64, 'w': 64, 'c': 512, 'p': 512}
#[cfg(feature = "va_th64")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 8929280;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4194304;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_th64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_th64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_th64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_th64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_tr_ct：transpose2d {'rows': 512, 'cols': 4096}
#[cfg(feature = "va_tr_ct")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 8388608;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4194304;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_tr_ct(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tr_ct(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tr_ct(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tr_ct(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_tr_tc：transpose2d {'rows': 4096, 'cols': 512}
#[cfg(feature = "va_tr_tc")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 8388608;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4194304;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_tr_tc(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tr_tc(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tr_tc(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_tr_tc(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_qkv：linear {'rows': 4096, 'k': 512, 'n': 512, 'w': 'hf', 'cores': 2, 'bias': 'f32'}
#[cfg(feature = "va_qkv")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        w: weights, bytes = 528384;
        act: workspace, bytes = 8388608;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4194304;
    const W_W: u64 = 0;
    const W_B: u64 = 524288;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_qkv(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_qkv(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_qkv(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_qkv(act().at(A_X).cast(), w().at(W_W).cast(), w().at(W_B).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_attn：attn_wide {'rows': 4096, 'heads': 1, 'head': 512}
#[cfg(feature = "va_attn")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 16777216;
    }
    const A_Q: u64 = 0;
    const A_K: u64 = 4194304;
    const A_V: u64 = 8388608;
    const A_O: u64 = 12582912;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_attn(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::va_attn(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::va_attn(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), act().at(A_O).cast());
        sync_cores();
        crate::vae_kernels::va_attn(act().at(A_Q).cast(), act().at(A_K).cast(), act().at(A_V).cast(), act().at(A_O).cast());
        sync_cores();
    }
}

/// va_up64：upsample2 {'h': 64, 'w': 64, 'c': 256, 'pin': 512, 'pout': 512}
#[cfg(feature = "va_up64")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 22839296;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 4866048;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_up64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up64(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_up128：upsample2 {'h': 128, 'w': 128, 'c': 256, 'pin': 512, 'pout': 512}
#[cfg(feature = "va_up128")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 87326720;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 17973248;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_up128(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up128(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up128(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up128(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}

/// va_up256：upsample2 {'h': 256, 'w': 256, 'c': 256, 'pin': 256, 'pout': 256}
#[cfg(feature = "va_up256")]
mod prog {
    use npu::prelude::*;
    npu::decls! {
        in_x: input, bytes = 64;
        out_y: output, bytes = 64;
        act: workspace, bytes = 171065344;
    }
    const A_X: u64 = 0;
    const A_Y: u64 = 34676736;
    #[npu::program(chiplets = 1, cores = 2)]
    pub fn step() {
        crate::vae_kernels::va_up256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
        crate::vae_kernels::va_up256(act().at(A_X).cast(), act().at(A_Y).cast());
        sync_cores();
    }
}
