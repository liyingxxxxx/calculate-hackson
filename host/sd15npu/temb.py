"""时间步嵌入 kernel（`sd_kernels::timestep_embed!`）的主机侧：常数表与 t 的打法。kernel 的排法见 crates/kernels/src/temb.rs。

- 常数表（模型常数，和权重一样打进 @w）：F = [f, f, 0…]、P = [0…, −π/2…, 0…]，各 N = ⌈2·half / 128⌉·128 个 f32，先 F 后 P；
  fᵢ 照 diffusers `get_timestep_embedding` 的写法在 torch 里用 f32 算（exp(−ln(10000) · i / half)，downscale_freq_shift = 0），
  这样设备上的 t · fᵢ 和 torch 乘的是同一个 f32。
- t：一行 32 B，8 份同一个 f32。
"""
from __future__ import annotations

import math

import numpy as np


def n_padded(half: int) -> int:
    return -(-2 * half // 128) * 128


def tables(half: int) -> bytes:
    import torch

    n = n_padded(half)
    f = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).numpy()
    F = np.zeros(n, np.float32)
    P = np.zeros(n, np.float32)
    F[:half], F[half:2 * half] = f, f
    P[half:2 * half] = np.float32(-math.pi / 2)
    return F.astype("<f4").tobytes() + P.astype("<f4").tobytes()


def t_row(t: float) -> bytes:
    return np.full(8, t, np.float32).astype("<f4").tobytes()


def reference(t: float, half: int) -> np.ndarray:
    """diffusers 的 f32 参照（SD1.5 UNet 的配置：flip_sin_to_cos = True、freq_shift = 0），[2·half] f32。"""
    import torch
    from diffusers.models.embeddings import get_timestep_embedding

    return get_timestep_embedding(torch.tensor([t], dtype=torch.float32), 2 * half, flip_sin_to_cos=True, downscale_freq_shift=0)[0].numpy()
