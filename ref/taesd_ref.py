"""TAESD 解码器的独立参照（CPU、torch）：照 diffusers `DecoderTiny.layers` 的模块结构逐层走，不读本仓的 layout / kernel。

两种：
- `decode_fp32`：diffusers 原样（fp32），看画质的基准；
- `decode_bf16`：同一结构，但按设备的数值口径（DESIGN D5）在每个「kernel 出口」舍 bf16 ——
  权重舍 bf16、卷积在 f32 里累加 + bias ⇒ 舍 ⇒ ReLU；Block 末尾 舍(舍(conv) + x) ⇒ ReLU；tanh(x/3)·3 ⇒ 舍。
  它和 `host/sd15npu/cvxsim.py`（numpy、按内存字节重放编出来的程序）是两份独立实现：两边对上了，接线和打包才算对。

输出都是「diffusers 的 x · 2 − 1 之前」的图（值域约 [0, 1]，[3, 512, 512] f32），和设备程序的 out_y 同一个口径。

测试输入（`make_latent`）：
- `image`：合成一张 512² 的测试图（色带、圆、细线、文字一样的高频格子），用 TAESD 的**编码器**编成 latent —— 分布是真的，
  解码出来还能和原图比（PSNR）；
- `randn:<种子>`：标准正态（UNet 空间的 latent 大致是单位方差），边界情形用；
- `zeros`、`big`（±8，过 tanh 软夹的饱和段）：边界。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_tiny(path: Path):
    from diffusers import AutoencoderTiny

    return AutoencoderTiny.from_pretrained(str(path), torch_dtype=torch.float32).eval()


def test_image(n: int = 512) -> np.ndarray:
    """合成测试图 [3, n, n]，值在 [0, 1]：低频色带 + 几个圆 + 细线格子（高频）+ 一块棋盘。"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    r = 0.5 + 0.5 * np.sin(2 * math.pi * (x + 0.3 * y))
    g = 0.5 + 0.5 * np.cos(2 * math.pi * (1.5 * y - 0.2 * x))
    b = np.clip(1.2 - 1.5 * np.hypot(x - 0.5, y - 0.5), 0, 1)
    img = np.stack([r, g, b])
    for cx, cy, rad, col in [(0.25, 0.3, 0.12, (1, 0.9, 0.1)), (0.7, 0.65, 0.18, (0.1, 0.2, 0.9)), (0.4, 0.8, 0.08, (0.9, 0.1, 0.2))]:
        m = np.hypot(x - cx, y - cy) < rad
        for c in range(3):
            img[c][m] = col[c]
    lines = ((np.arange(n) // 2) % 8 == 0)
    img[:, 384:448, :][:, :, lines] = 0.05
    cb = (((np.arange(n)[:, None] // 16) + (np.arange(n)[None, :] // 16)) % 2).astype(np.float32)
    img[:, 32:160, 352:480] = cb[32:160, 352:480][None]
    return img.astype(np.float32)


@torch.inference_mode()
def make_latent(vae, kind: str) -> np.ndarray:
    """[4, 64, 64] 的 UNet 空间 latent（TAESD 直接吃的那个）。"""
    if kind == "image":
        img = torch.from_numpy(test_image())[None]
        # AutoencoderTiny.encode 吃 [-1, 1]（diffusers 约定），吐 UNet 空间的 latent（scaling_factor = 1）
        return vae.encode(img * 2 - 1).latents[0].numpy().astype(np.float32)
    if kind.startswith("randn:"):
        return np.random.default_rng(int(kind.split(":", 1)[1])).standard_normal((4, 64, 64)).astype(np.float32)
    if kind == "zeros":
        return np.zeros((4, 64, 64), np.float32)
    if kind == "big":
        return (np.sign(np.random.default_rng(1).standard_normal((4, 64, 64))) * 8).astype(np.float32)
    raise ValueError(f"认不得的 latent：{kind}")


@torch.inference_mode()
def decode_fp32(vae, lat: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.asarray(lat, np.float32))[None]
    # 等价于 DecoderTiny.forward 去掉末尾的 x · 2 − 1
    x = torch.tanh(x / 3) * 3
    return vae.decoder.layers(x)[0].numpy()


def _b(t: torch.Tensor) -> torch.Tensor:
    """舍到 bf16 再回 f32（rne：torch 的 .to(bfloat16) 就是 rne）。"""
    return t.to(torch.bfloat16).to(torch.float32)


@torch.inference_mode()
def decode_bf16(vae, lat: np.ndarray) -> np.ndarray:
    from diffusers.models.unets.unet_2d_blocks import AutoencoderTinyBlock

    def conv(m: torch.nn.Conv2d, x):
        return _b(F.conv2d(x, _b(m.weight), m.bias, stride=m.stride, padding=m.padding))

    x = _b(torch.from_numpy(np.asarray(lat, np.float32))[None])  # 主机把 latent 打成 bf16
    x = _b(torch.tanh(x * np.float32(1 / 3)) * 3)
    for m in vae.decoder.layers:
        if isinstance(m, torch.nn.Conv2d):
            x = conv(m, x)
        elif isinstance(m, torch.nn.ReLU):
            x = torch.relu(x)
        elif isinstance(m, torch.nn.Upsample):
            assert m.mode == "nearest" and m.scale_factor in (2, 2.0)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        elif isinstance(m, AutoencoderTinyBlock):
            c0, r0, c1, r1, c2 = m.conv
            assert isinstance(r0, torch.nn.ReLU) and isinstance(m.skip, torch.nn.Identity)
            t = torch.relu(conv(c0, x))
            t = torch.relu(conv(c1, t))
            x = torch.relu(_b(conv(c2, t) + x))
        else:
            raise TypeError(f"参照没写 {type(m).__name__}")
    return x[0].numpy()


def metrics(got: np.ndarray, want: np.ndarray) -> dict:
    """比两张 [3, H, W] 的图（口径同 out_y：约 [0, 1]）：最大差、均方根差、余弦、PSNR（按 [0, 1] 夹过之后算）。"""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    d = got - want
    cos = float((got.ravel() @ want.ravel()) / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30))
    mse = float(np.mean((np.clip(got, 0, 1) - np.clip(want, 0, 1)) ** 2))
    return {"max_abs": float(np.abs(d).max()), "rms": float(np.sqrt(np.mean(d * d))), "cos": cos,
            "psnr_db": float("inf") if mse == 0 else 10 * math.log10(1 / mse),
            "finite": bool(np.isfinite(got).all())}


def to_png(img: np.ndarray, path: Path):
    from PIL import Image

    a = (np.clip(np.asarray(img, np.float32), 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path)
