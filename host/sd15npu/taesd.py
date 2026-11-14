"""TAESD 程序的打包与解包（偏移全部来自 layout-dump，见 `layout.py`）。

- 权重：diffusers 格式的 `diffusion_pytorch_model.safetensors`（madebyollin/taesd）⇒ `param_blk0.bin`（@w）。
  torch 的卷积权重 `[Cout][Cin][ky][kx]` ⇒ 模板要的 `[ky·3 + kx][Cin_pad][Cout_pad]` bf16（rne），bias f32 `[Cout_pad]`；
  补的通道是 0、没有 bias 的卷积写全零。
- 输入：latent `[4][64][64]`（**UNet 空间**，即 diffusers 管线里 `vae.decode` 之前、TAESD 直接吃的那个；scaling_factor = 1）
  ⇒ `[68][68][32]` bf16（NHWC、两圈 0 边框、通道 4..31 是 0）。ComfyUI 的 LATENT 是 VAE 原空间，先 × 0.18215（见 runner.py）。
- 输出：`[516][516][32]` bf16 ⇒ 内部 `[512][512]` 的前 3 个通道 ⇒ f32 `[3][512][512]`，值域约 [0, 1]（diffusers 的 `x · 2 − 1` 之前）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import bf16
from .layout import load


def layout() -> dict:
    return load("taesd")


def pack_params(L: dict, safetensors_path: Path) -> bytes:
    from safetensors.numpy import load_file

    sd = load_file(str(safetensors_path))
    blob = bytearray(L["consts"]["W_BYTES"])
    for c in L["convs"]:
        w = sd[c["key"] + ".weight"].astype(np.float32)  # [cout, cin, 3, 3]
        if w.shape != (c["cout"], c["cin"], 3, 3):
            raise ValueError(f"{c['key']}.weight 的形状 {w.shape}，layout 说是 ({c['cout']}, {c['cin']}, 3, 3)")
        t = np.zeros((9, c["cin_pad"], c["cout_pad"]), np.float32)
        t[:, :c["cin"], :c["cout"]] = w.transpose(2, 3, 1, 0).reshape(9, c["cin"], c["cout"])
        wb = bf16.bits(t).tobytes()
        assert len(wb) == c["w_len"]
        blob[c["w_off"]:c["w_off"] + len(wb)] = wb
        b = np.zeros(c["cout_pad"], np.float32)
        if c["bias"]:
            b[:c["cout"]] = sd[c["key"] + ".bias"].astype(np.float32)
        elif c["key"] + ".bias" in sd:
            raise ValueError(f"{c['key']} 在 layout 里标了没有 bias，权重文件里却有")
        bb = b.astype("<f4").tobytes()
        blob[c["b_off"]:c["b_off"] + len(bb)] = bb
    used = {c["key"] + s for c in L["convs"] for s in (".weight", ".bias")}
    left = sorted(k for k in sd if k.startswith("decoder.") and k not in used)
    if left:
        raise ValueError(f"解码器的这些权重没打进参数块：{left}")
    return bytes(blob)


def write_params(L: dict, safetensors_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "param_blk0.bin"
    data = pack_params(L, safetensors_path)
    p.write_bytes(data)
    (out_dir / "param_blk0.sha256").write_text(hashlib.sha256(data).hexdigest() + "  param_blk0.bin\n")
    return p


def pack_latent(L: dict, lat: np.ndarray) -> bytes:
    k = L["consts"]
    n, b, cin = k["LAT"], k["B"], k["CIN0"]
    lat = np.asarray(lat, np.float32)
    if lat.shape != (4, n, n):
        raise ValueError(f"latent 要 [4, {n}, {n}]（这份程序是 {8 * n}² 的档位），给的是 {list(lat.shape)}")
    x = np.zeros((n + 2 * b, n + 2 * b, cin), np.float32)
    x[b:b + n, b:b + n, :4] = lat.transpose(1, 2, 0)
    data = bf16.bits(x).tobytes()
    assert len(data) == k["IN_BYTES"]
    return data


def unpack_image(L: dict, out: bytes) -> np.ndarray:
    k = L["consts"]
    n, b, co = 8 * k["LAT"], k["B"], k["COUT"]
    if len(out) < k["OUT_BYTES"]:
        raise ValueError(f"out_y 要 {k['OUT_BYTES']} 字节，读回 {len(out)}")
    y = np.frombuffer(out[:k["OUT_BYTES"]], "<u2").reshape(n + 2 * b, n + 2 * b, co)
    return bf16.from_bits(y[b:b + n, b:b + n, :3]).transpose(2, 0, 1).copy()
