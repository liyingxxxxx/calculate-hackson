"""正式 VAE 解码器（分段程序，DESIGN D15 / D17）的打包与解包（偏移全部来自 layout-dump，见 `layout.py`）。

- 权重：SD1.5 的 `vae/diffusion_pytorch_model.safetensors`（AutoencoderKL，diffusers 格式，注意力是老键名 query / key / value / proj_attn）
  ⇒ 每段一个 `param_blk0.bin`（那一段的 @w）：
  - 卷积：torch 的 `[Cout][Cin][k][k]` ⇒ 模板要的 `[ky·k + kx][Cin_pad][Cout_pad]` bf16（rne），bias f32 `[Cout_pad]`；补的通道是 0。
    proj_attn 是 Linear `[Cout][Cin]`，设备上当 1×1 卷积。conv_out 把 diffusers 的 (x + 1) / 2 折进去：权重 × 0.5（bf16 里精确）、bias × 0.5 + 0.5。
  - GroupNorm：NHWC 原生的（DESIGN D16）w、b 就是 `[C]` f32；s2 注意力前面那个（calblas `group_norm!`，NCHW）按它的格式展开：
    NCHW 里每 `cols` 个像素一行，一行一个通道的值、写 8 份 ⇒ `[C · HW / cols][8]`。
  - 组掩码（NHWC GN 用）：每种通道数一份 `[32][C]` f32，第 g 行在第 g 组的 C / 32 个通道上是 1。
  - 注意力的 q、k、v：HF 原样 `[512][512]` bf16（`linear!` 的 `w = hf`）+ bias f32。
- 输入：latent `[4][64][64]`（**VAE 原空间**，即 ComfyUI 的 LATENT、diffusers `vae.decode` 吃的那个）⇒ `[68][68][32]` bf16（NHWC、两圈 0 边框、通道 4..31 是 0）。
- 段与段之间：上一段的 out_y 原样当下一段的 in_x（字节数相同，layout 里有断言）。
- 输出：s6 的 `[516][516][32]` bf16 ⇒ 内部 `[512][512]` 的前 3 个通道 ⇒ f32 `[3][512][512]`，**已经是 [0, 1] 的口径**（不夹）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import bf16
from .layout import load


def layout() -> dict:
    return load("vae")


def _conv_weight(c: dict, sd: dict) -> tuple[np.ndarray, np.ndarray]:
    w = sd[c["key"] + ".weight"].astype(np.float32)
    k, ci, co = c["k"], c["cin"], c["cout"]
    if c["linear"]:
        if w.shape != (co, ci):
            raise ValueError(f"{c['key']}.weight 的形状 {w.shape}，layout 说是 Linear ({co}, {ci})")
        w = w[:, :, None, None]
    if w.shape != (co, ci, k, k):
        raise ValueError(f"{c['key']}.weight 的形状 {w.shape}，layout 说是 ({co}, {ci}, {k}, {k})")
    b = sd[c["key"] + ".bias"].astype(np.float32)
    if c["fold"]:  # (x + 1) / 2 折进 conv_out
        w, b = w * np.float32(0.5), b * np.float32(0.5) + np.float32(0.5)
    t = np.zeros((k * k, c["cin_pad"], c["cout_pad"]), np.float32)
    t[:, :ci, :co] = w.transpose(2, 3, 1, 0).reshape(k * k, ci, co)
    bb = np.zeros(c["cout_pad"], np.float32)
    bb[:co] = b
    return t, bb


def _gn_rows(g: dict, v: np.ndarray) -> np.ndarray:
    """GroupNorm 的 w 或 b（[C] f32）⇒ `[C · HW / cols][8]` f32：第 r 行是 NCHW 里第 r · cols 个元素所在的通道。"""
    hw = g["res"] * g["res"]
    rows = g["c"] * hw // g["cols"]
    ch = (np.arange(rows) * g["cols"]) // hw
    return np.repeat(np.asarray(v, np.float32)[ch][:, None], 8, axis=1)


def pack_stage(L: dict, s: int, sd: dict) -> bytes:
    """第 s 段的 @w（param_blk0.bin）。"""
    st = L["stages"][s]
    blob = bytearray(st["w_bytes"])

    def put(off, data: bytes):
        blob[off:off + len(data)] = data

    for i in range(st["c0"], st["c1"]):
        c = L["convs"][i]
        w, b = _conv_weight(c, sd)
        wb = bf16.bits(w).tobytes()
        assert len(wb) == c["w_len"] and len(b) * 4 == c["b_len"]
        wo, bo = st["convs"][str(i)]
        put(wo, wb)
        put(bo, b.astype("<f4").tobytes())
    for j in range(st["g0"], st["g1"]):
        g = L["gns"][j]
        wo, bo = st["gns"][str(j)]
        for off, suffix in ((wo, ".weight"), (bo, ".bias")):
            v = sd[g["key"] + suffix].astype(np.float32)
            if v.shape != (g["c"],):
                raise ValueError(f"{g['key']}{suffix} 的形状 {v.shape}，layout 说是 ({g['c']},)")
            data = (v if g["nhwc"] else _gn_rows(g, v)).astype("<f4").tobytes()
            assert len(data) == g["len"]
            put(off, data)
    for c, off in st["masks"].items():
        c, g = int(c), L["consts"]["GROUPS"]
        m = (np.arange(c)[None, :] // (c // g) == np.arange(g)[:, None]).astype("<f4")
        put(off, m.tobytes())
    if st["attn"]:
        for (wo, bo), key in zip(st["qkv"], L["qkv_keys"]):
            w = sd[key + ".weight"].astype(np.float32)
            if w.shape != (L["consts"]["DA"], L["consts"]["DA"]):
                raise ValueError(f"{key}.weight 的形状 {w.shape}")
            put(wo, bf16.bits(w).tobytes())  # HF 原样 [N][K]
            put(bo, sd[key + ".bias"].astype("<f4").tobytes())
    return bytes(blob)


def used_keys(L: dict) -> set:
    keys = {c["key"] + s for c in L["convs"] for s in (".weight", ".bias")}
    keys |= {g["key"] + s for g in L["gns"] for s in (".weight", ".bias")}
    keys |= {k + s for k in L["qkv_keys"] for s in (".weight", ".bias")}
    return keys


def write_params(L: dict, safetensors_path: Path, out_root: Path) -> list[Path]:
    """各段的参数块 ⇒ `<out_root>/<段名>/param_blk0.bin`（+ sha256）。解码器的权重一个都不能漏打。"""
    from safetensors.numpy import load_file

    sd = load_file(str(safetensors_path))
    left = sorted(k for k in sd if (k.startswith("decoder.") or k.startswith("post_quant_conv.")) and k not in used_keys(L))
    if left:
        raise ValueError(f"解码器的这些权重没打进参数块：{left}")
    out = []
    for st in L["stages"]:
        d = out_root / st["name"]
        d.mkdir(parents=True, exist_ok=True)
        data = pack_stage(L, st["s"], sd)
        p = d / "param_blk0.bin"
        p.write_bytes(data)
        (d / "param_blk0.sha256").write_text(hashlib.sha256(data).hexdigest() + "  param_blk0.bin\n")
        out.append(p)
    return out


def pack_latent(L: dict, lat: np.ndarray) -> bytes:
    k = L["consts"]
    n, b, cin = k["LAT"], k["B"], k["CIN0"]
    lat = np.asarray(lat, np.float32)
    if lat.shape != (4, n, n):
        raise ValueError(f"latent 要 [4, {n}, {n}]（这份程序是 {8 * n}² 的档位），给的是 {list(lat.shape)}")
    x = np.zeros((n + 2 * b, n + 2 * b, cin), np.float32)
    x[b:b + n, b:b + n, :4] = lat.transpose(1, 2, 0)
    data = bf16.bits(x).tobytes()
    assert len(data) == L["stages"][0]["in_bytes"]
    return data


def unpack_image(L: dict, out: bytes) -> np.ndarray:
    k = L["consts"]
    n, b, co = 8 * k["LAT"], k["B"], k["COUT"]
    if len(out) < L["stages"][-1]["out_bytes"]:
        raise ValueError(f"out_y 要 {L['stages'][-1]['out_bytes']} 字节，读回 {len(out)}")
    y = np.frombuffer(out[:L["stages"][-1]["out_bytes"]], "<u2").reshape(n + 2 * b, n + 2 * b, co)
    return bf16.from_bits(y[b:b + n, b:b + n, :3]).transpose(2, 0, 1).copy()
