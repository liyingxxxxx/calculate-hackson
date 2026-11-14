"""SD1.5 UNet（UNet2DConditionModel）一次前向的独立参照（CPU、torch，B3）：照 diffusers 的模块结构逐层走，不读本仓的 layout / kernel。

两种：
- `forward_fp32`：diffusers 原样（`unet(x, t, encoder_hidden_states=ctx).sample`，fp32）；
- `forward_bf16`：同一结构，按设备的数值口径（DESIGN D5）在每个「kernel 出口」舍 bf16。kernel 怎么切照 DESIGN D18（UNet 全程在带边框的 NHWC 图上）：
  - 卷积（3×3、步长 2 的下采样、1×1 的 shortcut / proj_in / proj_out / to_out / q·k·v / 前馈）：权重舍 bf16、f32 累加 + bias（f32）⇒ 舍；带残差的 ⇒ 舍(舍(conv) + r)；
  - 时间步：sin / cos 嵌入（f32 ⇒ 舍）⇒ Linear ⇒ 舍 ⇒ SiLU ⇒ 舍 ⇒ Linear ⇒ 舍 ⇒ 每个 ResNet：SiLU ⇒ 舍 ⇒ time_emb_proj ⇒ 舍，
    **加进 conv1 的 bias**（f32 里 conv1.bias + 舍过的投影；设备上也是这么折的，不另走一遍整图）；
  - GroupNorm（+ SiLU）、LayerNorm：f32 里归一、仿射（w、b 不舍）⇒ 舍；带 SiLU 的再 舍(silu(·))；
  - 注意力：q、k、v 各舍一次；softmax(q·kᵀ / √d)·v 按 torch 语义在 f32 里算 ⇒ 舍（设备 kernel 内部还会把分数、p 舍成 bf16，那是 kernel 内部的事，算进容差）；
    交叉注意力的 k、v 是上下文（77 × 768）过 Linear ⇒ 舍；
  - GEGLU：两个 1×1 卷积分别出值 h 与门 g（各舍一次）⇒ h · gelu(g)（f32 里，精确的 erf 版）⇒ 舍；
  - 上采样、拼接（跳连）是纯搬运，不舍。
  它和设备程序的 numpy 模拟是两份独立实现：两边对上了，接线和打包才算对。

输入：`x` [4, 64, 64]（UNet 空间的 noisy latent）、`t`（可以是小数）、`ctx` [77, 768]（CLIP 的 last_hidden_state）；输出 [4, 64, 64]（预测的噪声）。
测试输入（`make_inputs`）：latent 用标准正态；ctx 用本仓 SD1.5 自带的 CLIP 文本编码器对一句提示词编出来（真分布），也可以给空串（CFG 的 uncond 那一路）。
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_unet(path: Path):
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel.from_pretrained(str(path), torch_dtype=torch.float32).eval()


@torch.inference_mode()
def encode_prompt(sd_dir: Path, prompt: str) -> np.ndarray:
    """CLIP 文本编码（CPU fp32，transformers）：[77, 768]。参照的输入，不是被测的东西。"""
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(str(sd_dir / "tokenizer"))
    te = CLIPTextModel.from_pretrained(str(sd_dir / "text_encoder"), torch_dtype=torch.float32).eval()
    ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
    return te(ids)[0][0].numpy().astype(np.float32)


def make_inputs(sd_dir: Path, seed: int = 0, prompt: str = "a photo of an astronaut riding a horse on mars") -> tuple[np.ndarray, np.ndarray]:
    x = np.random.default_rng(seed).standard_normal((4, 64, 64)).astype(np.float32)
    return x, encode_prompt(sd_dir, prompt)


@torch.inference_mode()
def forward_fp32(unet, x: np.ndarray, t: float, ctx: np.ndarray) -> np.ndarray:
    xs = torch.from_numpy(np.asarray(x, np.float32))[None]
    cs = torch.from_numpy(np.asarray(ctx, np.float32))[None]
    return unet(xs, torch.tensor([float(t)]), encoder_hidden_states=cs).sample[0].numpy()


def _b(t: torch.Tensor) -> torch.Tensor:
    """舍到 bf16 再回 f32（rne）。"""
    return t.to(torch.bfloat16).to(torch.float32)


def _conv(m: torch.nn.Conv2d, x, res=None, bias=None):
    """卷积 kernel 的出口：bf16 权重、f32 累加 + f32 bias ⇒ 舍；有残差再 舍(· + r)。bias 可以换成外面给的（conv1 折了时间步）。"""
    b = m.bias if bias is None else bias
    y = _b(F.conv2d(x, _b(m.weight), b, stride=m.stride, padding=m.padding))
    return y if res is None else _b(y + res)


def _lin1x1(m: torch.nn.Linear, x, res=None):
    """Linear 在设备上当 1×1 卷积（map [1, C, H, W]）：同 _conv。"""
    y = _b(F.conv2d(x, _b(m.weight)[:, :, None, None], m.bias))
    return y if res is None else _b(y + res)


def _gn(m: torch.nn.GroupNorm, x, silu: bool):
    y = _b(F.group_norm(x, m.num_groups, m.weight, m.bias, m.eps))
    return _b(F.silu(y)) if silu else y


def _ln(m: torch.nn.LayerNorm, x):
    """每像素沿通道（x 是 map [1, C, H, W]）。"""
    y = F.layer_norm(x.permute(0, 2, 3, 1), m.normalized_shape, m.weight, m.bias, m.eps).permute(0, 3, 1, 2)
    return _b(y)


def _attn(a, x, ctx_kv=None):
    """x：map [1, C, H, W]（已 LN）；ctx_kv：交叉注意力的 (k, v) [77, C]（已舍）。返回 o（map，已舍），to_out 在外面做。"""
    _, c, h, w = x.shape
    q = _lin1x1(a.to_q, x)[0].reshape(c, h * w).T  # [T, C]
    if ctx_kv is None:
        k = _lin1x1(a.to_k, x)[0].reshape(c, h * w).T
        v = _lin1x1(a.to_v, x)[0].reshape(c, h * w).T
    else:
        k, v = ctx_kv
    d = c // a.heads
    qh, kh, vh = (z.reshape(z.shape[0], a.heads, d).transpose(0, 1) for z in (q, k, v))
    o = torch.softmax(qh @ kh.transpose(1, 2) * (1.0 / math.sqrt(d)), dim=-1) @ vh  # [heads, T, d]
    o = _b(o.transpose(0, 1).reshape(h * w, c))
    return o.T.reshape(1, c, h, w)


def _block(b, x, ctx):
    """BasicTransformerBlock（x 是 map）：x + attn1(LN1 x) ⇒ + attn2(LN2 x, ctx) ⇒ + FF(LN3 x)。"""
    o = _attn(b.attn1, _ln(b.norm1, x))
    x = _lin1x1(b.attn1.to_out[0], o, res=x)
    kv = (_b(F.linear(ctx, _b(b.attn2.to_k.weight))), _b(F.linear(ctx, _b(b.attn2.to_v.weight))))
    o = _attn(b.attn2, _ln(b.norm2, x), ctx_kv=kv)
    x = _lin1x1(b.attn2.to_out[0], o, res=x)
    n3 = _ln(b.norm3, x)
    proj = b.ff.net[0].proj
    inner = proj.out_features // 2
    wh, wg = proj.weight[:inner], proj.weight[inner:]
    bh, bg = proj.bias[:inner], proj.bias[inner:]
    h = _b(F.conv2d(n3, _b(wh)[:, :, None, None], bh))
    g = _b(F.conv2d(n3, _b(wg)[:, :, None, None], bg))
    y = _b(h * F.gelu(g))
    return _lin1x1(b.ff.net[2], y, res=x)


def _t2d(t, x, ctx):
    assert not t.use_linear_projection and len(t.transformer_blocks) == 1
    h = _conv(t.proj_in, _gn(t.norm, x, silu=False))
    h = _block(t.transformer_blocks[0], h, ctx)
    return _conv(t.proj_out, h, res=x)


def _resnet(r, x, temb_act):
    """ResnetBlock2D：conv1 的 bias 折进时间步投影（f32 里加）。temb_act = 舍(silu(emb))。"""
    assert r.time_embedding_norm == "default" and r.output_scale_factor == 1
    tp = _b(F.linear(temb_act, _b(r.time_emb_proj.weight), r.time_emb_proj.bias))[0]
    h = _conv(r.conv1, _gn(r.norm1, x, silu=True), bias=r.conv1.bias + tp)
    sc = _conv(r.conv_shortcut, x) if r.conv_shortcut is not None else x
    return _conv(r.conv2, _gn(r.norm2, h, silu=True), res=sc)


@torch.inference_mode()
def forward_bf16(unet, x: np.ndarray, t: float, ctx: np.ndarray) -> np.ndarray:
    from diffusers.models.embeddings import get_timestep_embedding

    u = unet
    tp = u.time_proj
    e = _b(get_timestep_embedding(torch.tensor([float(t)]), tp.num_channels, flip_sin_to_cos=tp.flip_sin_to_cos,
                                  downscale_freq_shift=tp.downscale_freq_shift))
    te = u.time_embedding
    e = _b(F.linear(e, _b(te.linear_1.weight), te.linear_1.bias))
    e = _b(F.silu(e))
    e = _b(F.linear(e, _b(te.linear_2.weight), te.linear_2.bias))
    ta = _b(F.silu(e))  # 每个 ResNet 都先 SiLU 再投影
    c = _b(torch.from_numpy(np.asarray(ctx, np.float32)))  # 主机把上下文打成 bf16
    h = _b(torch.from_numpy(np.asarray(x, np.float32))[None])
    h = _conv(u.conv_in, h)
    skips = [h]
    for blk in u.down_blocks:
        atts = getattr(blk, "attentions", None)
        for i, r in enumerate(blk.resnets):
            h = _resnet(r, h, ta)
            if atts is not None:
                h = _t2d(atts[i], h, c)
            skips.append(h)
        if blk.downsamplers is not None:
            h = _conv(blk.downsamplers[0].conv, h)
            skips.append(h)
    mb = u.mid_block
    h = _resnet(mb.resnets[0], h, ta)
    h = _t2d(mb.attentions[0], h, c)
    h = _resnet(mb.resnets[1], h, ta)
    for blk in u.up_blocks:
        atts = getattr(blk, "attentions", None)
        for i, r in enumerate(blk.resnets):
            h = torch.cat([h, skips.pop()], dim=1)  # 拼接是纯搬运（设备上靠通道窗口）
            h = _resnet(r, h, ta)
            if atts is not None:
                h = _t2d(atts[i], h, c)
        if blk.upsamplers is not None:
            h = _conv(blk.upsamplers[0].conv, F.interpolate(h, scale_factor=2.0, mode="nearest"))
    h = _gn(u.conv_norm_out, h, silu=True)
    return _conv(u.conv_out, h)[0].numpy()


def metrics(got: np.ndarray, want: np.ndarray) -> dict:
    got, want = np.asarray(got, np.float64).ravel(), np.asarray(want, np.float64).ravel()
    d = got - want
    cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30))
    snr = 10 * math.log10(float(want @ want) / max(float(d @ d), 1e-30))
    return {"max_abs": float(np.abs(d).max()), "rms": float(np.sqrt(np.mean(d * d))), "cos": cos, "snr_db": snr, "finite": bool(np.isfinite(got).all())}


def main():
    """自检：两份参照对比（几组时间步、带提示词与空提示词），打印余弦与信噪比。"""
    import argparse
    import time

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="SD1.5 UNet 一次前向的两份独立参照（不上板）")
    ap.add_argument("--models", default=os.environ.get("SD15NPU_MODELS", str(root.parent.parent / "models")))
    ap.add_argument("--t", default="999,500.5,1")
    a = ap.parse_args()
    sd = Path(a.models) / "stable-diffusion-v1-5"
    unet = load_unet(sd / "unet")
    x, ctx = make_inputs(sd)
    _, ctx0 = make_inputs(sd, prompt="")
    for t in [float(v) for v in a.t.split(",")]:
        for name, c in (("提示词", ctx), ("空提示词", ctx0)):
            t0 = time.time()
            r32 = forward_fp32(unet, x, t, c)
            t1 = time.time()
            r16 = forward_bf16(unet, x, t, c)
            t2 = time.time()
            m = metrics(r16, r32)
            print(f"t = {t:g}、{name}：bf16 参照对 fp32 余弦 {m['cos']:.6f}、信噪比 {m['snr_db']:.1f} dB、最大差 {m['max_abs']:.4f}"
                  f"（fp32 {t1 - t0:.1f} s、bf16 参照 {t2 - t1:.1f} s）", flush=True)


if __name__ == "__main__":
    main()
