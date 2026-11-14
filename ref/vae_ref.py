"""正式 VAE 解码器（SD1.5 的 AutoencoderKL，B1b）的独立参照（CPU、torch）：照 diffusers 的模块结构逐层走，不读本仓的 layout / kernel。

两种：
- `decode_fp32`：diffusers 原样（`vae.decode`，fp32），看画质的基准；
- `decode_bf16`：同一结构，按设备的数值口径（DESIGN D5）在每个「kernel 出口」舍 bf16。kernel 怎么切照 DESIGN D13 / D14：
  - 卷积（含 1×1 的 post_quant_conv、shortcut、to_out）：权重舍 bf16、f32 累加 + bias（f32）⇒ 舍；带残差的 ⇒ 舍(舍(conv) + r)；
  - GroupNorm：输入是 bf16，均值、方差、归一、仿射全在 f32（w、b 不舍）⇒ 舍；带 SiLU 的再 舍(silu(·))；
  - 注意力：q、k、v 各是一个 `linear!`（权重舍 bf16、f32 + bias ⇒ 舍）；注意力本身按 torch 语义在 f32 里算（softmax(q·kᵀ/√512)·v）⇒ 舍。
    设备上 kernel 内部还会把分数、p 舍成 bf16（calblas `attention!` 的做法），这里不学——那是 kernel 内部的事，算进容差；
  - 最近邻上采样、排布转换、转置是纯搬运，不舍。
  它和 `host/sd15npu/cvxsim.py`（numpy、按内存字节重放编出来的程序）是两份独立实现：两边对上了，接线和打包才算对。

输出口径：[3, 512, 512] f32，**约 [0, 1]**（diffusers 的 x 在 [−1, 1]，这里换成 (x + 1) / 2，不夹）。
设备上把这一步折进 conv_out：权重 × 0.5、bias × 0.5 + 0.5（× 0.5 在 bf16 里是精确的），所以 bf16 参照在 [0, 1] 的口径上舍——
同一个数学，舍入点照设备来；fp32 参照不受影响。

测试输入（`make_latent`，都在 **VAE 原空间**，即 ComfyUI 的 LATENT、diffusers `vae.decode` 吃的那个，DESIGN F12）：
- `image`：TAESD 那张合成测试图（`taesd_ref.test_image`）经本 VAE 的**编码器**（取 latent 分布的均值）⇒ 分布是真的，解码还能和原图比；
- `randn:<种子>`：UNet 空间的标准正态 ÷ 0.18215（采样器出口大致是这个量级）；
- `zeros`、`big`（UNet 空间 ±8 ÷ 0.18215，远超正常范围的边界情形）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taesd_ref import metrics, test_image, to_png  # noqa: E402,F401（metrics、to_png 给调用方用）

SCALING = 0.18215  # SD1.5 VAE 的 scaling_factor（config.json）


def load_vae(path: Path):
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(str(path), torch_dtype=torch.float32).eval()


@torch.inference_mode()
def make_latent(vae, kind: str) -> np.ndarray:
    """[4, 64, 64] 的 VAE 原空间 latent。"""
    if kind == "image":
        img = torch.from_numpy(test_image())[None]
        return vae.encode(img * 2 - 1).latent_dist.mean[0].numpy().astype(np.float32)
    if kind.startswith("randn:"):
        z = np.random.default_rng(int(kind.split(":", 1)[1])).standard_normal((4, 64, 64))
        return (z / SCALING).astype(np.float32)
    if kind == "zeros":
        return np.zeros((4, 64, 64), np.float32)
    if kind == "big":
        return (np.sign(np.random.default_rng(1).standard_normal((4, 64, 64))) * 8 / SCALING).astype(np.float32)
    raise ValueError(f"认不得的 latent：{kind}")


@torch.inference_mode()
def decode_fp32(vae, lat: np.ndarray) -> np.ndarray:
    z = torch.from_numpy(np.asarray(lat, np.float32))[None]
    return ((vae.decode(z).sample[0] + 1) / 2).numpy()


def _b(t: torch.Tensor) -> torch.Tensor:
    """舍到 bf16 再回 f32（rne：torch 的 .to(bfloat16) 就是 rne）。"""
    return t.to(torch.bfloat16).to(torch.float32)


def _conv(m: torch.nn.Conv2d, x, res=None, w_scale: float = 1.0, b_shift: float = 0.0):
    """卷积 kernel 的出口：bf16 权重、f32 累加 + f32 bias ⇒ 舍；有残差再 舍(· + r)。w_scale / b_shift 给 conv_out 折 (x + 1) / 2 用。"""
    w = _b(m.weight * w_scale)
    b = m.bias * w_scale + b_shift if m.bias is not None else None
    y = _b(F.conv2d(x, w, b, stride=m.stride, padding=m.padding))
    return y if res is None else _b(y + res)


def _gn(m: torch.nn.GroupNorm, x, silu: bool):
    """group_norm! 的出口：f32 里归一、仿射（w、b 是 f32，不舍）⇒ 舍；带 SiLU 的再 舍(silu(·))。"""
    y = _b(F.group_norm(x, m.num_groups, m.weight, m.bias, m.eps))
    return _b(F.silu(y)) if silu else y


def _linear(m: torch.nn.Linear, x):
    """linear!（多行、HF 权重、f32 bias）的出口：f32 累加 + bias ⇒ 舍。"""
    return _b(F.linear(x, _b(m.weight), m.bias))


def _resnet(r, x):
    assert r.time_emb_proj is None and r.output_scale_factor == 1 and isinstance(r.nonlinearity, torch.nn.SiLU)
    h = _conv(r.conv1, _gn(r.norm1, x, silu=True))
    sc = _conv(r.conv_shortcut, x) if r.conv_shortcut is not None else x
    # dropout 推理时是恒等；conv2 带残差（shortcut 之后的 x）
    return _conv(r.conv2, _gn(r.norm2, h, silu=True), res=sc)


def _attn(a, x):
    assert a.heads == 1 and a.residual_connection and a.rescale_output_factor == 1 and a.spatial_norm is None
    _, c, hh, ww = x.shape
    t = _gn(a.group_norm, x, silu=False)[0].reshape(c, hh * ww).T  # [token, c]：转置是纯搬运
    q, k, v = _linear(a.to_q, t), _linear(a.to_k, t), _linear(a.to_v, t)
    s = (q @ k.T) * (1.0 / np.sqrt(c))
    o = _b(torch.softmax(s, dim=-1) @ v)  # [token, c]
    o = o.T.reshape(1, c, hh, ww)
    # to_out 在设备上是 1×1 卷积（带残差），数学同 Linear：权重 [c_out, c_in] ⇒ [c_out, c_in, 1, 1]
    lin = a.to_out[0]
    y = _b(F.conv2d(o, _b(lin.weight)[:, :, None, None], lin.bias))
    return _b(y + x)


@torch.inference_mode()
def decode_bf16(vae, lat: np.ndarray) -> np.ndarray:
    d = vae.decoder
    x = _b(torch.from_numpy(np.asarray(lat, np.float32))[None])  # 主机把 latent 打成 bf16
    x = _conv(vae.post_quant_conv, x)
    x = _conv(d.conv_in, x)
    mb = d.mid_block
    assert len(mb.resnets) == 2 and len(mb.attentions) == 1
    x = _resnet(mb.resnets[0], x)
    x = _attn(mb.attentions[0], x)
    x = _resnet(mb.resnets[1], x)
    for u in d.up_blocks:
        for r in u.resnets:
            x = _resnet(r, x)
        if u.upsamplers is not None:
            up = u.upsamplers[0]
            assert up.interpolate and up.use_conv and not up.use_conv_transpose
            x = _conv(up.conv, F.interpolate(x, scale_factor=2, mode="nearest"))
    x = _gn(d.conv_norm_out, x, silu=True)
    # conv_out 折进 (x + 1) / 2：权重 × 0.5、bias × 0.5 + 0.5（同设备的打包，见模块文档）
    return _conv(d.conv_out, x, w_scale=0.5, b_shift=0.5)[0].numpy()


def main():
    """自检：每种 latent 出两份参照，比 bf16 参照对 fp32、fp32 对原图（image），图写到 out/vae-ref/<latent>/。"""
    import argparse
    import time

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="正式 VAE 解码器的两份独立参照（不上板）")
    ap.add_argument("--latent", default="image,randn:0,zeros,big")
    # 权重目录同 host/sd15npu/runner.py 的 models_dir()：缺省 CALCULET_HOME 的上一层的 models/，可用 SD15NPU_MODELS 改
    ap.add_argument("--models", default=os.environ.get("SD15NPU_MODELS", str(root.parent.parent / "models")))
    a = ap.parse_args()
    vae = load_vae(Path(a.models) / "stable-diffusion-v1-5" / "vae")
    for kind in a.latent.split(","):
        lat = make_latent(vae, kind)
        t0 = time.time()
        r32 = decode_fp32(vae, lat)
        t1 = time.time()
        r16 = decode_bf16(vae, lat)
        t2 = time.time()
        m = metrics(r16, r32)
        line = f"{kind}：bf16 参照对 fp32 余弦 {m['cos']:.6f}、最大差 {m['max_abs']:.4f}、PSNR {m['psnr_db']:.1f} dB（fp32 {t1 - t0:.1f} s、bf16 参照 {t2 - t1:.1f} s）"
        if kind == "image":
            line += f"；fp32 对原图 PSNR {metrics(r32, test_image())['psnr_db']:.1f} dB"
        print(line, flush=True)
        out = root / "out" / "vae-ref" / kind.replace(":", "-")
        out.mkdir(parents=True, exist_ok=True)
        to_png(r32, out / "ref_fp32.png")
        to_png(r16, out / "ref_bf16.png")


if __name__ == "__main__":
    main()
