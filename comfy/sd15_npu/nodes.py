"""节点实现。执行器在本仓 host/sd15npu/runner.py（命令行 tools/taesd.py 用的是同一份），这里只做 ComfyUI 的约定换算：

- ComfyUI 的 LATENT 在 VAE 的原空间（KSampler 出口做过 `process_latent_out`，即 ÷ 0.18215）；TAESD（diffusers 格式，scaling_factor = 1）
  吃的是 UNet 空间 ⇒ 喂进去之前 × 0.18215。ComfyUI 自己加载 taesd 时也是这么设的（nodes.py `load_taesd`：vae_scale = 0.18215）。
- 解码器出口约 [0, 1]；ComfyUI 的 IMAGE 是 [B, H, W, 3]、[0, 1] ⇒ 夹到 [0, 1]、换成 NHWC。
  （ComfyUI 内置 TAESD 是「[0,1] ⇒ ×2−1 ⇒ VAE 的 (x+1)/2 ⇒ 夹」，绕一圈结果相同。）
- 只做 SD1.5 的 latent（4 通道）、512×512（latent 64×64）；别的尺寸点名拒绝（DESIGN D4：形状是编译期常量，按档位加）。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("SD15NPU_ROOT") or Path(__file__).resolve().parents[2])
if str(ROOT / "host") not in sys.path:
    sys.path.insert(0, str(ROOT / "host"))

from sd15npu import runner  # noqa: E402

SD15_SCALE = 0.18215
"""comfy/latent_formats.py `SD15.scale_factor`。"""

_decoders: dict = {}


class TAESDDecodeNPU:
    DESCRIPTION = ("SD1.5 的 TAESD 解码（512×512）。npu：在 Calculet NPU 上跑整步程序（要 SD15NPU_ALLOW_BOARD=1）；"
                   "npu-sim：CPU 上 numpy 模拟同一份设备程序（不是 NPU，演示与查接线用）；cpu：diffusers fp32（CPU 兜底，会计数）。")
    CATEGORY = "sd15-npu"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "backend": (list(runner.BACKENDS), {"default": os.environ.get("SD15NPU_BACKEND", "npu-sim")}),
        }}

    def decode(self, samples, backend):
        lat = samples["samples"]
        if lat.ndim != 4 or lat.shape[1] != 4:
            raise ValueError(f"sd15-npu：要 SD1.5 的 4 通道 latent [B, 4, H, W]，给的是 {list(lat.shape)}")
        h, w = lat.shape[2], lat.shape[3]
        if (h, w) != (64, 64):
            raise ValueError(f"sd15-npu：TAESD 的 NPU 程序现在只有 512×512 的档位（latent 64×64），这个是 {w * 8}×{h * 8}")
        dec = _decoders.get(backend) or _decoders.setdefault(backend, runner.TaesdDecoder(backend))
        out = []
        for b in range(lat.shape[0]):
            img, info = dec.decode((lat[b].to(torch.float32) * SD15_SCALE).cpu().numpy())
            logging.getLogger("sd15_npu").info(
                "[sd15-npu] TAESD 解码 第 %d/%d 张：backend=%s，%.2f s%s", b + 1, lat.shape[0], info["backend"], info["seconds"],
                f"，dort 墙钟 {info['dort_wall_s']:.3f} s" if "dort_wall_s" in info else "")
            out.append(torch.from_numpy(np.clip(img, 0.0, 1.0).transpose(1, 2, 0).copy()))
        return (torch.stack(out),)


class VAEDecodeNPU:
    """正式 VAE（AutoencoderKL）解码，分段设备程序（DESIGN D15 / D17）。ComfyUI 的 LATENT 本来就在 VAE 原空间 ⇒ 原样喂，不乘 0.18215。"""

    DESCRIPTION = ("SD1.5 的正式 VAE 解码（512×512，画质同 ComfyUI 自带的 VAEDecode）。npu：在 Calculet NPU 上依次跑各段整步程序"
                   "（要 SD15NPU_ALLOW_BOARD=1）；npu-sim：CPU 上 numpy 模拟同一份设备程序（不是 NPU，一张约半分钟）；cpu：diffusers fp32（CPU 兜底，会计数）。")
    CATEGORY = "sd15-npu"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "backend": (list(runner.BACKENDS), {"default": os.environ.get("SD15NPU_BACKEND", "npu-sim")}),
        }}

    def decode(self, samples, backend):
        lat = samples["samples"]
        if lat.ndim != 4 or lat.shape[1] != 4:
            raise ValueError(f"sd15-npu：要 SD1.5 的 4 通道 latent [B, 4, H, W]，给的是 {list(lat.shape)}")
        h, w = lat.shape[2], lat.shape[3]
        if (h, w) != (64, 64):
            raise ValueError(f"sd15-npu：正式 VAE 的 NPU 程序现在只有 512×512 的档位（latent 64×64），这个是 {w * 8}×{h * 8}")
        key = ("vae", backend)
        dec = _decoders.get(key) or _decoders.setdefault(key, runner.VaeDecoder(backend))
        out = []
        for b in range(lat.shape[0]):
            img, info = dec.decode(lat[b].to(torch.float32).cpu().numpy())
            logging.getLogger("sd15_npu").info(
                "[sd15-npu] VAE 解码 第 %d/%d 张：backend=%s，%.2f s%s", b + 1, lat.shape[0], info["backend"], info["seconds"],
                f"，dort 墙钟合计 {info['dort_wall_s']:.3f} s（各段 {info['stages']}）" if "dort_wall_s" in info else "")
            out.append(torch.from_numpy(np.clip(img, 0.0, 1.0).transpose(1, 2, 0).copy()))
        return (torch.stack(out),)


NODE_CLASS_MAPPINGS = {"SD15NPU_TAESDDecode": TAESDDecodeNPU, "SD15NPU_VAEDecode": VAEDecodeNPU}
NODE_DISPLAY_NAME_MAPPINGS = {"SD15NPU_TAESDDecode": "SD15 NPU · TAESD Decode", "SD15NPU_VAEDecode": "SD15 NPU · VAE Decode"}
