"""解码器的三种执行器（ComfyUI 节点与 `tools/taesd.py`、`tools/vae.py` 共用）：TAESD（`TaesdDecoder`）、正式 VAE（`VaeDecoder`，分段程序，DESIGN D15 / D17）。

| backend | 谁算 | 说明 |
| --- | --- | --- |
| `cpu` | CPU（diffusers fp32） | **CPU 兜底**：每用一次计一次数、打出来（AGENTS.md 硬规矩 5） |
| `npu-sim` | CPU 上的 numpy 模拟 | 跑的是设备程序编出来的 `.cvx`（同一份偏移、同一份参数块），**不是 NPU**：查接线、演示整条链，结果按模拟标 |
| `npu` | NPU（dort） | 要 `SD15NPU_ALLOW_BOARD=1` 和带真权重的 calbin（`tools/taesd.py calbin`）；没开就报错，**不静默落回 CPU** |

TAESD 吃 UNet 空间的 latent `[4, 64, 64]`（ComfyUI 的 LATENT 要先 × 0.18215，见 comfy/sd15_npu）；正式 VAE 吃 VAE 原空间的（ComfyUI 的 LATENT 原样）。
输出都是 `[3, 512, 512]` f32，值域约 [0, 1]。
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from . import ROOT, taesd, vae

BACKENDS = ("cpu", "npu-sim", "npu")
CVX = ROOT / "target" / "dopamine-unknown-none-cvx" / "npu" / "taesd.cvx"
KERNELS_RS = ROOT / "crates" / "taesd" / "src" / "kernels.rs"
PARAMS = ROOT / "out" / "taesd" / "params"
CALBIN = ROOT / "out" / "taesd-real" / "cal"

cpu_fallbacks = 0
"""本进程里落回 CPU 解码的次数（打进日志与节点输出）。"""
log = logging.getLogger("sd15_npu")


def models_dir() -> Path:
    return Path(os.environ.get("SD15NPU_MODELS", ROOT.parent.parent / "models"))


def taesd_dir() -> Path:
    return Path(os.environ.get("SD15NPU_TAESD", models_dir() / "taesd"))


def build_device_pass():
    """设备遍（cargo，几秒）：出 taesd.cvx。要 tools/env.sh 设过的环境（隔离的工具链）。"""
    subprocess.run(["cargo", "build", "-q", "-p", "taesd", "--lib", "--target", "dopamine-unknown-none-cvx"], cwd=ROOT, check=True)


def ensure_params(L: dict) -> Path:
    p = PARAMS / "param_blk0.bin"
    src = taesd_dir() / "diffusion_pytorch_model.safetensors"
    if not p.exists() or p.stat().st_mtime < max(src.stat().st_mtime, (ROOT / "crates/taesd/src/layout.rs").stat().st_mtime):
        taesd.write_params(L, src, PARAMS)
    return p


class TaesdDecoder:
    def __init__(self, backend: str = "cpu"):
        if backend not in BACKENDS:
            raise ValueError(f"backend 要是 {BACKENDS} 之一，给的是 {backend!r}")
        self.backend = backend
        self._vae = None
        self.L = None

    def _cpu(self, lat):
        global cpu_fallbacks
        import sys

        sys.path.insert(0, str(ROOT / "ref"))
        import taesd_ref

        if self._vae is None:
            self._vae = taesd_ref.load_tiny(taesd_dir())
        cpu_fallbacks += 1
        log.warning("[sd15-npu] ⚠ TAESD 解码落在 CPU（diffusers fp32），本进程第 %d 次", cpu_fallbacks)
        return taesd_ref.decode_fp32(self._vae, lat)

    def decode(self, lat: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        info = {"backend": self.backend}
        if self.backend == "cpu":
            img = self._cpu(lat)
        else:
            self.L = self.L or taesd.layout()
            data = taesd.pack_latent(self.L, lat)
            params = ensure_params(self.L)
            if self.backend == "npu-sim":
                from . import cvxsim

                if not CVX.exists() or CVX.stat().st_mtime < max(f.stat().st_mtime for f in (ROOT / "crates").rglob("*.rs")):
                    build_device_pass()
                sim = cvxsim.simulate(CVX, KERNELS_RS, self.L["consts"], {"in_x": data, "w": params.read_bytes()})
                out = sim.mem["out_y"].tobytes()
                info["sim"] = sim.report()
                log.info("[sd15-npu] TAESD 解码：numpy 模拟设备程序（不是 NPU）%s", sim.report())
            else:
                from . import dort

                if not (CALBIN / "param_blk0.bin").exists():
                    raise FileNotFoundError(f"没有带真权重的 calbin：{CALBIN}（先 tools/taesd.py calbin）")
                out, d = dort.run_raw(CALBIN, "taesd", data, "in_x", "out_y", log=ROOT / "out" / "taesd-real" / "dort.log")
                info.update(d)
            img = taesd.unpack_image(self.L, out)
        info["seconds"] = time.time() - t0
        return img, info


# ───────────────────────── 正式 VAE（分段程序，DESIGN D15 / D17）─────────────────────────

VAE_KERNELS_RS = ROOT / "crates" / "vae" / "src" / "kernels.rs"
VAE_PARAMS = ROOT / "out" / "vae" / "params"
VAE_CALBINS = ROOT / "out" / "vae-real"


def vae_dir() -> Path:
    return Path(os.environ.get("SD15NPU_VAE", models_dir() / "stable-diffusion-v1-5" / "vae"))


def vae_stage_cvx(name: str) -> Path:
    """第几段的 .cvx：带 feature 的用自己的 target 目录（同 tools/calbin.sh；DESIGN F10）。"""
    return ROOT / "target" / "features" / f"vae-{name}" / "dopamine-unknown-none-cvx" / "npu" / "vae.cvx"


def build_vae_stage(name: str):
    """设备遍（cargo，几秒）：出第几段的 vae.cvx。要 tools/env.sh 设过的环境（隔离的工具链）。"""
    subprocess.run(["cargo", "build", "-q", "-p", "vae", "--lib", "--target", "dopamine-unknown-none-cvx", "--features", name,
                    "--target-dir", str(ROOT / "target" / "features" / f"vae-{name}")], cwd=ROOT, check=True)


def ensure_vae_params(L: dict) -> list[Path]:
    """各段的参数块；比权重文件或 layout.rs 旧就重打。"""
    src = vae_dir() / "diffusion_pytorch_model.safetensors"
    ps = [VAE_PARAMS / st["name"] / "param_blk0.bin" for st in L["stages"]]
    newest = max(src.stat().st_mtime, (ROOT / "crates/vae/src/layout.rs").stat().st_mtime)
    if not all(p.exists() and p.stat().st_mtime >= newest for p in ps):
        vae.write_params(L, src, VAE_PARAMS)
    return ps


class VaeDecoder:
    """正式 VAE 解码：各段依次跑，上一段的 out_y 原样当下一段的 in_x（段间的特征图经主机转手，不在 CPU 上算）。"""

    def __init__(self, backend: str = "cpu"):
        if backend not in BACKENDS:
            raise ValueError(f"backend 要是 {BACKENDS} 之一，给的是 {backend!r}")
        self.backend = backend
        self._vae = None
        self.L = None

    def _cpu(self, lat):
        global cpu_fallbacks
        import sys

        sys.path.insert(0, str(ROOT / "ref"))
        import vae_ref

        if self._vae is None:
            self._vae = vae_ref.load_vae(vae_dir())
        cpu_fallbacks += 1
        log.warning("[sd15-npu] ⚠ VAE 解码落在 CPU（diffusers fp32），本进程第 %d 次", cpu_fallbacks)
        return vae_ref.decode_fp32(self._vae, lat)

    def decode(self, lat: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        info = {"backend": self.backend}
        if self.backend == "cpu":
            img = self._cpu(lat)
        else:
            self.L = self.L or vae.layout()
            data = vae.pack_latent(self.L, lat)
            params = ensure_vae_params(self.L)
            stages = {}
            if self.backend == "npu-sim":
                from . import cvxsim

                newest = max(f.stat().st_mtime for f in (ROOT / "crates").rglob("*.rs"))
                for st, p in zip(self.L["stages"], params):
                    cvx = vae_stage_cvx(st["name"])
                    if not cvx.exists() or cvx.stat().st_mtime < newest:
                        build_vae_stage(st["name"])
                    sim = cvxsim.simulate(cvx, VAE_KERNELS_RS, self.L["consts"], {"in_x": data, "w": p.read_bytes()})
                    data = sim.mem["out_y"].tobytes()
                    stages[st["name"]] = sim.report()
                    del sim
                log.info("[sd15-npu] VAE 解码：numpy 模拟各段设备程序（不是 NPU）")
            else:
                from . import dort

                for st in self.L["stages"]:
                    cal = VAE_CALBINS / st["name"] / "cal"
                    if not (cal / "param_blk0.bin").exists():
                        raise FileNotFoundError(f"没有带真权重的 calbin：{cal}（先 tools/vae.py calbin）")
                    data, d = dort.run_raw(cal, "vae", data, "in_x", "out_y", log=VAE_CALBINS / "dort.log")
                    stages[st["name"]] = round(d["dort_wall_s"], 3)
                info["dort_wall_s"] = sum(stages.values())
            info["stages"] = stages
            img = vae.unpack_image(self.L, data)
        info["seconds"] = time.time() - t0
        return img, info
