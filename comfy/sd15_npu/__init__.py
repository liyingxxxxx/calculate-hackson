"""ComfyUI 自定义节点：sd15-npu（Calculet NPU 上的 SD1.5）。现在有一个节点：

- **SD15 NPU · TAESD Decode**（`SD15NPU_TAESDDecode`）：LATENT ⇒ IMAGE，用 TAESD 解码器（512² 档位）。
  backend = `npu`（真机，要 SD15NPU_ALLOW_BOARD=1）/ `npu-sim`（CPU 上 numpy 模拟设备程序，不是 NPU）/ `cpu`（diffusers fp32，CPU 兜底）。

装法：`bash comfy/setup_comfyui.sh` 把本目录链进 ComfyUI/custom_nodes；不改 ComfyUI 源码。细节见本目录 README.md。
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
