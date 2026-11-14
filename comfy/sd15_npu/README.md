# ComfyUI 节点：sd15_npu

ComfyUI v0.45.3 的自定义节点（`bash comfy/setup_comfyui.sh` 把本目录链进 `ComfyUI/custom_nodes/sd15_npu`，不改 ComfyUI 源码）。

## SD15 NPU · TAESD Decode（`SD15NPU_TAESDDecode`）

| 输入 | 说明 |
| --- | --- |
| `samples`（LATENT） | SD1.5 的 latent，`[B, 4, 64, 64]`（512×512）。别的尺寸点名拒绝：设备程序的形状是编译期常量，现在只有这一个档位 |
| `backend` | `npu`：Calculet NPU 上跑整步程序（`out/taesd-real/cal`）；`npu-sim`：CPU 上 numpy 重放同一份设备程序（**不是 NPU**）；`cpu`：diffusers fp32（CPU 兜底） |

输出 IMAGE `[B, 512, 512, 3]`，[0, 1]。缺省后端取环境变量 `SD15NPU_BACKEND`，没设就是 `npu-sim`。

约定换算（ComfyUI v0.45.3 核过）：LATENT 在 VAE 原空间 ⇒ × 0.18215 进 TAESD（UNet 空间）；解码器出 [0, 1] ⇒ 夹一下就是 IMAGE。

### 后端与环境变量

| 变量 | 作用 |
| --- | --- |
| `SD15NPU_BACKEND` | 节点里 backend 的缺省值 |
| `SD15NPU_ALLOW_BOARD=1` | 🚨 允许 `npu` 后端真去调 `dort calbin run`。**负责人确认现场排到了板子才设**；没设时 `npu` 直接报错，不落回 CPU |
| `SD15NPU_DORT_INDEX` / `SD15NPU_BOARD_LOCK` | dort 的设备下标（`-i`）与板锁文件（`--board-lock`），按现场的排队规矩给 |
| `SD15NPU_MODELS` / `SD15NPU_TAESD` | 权重目录（缺省 `<CALCULET_HOME>/../models`、其下的 `taesd`） |

`npu` 后端要先出带真权重的 calbin：`$CALCULET_HOME/venv/bin/python -I tools/taesd.py calbin`。
`npu-sim` 第一次用会自己编设备遍、打参数块（几秒），每张图模拟 6–7 秒（32 核）。`cpu` 每用一次计数、打 warning 日志。

### 工作流

`comfy/workflows/sd15_taesd_npu.api.json`：DiffusersLoader（`models/diffusers/stable-diffusion-v1-5`，setup 脚本链好的）→ CLIPTextEncode ×2 → KSampler
→ **SD15 NPU · TAESD Decode** → SaveImage；旁边一路 ComfyUI 自带的 VAEDecode 作对照。浏览器里拖进去，或无界面：

```bash
$CALCULET_HOME/comfy-venv/bin/python -I comfy/run_workflow.py --steps 20 [--backend npu-sim|npu|cpu]
```

现在只有解码这一段写了 NPU 程序；CLIP、UNet（KSampler）还是 ComfyUI 原生节点在 CPU 上跑（B2–B4 逐段换掉）。
