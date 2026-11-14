# 真卡性能（真机实测，2026-11-14，npu3）

所有的数都在 npu3 的一块真卡上跑出来，原始记录在 [`results/npu3-20261114/perf.json`](results/npu3-20261114/perf.json)；
它是从同目录下各次运行的 `metrics.json`、`kernels.json`、`board_summary.json`、ComfyUI 日志汇总来的。
「估算」和「CPU 对照」单独标出来，不和真机实测混在一起。

## 环境与口径

| 项 | 值 |
| --- | --- |
| 卡 | npu3 的 `/dev/calculet0`，PCI 0000:07:00.0，2 个芯粒，PCIe 16 GT/s x4；板上 DRT rt 3.1 / ABI 1.1 |
| 工具链 | rustc 1.98.1-calculet、docc 0.4.1、dort 0.4.1（赛方公开栈） |
| 程序 | 单芯粒、两核（`cores = 2`）；激活 bf16，卷积在 TU 里 f32 累加 |
| 主机 | AMD Ryzen 9 9955HX（16 核 32 线程），Linux x86_64 |
| 计时口径 | **墙钟**：主机上一次 `dort calbin run` 进程从起到退，含 configure、部署参数块、写输入、跑、读回输出。**推理（估算）**：整次墙钟的中位数减去同一个 calbin `--deploy-only` 的墙钟 |

dort 0.4.1 没有常驻会话（DESIGN F1），每次调用都要重新部署。所以这里给的「整次墙钟」就是应用里真实要付的时间。

## 1. TAESD 解码器，512×512（整个解码器是一个设备程序）

计算量 141.4 GFLOPs / 张（35 个卷积、3 次上采样、1 次 tanh）。每次调用写入 0.3 MB 的 latent、读回 17 MB 的图；参数块 2.6 MB。

| 项 | 真机实测 |
| --- | --- |
| 整次墙钟（部署 + 推理 + 读写） | **中位数 0.113 s**（两次上卡共 15 遍：最快 0.111 s，最慢 0.115 s） |
| 只部署（`--deploy-only`，参数块 2.6 MB） | 0.003 s（板上运行时日志：参数块 2 MiB 用时 1 ms） |
| 推理（估算 = 整次 − 只部署） | **约 0.11 s**，折合约 1.29 TFLOPS（估算，含读回 17 MB） |
| 确定性 | 每组都是 3 遍逐字节一致；从 GitHub 重新克隆、从零重建后再跑，出的图和第一次逐字节相同 |
| 正确性（对 bf16 口径的 torch 参照） | 52.7–65.8 dB（四组 latent）；对 diffusers fp32 46.8–59.4 dB |

| 每组 latent | 只部署 | 整次 3 遍 | 推理（估算） |
| --- | --- | --- | --- |
| image（第一次上卡） | 0.003 s | 0.111 / 0.112 / 0.112 s | 0.108 s |
| randn:0 | 0.004 s | 0.115 / 0.114 / 0.114 s | 0.110 s |
| zeros | 0.003 s | 0.112 / 0.112 / 0.113 s | 0.109 s |
| big | 0.004 s | 0.114 / 0.114 / 0.114 s | 0.110 s |
| image（第二次，从 GitHub 克隆后） | 0.003 s | 0.112 / 0.113 / 0.114 s | 0.110 s |

**对照：同一台机器的 CPU**（diffusers fp32，torch 16 线程，5 遍中位数；本机 CPU 实测，不是 NPU 的数）：TAESD 解码 0.185 s。
按整次墙钟比，NPU 快约 **1.6 倍**；这个比较里 NPU 的数还包含了部署和读写。两边精度不同：NPU 是 bf16，CPU 是 fp32。

粗拆一下 0.11 s 花在哪（估算）：单 kernel 测试里，up_256 搬了 8.7 MB 进、34 MB 出，用了 27 ms，折合约 1.6 GB/s；
照这个速度，TAESD 读回 17 MB 约 10 ms，剩下约 0.1 s 是在卡上算，约 1.4 TFLOPS。

## 1b. 正式 VAE 解码器，512×512（3 段程序，主机串起来）

计算量 2514.5 GFLOPs / 张，37 个卷积、30 个 GroupNorm、单头 512 维的注意力，和算子清单里的总数对得上。
docc 每核的代码放不下整个解码器，所以分段：每段一次 `dort calbin run`，上一段读回的特征图原样当下一段的输入（DESIGN F22 / D15 / D17）。
原始记录：优化后在 `results/npu3-20261114/vae-opt/`，第一版在 `results/npu3-20261114/vae/`。

**现在（2026-11-14 17:58，优化后，真机实测）**

| 项 | 真机实测 |
| --- | --- |
| 正确性 | 3 遍逐字节一致；对 bf16 口径的 torch 参照余弦 0.999998、**PSNR 58.3 dB**，对 diffusers fp32 56.9 dB（和优化前一样） |
| ComfyUI 里的出图 | 和 ComfyUI 自带的 VAEDecode（CPU）比 **55.6 dB、最大差 5/255**；节点 **0.99 s**（3 段 dort 墙钟合计 0.977 s） |
| 3 段整次墙钟之和 | **中位数 0.98 s**（3 遍 0.97–0.98 s）：s1 0.02 s、s2（注意力）0.10 s、s3 0.85 s |
| 只部署（3 段参数块共 96 MB） | 合计 0.04 s |
| 推理（估算） | 约 0.94 s，折合约 2.7 TFLOPS |

**对照：同一台机器的 CPU**（diffusers fp32，torch 16 线程，5 遍中位数）：1.71 s。**NPU 快约 1.75 倍**（NPU 的数含部署与读写；精度不同：bf16 对 fp32）。

### 优化的过程（每一步都真机实测、对参照，3 遍逐字节一致）

先用 `tools/prof.py` 在卡上给每个 kernel 单独计时：每个 kernel 一个小程序，就地连调 4 遍，扣掉一次调用的固定开销。
按测出来的大头逐个改：

| 步骤 | 改了什么 | 整条（中位数） |
| --- | --- | --- |
| 第一版（6 段） | GroupNorm 走「排布转换 + calblas group_norm!」，128² 以上的 3×3 卷积走原路 | 5.18 s |
| ① 卷积的列块快路 | 快路放不下整行时按列切块，每块读 K 段进 VB（`sd_kernels::conv2d!`） | 4.51 s |
| ② NHWC 原生 GroupNorm | 统计一遍 + 归一一遍，直接在带边框的图上算，不转排布（`gn_stats!` / `gn_apply!`，DESIGN D16） | 1.23 s |
| ③ 并段（6 段 ⇒ 3 段） | ② 之后代码小了，128² 以上合成一段，大特征图不再经主机转手（D17） | **0.98 s** |

逐 kernel 的前后（真机实测，一遍的时间；`results/npu3-20261114/vae-opt/prof/`）：

| kernel | 优化前 | 优化后 |
| --- | --- | --- |
| 3×3 卷积 512→512 @128² | 44.1 ms（1.75 TFLOPS，原路） | 18.0 ms（4.28 TFLOPS，列块快路） |
| 3×3 卷积 512→512 @256²（上采样后） | 175.9 ms（1.76 TFLOPS） | 70.7 ms（4.38 TFLOPS） |
| 3×3 卷积 128→128 @512² | 29.6 ms（2.62 TFLOPS） | 16.8 ms（4.62 TFLOPS） |
| 3×3 卷积 256→256 @512²（上采样后） | 159.0 ms（1.95 TFLOPS） | 69.6 ms（4.45 TFLOPS） |
| conv_out 128→3（补到 32）@512² | 29.1 ms | 9.8 ms |
| GroupNorm + SiLU @512² × 128 | 244 ms（转换 100 + GN 45 + 转换 99） | 21.6 ms（统计 19.3 + 归一 2.3） |
| GroupNorm + SiLU @512² × 256 | 487 ms（199 + 90 + 198） | 26.5 ms（20.6 + 5.8） |
| GroupNorm + SiLU @256² × 256 | 122 ms | 7.3 ms |
| GroupNorm + SiLU @128² × 512 | 79 ms | 3.0 ms |
| GroupNorm + SiLU @64² × 512 | 15 ms | 1.6 ms |

现在 0.98 s 大致花在（估算）：卷积约 0.59 s（4.2–4.6 TFLOPS），GroupNorm 约 0.24 s（其中 512² 的统计那一遍一次约 20 ms、占大头），
注意力那一段 0.10 s（`attn_wide!` 67 ms，0.51 TFLOPS），其余是清零、上采样与读写。
接下来能抠的：GroupNorm 统计那一遍（512² 只有约 3.5–6.5 GB/s）、`attn_wide!`、把注意力那一段也并进去。

## 1c. SD1.5 UNet 一次前向（CFG 两路，20 段程序，第一版）

2026-11-14 在同一块卡上跑（`tools/unet.py board --t 500 --runs 2`，结果在 `results/npu3-20261114/unet/`）。
输入：latent 是标准正态（种子 0），t = 500；两路的上下文是本仓 SD1.5 的 CLIP 对「a photo of an astronaut riding a horse on mars」和空串编出来的。

| 项 | 值 |
| --- | --- |
| 正确性 | 2 遍逐字节一致；对 bf16 口径的 torch 参照：有提示词 余弦 0.999995、49.8 dB，空提示词 50.1 dB；对 diffusers fp32 49.4 / 48.0 dB |
| 一步（CFG 两路）整次墙钟 | **12.35 s**（20 段各自 `dort calbin run` 的墙钟之和，2 遍的中位） |
| 只部署（20 段合计） | 0.56 s（参数块合计 1.62 GiB，约 3 GB/s） |
| 推理（估算） | 约 11.8 s（整次 − 只部署） |
| 对照：同机 CPU | diffusers fp32 一次前向约 1 s（`ref/unet_ref.py` 里量的，CPU 实测，不是 NPU 的数）⇒ CFG 两路约 2 s。**这一版 NPU 比 CPU 慢约 6 倍** |

时间花在哪（逐 kernel 计时，真机实测：每个 kernel 单独一个程序在卡上连调 4 遍；各 kernel 一次的时间 × 一步里调的次数，合计 11.6 s，和整次对得上）：

| kernel | 一步调几次 | 一次 | 一步合计 | 折合 |
| --- | --- | --- | --- | --- |
| 64² 自注意力（calblas `attention!`，4736 token、8 头 × 64 维） | 10 | 625 ms | **6.25 s** | 0.07 TFLOPS |
| 32² 自注意力 | 10 | 58 ms | 0.58 s | 0.10 TFLOPS |
| 16² 前馈 1×1 卷积 1280 → 5120 | 20 | 21 ms | 0.41 s | 0.16 TFLOPS |
| 16² 3×3 卷积 1280 → 1280（带残差） | 10 | 37 ms | 0.37 s | 0.21 TFLOPS |
| 其余（卷积为主） | | | 约 4 s | 输入 ≥ 1280 通道的卷积走原路，0.1–0.2 TFLOPS |
| 按种类 | | | 注意力 7.19 s、卷积 4.03 s、linear 0.14 s、geglu 0.10 s、LayerNorm 0.06 s、GroupNorm 0.07 s | |

注意力慢在读 K / V 的次数：kernel 每 16 行 q 把每块 128 个 key 的 K、V 各从 DDR 读一遍。先用 `transpose2d!` 把 K 转好再算（`k = transposed`）也是 627 ms
（`prof_attention_k_transposed.json`），说明慢的不是读的方式。下一步换成「一块 K / V 给多块 q 用」的写法（PROGRESS U6）。
每段的时间、部署与参数块大小在 `results/npu3-20261114/unet/breakdown.json`。

## 2. 单 kernel（7 个 kernel、20 个用例，各调一次 dort）

每次调用都要建进程、configure、部署、读写，所以小 kernel 的墙钟基本就是一次调用的固定开销；大的卷积主要花在搬数据上。

| 测试 | kernel | 计算量 | 进 / 出 | 墙钟（各用例） |
| --- | --- | --- | --- | --- |
| temb（5 例） | 时间步嵌入 320 维 | — | 64 B / 768 B | 1.3–1.5 ms（**一次 dort 调用的固定开销约 1.3 ms**） |
| tanh（4 例） | tanh(x/3)·3，64² × 32 | — | 0.3 MB / 0.3 MB | 1.8–2.1 ms |
| conv_in（4 例） | 3×3 卷积 32→64 @64² + ReLU | 0.15 GFLOP | 0.3 MB / 0.6 MB | 2.2–2.5 ms |
| convr_64（4 例） | 3×3 卷积 64→64 @64² + 残差 + ReLU | 0.30 GFLOP | 1.2 MB / 0.6 MB | 2.8–3.2 ms |
| up_256 | 最近邻 ×2，256² → 512²，64 通道 | — | 8.7 MB / 34 MB | 26.7 ms |
| conv_out | 3×3 卷积 64→32 @512² | 9.7 GFLOP | 34 MB / 17 MB | 35.4 ms |
| convr_512 | 3×3 卷积 64→64 @512² + 残差 + ReLU | 19.3 GFLOP | 68 MB / 34 MB | 69.9 ms |

## 3. ComfyUI 原生工作流（512²，euler 20 步，cfg 7.5，seed 42）

解码用 TAESD（`comfy/workflows/sd15_taesd_npu.api.json`）：

| 项 | 第一次（board_check --comfy） | 第二次（从 GitHub 克隆后 run_example.sh --comfy） |
| --- | --- | --- |
| 整条（提交到出图） | 50.1 s（ComfyUI 记的执行 49.3 s） | 50.1 s（48.9 s） |
| 其中 TAESD 解码节点（**NPU**） | 0.12 s（dort 墙钟 0.113 s） | 0.12 s（0.114 s） |
| 其中 CLIP + UNet 20 步（**CPU**） | 约 49 s | 约 49 s |
| 服务启动 | 4.0 s | 4.0 s |

解码用正式 VAE（`comfy/workflows/sd15_vae_npu.api.json`，真机实测）：优化后（17:58）整条 52.1 s（ComfyUI 记的执行 50.1 s），
其中 **SD15 NPU · VAE Decode 节点 0.99 s**（3 段 dort 墙钟合计 0.977 s）；优化前（17:24）节点 5.89 s。CLIP + UNet 20 步在 CPU 上约 49 s。
出的图和旁边 ComfyUI 自带的 VAEDecode 相比是 55.6 dB。

**如实说**：整条工作流里，NPU 现在只做最后的解码。用 TAESD 时约占时间的 0.2%、计算量的约 0.4%；用正式 VAE 时约占时间的 2%、计算量的约 7%。
UNet 20 步 × 2 路（CFG）× 803 GFLOPs ≈ 32 TFLOPs，全在 CPU 上，这是接下来要搬上 NPU 的大头（PROGRESS 的 B1b–B4）。

## 4. 编译（本机实测，不上卡）

| 项 | 用时 |
| --- | --- |
| 设备遍（TAESD，cargo 增量） | < 0.1 s |
| docc 出 calbin（TAESD，带真权重） | 4.5 s |
| 上板检查一条命令（`board_check.py --comfy`，6 步） | 约 1.5 分钟，其中 ComfyUI 那步 54 s |

## 还没量的

- 卡上每个 kernel 的耗时：dort 0.4.1 的 `calbin run` 只给整次的结果，上面只有整次墙钟和据此的估算。
- CLIP 的真卡性能：还没写成设备程序（B2）。UNet 只测了一组输入（t = 500），还没接进 ComfyUI 的采样循环。
- 正式 VAE 只测了一组 latent（image）。
