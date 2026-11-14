# sd15-npu：在 Calculet NPU 上跑 Stable Diffusion 1.5，由 ComfyUI 原生工作流发起

Calculet NPU CalVX 黑客松的参赛工程（路线 B）。算子用 Rust 写（`#[npu::kernel]`，复用 calBLAS 的模板），每个模型编成一份 CalVX 整步程序，
docc 出 calbin，dort 在真机上跑；ComfyUI 侧是自定义节点，不改 ComfyUI 源码。

## 当前状态（2026-11-14，TAESD 与正式 VAE 解码器都已在真卡上跑通）

- **B1（进行中）：TAESD 解码器整个编成一个设备程序**——35 个卷积、3 次上采样、1 次 tanh、入口清零，单芯粒两核。
  设备遍编过，docc 出 calbin（带真权重）4.7 s、`doctor` 拒收 0 条；numpy 照编出来的 `.cvx` 重放（接线检查全过），
  对 bf16 口径的独立参照余弦 0.999999 / PSNR 60.6 dB，对 diffusers fp32 PSNR 55.9 dB（合成测试图的 latent）。
  **真机实测**（2026-11-14，npu3 的卡）：3 遍逐字节一致，对 bf16 参照余弦 0.9999987 / PSNR 60.3 dB，对 fp32 55.9 dB；只部署 0.003 s，推理约 0.11 s。
- **ComfyUI 原生工作流**跑通：DiffusersLoader → CLIPTextEncode → KSampler → **SD15 NPU · TAESD Decode** → SaveImage。
  解码节点在真卡上走 `npu`（dort，**真机实测**：节点 0.12 s，没有落回 CPU）；没有卡时走 `npu-sim`（CPU 上模拟设备程序，**不是 NPU**）。
  CLIP、UNet（KSampler）还在 CPU 上。
- 单 kernel 测试台（7 个测试 × 边界用例）、模拟器自测（7 种接线错都抓得到）、一键重建（在全新目录里从零验过）与一键自检都在。
- **上板一条命令**（`tools/board_check.py`）：先只读查卡，再逐个 kernel、再整个解码器，任何一步不对就停；没卡时用 dort 的测试替身把整套空跑过一遍。
  dort 已经静态解析过我们全部 8 个 calbin，张量名与字节数和主机侧一致。2026-11-14 在 npu3 的真卡上 6 步全过（PROGRESS「上板准备」）。
- A1 的第一个新 kernel：**时间步嵌入**（t 可以是小数，f32 规约 + sin / cos 近似），真机实测过了；板上 `cos_approx` 的分辨率是 1/128（DESIGN F19）。
- **B1b：正式 VAE 解码器**（AutoencoderKL，2.5 TFLOPs / 张）切成几段设备程序（docc 每核的代码放不下整个解码器，DESIGN F22 / D15；优化后 3 段，D17）。
  **真机实测**（2026-11-14，npu3）：3 遍逐字节一致，对 bf16 参照 58.3 dB、对 fp32 56.9 dB；ComfyUI 节点 **SD15 NPU · VAE Decode** 出的图和 ComfyUI 自带的 VAEDecode 相比 55.6 dB。
  按真卡逐 kernel 计时优化之后 **0.98 s**（第一版 5.18 s），**比同机 CPU fp32（1.71 s）快约 1.75 倍**：卷积的列块快路、NHWC 原生 GroupNorm、并成 3 段（`test/PERF.md`）。
- **UNet 一次前向（CFG 两路）也在真卡上跑通**（对 bf16 参照 49.8 / 50.1 dB，2 遍逐字节一致）：第一版一步 12.35 s，注意力换成本仓的 `flash_attn!`、宽通道卷积改走 `linear!`、自注意力压成稠密 token 并一次取 128 行之后 **3.12 s**（5 段），同机 CPU fp32 约 2.4 s，还慢约 30%（`test/PERF.md` 1c）；还没接进 ComfyUI 的采样循环。
- 没做的：CLIP 的设备程序（B2）、UNet 接进 ComfyUI 与去噪循环（B3 U7、B4）。逐项细账见 [PROGRESS.md](PROGRESS.md)。

关键事实：公开工具链的 dort 每次调用都重新部署权重、没有常驻会话，所以去噪循环要放进设备程序里（[docs/DESIGN.md](docs/DESIGN.md) 的 F1 / D2）。

## 在比赛的真卡环境里复现

前提：Linux x86_64；git、curl、tar；内存 ≥ 16 GiB、磁盘 ≥ 20 GiB；网络能到 115.159.82.89（http）、腾讯云 COS、huggingface.co、
download.pytorch.org、pypi.org、github.com、astral.sh、sh.rustup.rs。重建与自检不需要 GPU，也不需要 NPU；上板那一步要一块跑 DeviceRT 的卡。

```bash
# 1. 取代码：放进一个空目录（下面叫 CALCULET_HOME），本仓的目录名必须是 sd15-npu
mkdir -p ~/calculet && cd ~/calculet
git clone https://github.com/liyingxxxxx/calculate-hackson.git sd15-npu

# 2. 从零重建（幂等，断了重跑；约 10 分钟，看网速）：工具链、三个公开仓库、CPU 参照环境、权重（下到 ~/models，即 CALCULET_HOME 的上一层）、
#    ComfyUI，最后跑 tools/verify_all.sh（12 项全 ✓ 才算好）
bash sd15-npu/tools/bootstrap.sh
source ~/calculet/env.sh                  # 以后每个 shell 先来这一句

# 3. 上真卡（确认这块卡现在可以用；板锁文件按现场规矩给）
cd ~/calculet/sd15-npu
bash test/run_example.sh                                    # 先不上卡：npu-sim 跑一遍简单示例
SD15NPU_ALLOW_BOARD=1 SD15NPU_BOARD_LOCK=<板锁文件> bash test/run_example.sh --comfy        # 简单示例 + ComfyUI，在真卡上
SD15NPU_ALLOW_BOARD=1 SD15NPU_BOARD_LOCK=<板锁文件> $CALCULET_HOME/venv/bin/python -I tools/board_check.py --comfy   # 完整上板检查
```

上板的结果应该和 [`test/README.md`](test/README.md) 里 2026-11-14 npu3 那一份对得上：TAESD 对 bf16 参照约 60 dB、3 遍逐字节一致，7 个单 kernel 全过。
卡不满足条件时，脚本只读查完就停，不往卡上发作业：比如板上不是 DeviceRT（dort 报「拒」），要找管卡的人重刷，⛔ 自己刷。

`bootstrap.sh` 依次做这些：
1. 写入口 `env.sh`；
2. 装 uv、rustup（缺了才装，装进隔离目录）；
3. 跑赛方 install.sh：**先核 sha256**，和 2026-11-13 读过的那份一致才执行；
4. 把赛方三个公开仓库切到钉住的提交；
5. 编一遍 calBLAS 的 tiny 设备遍，作自检；
6. 建 CPU 参照环境、下权重（钉 revision、核 sha256）；
7. 装 ComfyUI v0.45.3；
8. 跑 `tools/verify_all.sh`。

装好之后的目录布局：`<CALCULET_HOME>/{sd15-npu, env.sh, rustup, cargo, venv, comfy-venv, ComfyUI, work, logs}`，权重在 `<CALCULET_HOME>/../models`。

每个 shell 先 `source env.sh`：它隔离 RUSTUP_HOME / CARGO_HOME，并把工具链自带的 docc / dort 放到 PATH 最前。机器上另装着老版本（比如 npu3 的 `/usr/local/bin` 里有 dort 0.2.8）也不会用错，
`tools/check_env.sh` 会核。版本（统一复测要对得上）：rustc 1.98.1-calculet（nightly-2026-11-13）、docc 0.4.1、dort 0.4.1；
calblas / caldnn / npu 0.1.0（赛方 cargo 源）；torch 2.15.0+cpu、diffusers 0.43.0；ComfyUI v0.45.3（71096d1a）。

## 常用命令

```bash
source <CALCULET_HOME>/env.sh && cd <CALCULET_HOME>/sd15-npu
PY=$CALCULET_HOME/venv/bin/python

bash tools/verify_all.sh                 # 不上板能做的全部检查（VERIFY_FULL=1 再加 ComfyUI 整条工作流）
$PY -I tools/taesd.py calbin             # TAESD：打包权重 ⇒ 设备遍 ⇒ docc ⇒ out/taesd-real/cal（带真权重）+ doctor
$PY -I tools/taesd.py sim                # TAESD：numpy 重放设备程序、对两份参照 ⇒ out/taesd-sim/image/{sim,ref_fp32,…}.png
$PY -I tools/vae.py calbin               # 正式 VAE：各段参数块 ⇒ 各段 calbin（约 20 s）
$PY -I tools/vae.py sim                  # 正式 VAE：各段依次模拟、对两份参照（约 25 s）⇒ out/vae-sim/image/
SD15NPU_ALLOW_BOARD=1 SD15NPU_BOARD_LOCK=… $PY -I tools/prof.py board   # 🚨 真卡上逐个 kernel 计时（先 tools/prof.py gen、build）
$PY -I ref/ktest.py all                  # 单 kernel 测试台：全部用例模拟 + 出 calbin
bash comfy/run_comfyui.sh                # 起 ComfyUI（http://127.0.0.1:8188），把 comfy/workflows/*.json 拖进去
$CALCULET_HOME/comfy-venv/bin/python -I comfy/run_workflow.py --steps 20    # 无界面跑工作流

# 🚨 上板（负责人确认这块卡可以用才做；缺省一律拒绝）：一条命令跑完，任何一步不对就停
SD15NPU_ALLOW_BOARD=1 [SD15NPU_DORT_INDEX=0] [SD15NPU_BOARD_LOCK=…] $PY -I tools/board_check.py [--comfy]
#   重出 calbin ⇒ 静态核对（tools/preflight.py）⇒ 只读查卡（OS / RT、要 DeviceRT、没别人在用）⇒ 7 个单 kernel 全部用例 ⇒ TAESD（只部署 + 整次 3 遍）
$PY -I ref/test_board_kit.py             # 没卡时：拿 dort 的测试替身把上面整套空跑一遍（verify_all 里也跑）
```

## 目录

| 路径 | 是什么 |
| --- | --- |
| `crates/kernels` | 本仓的 kernel 模板：`conv2d!`（抄 calblas，加 relu / res_relu）、`tanh_scale!`、`timestep_embed!`（新写） |
| `crates/taesd` | TAESD 解码器的整步程序：`layout.rs`（偏移表，设备与主机共用）、`kernels.rs`（18 个特化）、`step.rs`（入口） |
| `crates/vae` | 正式 VAE 解码器的 3 段程序（feature `s1`…`s3`）：`layout.rs`（段表与偏移）、`kernels.rs`（特化）、`step.rs`（3 个入口） |
| `crates/unet` | SD1.5 UNet 的分段程序（现在 5 段，feature `s1`…`s5`）：`kernels.rs`、`step.rs`、`plan.json` 都由 `tools/unet_gen.py` 生成（偏移只算一份），`kernel_sizes.json` 是各 kernel 的代码量 |
| `crates/prof`、`crates/uprof` | 真卡上逐个 kernel 计时的小程序（`tools/prof.py [--model unet] gen` 生成） |
| `crates/ktest` | 单 kernel 测试台：每个 feature 一个只调一个 kernel 的整步程序 |
| `crates/layout-dump` | 主机程序：把 layout.rs 打成 JSON 给 Python 用（偏移只有 Rust 这一份） |
| `crates/smoke` | 冒烟整步程序：验「经 cargo 源依赖 calblas ⇒ CalVX ⇒ calbin」这条路 |
| `host/sd15npu` | 主机侧 Python：bf16、打包 / 解包、`.cvx` 模拟器、dort 包装（缺省不上板）、三种执行器 |
| `ref/` | CPU 参照：环境脚本、算子清单、TAESD 独立参照、单 kernel 测试台、模拟器自测 |
| `comfy/` | ComfyUI：钉版本安装、拉起脚本、自定义节点 `sd15_npu`、工作流 JSON、无界面跑工作流 |
| `tools/` | `bootstrap.sh`（新机器从零）、`verify_all.sh`（全部自检）、`env.sh` / `check_env.sh`（隔离环境）、`calbin.sh`、`taesd.py` |
| `docs/` | `DESIGN.md`（设计与关键事实）、`HANDOFF.md`（交接文档）、`赛题原文.md`、`inventory/`（SD1.5 算子清单，脚本生成） |
| `test/` | 简单示例 `run_example.sh`、真卡性能 `PERF.md`、真卡上跑出来的结果（`results/npu3-20261114/`：图、指标、dort 日志、perf.json） |

## NPU / CPU 分工

TAESD 解码、正式 VAE 解码与 7 个单 kernel 有**真机实测**（2026-11-14，npu3）；别的数是 CPU 上的模拟与参照。照「真机实测 / 模拟 / 估算」分开标。

| 计算 | 在哪 | 说明 |
| --- | --- | --- |
| TAESD 解码：tanh 软夹、35 个 3×3 卷积（含 ReLU、残差）、3 次最近邻上采样、清零 | **NPU**（设备程序 `crates/taesd`） | 141 GFLOPs / 张（512²）；`npu` 后端真机实测过（ComfyUI 工作流里也是）。`npu-sim` 是在 CPU 上重放，不算 NPU 贡献 |
| latent 打包（× 0.18215、转 bf16、NHWC 带边框）、输出解包（取 RGB、夹到 [0, 1]） | CPU | 输入输出的打包解包（DESIGN D6） |
| 正式 VAE 解码：37 个卷积、30 个 GroupNorm + SiLU、注意力、3 次上采样 | **NPU**（设备程序 `crates/vae`，3 段） | 2514.5 GFLOPs / 张，真机 0.98 s；`npu` 后端真机实测过。段之间的特征图（4.5 MB）经主机转手（只搬，不在 CPU 上算） |
| UNet 一次前向（CFG 两路）：卷积、GroupNorm、LayerNorm、自 / 交叉注意力、GEGLU、时间步那一路、上下采样、跳连拼接 | **NPU**（设备程序 `crates/unet`，5 段） | `tools/unet.py board` 真机实测过（一步 3.12 s，还比 CPU 慢约 30%）；段间的张量经主机按偏移转手（只搬，不在 CPU 上算）；**还没接进 ComfyUI** |
| CLIP 文本编码、ComfyUI 工作流里的 UNet 去噪（KSampler） | CPU（ComfyUI 原生节点） | B2、B3 U7、B4 要搬到 NPU；现在工作流里它们都在 CPU 上 |
| `cpu` 后端的 TAESD / 正式 VAE 解码 | CPU（diffusers fp32） | 兜底；每用一次计数并打进日志，不静默 |

## 时间（分开记）

真卡性能的全表（TAESD 两次上卡 15 遍、正式 VAE 优化前后与逐 kernel 计时、20 个单 kernel、ComfyUI、CPU 对照）在 [`test/PERF.md`](test/PERF.md)，原始记录在 `test/results/npu3-20261114/perf.json`。

| 阶段 | TAESD 512² | 口径 |
| --- | --- | --- |
| 编译：设备遍 | < 1 s | 本机实测（增量） |
| 编译：docc from-calvx | 4.5 s | npu3 实测（`out/taesd-real/timing.txt`；原机器 4.7 s） |
| 加载（部署） | 0.003 s | **真机实测**（npu3，`dort --deploy-only` 的墙钟；参数块 2.6 MB） |
| 预热 | 没有单独一步 | dort 每次调用都是部署 + 跑一次（DESIGN F1）；第 1 遍与后两遍一样（0.111 / 0.112 / 0.112 s） |
| 推理（含写入 0.3 MB、读回 17 MB） | 约 0.11 s | **真机实测**的整次中位 0.112 s 减只部署，是估算（`out/taesd-board/image/metrics.json`） |
| ComfyUI 节点里一次解码（`npu`） | 0.12 s | **真机实测**（含打包解包；dort 墙钟 0.113 s） |
| numpy 模拟 | 6–7 s | 本机（32 核）实测，只是查接线用，不是性能 |

## 验收口径

- 单 kernel：`ref/ktest.py`，期望按 torch 语义另算；比「最大差 ≤ 2 ULP(输出最大值)」与余弦，ULP 分布另报。上板门槛按 kernel 类型另定（`BOARD_TOL`，2026-11-14 按真机实测收紧：卷积 2 ULP、tanh 1 ULP、temb 3 ULP、上采样逐位）。
- 整模型：`tools/taesd.py`，对 bf16 口径的 torch 参照（逐层舍 bf16）与 diffusers fp32；报余弦、最大差、PSNR。不承诺逐位（exp / rsqrt / tanh 是近似指令）。
- 接线：`.cvx` 模拟器查对齐、越界、原地、屏障、未初始化内存；`ref/test_cvxsim.py` 拿故意改坏的程序验它真能抓到。

## 已知限制

- **真机实测只覆盖 TAESD 与它的 7 个单 kernel**（2026-11-14，npu3 的一块卡、一次，latent 只测了 image）；CLIP、UNet、正式 VAE 还在 CPU 上。
  公开环境里没有指令级模拟器，kernel 内部只能上板验；模拟器按精确的 tanh / cos 算，板上是近似指令。
- 时间步嵌入：板上 `cos_approx` 的输出落在 1/128 的网格上（最大差 ≈ 0.02，DESIGN F19），接 UNet 时要看端到端影响。
- 只有 512×512（latent 64×64）一个档位：形状是编译期常量；别的尺寸节点会点名拒绝。
- TAESD 是 SD 的轻量解码器（画质低于正式 VAE，合成测试图来回一趟 PSNR 27.3 dB，fp32 与模拟一样）；正式 VAE 是下一步。
- dort 每次调用都重新部署权重（2.6 MB，TAESD 不大）；连续推理要开 `double_buffer`（doctor 现在报两条预期内的警告）。

## 许可

本仓代码待定（ComfyUI 是 GPL-3.0，节点要 import 它，倾向 GPL 兼容的许可）。模型：SD1.5 是 CreativeML OpenRAIL-M、TAESD 是 MIT；权重、参数块、calbin 都不进仓，
怎么重出见上面的命令。抄来的代码见 [VENDORED.md](VENDORED.md)。
