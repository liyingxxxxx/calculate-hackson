# 交接文档：Calculet NPU 黑客松 · 在 NPU 上跑 SD1.5 并接 ComfyUI（项目 sd15-npu）

> 把这份文档交给接手的人（或 AI 编码助手）当第一条消息。仓库：**https://github.com/liyingxxxxx/calculate-hackson**（公开）。
> - `main` 分支：稳定版，在真卡上验过（2026-11-14，npu3）：TAESD 与正式 VAE 两个解码器都在 NPU 上；
> - `wip/vae-decoder` 分支：正式 VAE 的开发分支，真卡验过之后已合进 `main`。
>
> `tools/make_handoff.py` 把本文和附录拼成 `<CALCULET_HOME>/交接提示词.md`（附录 A 是赛题原文，附录 B 是仓库每个文件的全文），并打一个 tar 包，给连不上 GitHub 的时候用。

## 0. 接手先做这几步（在比赛提供的真卡环境里）

```bash
mkdir -p ~/calculet && cd ~/calculet                  # 任意空目录，下面叫 CALCULET_HOME
git clone https://github.com/liyingxxxxx/calculate-hackson.git sd15-npu   # 目录名必须是 sd15-npu
bash sd15-npu/tools/bootstrap.sh                      # 从零重建，约 10 分钟；最后自动跑 verify_all.sh，12 项全 ✓ 才算好
source ~/calculet/env.sh && cd sd15-npu               # 以后每个 shell 先 source env.sh

bash test/run_example.sh                              # 简单示例，先不上卡（npu-sim：CPU 上模拟，不是 NPU 的数）
# 确认这块卡现在可以用、板锁文件是哪个之后：
SD15NPU_ALLOW_BOARD=1 SD15NPU_BOARD_LOCK=<板锁文件> bash test/run_example.sh --comfy        # TAESD 解码
SD15NPU_ALLOW_BOARD=1 SD15NPU_BOARD_LOCK=<板锁文件> bash test/run_example.sh --vae --comfy  # 正式 VAE 解码（3 段，约 1 s）
SD15NPU_ALLOW_BOARD=1 SD15NPU_BOARD_LOCK=<板锁文件> $CALCULET_HOME/venv/bin/python -I tools/board_check.py --vae --comfy
```

这几步 2026-11-14 在 npu3 上照做过：从 GitHub 克隆到全新目录，从零重建约 6 分钟、自检全过；真卡上出的图和仓里存的逐字节相同。
跑出来的结果应该和 [`test/README.md`](../test/README.md) 里 2026-11-14 npu3 那一份对得上：
TAESD 3 遍逐字节一致、对 bf16 参照约 60 dB、对 fp32 约 56 dB；7 个单 kernel 全过；ComfyUI 出图，节点日志写 `backend=npu`。
对不上就停，把 `out/board/summary-*.json` 和那一步的输出发给负责人。

然后读仓库里的 `AGENTS.md`（硬规矩、写程序的规矩）、`PROGRESS.md`（做到哪了）、`docs/DESIGN.md`（关键事实 F1–F24、设计决定 D1–D17），
再从第 8 节「下一步」接着干。不清楚的先问负责人，别猜。

## 1. 边界

### 1.1 已经做完、可以直接用的

- **真机实测（2026-11-14，npu3）**：`tools/board_check.py --comfy` 6 步全过。
  - 单 kernel：7 个 kernel、20 个用例（19 个必过）全过；
  - TAESD 整个解码器：3 遍逐字节一致，对 bf16 参照 60.3 dB、对 fp32 55.9 dB，只部署 0.003 s、推理约 0.11 s；
    另外三组 latent（randn:0、zeros、big）也在真卡上过了，最差的 randn:0 也有 52.7 dB；
  - ComfyUI 原生工作流：解码走 npu，没有落回 CPU。
  - **正式 VAE 解码器**（分段程序，现在 3 段）：3 遍逐字节一致，对 bf16 参照 58.3 dB；ComfyUI 节点 SD15 NPU · VAE Decode 出的图和自带 VAEDecode 相比 55.6 dB。
    按真卡逐 kernel 计时优化后 **0.98 s**（第一版 5.18 s），比同机 CPU fp32（1.71 s）快约 1.75 倍（`test/PERF.md`）。
  - **UNet 一次前向（CFG 两路）**：2 遍逐字节一致，对 bf16 参照 49.8 / 50.1 dB、对 fp32 49.4 / 48.0 dB；第一版一步 12.35 s，
    注意力换成本仓的 `flash_attn!`（DESIGN D21）8.19 s，宽通道卷积改走 `linear!`（D22）5.16 s，自注意力压成稠密 token（D23）4.18 s，自注意力一次 128 行 q（D24）**3.12 s**（5 段，只部署 0.50 s），
    **还比同机 CPU（约 2.4 s）慢约 30%**；现在一步的 kernel 合计 2.57 s：注意力 1.04 s、linear 0.79 s、卷积 0.49 s，`test/PERF.md` 1c。
  - 结果与图在 `test/results/npu3-20261114/`；性能全表在 `test/PERF.md`（TAESD 整次中位数 0.113 s，比同机 CPU fp32 快约 1.6 倍）。
- 编译链：赛方工具链装在隔离目录，经赛方 cargo 源依赖 calblas；整步程序 ⇒ CalVX ⇒ docc 出 calbin，doctor 拒收 0 条。
- B1 的 TAESD 解码器：整步程序在 `crates/taesd`，有带真权重的 calbin。numpy 模拟器照编出来的 `.cvx` 重放，查接线；对两份独立参照是 60.6 / 55.9 dB。
- ComfyUI v0.45.3 + 自定义节点 `SD15 NPU · TAESD Decode`，三个后端：`npu`（真卡）/ `npu-sim` / `cpu`。
- 单 kernel 测试台（7 个测试）、模拟器自测、上板前静态核对、上板一条命令；没卡时可以用 dort 的测试替身把上板流程空跑一遍。
- 一键重建 `tools/bootstrap.sh`、一键自检 `tools/verify_all.sh`（12 项，`VERIFY_FULL=1` 时 13 项）、简单示例 `test/run_example.sh`。

### 1.2 没做的（⛔ 不能说成做完了）

- **真机实测只覆盖 TAESD 和它的 7 个单 kernel**：一块卡（npu3）、一天，TAESD 测了四组 latent。别的数都是 CPU 上的模拟与参照。`npu-sim` 不是 NPU，不算 NPU 贡献。
- ComfyUI 工作流里的 CLIP、UNet（KSampler）现在是 ComfyUI 原生节点在 **CPU** 上跑；NPU 的 UNet 还没接进去（B3 U7），CLIP（B2）、去噪循环进设备（B4）还没写。
- UNet 只在一组输入（t = 500）上真机对过参照；新 kernel（`ln_nhwc!`、`geglu!`、`copy_win!`、`bias_fold!`）没有单独的 ktest 项。
- 正式 VAE 只测了一组 latent（image）。
- 时间步嵌入：板上 `cos_approx` 的输出只有 1/128 的分辨率（DESIGN F19），对 UNet 有多大影响还没测。
- 只有 512×512 一个档位。

### 1.3 允许 / 不允许

| 可以 | 不可以 |
| --- | --- |
| 用赛方公开栈做交付件：隔离的 calculet 工具链、赛方 cargo 源、三个公开仓库 | 交付件依赖任何非公开的设备栈、cargo 源或资料 |
| 只读看卡的状态（`dort device`，确认能用之后）；`dort calbin parse` 是纯静态的、不开设备，随时可以 | 没确认就跑 `dort calbin run`；`dort device reprobe`、刷写、`rmmod / insmod`；kill 在飞的 dort |
| 卡有问题（不是 DeviceRT、被拒、挂死）就停，告诉负责人、找管卡的人 | 自己刷 DeviceRT、自己复位共享卡 |
| 改仓库后跑 `verify_all.sh`、更新 `PROGRESS.md`、重出交接材料（`tools/make_handoff.py`） | 把模拟 / 参照的数写成「真机实测」；静默地落回 CPU |
| 负责人要求时提交、推送（身份用负责人给的） | 自作主张提交、推送 |

## 2. 比赛是什么

**Calculet NPU CalVX 算子与端到端模型移植黑客松**，原粒半导体赛道。赛题页 http://115.159.82.89/ ，原文在 `docs/赛题原文.md`。
- 路线 A：补算子。
- 路线 B：端到端。原生前端下发、计算全在 NPU、不靠 CPU 兜底；标杆是「一键拉起 ComfyUI 工作流、稳定出图、支持多尺寸」。

评委先看授权合规与基础正确性（对独立参照在容差内），再看功能、性能 / 应用价值、可复现性、NPU 实际贡献。
要写清哪些在 NPU、哪些回退 CPU；编译、加载、预热、推理时间分开记；真机实测、仿真、估算分开标。提交方式与 FAQ 还没发布。

## 3. 我们的选择

- **主线：路线 B，SD1.5 + ComfyUI，先只做 512×512。** calBLAS 已有 SD 要的一大半模板（卷积、GroupNorm、注意力、线性层），
  工作量主要在整模型拼装、权重打包、主机侧与 ComfyUI 节点。
- 已经用 **TAESD**（轻量解码器）把「NPU 解码 ⇒ ComfyUI 出图」打通，真卡上跑过（B1）。
- 接下来：正式 VAE（B1b，进行中）⇒ CLIP（B2）⇒ UNet（B3）⇒ 去噪循环放进设备（B4，DESIGN D2）⇒ 性能与多尺寸（B5）。

## 4. 硬规矩（全文在 AGENTS.md）

1. 只用赛方公开栈：`calculet` 工具链（赛方 install.sh 装）、经赛方 cargo 源依赖 npu / caldnn / calblas 0.1.0。
2. 每个 shell 先 `source <CALCULET_HOME>/env.sh`：隔离 RUSTUP_HOME / CARGO_HOME，把工具链自带的 docc / dort 放到 PATH 最前；`tools/check_env.sh` 守着。
   机器上可能另装着老版本，npu3 的 `/usr/local/bin` 里就有 dort 0.2.8、docc 0.1.4，⛔ 用它们。
3. **上卡要先确认**：确认卡能用、板锁文件是哪个，才设 `SD15NPU_ALLOW_BOARD=1`（上板脚本缺省拒绝）。
   上了板也不复位、不刷写、不 SIGKILL 在飞的作业；挂死就停、找管卡的人。
4. 现场另发的、未确认公开的资料不贴给 AI 工具。
5. CPU 承担的计算要记账（README 的「NPU / CPU 分工」）；运行时落回 CPU 要计数、打日志，不静默。
6. 数值验收用独立参照（torch / diffusers 另写），不照着 kernel 写；近似指令不承诺逐位。
7. 抄来的代码记 `VENDORED.md`。权重、参数块、calbin 不进仓。

## 5. 复现的细节

前提：
- Linux x86_64；git、curl、tar；内存 ≥ 16 GiB、磁盘 ≥ 20 GiB；
- 网络能到 115.159.82.89（http）、腾讯云 COS、huggingface.co、download.pytorch.org、pypi.org、github.com、astral.sh、sh.rustup.rs。

`bootstrap.sh` 幂等，断了重跑就行。它依次做：
1. 写入口 `env.sh`；
2. 装 uv、rustup（缺了才装，装进隔离目录）；
3. 下载赛方 install.sh，**核 sha256**（`3c175657…`）。不一致就停，先读一遍新的 install.sh，再改脚本里的值；
4. 三个公开仓库切到钉住的提交：calBLAS 32d2bb8、calDNN 1e06c41、npu 172aad2；
5. 编一遍 calBLAS tiny 的设备遍；
6. 建 CPU 参照环境（`ref/requirements.lock`），下权重：SD1.5 revision 451f4fe、TAESD 614f768，下到 CALCULET_HOME 的上一层的 `models/`，sha256 写进 `download-verification.json`；
7. 装 ComfyUI v0.45.3（单独的 venv，`comfy/requirements.lock`）；
8. 跑 `verify_all.sh`。

**核对（全过才算恢复好）**：
- `verify_all.sh` 12 项全 ✓（`VERIFY_FULL=1` 时 13 项）；
- TAESD 参数块的 sha256 是 `ed33d7027ee760b51cd163eab9e1cd2f1a2843e1921ac397a4ffab174d404955`；
- `tools/taesd.py check` 里 image 那组：对 bf16 参照余弦 0.999999、PSNR ≈ 60.6 dB，对 fp32 ≈ 55.9 dB。

版本：rustc 1.98.1-calculet（nightly-2026-11-13）、docc 0.4.1、dort 0.4.1、torch 2.15.0+cpu、diffusers 0.43.0、ComfyUI v0.45.3。

坑：
- 赛方 install.sh 最后一步往 `$CARGO_HOME/bin` 链 docc / dort，目录不在会半途退出。bootstrap 已经先建好了这个目录。
- 本仓的目录名必须是 `sd15-npu`：env.sh 和各脚本都按 `<CALCULET_HOME>/sd15-npu` 找它。

## 6. 上真卡

卡要满足这三条（DESIGN F18 / F20）：设备端 `OS up / RT up`；**板上跑 DeviceRT**；没有别的进程在用。
板上不是 DeviceRT 时，dort 开设备会被拒，`dort device` 里写「拒」。这要管卡的人重刷，⛔ 自己刷。

`board_check.py` 一条命令依次做：
1. 重出全部 calbin；
2. 静态核对（`tools/preflight.py`，不开设备）；
3. 只读查卡，不满足那三条就停；
4. 逐个 kernel：up_256 验两核分行 ⇒ tanh / temb 标定近似指令 ⇒ 各个卷积；
5. TAESD 整个解码器：只部署一次，再整次跑 3 遍，查逐字节一致、对两份参照，部署与推理分开计时；
6. ComfyUI 整条工作流（加 `--comfy` 时），解码走 npu。

任何一步不对就停、退出 1，记录在 `out/board/summary-<时间>.json`。整个解码器算错时，用单 kernel 的结果二分。

板锁：dort 的缺省是 `/opt/data/locks/board.lock`，现场若有自己的规矩，用 `SD15NPU_BOARD_LOCK` 给。
npu3 的锁按 PCI 地址起名，负责人定的是 `/opt/data/locks/board-0000:07:00.0.lock`。
**板锁要和别人锁同一个文件**才防得住撞车，自己另建一个没用。设备下标用 `SD15NPU_DORT_INDEX`，缺省 0。

npu3 上的实测：一次 `dort calbin run`，小 kernel 约 2 ms，TAESD 整次 0.11 s；部署 2.6 MB 的参数块 1 ms。
所以 F1「每次调用都重新部署」对 TAESD 不是瓶颈。UNet（bf16 约 1.7 GB）部署要多久，还要实测。

## 7. 最要紧的几条事实（全表在 DESIGN.md）

- **F1** `dort calbin run` 是一次性的：每次都部署权重、喂一个输入张量、读一个输出张量；没有常驻会话，也没有 Python 绑定。
  所以所有输入要打进一个 input 声明（D1），去噪循环要放进设备（D2）。
- **F11** 按 `num_cores()` 分行的单核 kernel（`upsample2!`）要两核都调。真卡上验过。
- **F12** ComfyUI 的 LATENT 在 VAE 原空间，喂 TAESD 前要 × 0.18215。
- **F14 / F15** docc 0.4.1 没有 `vec.round`；VU 和标量做运算时，向量要在一行 512 B 以内，或者是整行的倍数。
- **F16** 公开环境里除了真机，没有能执行 kernel 的东西 ⇒ 我们的 numpy 模拟器只查到 kernel 调用这一级，kernel 内部对不对要上板才知道。
- **F17** calDNN 的图运算里没有二维卷积、上采样、GroupNorm ⇒ SD 照 calBLAS yolo26 的路子走：自己的作业表 + 整步程序。
- **F19**（真机实测）`cos_approx` 的输出落在 1/128 的网格上；`tanh_approx`（f32）几乎精确；卷积最差 1 ULP(max)。
- **F21** 正式 VAE 要用的模板 docc 都收，只有 `attention!` 头维 512 不收（TU 的操作数放不下）⇒ 新写按头维分段的 `attn_wide!`。
- **F22** docc 每核的代码不能超过约 1 MB（RISC-V 跳转 ±1 MB）⇒ 正式 VAE 切成几段程序、主机串起来（D15；优化后 3 段，D17）；128² 的 calblas `group_norm!` 单个就超，本仓抄一份加 `parts`。
- **F23 / F24**（真机实测）正式 VAE 第一版比 CPU 慢约 3 倍；逐 kernel 计时发现大头是 GroupNorm 的排布转换（1.3 GB/s）⇒ NHWC 原生 GN（D16）+ 卷积列块快路 + 并成 3 段（D17）⇒ 0.98 s。

## 8. 进度与下一步（细账在 PROGRESS.md）

| 阶 | 内容 | 状态 |
| --- | --- | --- |
| A0 | 环境、编译链、CPU 参照、算子清单、ComfyUI、模拟器、测试台、一键重建 | ✅ |
| A1 | 新算子 | 时间步嵌入真机实测过；quick_gelu、NHWC GroupNorm 未写 |
| B1 | TAESD 解码上 NPU + ComfyUI 节点 | ✅ **真机跑通**（2026-11-14 npu3） |
| B1b | 正式 VAE 解码器上 NPU | ✅ **真机跑通**（3 段，58.3 dB，0.98 s，比 CPU 快约 1.75 倍；ComfyUI 节点 0.99 s）；单 kernel 测试待补（V2b） |
| B3 | UNet 一次前向 | **真机跑通**（约 50 dB），一步 12.35 ⇒ 8.19 ⇒ 5.16 ⇒ 4.18 ⇒ 3.12 s（注意力、宽通道卷积、稠密 token、128 行 q），5 段；接 ComfyUI（U7）未做 |
| B2、B4、B5 | CLIP、去噪循环进设备、性能与多尺寸 | 未开始 |

接着做什么：

0. **UNet 提速（PROGRESS U6）**：① 注意力已换成本仓的 `flash_attn!`（U6a，DESIGN D21；⛔ 别在 VB 里对 [N, 1] 的向量按行切了再 bcast，docc 收但真卡算错，F28）；
   ② 宽通道的卷积已改走 `linear!`（U6b，DESIGN D22）；自注意力已压成稠密 token、一次 128 行（U6c / U6e，D23 / D24）；③ 剩下：部署 0.50 s / 步（dort 没有常驻）、64² 自注意力 0.84 s、linear 0.79 s；改完 `tools/unet_gen.py kernels && probe && program`，
   `tools/unet.py sim` 对参照，`tools/prof.py --model unet gen/build/board` 量 kernel，`tools/unet.py board` 量整步。然后 U7 接 ComfyUI。
1. **正式 VAE 还能抠的**（PROGRESS「B1b」的 V8）：GroupNorm 统计那一遍、`attn_wide!`、把注意力那一段也并进去。量性能用 `tools/prof.py`（真卡上逐个 kernel 计时）。
   正式 VAE 已经真机跑通，下面是当时的做法，留作写 UNet 的参考：
   - 结构：post_quant_conv ⇒ conv_in(4→512) @64² ⇒ 中间块（ResNet、单头 512 维注意力、ResNet）⇒ 4 个上采样段（通道 512 / 512 / 256 / 128）⇒ GN+SiLU ⇒ conv_out。
   - GroupNorm 先走「排布转换 + calblas `group_norm!`」（D13）；注意力走稠密 token + `attn_wide!`（D14）。
   - 剩下的步骤：V2b 单 kernel 测试与模拟器 ⇒ V3 整步程序 `crates/vae` 与打包 ⇒ V4 模拟对参照 ⇒ V5 ComfyUI 节点 ⇒ V6 上板。
2. **B2 CLIP**：quick_gelu（x · σ(1.702x)，新写一个逐元素 kernel）、因果注意力（`attention!` 的 `mask = full`）、77 个 token 补到 128。
3. **B3 UNet 一次前向**：时间步嵌入（已写）⇒ 线性层 + SiLU ⇒ 各 ResNet；交叉注意力（77 ⇒ 128 补齐）；GEGLU 拆两个 `linear!`。
4. 每加一种 kernel：在 `host/sd15npu/cvxsim.py` 加一个 `_k_<模板>`，在 `crates/ktest` 加一个测试，上板先单 kernel、再整个程序。

## 9. 未决 / 要问组委会

- 常驻会话、Python 绑定、跳过部署：有了这些，D2 才能换成「每步调一次 UNet」；
- 能不能提供指令级模拟器（calBLAS AGENTS.md 里写的 calvxus `cvx run`）；
- 现场板子的排队方式、单次时长、板锁文件；统一复测用的设备、固件、工具链版本；
- 开赛前写的代码算不算「已有成果」；能不能给 calDNN 加 `Op` 变体（路线 A 的「被图调用」）；提交方式与 FAQ。
