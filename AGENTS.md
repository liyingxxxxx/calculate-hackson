# 给 AI 编码助手（和接手的人）的约定

本仓是 Calculet NPU CalVX 黑客松的参赛工程：在 Calculet NPU 上跑 Stable Diffusion 1.5，由 ComfyUI 原生工作流发起（路线 B）。
现在到哪看 [PROGRESS.md](PROGRESS.md)，设计与关键事实看 [docs/DESIGN.md](docs/DESIGN.md)，交接看 [docs/HANDOFF.md](docs/HANDOFF.md)，赛题原文在 [docs/赛题原文.md](docs/赛题原文.md)，真卡上的结果与简单示例在 [test/](test/README.md)。**每做完一项就更新 PROGRESS.md**（勾上、写日期与证据）。

## 硬规矩（违反任何一条都要先问负责人）

1. **只用赛方的公开栈。** 工具链是 `calculet`（rustc 1.98.1 + docc + dort，赛方 install.sh 装的），基础库经赛方 cargo 源依赖
   （npu、npu-macros、caldnn、calblas）。机器上若还装着别的（非公开的）设备栈、工具链或 cargo 源，一律不用、不引用、不抄；交付件（`verify_all.sh`、`bootstrap.sh`、上板脚本）只依赖公开栈，目标是在**比赛提供的真卡环境**里照 README 复现。
2. **先进隔离环境再干活。** 每个 shell 先 `source $CALCULET_HOME/env.sh`（隔离的 RUSTUP_HOME / CARGO_HOME，PATH 最前面是工具链自带的 docc / dort）；
   `tools/check_env.sh` 守着这一条（机器上另装着老版本的 docc / dort 时它会报），构建脚本先跑它，不过就停。
3. **不碰本机的板子**（`/dev/calculet*`）：没获准之前不跑任何会开设备的 dort 子命令（`dort --help` 可以，不带参数的 `dort` 会读设备信息，也不跑）。
   现场共享板按排队来；不刷写、不复位（`dort device reprobe`、`rmmod / insmod` 一律不做）；不 SIGKILL 在飞的作业；挂死就停、找工程师。
4. **没公开的资料不进 AI 工具。** 三个公开仓库与赛题页按公开处理；现场另发的 SDK 文档、FAQ、板子说明，确认公开前不贴给 AI。
5. **CPU 承担的计算要记账。** 任何落回 CPU 的运算都要在 README 的「NPU / CPU 分工」里列出；运行时落回要计数、打出来，不静默。
6. **数值验收用独立参照。** 参照按 torch 的语义另写（numpy / diffusers），不照着 kernel 写；KS1 的 exp / rsqrt 是查表近似，
   不承诺和 CPU 逐位相同，用容差，并写进已知限制。真机实测、估算分开标；编译、加载、预热、推理分开记。
7. **复制别处的代码要记账**（`VENDORED.md`：出处、提交、许可、改了什么），包括从 calBLAS 例子、diffusers、ComfyUI 抄的片段。
8. **注释与文档用中文，写「为什么」**；新 kernel 在模块文档里写清形状约束、排布、数值口径、哪个测试守着它。
9. 权重、参数块、calbin 都不进仓（`.gitignore`）；怎么重出写在 README。

## 常用命令

`CALCULET_HOME` 是本仓的上一层，权重在它的上一层的 `models/`。新机器从零：`bash tools/bootstrap.sh`。

```bash
source $CALCULET_HOME/env.sh             # 每个 shell 先来这一句
bash tools/verify_all.sh                 # 不上板能做的全部检查；改完东西、交之前都跑（VERIFY_FULL=1 再加 ComfyUI 整条工作流）
tools/check_env.sh                       # 环境自检
tools/calbin.sh <crate> [--features f] [--params 目录]   # 整步程序 ⇒ calbin ⇒ docc doctor（不上板）
$CALCULET_HOME/venv/bin/python -I tools/taesd.py {params|calbin|sim|check}   # TAESD（B1）
$CALCULET_HOME/venv/bin/python -I tools/vae.py {params|calbin|sim|check}     # 正式 VAE（B1b，分段程序，现在 3 段）
$CALCULET_HOME/venv/bin/python -I ref/ktest.py {gen|sim|calbin} <测试>      # 单 kernel 测试台
$CALCULET_HOME/venv/bin/python -I ref/test_cvxsim.py                        # 模拟器自测
bash comfy/run_comfyui.sh                # ComfyUI（venv 在 $CALCULET_HOME/comfy-venv）
bash ref/setup_ref_env.sh                # CPU 参照环境与权重
```

上板一条命令：`SD15NPU_ALLOW_BOARD=1 $CALCULET_HOME/venv/bin/python -I tools/board_check.py`（重出 calbin ⇒ 静态核对 ⇒ 只读查卡 ⇒ 逐个 kernel ⇒ 整个解码器，
任何一步不对就停）。缺省拒绝，要 `SD15NPU_ALLOW_BOARD=1`——**只有负责人确认这块卡可以用才设**。卡要跑 DeviceRT（DESIGN F18）；
板上不是 DeviceRT 时 dort 会被拒，那要管卡的人重刷，⛔ 自己刷。

## 写整步程序的规矩（本仓踩出来的，补 calBLAS AGENTS.md）

- 偏移表只有 Rust 一份（`crates/<程序>/src/layout.rs`）：主机经 `cargo run -p layout-dump` 读 JSON，Python 里不另算。`decls!` 的字节数只收字面量，旁边写 const 断言核对。
- 按 `num_cores()` 分行的单核 kernel（`upsample2!`、`tanh_scale!`）**两核都调**；只调一次、不分行的（`fill16k!`）放在 `if core_rank() == 0` 里或按核给不同的区。
- workspace 的初值不可靠：入口先清零（`fill16k!`），模拟器把 workspace 填成 NaN 来查这一条。
- 带 feature 编设备遍要用自己的 `--target-dir`（cargo 切回编过的 feature 不重编，`npu/<crate>.cvx` 会是上一个组合的）。
- 宏参数别和宏里的 `const` 同名（`c = C` 会展开成 `const C: usize = C;`）。
- **每核的代码不能超过约 1 MB**（docc 0.4.1，DESIGN F22）：docc 会展开 ≤ 64 轮的循环，单个 kernel 也可能很大（128² 的 calblas group_norm! 约 1.1 MB）。放不下就切段（crates/vae）；calbin 里只有调到的 kernel 进 ELF。
- 改了入口或 kernel 特化：`tools/taesd.py check`（或对应程序的 check）+ `ref/test_cvxsim.py` 必须过；新 kernel 要在 `crates/ktest` 加一个测试。

## 写 kernel 前先读

- 公开仓库的克隆在 `$CALCULET_HOME/work/{calBLAS,calDNN,npu}`（bootstrap 钉在 2026-11-13 的下发版；原机器上 `dl/repos/` 是只读的原样快照）。
- `npu/SYNTAX.md`（kernel 语法）；calBLAS 的 `README.md`（覆盖表与「绕开 docc 现状的写法」）、`AGENTS.md`（整步程序、两核、数值的坑）；
  `src/kernels/vision.rs`（卷积、GroupNorm、上采样、空间注意力）；`examples/yolo26/deploy/DEPLOY.md`（卷积网络整套做法与 docc 的规矩）。
- 两核：`cores = 2` 的 kernel 调完必须 `sync_cores()`；不能原地（同一块既输入又输出）；单核 kernel 看它分不分行（见上一节）。
- 本仓的现成例子：`crates/taesd`（整步程序的写法）、`crates/vae`（大模型切段：段表、按段的偏移、一个 feature 一段）、`crates/unet`（**生成器出的**分段程序：`tools/unet_gen.py` 量 kernel 代码、自动切段、偏移只算一份写进 step.rs 与 plan.json，别手改）、`crates/kernels`（模板的写法）、`crates/ktest`（单 kernel 测试）。
- 性能先量再改：`tools/prof.py [--model unet]` 在真卡上逐个 kernel 计时（每个 kernel 单独一个程序连调几遍），按「一次的时间 × 调的次数」找大头。
