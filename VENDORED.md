# 抄来的代码（出处、版本、许可、改了什么）

AGENTS.md 硬规矩 7：从别处复制的代码逐条记在这里。赛方说 calBLAS 的 kernel 模板与片段可以直接复用（赛题文档「技术链路」：
「可以直接复用 calBLAS 的 kernel 模板与片段」）；calblas 0.1.0 的 Cargo.toml 没写 license 字段。

| 本仓位置 | 出处 | 版本 | 许可 | 改了什么 |
| --- | --- | --- | --- | --- |
| `crates/kernels/src/conv.rs` 的 `conv2d!` | calblas 0.1.0（赛方 cargo 源）`src/kernels/vision.rs` 第 93–283 行；同 calBLAS 仓库 32d2bb8 | 文件 sha256 `218ca5a4295340451815e821ce737ca31fc6f328b4b28ddd3e7955ce41d41754` | 未声明（赛方允许复用模板） | ① `$crate::kernels::BORDER` / `Cols`、`$crate::halves!` / `rd!` / `logistic!` 改成经 `$crate::calblas::…`（宏搬到别的 crate 里要这么指回去）；② 尾处理的激活拆成 `@pre`（残差之前）/ `@post`（残差之后）两段，加 `act = relu`（bias 后 max(y, 0)）与 `act = res_relu`（残差之后 max(y, 0)）；silu / none 的行为不变；③（2026-11-14）加「列块快路」`TILED` 分支与 `conv_tile` 常量函数：快路放不下整行时按列切块、每块读 K 段进 VB，块内算法同快路（正式 VAE 128² 以上的 3×3 卷积 1.75–2.6 ⇒ 4.2–4.6 TFLOPS，真机实测；原来走快路的形状编出来逐字节不变）。其余逐字不动 |

| `crates/kernels/src/attn.rs` 的 `attn_wide!` | calblas 0.1.0 `src/kernels/attention.rs` 的 `attention!`（第 492–712 行：q / o 的 `[T·H, 16, D]` 视图、kᵀ 的列主序读、两核分作业、分数 / p / 行和 / 收尾的数值口径）；`src/kernels/vision.rs` 的 `spatial_attn!`（第 664–741 行：整行分数存在 VB 里、`rows()` 上的拷贝与 `load_v`）；同 calBLAS 仓库 32d2bb8 | attention.rs sha256 `85154942f94ae2d948dc8cf0de26554d3cc98b96f13477a0a015b3ebaece60f2`、vision.rs 同上 | 未声明（赛方允许复用模板） | 新组合的一个宏，不是逐字搬：① 头维按 128 一段切（q·kᵀ 在 TU 里沿段累加、p·v 按段各算一遍），为了 D = 512 时 rhs 放得进 GIB1（DESIGN F21 / D14）；② 只留自注意力、不带掩码、k 不转置这一条路；③ 分数只算一遍、整行存在 VB（attention! 是两遍各算一次） |

| `crates/kernels/src/norm.rs` 的 `group_norm!` | calblas 0.1.0 `src/kernels/vision.rs` 第 523–600 行；同 calBLAS 仓库 32d2bb8 | 文件 sha256 `218ca5a4295340451815e821ce737ca31fc6f328b4b28ddd3e7955ce41d41754` | 未声明（赛方允许复用模板） | ① `$crate::kernels::Cols`、`$crate::halves!` / `rd!` / `logistic!` 改成经 `$crate::calblas::…`；② 加可选参数 `parts = N`（一组切几段进 VB；不给就是 calblas 原来的算法），为了 128² × 512 时代码不超出 docc 的跳转范围（DESIGN F22）；③ 断言多核一条「段数整除一组的行数」。其余逐字不动 |

| `crates/kernels/src/pointwise.rs` 的 `geglu!` 里的 erf-GELU 算式 | calblas 0.1.0 `src/frag.rs` 的 `act!(gelu_erf)`（Abramowitz–Stegun 7.1.26 的系数与步骤） | 同上 | 未声明（赛方允许复用模板） | 只抄了算式（系数与运算次序），套进本仓新写的逐元素 kernel，临时块在 kernel 入口申请一套 |

只是**调用**、没有复制的（经 cargo 源依赖，不进本仓）：calblas 的 `upsample2!`、`fill16k!`、`halves!`、`rd!` / `wr!`、`Line` / `Rows`；npu 的原语。

参考了思路、代码自己写的（不算抄，记一笔方便对照）：

| 本仓位置 | 参考了 | 说明 |
| --- | --- | --- |
| `host/sd15npu/cvxsim.py` | calBLAS `examples/yolo26/deploy/sim.py` | 那份按作业表重放；本仓改成**解析编出来的 `.cvx` 入口**重放，并加了对齐、越界、原地、屏障、未初始化内存的检查 |
| `crates/ktest` + `ref/ktest.py` | calBLAS `examples/yolo26/deploy/ktest.py` | 那份用脚本拼 MLIR 文本（calBLAS AGENTS.md 现在禁止）；本仓每个测试是一个 `npu::decls!` + `#[npu::program]` 的整步程序 |
| `crates/layout-dump`（`#[path]` 引设备 crate 的 layout.rs） | calBLAS `examples/tiny/pack` | 同样的「设备与主机共用一份偏移表」做法 |
| `ref/taesd_ref.py` 的模块结构 | diffusers 0.43.0 `DecoderTiny` / `AutoencoderTinyBlock`（Apache-2.0） | 只照它的结构逐层调 torch 的 `F.conv2d` 等，没有复制它的代码；fp32 那份直接调 diffusers 本身 |
| `ref/unet_ref.py` 的模块结构 | diffusers 0.43.0 `UNet2DConditionModel`、`ResnetBlock2D`、`Transformer2DModel`、`BasicTransformerBlock`、`GEGLU`（Apache-2.0） | 同上：照结构逐层调 torch 的算子，没有复制它的代码；fp32 那份直接调 `unet(...)` |
| `ref/vae_ref.py` 的模块结构 | diffusers 0.43.0 `AutoencoderKL` 的 `Decoder` / `ResnetBlock2D` / `Attention`（Apache-2.0） | 同上：照结构逐层调 torch 的 `F.conv2d`、`F.group_norm`、`F.linear`、`softmax`，没有复制它的代码；fp32 那份直接调 `vae.decode` |
| `comfy/sd15_npu/nodes.py` 的缩放约定 | ComfyUI v0.45.3（GPL-3.0）`nodes.py` 的 `load_taesd`、`comfy/latent_formats.py` 的 `SD15` | 只核了常数 0.18215 与 [0, 1] 的口径，没有复制代码 |
