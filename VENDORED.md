# 抄来的代码（出处、版本、许可、改了什么）

AGENTS.md 硬规矩 7：从别处复制的代码逐条记在这里。赛方说 calBLAS 的 kernel 模板与片段可以直接复用（赛题文档「技术链路」：
「可以直接复用 calBLAS 的 kernel 模板与片段」）；calblas 0.1.0 的 Cargo.toml 没写 license 字段。

| 本仓位置 | 出处 | 版本 | 许可 | 改了什么 |
| --- | --- | --- | --- | --- |
| `crates/kernels/src/conv.rs` 的 `conv2d!` | calblas 0.1.0（赛方 cargo 源）`src/kernels/vision.rs` 第 93–283 行；同 calBLAS 仓库 32d2bb8 | 文件 sha256 `218ca5a4295340451815e821ce737ca31fc6f328b4b28ddd3e7955ce41d41754` | 未声明（赛方允许复用模板） | ① `$crate::kernels::BORDER` / `Cols`、`$crate::halves!` / `rd!` / `logistic!` 改成经 `$crate::calblas::…`（宏搬到别的 crate 里要这么指回去）；② 尾处理的激活拆成 `@pre`（残差之前）/ `@post`（残差之后）两段，加 `act = relu`（bias 后 max(y, 0)）与 `act = res_relu`（残差之后 max(y, 0)）；silu / none 的行为不变。其余逐字不动 |


只是**调用**、没有复制的（经 cargo 源依赖，不进本仓）：calblas 的 `upsample2!`、`fill16k!`、`halves!`、`rd!` / `wr!`、`Line` / `Rows`；npu 的原语。

参考了思路、代码自己写的（不算抄，记一笔方便对照）：

| 本仓位置 | 参考了 | 说明 |
| --- | --- | --- |
| `host/sd15npu/cvxsim.py` | calBLAS `examples/yolo26/deploy/sim.py` | 那份按作业表重放；本仓改成**解析编出来的 `.cvx` 入口**重放，并加了对齐、越界、原地、屏障、未初始化内存的检查 |
| `crates/ktest` + `ref/ktest.py` | calBLAS `examples/yolo26/deploy/ktest.py` | 那份用脚本拼 MLIR 文本（calBLAS AGENTS.md 现在禁止）；本仓每个测试是一个 `npu::decls!` + `#[npu::program]` 的整步程序 |
| `crates/layout-dump`（`#[path]` 引设备 crate 的 layout.rs） | calBLAS `examples/tiny/pack` | 同样的「设备与主机共用一份偏移表」做法 |
| `ref/taesd_ref.py` 的模块结构 | diffusers 0.43.0 `DecoderTiny` / `AutoencoderTinyBlock`（Apache-2.0） | 只照它的结构逐层调 torch 的 `F.conv2d` 等，没有复制它的代码；fp32 那份直接调 diffusers 本身 |
| `comfy/sd15_npu/nodes.py` 的缩放约定 | ComfyUI v0.45.3（GPL-3.0）`nodes.py` 的 `load_taesd`、`comfy/latent_formats.py` 的 `SD15` | 只核了常数 0.18215 与 [0, 1] 的口径，没有复制代码 |
