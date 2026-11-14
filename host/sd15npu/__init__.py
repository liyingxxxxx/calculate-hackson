"""sd15-npu 的主机侧（Python）：打包 / 解包、整步程序的 numpy 模拟、调 dort（受开关保护）、给 ComfyUI 节点用的执行器。

模块：
- `bf16`    ：f32 ⇄ bf16（rne），与设备上 `Cvt::default()` 的舍入一致
- `layout`  ：读 Rust 的偏移表（`cargo run -p layout-dump`），Python 不自己算偏移
- `cvxsim`  ：读编出来的 `.cvx`，照入口函数的调用序列逐个 kernel 在 numpy 里重放（查接线：对齐、越界、原地、漏屏障、读没清零的内存）
- `taesd`   ：TAESD 程序的打包（权重 ⇒ param_blk0.bin、latent ⇒ in_x）与解包（out_y ⇒ 图）
- `dort`    ：`dort calbin run` 的包装；**缺省不碰板子**，要 SD15NPU_ALLOW_BOARD=1 才真调
- `runner`  ：TAESD 解码的三种执行器（cpu / npu-sim / npu），ComfyUI 节点与命令行共用
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
"""本仓根目录。"""
