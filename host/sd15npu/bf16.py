"""f32 ⇄ bf16（round-to-nearest-even，NaN 保持 NaN）：设备上 `Cvt::default()` 的舍入就是 rne（CalVX 程序级的缺省数值合同）。"""
import numpy as np


def bits(a) -> np.ndarray:
    """f32（或能转成 f32 的）⇒ bf16 的位（uint16，rne）。"""
    f = np.ascontiguousarray(a, np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(f)
    if nan.any():
        r[nan] = 0x7FC0
    return r


def from_bits(u) -> np.ndarray:
    """bf16 的位（uint16）⇒ f32。"""
    return (np.ascontiguousarray(u, np.uint16).astype(np.uint32) << 16).view(np.float32)


def rnd(a) -> np.ndarray:
    """f32 舍到 bf16 再放回 f32（逐元素「这一步出口舍一次」）。"""
    return from_bits(bits(a))
