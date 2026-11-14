"""SD1.5 UNet（B3）的主机侧：参数块打包、输入打包 / 输出解包、各段的执行器（numpy 模拟或真卡）。

偏移不在这里算：全部读 `crates/unet/plan.json`（`tools/unet_gen.py program` 出的，和设备入口 `crates/unet/src/step.rs` 是同一次计算）。
主机在这里做的只有**搬字节**（权重排布、bf16 舍入、把上一段的输出按偏移拼成下一段的输入），不做任何模型里的计算（AGENTS.md 硬规矩 5）。

权重条目（plan.json 的 `wsize` 的键）怎么打：
- `conv_w|模块`：Conv2d [Cout, Cin, k, k] ⇒ [k·k][Cin][Cout] bf16（conv_in 的输入 4 通道、conv_out 的输出 4 通道补零到 32）；`conv_b|模块`：f32 [Cout]。
- `lin_w|模块|方式`：Linear [N, K] 当 1×1 卷积 ⇒ [K][N] bf16。方式：`plain` 原样；`qkv` 输出按头补零（头 h 的第 d 维放到 h·Dp + d）；
  `out` 输入按头补零；`ff_h` / `ff_g` 取 GEGLU 投影的前 / 后一半。`lin_b|模块|方式`：f32；`zero_b|C`：全零 f32（q / k / v 没有 bias）。
- `kv_w|模块`：交叉注意力的 to_k / to_v，calblas linear! `w = hf` 的 HF 原样 [N][768]，行按头补零。
- `norm_w|模块`、`norm_b|模块`：GroupNorm / LayerNorm 的 w、b（f32）；`gmask|C`：GroupNorm 的组掩码 f32 [32][C]。
- `amask|h`：自注意力的 key 掩码 bf16 [T]（带边框的图当 token：内部 0、边框与补齐 −MAX）；`amask|x`：交叉注意力的（77 行 0、补的 −MAX）。
- 时间步：`temb_fp`（sin / cos 的常数表，见 temb.py）、`l1_*`、`l2_*`（time_embedding 的两层，K 补到 384）、
  `lp_*`（22 个 ResNet 的 time_emb_proj 按执行次序竖着拼成一个 [NB][1280]）、`c1_b`（22 个 conv1 的 bias 拼起来，设备上加上投影后当 conv1 的 bias）。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import ROOT, bf16, temb

PLAN = ROOT / "crates" / "unet" / "plan.json"
KERNELS_RS = ROOT / "crates" / "unet" / "src" / "kernels.rs"
PARAMS = ROOT / "out" / "unet" / "params"
CALBINS = ROOT / "out" / "unet-real"


def plan() -> dict:
    return json.loads(PLAN.read_text())


def stage_cvx(feature: str) -> Path:
    """第几段的 .cvx：带 feature 的用自己的 target 目录（同 tools/calbin.sh）。"""
    return ROOT / "target" / "features" / f"unet-{feature}" / "dopamine-unknown-none-cvx" / "npu" / "unet.cvx"


def tpad(h: int, B: int = 2) -> int:
    return -(-((h + 2 * B) ** 2) // 128) * 128


# ───────────────────────── 权重 ─────────────────────────

def _bf(a) -> bytes:
    return bf16.bits(np.asarray(a, np.float32)).astype("<u2").tobytes()


def _f32(a) -> bytes:
    return np.asarray(a, np.float32).astype("<f4").tobytes()


class Packer:
    """权重条目 ⇒ 字节（按 plan.json 的键）。"""

    def __init__(self, unet, P: dict):
        self.P = P
        self.c = P["consts"]
        self.mods = dict(unet.named_modules())
        self.unet = unet

    def _head_rows(self, w: np.ndarray, c: int) -> np.ndarray:
        """[c, …] 的行按头补零：头 h 的第 d 行 ⇒ h·Dp + d。"""
        H, dp = self.c["HEADS"], self.c["DP"][str(c)]
        dr = c // H
        out = np.zeros((H * dp,) + w.shape[1:], np.float32)
        for h in range(H):
            out[h * dp:h * dp + dr] = w[h * dr:(h + 1) * dr]
        return out

    def entry(self, key: str) -> bytes:
        kind, _, rest = key.partition("|")
        m = self.mods.get(rest.split("|")[0]) if rest else None
        meta = self.P["wmeta"].get(key, {})
        if kind == "conv_w":
            w = m.weight.detach().numpy()
            co, ci, k, _ = w.shape
            if meta.get("hf"):  # 宽通道走 linear!（DESIGN D22）：每个抽头一块 HF 原样 [N][Cin]，抽头按 ky·k + kx 排；补的输出通道是 0
                out = np.zeros((k * k, meta["n"], meta["cin"]), np.float32)
                out[:, :co, :ci] = w.transpose(2, 3, 0, 1).reshape(k * k, co, ci)
                return _bf(out)
            out = np.zeros((k, k, meta["cin"], meta["cout"]), np.float32)  # 补的输入 / 输出通道是 0（conv_in 4 ⇒ 32、conv_out 4 ⇒ 32、x3 640 ⇒ 768）
            out[:, :, :ci, :co] = w.transpose(2, 3, 1, 0)
            return _bf(out)
        if kind == "conv_b":
            b = m.bias.detach().numpy()
            out = np.zeros(meta.get("n", meta["cout"]), np.float32)
            out[:b.size] = b
            return _f32(out)
        if kind in ("lin_w", "lin_b"):
            mode = rest.split("|")[1]
            w = m.weight.detach().numpy()
            if w.ndim == 4:  # 1×1 的 Conv2d（proj_in）当 Linear
                w = w[:, :, 0, 0]
            b = m.bias.detach().numpy() if m.bias is not None else None
            n = w.shape[0]
            if mode in ("ff_h", "ff_g"):
                sl = slice(0, n // 2) if mode == "ff_h" else slice(n // 2, n)
                w, b = w[sl], (b[sl] if b is not None else None)
            if kind == "lin_b":
                out = np.zeros(meta.get("n", b.size), np.float32)  # linear! 的 N 补到 128 的倍数时补 0
                out[:b.size] = b
                return _f32(out)
            if mode == "qkv":
                kn = self._head_rows(w, w.shape[0]).T  # [K][H·Dp]
            elif mode == "out":
                kn = self._head_rows(w.T, w.shape[1])  # [H·Dp][N]
            else:
                kn = w.T
            full = np.zeros((meta["cin"], meta.get("n", meta["cout"])), np.float32)  # 补的输入通道（行）、补的输出通道（列）是 0
            full[:kn.shape[0], :kn.shape[1]] = kn
            # 1×1 卷积要 [K][N]；calblas linear! 的 `w = hf` 要 HF 原样 [N][K]
            return _bf(full.T if meta.get("hf") else full)
        if kind == "zero_b":
            return _f32(np.zeros(int(rest), np.float32))
        if kind == "kv_w":
            w = m.weight.detach().numpy()
            return _bf(self._head_rows(w, w.shape[0]))
        if kind in ("norm_w", "norm_b"):
            return _f32((m.weight if kind == "norm_w" else m.bias).detach().numpy())
        if kind == "gmask":
            c = int(rest)
            mk = np.zeros((32, c), np.float32)
            for gi in range(32):
                mk[gi, gi * (c // 32):(gi + 1) * (c // 32)] = 1
            return _f32(mk)
        if kind == "amask":
            neg = np.full(1, self.c["NEG_BF16"], "<u2")
            if rest == "x":
                u = np.zeros(self.c["CTX_ROWS"], "<u2")
                u[self.c["CTX_REAL"]:] = neg
                return u.tobytes()
            h, B = int(rest), self.c["B"]
            wb = h + 2 * B
            u = np.full(tpad(h, B), neg[0], "<u2")
            idx = np.arange(wb * wb).reshape(wb, wb)[B:B + h, B:B + h].ravel()
            u[idx] = 0
            return u.tobytes()
        te = self.unet.time_embedding
        if key == "temb_fp":
            return temb.tables(160)
        if key == "l1_w":
            w = np.zeros((1280, self.c["TEMB_K"]), np.float32)
            w[:, :320] = te.linear_1.weight.detach().numpy()
            return _bf(w)
        if key in ("l1_b", "l2_b"):
            return _f32((te.linear_1 if key == "l1_b" else te.linear_2).bias.detach().numpy())
        if key == "l2_w":
            return _bf(te.linear_2.weight.detach().numpy())
        if key in ("lp_w", "lp_b", "c1_b"):
            nb = self.P["nb"]
            rs = resnets(self.unet)
            if key == "lp_w":
                out = np.zeros((nb, 1280), np.float32)
                for r, o in zip(rs, self.P["toffs"]):
                    out[o:o + r.conv1.out_channels] = r.time_emb_proj.weight.detach().numpy()
                return _bf(out)
            out = np.zeros(nb, np.float32)
            for r, o in zip(rs, self.P["toffs"]):
                src = r.time_emb_proj.bias if key == "lp_b" else r.conv1.bias
                out[o:o + r.conv1.out_channels] = src.detach().numpy()
            return _f32(out)
        raise KeyError(f"不认得的权重条目 {key}")


def resnets(unet) -> list:
    out = []
    for blk in unet.down_blocks:
        out += list(blk.resnets)
    out += list(unet.mid_block.resnets)
    for blk in unet.up_blocks:
        out += list(blk.resnets)
    return out


def write_params(unet, P: dict, out_dir: Path = PARAMS) -> list[Path]:
    """各段的参数块 out/unet/params/<段>/param_blk0.bin（按 plan.json 的偏移）。"""
    pk = Packer(unet, P)
    paths = []
    cache = {}
    for st in P["stages"]:
        buf = bytearray(st["w"]["bytes"])
        for key, (off, n) in st["w"]["entries"].items():
            if key not in cache:
                cache[key] = pk.entry(key)
            d = cache[key]
            if len(d) != n:
                raise ValueError(f"{key}：打出 {len(d)} 字节，plan 要 {n}")
            buf[off:off + n] = d
        p = out_dir / st["feature"] / "param_blk0.bin"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes(buf))
        paths.append(p)
    (out_dir / "stamp.json").write_text(json.dumps({"plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(), "time": time.strftime("%F %T")}))
    return paths


def params_fresh(out_dir: Path = PARAMS) -> bool:
    s = out_dir / "stamp.json"
    return s.exists() and json.loads(s.read_text())["plan_sha256"] == hashlib.sha256(PLAN.read_bytes()).hexdigest()


# ───────────────────────── 输入、输出 ─────────────────────────

def pack_inputs(P: dict, x, t: float, ctxs: list[np.ndarray]) -> dict:
    """主机给的张量：{(名字, 路): 字节}。x：[4, 64, 64]（两路同一个）或两路各一个的列表；ctxs：两路的上下文 [77, 768]（有提示词、空提示词）。
    两路共用一个时间步 t（CFG 的两路总是同一个 t）。latent 打成带边框的 NHWC bf16（4 通道补到 32），上下文打成 [128][768] bf16（补的行是 0）。"""
    c, bufs, ins = P["consts"], P["bufs"], P["inputs"]
    B = c["B"]
    h = 64
    xs = list(x) if isinstance(x, (list, tuple)) else [x, x]
    store = {}
    for br, ctx in enumerate(ctxs):
        xm = np.zeros(bufs[ins["x"]]["bytes"] // 2, "<u2")
        v = xm[:(h + 2 * B) ** 2 * 32].reshape(h + 2 * B, h + 2 * B, 32)
        v[B:B + h, B:B + h, :4] = bf16.bits(np.asarray(xs[br], np.float32).transpose(1, 2, 0))
        store[(ins["x"], br)] = xm.tobytes()
        cm = np.zeros(bufs[ins["ctx"]]["bytes"] // 2, "<u2")
        cm[:c["CTX_REAL"] * c["CTX_DIM"]] = bf16.bits(np.asarray(ctx, np.float32)).ravel()
        store[(ins["ctx"], br)] = cm.tobytes()
    tb = np.zeros(bufs[ins["t"]]["bytes"] // 4, "<f4")
    tb[:8] = np.float32(t)
    store[(ins["t"], None)] = tb.tobytes()
    return store


def unpack_eps(P: dict, store: dict) -> list[np.ndarray]:
    name, B, h = P["outputs"]["eps"], P["consts"]["B"], 64
    out = []
    for br in (0, 1):
        u = np.frombuffer(store[(name, br)], "<u2")[:(h + 2 * B) ** 2 * 32].reshape(h + 2 * B, h + 2 * B, 32)
        out.append(bf16.from_bits(u[B:B + h, B:B + h, :4]).transpose(2, 0, 1).copy())
    return out


def assemble(P: dict, st: dict, store: dict) -> bytes:
    """这一段的 in_x：按偏移把主机与前面各段出的张量拼起来。"""
    r = st["in"]
    buf = bytearray(r["bytes"])
    for name, off in r["sh"].items():
        d = store[(name, None)]
        buf[off:off + len(d)] = d
    for name, off in r["br"].items():
        for br in (0, 1):
            d = store[(name, br)]
            at = r["br_base"] + br * r["br_stride"] + off
            buf[at:at + len(d)] = d
    return bytes(buf)


def scatter(P: dict, st: dict, out: bytes, store: dict):
    """这一段的 out_y ⇒ 各张量（存起来给后面的段）。"""
    r, bufs = st["out"], P["bufs"]
    for name, off in r["sh"].items():
        store[(name, None)] = out[off:off + bufs[name]["bytes"]]
    for name, off in r["br"].items():
        for br in (0, 1):
            at = r["br_base"] + br * r["br_stride"] + off
            store[(name, br)] = out[at:at + bufs[name]["bytes"]]


def run(P: dict, x, t, ctxs, backend: str, params: list[Path], log_dir: Path | None = None, verbose: bool = False) -> tuple[list, dict]:
    """一次 UNet 前向（CFG 两路）：各段依次跑。backend：`npu-sim`（numpy 模拟 .cvx，不是 NPU；params 是各段参数块）或
    `npu`（真卡，dort；参数块已经打进 out/unet-real/<段>/cal，params 不用）。"""
    store = pack_inputs(P, x, t, ctxs)
    info = {"backend": backend, "stages": {}}
    t0 = time.time()
    for i, st in enumerate(P["stages"]):
        data = assemble(P, st, store)
        f = st["feature"]
        if backend == "npu-sim":
            from . import cvxsim

            sim = cvxsim.simulate(stage_cvx(f), KERNELS_RS, {}, {"in_x": data, "w": params[i].read_bytes()}, verbose=verbose)
            out = sim.mem["out_y"].tobytes()
            info["stages"][f] = sim.report()
            del sim
        elif backend == "npu":
            from . import dort

            cal = CALBINS / f / "cal"
            out, d = dort.run_raw(cal, "unet", data, "in_x", "out_y", log=(log_dir or CALBINS) / "dort.log")
            info["stages"][f] = round(d["dort_wall_s"], 3)
        else:
            raise ValueError(backend)
        scatter(P, st, out, store)
    info["seconds"] = time.time() - t0
    return unpack_eps(P, store), info
