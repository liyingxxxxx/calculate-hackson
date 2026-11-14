"""SD1.5 UNet（B3）设备程序的生成器：从 diffusers 的模块结构走一遍，出 kernel 特化、算子序列、按代码量自动切段的各段入口与偏移表。

为什么生成：UNet 有约 240 个卷积类算子、61 个 GroupNorm、48 个 LayerNorm、32 个注意力，手写入口和偏移表既慢又容易错；
docc 每核的代码不能超过约 1 MB（DESIGN F22），要按各 kernel 的实际代码量切段。**偏移只算一份**：生成器算好，写进
`crates/unet/src/step.rs`（设备，字面量）与 `crates/unet/plan.json`（主机打包、执行器读），两边出自同一次计算（DESIGN D20）。

切法照 DESIGN D18：全程带边框的 NHWC 图；投影都当 1×1 卷积；注意力把整张带边框的图当 token（边框、补齐用掩码去掉，头维补零到 32 的倍数）；
GroupNorm / LayerNorm 用本仓的 NHWC 版；GEGLU = 两个 1×1 卷积 + `geglu!`；时间步那一路在第一段算一次，全部 ResNet 的 conv1 bias 一起折好往后传。
CFG 的两路（有提示词、空提示词）在**同一次调用里**先后跑（D19）：权重部署一次、两路共用；段内的临时图两路共用
（只写内部 ⇒ 边框一直是 0；注意力的输出图边框不是 0，但只有 1×1 卷积读它，只读内部）。

段间：每段的 in_x 是「这一段要读、前面的段（或主机）出的」张量，out_y 是「这一段出的、后面的段要读的」张量（跳连跨好几段时由主机转手，
不经中间的段）；主机只按 plan.json 的偏移搬字节，不算任何东西。

用法（先 source env.sh；参照环境的 Python：`$CALCULET_HOME/venv/bin/python -I tools/unet_gen.py …`）：
  kernels      ⇒ crates/unet/src/kernels.rs（全部特化）
  probe        量各 kernel 的代码大小：把 kernel 分批编成几个探针程序（crates/unet 的 z* feature），从 ELF 的符号表按 kernel 归账
               ⇒ crates/unet/kernel_sizes.json（切段用；kernel 变了要重量）
  program      ⇒ 切段、crates/unet/src/step.rs、Cargo.toml 的 feature、crates/unet/plan.json
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNET = ROOT / "crates" / "unet"
B = 2
MARGIN_PX = 128  # 卷积的快路会多读输入末尾 16 个像素：每张图后面留余量（同 VAE）
HEADS = 8
DP = {320: 64, 640: 96, 1280: 160}  # 头维补零到 32 的倍数（真头维 40 / 80 / 160）
CTX_ROWS, CTX_DIM, CTX_REAL = 128, 768, 77  # 上下文 77 个 token 补到 128（key 行数要是 128 的倍数），补的用掩码去掉
EPS_RES, EPS_T2D, EPS_LN = 1e-5, 1e-6, 1e-5
TEMB_K = 384  # 时间步嵌入 320 维补到 384（linear! 的 K 要 128 的倍数）
TROWS = 16  # 时间步那一路的 linear! 至少 16 行（只有第 0 行有用）
NEG_BF16 = 0xFF61  # bf16 的 −3.0e38（注意力掩码；calblas 的 NEG）
BUDGET = 900 * 1024  # 每核代码的上限（docc 0.4.1 的 JAL ±1 MB，留一成余量）
BASE_CODE = 4 * 1024  # 入口外的固定代码（运行库、main）
CALL_CODE = 192  # 入口里每次 kernel 调用（含屏障、4–6 个指针参数）约多少字节：探针程序的入口实测约 180 B / 次


def a4k(n: int) -> int:
    return (n + 4095) // 4096 * 4096


def a16k(n: int) -> int:
    return (n + 16383) // 16384 * 16384


def tpad(h: int) -> int:
    """带边框的图当 token：(h + 4)² 补到 128 的倍数。"""
    return -(-((h + 2 * B) ** 2) // 128) * 128


def map_bytes(h: int, c: int) -> int:
    """一张带边框的图在内存里占多少：token 补齐的行 + 余量（都清零）。"""
    return a4k((tpad(h) + MARGIN_PX) * c * 2)


# ───────────────────────── kernel 特化 ─────────────────────────

class Kernels:
    """名字 ⇒ {kind, spec, macro}。名字由形状定（同形状共用一份代码）。"""

    def __init__(self):
        self.k = {}

    def add(self, name, kind, spec, macro):
        if name in self.k:
            assert self.k[name]["spec"] == spec, name
        else:
            self.k[name] = {"kind": kind, "spec": spec, "macro": macro}
        return name

    def conv(self, h, cin, cout, k, s=1, res=False, pout=None, pres=None):
        """cin 同时是输入图的行距（补的通道是 0、权重也补 0）；pout / pres：输出、残差图的行距（缺省 = cout）。"""
        pout, pres = pout or cout, pres or cout
        name = (f"un_c{k}_{h}_{cin}_{cout}" + ("_s2" if s == 2 else "") + ("_r" if res else "")
                + (f"_o{pout}" if pout != cout else "") + (f"_q{pres}" if res and pres != pout else ""))
        r = f", res = ({pres}, 0)" if res else ""
        assert not (k == 1 and h == 32 and cin == 640), "32² × 640 通道输入的 1×1 卷积 docc 排不下（DESIGN F26），走 lin_rows"
        return self.add(name, "conv2d", {"h": h, "cin": cin, "cout": cout, "k": k, "s": s, "res": res, "pout": pout, "pres": pres},
                        f"sd_kernels::conv2d!(pub {name}: h = {h}, w = {h}, cin = {cin}, cout = {cout}, k = {k}, s = {s}, "
                        f"pin = ({cin}, 0), pout = ({pout}, 0), act = none{r});")

    def lin_rows(self, h, k, n, bias):
        """带边框的图整张当稠密的行（tpad(h) 行）过 calblas linear!（w = hf）：边框行也算（无 bias 时输入边框是 0 ⇒ 输出边框还是 0）。"""
        t = tpad(h)
        name = f"un_lr_{h}_{k}_{n}" + ("_b" if bias else "")
        return self.add(name, "linear", {"rows": t, "k": k, "n": n, "bias": bias},
                        f"calblas::linear!(pub {name}: rows = {t}, k = {k}, n = {n}, w = hf, cores = 2{', bias = f32' if bias else ''});")

    def gn(self, h, c, act):
        eps = EPS_RES if act == "silu" else EPS_T2D
        st = self.add(f"un_gs_{h}_{c}", "gn_stats", {"h": h, "c": c}, f"sd_kernels::gn_stats!(pub un_gs_{h}_{c}: h = {h}, w = {h}, c = {c}, p = {c});")
        ap = self.add(f"un_ga_{h}_{c}_{act}", "gn_apply", {"h": h, "c": c, "act": act},
                      f"sd_kernels::gn_apply!(pub un_ga_{h}_{c}_{act}: h = {h}, w = {h}, c = {c}, p = {c}, groups = 32, eps = {eps:g}f32, act = {act});")
        return st, ap

    def ln(self, h, c):
        return self.add(f"un_ln_{h}_{c}", "ln_nhwc", {"h": h, "c": c},
                        f"sd_kernels::ln_nhwc!(pub un_ln_{h}_{c}: h = {h}, w = {h}, c = {c}, p = {c}, eps = {EPS_LN:g}f32);")

    def attn(self, h, c, cross):
        """本仓 flash_attn!（DESIGN D21）：一块 K / V 给 64 行 q 用（两遍法，数值同 calblas attention!；K 按原样 [key, 头 · d] 读）。
        calblas attention! 每 16 行 q 把每块 key 走一串指令，64² 的自注意力一次 625 ms（0.07 TFLOPS，DESIGN F27）。"""
        t, d = tpad(h), DP[c]
        tk = CTX_ROWS if cross else t
        name = f"un_at_{h}_{'x' if cross else 's'}"
        return self.add(name, "flash_attn", {"h": h, "c": c, "cross": cross},
                        f"sd_kernels::flash_attn!(pub {name}: rows = {t}, keys = {tk}, heads = {HEADS}, head = {d}, scale = {c // HEADS});")

    def tr(self, rows, cols):
        name = f"un_tr_{rows}_{cols}"
        return self.add(name, "transpose2d", {"rows": rows, "cols": cols}, f"calblas::transpose2d!(pub {name}: rows = {rows}, cols = {cols});")

    def kv(self, n):
        return self.add(f"un_kv_{n}", "linear", {"rows": CTX_ROWS, "k": CTX_DIM, "n": n, "bias": False},
                        f"calblas::linear!(pub un_kv_{n}: rows = {CTX_ROWS}, k = {CTX_DIM}, n = {n}, w = hf, cores = 2);")

    def geglu(self, h, c4):
        n = (h + 2 * B) ** 2 * c4
        return self.add(f"un_gg_{h}", "geglu", {"n": n}, f"sd_kernels::geglu!(pub un_gg_{h}: n = {n});")

    def up(self, h, c, cw):
        return self.add(f"un_up_{h}_{c}_{cw}", "upsample2", {"h": h, "c": cw, "p": c},
                        f"calblas::upsample2!(pub un_up_{h}_{c}_{cw}: h = {h}, w = {h}, c = {cw}, pin = {c}, pout = {c});")

    def copy(self, h, c, pout):
        name = f"un_cp_{h}_{c}_{pout}"
        return self.add(name, "copy_win", {"h": h, "c": c, "pout": pout},
                        f"sd_kernels::copy_win!(pub {name}: h = {h}, w = {h}, c = {c}, pin = {c}, pout = {pout});")

    def lin16(self, k, n):
        name = f"un_l16_{k}_{n}"
        return self.add(name, "linear", {"rows": TROWS, "k": k, "n": n, "bias": True},
                        f"calblas::linear!(pub {name}: rows = {TROWS}, k = {k}, n = {n}, w = hf, cores = 2, bias = f32);")

    def misc(self, nb):
        self.add("un_temb", "timestep_embed", {}, "sd_kernels::timestep_embed!(pub un_temb: half = 160);")
        self.add("un_silu", "unary", {}, f"calblas::unary!(pub un_silu: silu, n = {TROWS * 1280});")
        self.add("un_bf", "bias_fold", {}, f"sd_kernels::bias_fold!(pub un_bf: n = {nb});")
        self.add("un_zero", "fill16k", {}, "calblas::fill16k!(pub un_zero: value = 0.0f32);")


# 只让 0 号核调的（不分行的单核 kernel）；别的两核都调（cores = 2，或按 num_cores() 分行的单核 kernel）
CORE0 = {"timestep_embed", "unary", "bias_fold"}


# ───────────────────────── 算子序列（一路）─────────────────────────

class Graph:
    """张量（名字 ⇒ 描述）+ 算子（执行次序，按「单元」分组：只在单元之间切段）。
    张量的 scope：`br` 每路一份（CFG 的两路各一份）、`sh` 两路共用（时间步那一路、GN 的统计）。"""

    def __init__(self):
        self.bufs = {}
        self.ops = []
        self.units = []
        self.wsize = {}  # 权重条目 ⇒ 字节
        self.wmeta = {}  # 权重条目 ⇒ 形状（主机打包用：cin / cout 含补的通道）

    def map(self, h, c, tag="m", scope="br", p=None):
        """带边框的图 [h + 4][h + 4][p]，前 c 个通道有用（p > c 时补的通道一直是 0）。"""
        name = f"{tag}{len(self.bufs)}"
        p = p or c
        self.bufs[name] = {"kind": "map", "h": h, "c": c, "p": p, "bytes": map_bytes(h, p), "scope": scope}
        return name

    def dense(self, nbytes, tag, scope="br"):
        name = f"{tag}{len(self.bufs)}"
        self.bufs[name] = {"kind": "dense", "bytes": a4k(nbytes), "scope": scope}
        return name

    def unit(self, name, shared=False):
        self.units.append({"name": name, "shared": shared})

    def op(self, K, kernel, args):
        """args：("b", 张量, 字节偏移) 或 ("w", 权重条目)；最后一个 ("b", …) 是输出（copy / 上采样的窗口也是）。"""
        self.ops.append({"unit": len(self.units) - 1, "kernel": kernel, "kind": K.k[kernel]["kind"], "args": args})

    def w(self, key, nbytes, **meta):
        if key in self.wsize:
            assert self.wsize[key] == nbytes and self.wmeta[key] == meta, key
        self.wsize[key] = nbytes
        self.wmeta[key] = meta
        return ("w", key)


def b(name, off=0):
    return ("b", name, off)


def resnet_list(unet):
    """全部 ResNet（执行次序）。时间步投影按这个次序拼成一个大 Linear。"""
    out = []
    for blk in unet.down_blocks:
        out += list(blk.resnets)
    out += list(unet.mid_block.resnets)
    for blk in unet.up_blocks:
        out += list(blk.resnets)
    return out


def nb_total(unet) -> tuple[int, list[int]]:
    """全部 ResNet 的 conv1 输出通道之和（补到 256 的倍数：bias_fold! 与两核的 linear! 块数要求）与各自的起点。"""
    offs, at = [], 0
    for r in resnet_list(unet):
        offs.append(at)
        at += r.conv1.out_channels
    return -(-at // 256) * 256, offs


def build(unet):
    """走一遍 UNet：登记 kernel 特化与一路的算子序列。返回 (Kernels, Graph, 附加信息)。"""
    K, g = Kernels(), Graph()
    for name, mod in unet.named_modules():
        mod._sdk = name
    nb, toffs = nb_total(unet)
    K.misc(nb)

    # ── 时间步那一路（两路共用，只在第一段算一次）──
    g.unit("time", shared=True)
    t_in = g.dense(64, "t", "sh")
    te = g.dense(TROWS * TEMB_K * 2, "temb", "sh")
    e1, e1s = g.dense(TROWS * 1280 * 2, "e", "sh"), g.dense(TROWS * 1280 * 2, "e", "sh")
    e2, e2s = g.dense(TROWS * 1280 * 2, "e", "sh"), g.dense(TROWS * 1280 * 2, "e", "sh")
    tp = g.dense(TROWS * nb * 2, "tp", "sh")
    tb = g.dense(nb * 4, "tb", "sh")
    g.op(K, "un_temb", [b(t_in), g.w("temb_fp", 2 * TEMB_K * 4), b(te)])
    g.op(K, K.lin16(TEMB_K, 1280), [b(te), g.w("l1_w", 1280 * TEMB_K * 2), g.w("l1_b", 1280 * 4), b(e1)])
    g.op(K, "un_silu", [b(e1), b(e1s)])
    g.op(K, K.lin16(1280, 1280), [b(e1s), g.w("l2_w", 1280 * 1280 * 2), g.w("l2_b", 1280 * 4), b(e2)])
    g.op(K, "un_silu", [b(e2), b(e2s)])
    g.op(K, K.lin16(1280, nb), [b(e2s), g.w("lp_w", nb * 1280 * 2), g.w("lp_b", nb * 4), b(tp)])
    g.op(K, "un_bf", [g.w("c1_b", nb * 4), b(tp), b(tb)])

    st = g.dense(4 * 2560 * 4, "st", "sh")  # GN 的统计（每核 Σx、Σx²），最宽 2560 通道
    hw = lambda name: (g.bufs[name]["h"], g.bufs[name]["c"])  # noqa: E731

    pp = lambda name: g.bufs[name]["p"]  # noqa: E731

    def conv(m, x, y, res=None, bias=None, s=1):
        k = m.kernel_size[0]
        cx = pp(x)  # 输入的补齐通道也乘（权重补 0）
        hy, cy = hw(y)
        name = K.conv(hy, cx, cy, k, s=s, res=res is not None, pout=pp(y), pres=pp(res) if res else None)
        args = [b(x), g.w(f"conv_w|{m._sdk}", k * k * cx * cy * 2, cin=cx, cout=cy, k=k), bias or g.w(f"conv_b|{m._sdk}", cy * 4, cout=cy)]
        if res is not None:
            args.append(b(res))
        g.op(K, name, args + [b(y)])

    def lin(m, x, y, mode, res=None):
        """Linear（或 1×1 的 Conv2d）当 1×1 卷积。mode：plain / qkv（输出按头补零，无 bias）/ out（输入按头补零）/ ff_h、ff_g（GEGLU 投影的两半）。
        32² × 640 通道输入、不带残差的走 calblas linear!（整张带边框的图当稠密行，DESIGN F26）：输出的边框行是 bias（qkv 无 bias ⇒ 还是 0），
        这几张图（proj_in 的输出、q / k / v、前馈的两半）只有 LayerNorm、注意力、geglu、1×1 卷积读，边框不碍事。"""
        cx = pp(x)
        hy, cy = hw(y)
        rows = hy == 32 and cx == 640 and res is None
        meta = {"cin": cx, "cout": cy, "mode": mode, "hf": rows}
        if rows:
            assert pp(y) == cy
            name = K.lin_rows(hy, cx, cy, bias=mode != "qkv")
            args = [b(x), g.w(f"lin_w|{m._sdk}|{mode}|hf", cx * cy * 2, **meta)]
            if mode != "qkv":
                args.append(g.w(f"lin_b|{m._sdk}|{mode}", cy * 4, mode=mode))
            g.op(K, name, args + [b(y)])
            return
        name = K.conv(hy, cx, cy, 1, res=res is not None, pout=pp(y), pres=pp(res) if res else None)
        bias = g.w(f"zero_b|{cy}", cy * 4) if mode == "qkv" else g.w(f"lin_b|{m._sdk}|{mode}", cy * 4, mode=mode)
        args = [b(x), g.w(f"lin_w|{m._sdk}|{mode}", cx * cy * 2, **{**meta, "hf": False}), bias]
        if res is not None:
            args.append(b(res))
        g.op(K, name, args + [b(y)])

    def gn(m, x, y, act):
        h, c = hw(x)
        gs, ga = K.gn(h, c, act)
        g.op(K, gs, [b(x), b(st)])
        g.op(K, ga, [b(x), b(st), g.w(f"norm_w|{m._sdk}", c * 4), g.w(f"norm_b|{m._sdk}", c * 4), g.w(f"gmask|{c}", 32 * c * 4), b(y)])

    def ln(m, x, y):
        h, c = hw(x)
        g.op(K, K.ln(h, c), [b(x), g.w(f"norm_w|{m._sdk}", c * 4), g.w(f"norm_b|{m._sdk}", c * 4), b(y)])

    def resnet(r, ri, x):
        h, cin = hw(x)
        cout = r.conv1.out_channels
        a = g.map(h, cin)
        gn(r.norm1, x, a, "silu")
        t = g.map(h, cout)
        conv(r.conv1, a, t, bias=b(tb, toffs[ri] * 4))
        u = g.map(h, cout)
        gn(r.norm2, t, u, "silu")
        sc = x
        if r.conv_shortcut is not None:
            sc = g.map(h, cout)
            conv(r.conv_shortcut, x, sc)
        o = g.map(h, cout)
        conv(r.conv2, u, o, res=sc)
        return o

    def attention(a, x, ctx, cross):
        h, c = hw(x)
        hd = HEADS * DP[c]
        q = g.map(h, hd)
        lin(a.to_q, x, q, "qkv")
        if cross:
            k, v = g.dense(CTX_ROWS * hd * 2, "k"), g.dense(CTX_ROWS * hd * 2, "v")
            for m, buf in ((a.to_k, k), (a.to_v, v)):
                g.op(K, K.kv(hd), [b(ctx), g.w(f"kv_w|{m._sdk}", hd * CTX_DIM * 2), b(buf)])
        else:
            k, v = g.map(h, hd), g.map(h, hd)
            lin(a.to_k, x, k, "qkv")
            lin(a.to_v, x, v, "qkv")
        o = g.map(h, hd)
        mk = g.w("amask|x", CTX_ROWS * 2) if cross else g.w(f"amask|{h}", tpad(h) * 2)
        g.op(K, K.attn(h, c, cross), [b(q), b(k), b(v), mk, b(o)])
        return o

    def block(blk, x, ctx, name):
        """BasicTransformerBlock。拆成三个单元（自注意力 / 交叉注意力 / 前馈）：64² 的一个 Transformer 代码量超过一段（flash_attn! 约 350 KB）。"""
        h, c = hw(x)
        n1 = g.map(h, c)
        ln(blk.norm1, x, n1)
        o = attention(blk.attn1, n1, ctx, False)
        x1 = g.map(h, c)
        lin(blk.attn1.to_out[0], o, x1, "out", res=x)
        g.unit(f"{name}.cross")
        n2 = g.map(h, c)
        ln(blk.norm2, x1, n2)
        o2 = attention(blk.attn2, n2, ctx, True)
        x2 = g.map(h, c)
        lin(blk.attn2.to_out[0], o2, x2, "out", res=x1)
        g.unit(f"{name}.ff")
        n3 = g.map(h, c)
        ln(blk.norm3, x2, n3)
        hh, gg, yy = g.map(h, 4 * c), g.map(h, 4 * c), g.map(h, 4 * c)
        proj = blk.ff.net[0].proj
        lin(proj, n3, hh, "ff_h")
        lin(proj, n3, gg, "ff_g")
        g.op(K, K.geglu(h, 4 * c), [b(hh), b(gg), b(yy)])
        # 32² 时 x3 补到 768 通道：proj_out（1×1，640 通道输入）docc 排不下，补成 768 通道输入就排得下（DESIGN F26）
        x3 = g.map(h, c, p=768 if (h == 32 and c == 640) else None)
        lin(blk.ff.net[2], yy, x3, "plain", res=x2)
        return x3

    def t2d(t, x, ctx, name):
        assert not t.use_linear_projection and len(t.transformer_blocks) == 1
        h, c = hw(x)
        a = g.map(h, c)
        gn(t.norm, x, a, "none")
        p = g.map(h, c)
        lin(t.proj_in, a, p, "plain")  # 1×1 的 Conv2d：同 Linear（32² 时走 linear!）
        y = block(t.transformer_blocks[0], p, ctx, name)
        o = g.map(h, c)
        conv(t.proj_out, y, o, res=x)
        return o

    def cat(x, skip):
        h, c1 = hw(x)
        _, c2 = hw(skip)
        y = g.map(h, c1 + c2)
        g.op(K, K.copy(h, c1, c1 + c2), [b(x), b(y)])
        g.op(K, K.copy(h, c2, c1 + c2), [b(skip), b(y, c1 * 2)])
        return y

    def upsample(x):
        h, c = hw(x)
        y = g.map(2 * h, c)
        at = 0
        while at < c:
            cw = min(256, c - at)
            g.op(K, K.up(h, c, cw), [b(x, at * 2), b(y, at * 2)])
            at += cw
        return y

    g.unit("conv_in")
    x_in = g.map(64, 32, "x")
    ctx = g.dense(CTX_ROWS * CTX_DIM * 2, "ctx")
    x = g.map(64, 320)
    conv(unet.conv_in, x_in, x)
    skips = [x]
    rs = iter(range(len(resnet_list(unet))))
    for i, blk in enumerate(unet.down_blocks):
        atts = getattr(blk, "attentions", None)
        for j, r in enumerate(blk.resnets):
            g.unit(f"down{i}.res{j}")
            x = resnet(r, next(rs), x)
            if atts is not None:
                g.unit(f"down{i}.attn{j}")
                x = t2d(atts[j], x, ctx, f"down{i}.attn{j}")
            skips.append(x)
        if blk.downsamplers is not None:
            g.unit(f"down{i}.down")
            h, c = hw(x)
            y = g.map(h // 2, c)
            conv(blk.downsamplers[0].conv, x, y, s=2)
            x = y
            skips.append(x)
    mb = unet.mid_block
    g.unit("mid.res0")
    x = resnet(mb.resnets[0], next(rs), x)
    g.unit("mid.attn")
    x = t2d(mb.attentions[0], x, ctx, "mid.attn")
    g.unit("mid.res1")
    x = resnet(mb.resnets[1], next(rs), x)
    for i, blk in enumerate(unet.up_blocks):
        atts = getattr(blk, "attentions", None)
        for j, r in enumerate(blk.resnets):
            g.unit(f"up{i}.res{j}")
            x = cat(x, skips.pop())
            x = resnet(r, next(rs), x)
            if atts is not None:
                g.unit(f"up{i}.attn{j}")
                x = t2d(atts[j], x, ctx, f"up{i}.attn{j}")
        if blk.upsamplers is not None:
            g.unit(f"up{i}.up")
            u = upsample(x)
            y = g.map(*hw(u))
            conv(blk.upsamplers[0].conv, u, y)
            x = y
    g.unit("out")
    a = g.map(64, 320)
    gn(unet.conv_norm_out, x, a, "silu")
    eps = g.map(64, 32, "eps")
    conv(unet.conv_out, a, eps)
    assert not skips and next(rs, None) is None
    info = {"nb": nb, "toffs": toffs, "inputs": {"x": x_in, "ctx": ctx, "t": t_in}, "outputs": {"eps": eps}, "st": st}
    return K, g, info


def load_unet():
    import torch
    from diffusers import UNet2DConditionModel

    sys.path.insert(0, str(ROOT / "host"))
    from sd15npu import runner

    return UNet2DConditionModel.from_pretrained(str(runner.models_dir() / "stable-diffusion-v1-5" / "unet"), torch_dtype=torch.float32).eval()


# ───────────────────────── kernels.rs ─────────────────────────

KIND_ORDER = ["conv2d", "gn_stats", "gn_apply", "ln_nhwc", "flash_attn", "transpose2d", "linear", "geglu", "upsample2", "copy_win",
              "timestep_embed", "unary", "bias_fold", "fill16k"]


def cmd_kernels(a):
    K, g, info = build(load_unet())
    lines = ["//! SD1.5 UNet 的 kernel 特化（**`tools/unet_gen.py kernels` 生成，不要手改**）。形状是编译期常量：每个形状一份，名字带 `un_` 前缀。",
             "//! 模拟器（`host/sd15npu/cvxsim.py`）按这里的宏参数认 kernel 的语义（字面量）；calbin 里只有这一段调到的 kernel 进 ELF。",
             "//! 卷积全是 `act = none`（SiLU 在 GroupNorm 那一步做）；投影（q / k / v、to_out、前馈）当 1×1 卷积；注意力头维补零（40 ⇒ 64、80 ⇒ 96），",
             "//! `scale` 给真头维；交叉注意力的 key 是上下文的 128 行（77 行有用，补的用掩码去掉）。", ""]
    for kind in KIND_ORDER:
        ks = [v for v in K.k.values() if v["kind"] == kind]
        if ks:
            lines += [f"// ── {kind}（{len(ks)} 个）──"] + [v["macro"] for v in ks] + [""]
    (UNET / "src").mkdir(parents=True, exist_ok=True)
    (UNET / "src" / "kernels.rs").write_text("\n".join(lines))
    print(f"crates/unet/src/kernels.rs：{len(K.k)} 个 kernel；一路 {len(g.ops)} 个算子、{len(g.units)} 个单元；时间步投影合计 {info['nb']} 个通道")


# ───────────────────────── Cargo.toml / lib.rs ─────────────────────────

def write_cargo(features: list[str]):
    (UNET / "Cargo.toml").write_text(
        '[package]\nname = "unet"\nversion = "0.1.0"\nedition = "2024"\npublish = false\n'
        'description = "SD1.5 UNet 的分段整步程序（B3）：一次去噪步的 UNet 前向、CFG 两路（tools/unet_gen.py 生成入口）"\n\n'
        '[lib]\ncrate-type = ["rlib"]\n\n[dependencies]\nnpu.workspace = true\ncalblas.workspace = true\nsd-kernels = { path = "../kernels" }\n\n'
        '# s* 是各段程序，z* 是量代码大小的探针（tools/unet_gen.py probe）；一次只开一个\n[features]\n'
        + "".join(f"{f} = []\n" for f in features))


LIB_RS = """//! SD1.5 UNet（B3）的分段整步程序：每个 feature 一段（`s1` …），各是一个单芯粒两核的整步程序；`z*` 是量代码大小的探针。
//! 入口与偏移全由 `tools/unet_gen.py program` 生成（`src/step.rs`、`plan.json`），kernel 特化由 `tools/unet_gen.py kernels` 生成（`src/kernels.rs`）。
//! 一段 = 若干「单元」（ResNet、Transformer、上 / 下采样……）× CFG 两路；段间的张量经主机转手（只搬字节）。
#![no_std]
#![allow(unused_imports, dead_code)]

pub mod kernels;
mod step;
"""


# ───────────────────────── probe：量各 kernel 的代码大小 ─────────────────────────

def _sections(d: bytes):
    shoff, = struct.unpack_from("<I", d, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x2E)
    return [struct.unpack_from("<IIIIIIIIII", d, shoff + i * shentsize) for i in range(shnum)], shstrndx


def elf_symbols(path: Path) -> dict:
    """ELF32 的符号表：名字 ⇒ 大小（只要 FUNC）。docc 出的 ELF 带符号表，kernel 的代码在 `<入口>.<kernel>[.N]` 这些符号里。"""
    d = path.read_bytes()
    secs, _ = _sections(d)
    out = {}
    for s in secs:
        if s[1] != 2:  # SHT_SYMTAB
            continue
        strtab = secs[s[6]]
        for i in range(s[5] // 16):
            nm, _, size, info, _, _ = struct.unpack_from("<IIIBBH", d, s[4] + i * 16)
            if info & 0xF != 2:
                continue
            o = strtab[4] + nm
            out[d[o:d.index(b"\0", o)].decode()] = size
    return out


def text_size(path: Path) -> int:
    d = path.read_bytes()
    secs, shstrndx = _sections(d)
    strs = secs[shstrndx]
    for s in secs:
        o = strs[4] + s[0]
        if d[o:d.index(b"\0", o)] == b".text":
            return s[5]
    raise ValueError(f"{path} 没有 .text")


def kernel_code(syms: dict, entry: str = "step") -> dict:
    """按 kernel 归账：`step.<kernel>` 与 `step.<kernel>.<n>` 都算它的。"""
    out = {}
    for name, size in syms.items():
        m = re.match(rf"^{entry}\.(\w+?)(?:\.\d+)?$", name)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + size
    return out


GUESS = {"conv2d": 120_000, "flash_attn": 120_000, "gn_stats": 30_000, "gn_apply": 30_000, "ln_nhwc": 40_000}
NARGS = {"gn_stats": 2, "gn_apply": 6, "ln_nhwc": 4, "flash_attn": 5, "transpose2d": 2, "geglu": 3, "upsample2": 2, "copy_win": 2,
         "timestep_embed": 3, "unary": 2, "bias_fold": 3}


def probe_batches(K: Kernels) -> list[list[str]]:
    names = [n for n in K.k if K.k[n]["kind"] != "fill16k"]
    out, cur, acc = [], [], 0
    for n in names:
        s = GUESS.get(K.k[n]["kind"], 15_000)
        if cur and acc + s > 560_000:
            out.append(cur)
            cur, acc = [], 0
        cur.append(n)
        acc += s
    return out + ([cur] if cur else [])


def probe_program(feature: str, kernels: list[str], K: Kernels) -> str:
    """一个探针程序：每个 kernel 调一次（指针随便给，只编不跑）。"""
    lines = [f'#[cfg(feature = "{feature}")]', f"mod {feature} {{", "    use super::*;", "    npu::decls! {",
             "        in_x: input, bytes = 64;", "        out_y: output, bytes = 64;", "        w: weights, bytes = 67108864;",
             "        act: workspace, bytes = 268435456;", "    }", "    #[npu::program(chiplets = 1, cores = 2)]", "    pub fn step() {"]
    for n in kernels:
        kind, sp = K.k[n]["kind"], K.k[n]["spec"]
        if kind == "conv2d":
            na = 5 if sp["res"] else 4
        elif kind == "linear":
            na = 4 if sp["bias"] else 3
        else:
            na = NARGS[kind]
        args = ", ".join(f"act().at({i * 33554432}).cast()" for i in range(na))
        lines += [f"        {n}({args});", "        sync_cores();"]
    lines += ["    }", "}", ""]
    return "\n".join(lines)


def build_feature(feature: str, out: Path) -> tuple[bool, str]:
    """tools/calbin.sh unet --features <f>（不带参数块：只核结构）。"""
    r = subprocess.run([str(ROOT / "tools" / "calbin.sh"), "unet", "--features", feature, "-o", str(out)], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)[-3000:]


def cmd_probe(a):
    K, g, info = build(load_unet())
    batches = probe_batches(K)
    feats = [f"z{i + 1}" for i in range(len(batches))]
    write_cargo(feats)
    (UNET / "src" / "lib.rs").write_text(LIB_RS)
    body = ["//! 量代码大小的探针（`tools/unet_gen.py probe` 生成）：每个 z* feature 把一批 kernel 各调一次。", "use crate::kernels::*;",
            "use npu::prelude::*;", ""]
    for f, ks in zip(feats, batches):
        body.append(probe_program(f, ks, K))
    (UNET / "src" / "step.rs").write_text("\n".join(body))
    sizes = {}
    t0 = time.time()
    todo = [(f, ks) for f, ks in zip(feats, batches) if not a.only or f in a.only.split(",")]
    print(f"{len(batches)} 个探针程序：" + "；".join(f"{f} {len(ks)} 个" for f, ks in zip(feats, batches)), flush=True)
    for f, ks in todo:
        out = ROOT / "out" / "unet-probe" / f
        ok, log = build_feature(f, out)
        if not ok:
            print(f"✗ {f}（{'、'.join(ks)}）\n{log}")
            sys.exit(1)
        per_core = [kernel_code(elf_symbols(out / "cal" / "unet" / f"ccu{c}_chip0.elf")) for c in (0, 1)]
        tsz = max(text_size(out / "cal" / "unet" / f"ccu{c}_chip0.elf") for c in (0, 1))
        for k in ks:
            sizes[k] = max(pc.get(k, 0) for pc in per_core)
        print(f"✓ {f}：{len(ks)} 个 kernel，.text {tsz / 1024:.0f} KB（{time.time() - t0:.0f} s）", flush=True)
    old = json.loads((UNET / "kernel_sizes.json").read_text()) if (UNET / "kernel_sizes.json").exists() else {}
    old.update(sizes)
    (UNET / "kernel_sizes.json").write_text(json.dumps(dict(sorted(old.items())), indent=1))
    big = sorted(sizes.items(), key=lambda kv: -kv[1])[:12]
    print("最大的：" + "、".join(f"{k} {v / 1024:.0f} KB" for k, v in big))
    print(f"合计 {sum(old.values()) / 1024:.0f} KB（{len(old)} 个 kernel）⇒ crates/unet/kernel_sizes.json")


# ───────────────────────── program：切段、偏移、入口 ─────────────────────────

def split_stages(g: Graph, sizes: dict, budget: int, call_code: int) -> list[list[int]]:
    """按单元的次序贪心装段：这一段用到的 kernel 的代码（各算一次）+ 入口里的调用（两路各一次）不超过 budget。"""
    unit_ops = [[o for o in g.ops if o["unit"] == u] for u in range(len(g.units))]
    stages, cur, ks, calls = [], [], set(), 0

    def est(kset, n):
        return BASE_CODE + sum(sizes[k] for k in kset) + sizes.get("un_zero", 2000) + n * call_code

    for u, ops in enumerate(unit_ops):
        nk = {o["kernel"] for o in ops}
        n = len(ops) * (1 if g.units[u]["shared"] else 2)
        if cur and est(ks | nk, calls + n) > budget:
            stages.append(cur)
            cur, ks, calls = [], set(), 0
        cur.append(u)
        ks |= nk
        calls += n
        if est(ks, calls) > budget:
            raise SystemExit(f"单元 {g.units[u]['name']} 一个就放不下（估 {est(ks, calls) / 1024:.0f} KB）")
    return stages + [cur]


def layout(g: Graph, info: dict, stages: list[list[int]]) -> list[dict]:
    """每段的 in_x / out_y / act / w 的偏移。"""
    unit_stage = {u: s for s, us in enumerate(stages) for u in us}
    prod, cons = {}, {}  # 张量 ⇒ 写它的段 / 读它的段
    for o in g.ops:
        s = unit_stage[o["unit"]]
        bargs = [x for x in o["args"] if x[0] == "b"]
        for x in bargs[:-1]:
            cons.setdefault(x[1], set()).add(s)
        prod.setdefault(bargs[-1][1], set()).add(s)
    host_in = set(info["inputs"].values())
    final = set(info["outputs"].values())
    for name, ss in prod.items():
        assert len(ss) == 1 or name == info["st"], f"{name} 由好几段写：{ss}"
    out = []
    for s, us in enumerate(stages):
        ops = [o for o in g.ops if unit_stage[o["unit"]] == s]
        used = []
        for o in ops:
            for x in o["args"]:
                if x[0] == "b" and x[1] not in used:
                    used.append(x[1])
        ins, outs, temps = [], [], []
        for name in used:
            p = prod.get(name, set())
            if name in host_in or (name != info["st"] and p and min(p) < s):
                ins.append(name)
            elif name != info["st"] and (name in final or any(c > s for c in cons.get(name, ()))):
                outs.append(name)
            else:
                temps.append(name)

        def region(names, pad16=False):
            sh = [n for n in names if g.bufs[n]["scope"] == "sh"]
            br = [n for n in names if g.bufs[n]["scope"] == "br"]
            off, at = {}, 0
            for n in sh:
                off[n] = at
                at += g.bufs[n]["bytes"]
            base, bo = at, {}
            for n in br:
                bo[n] = at - base
                at += g.bufs[n]["bytes"]
            stride = at - base
            total = base + 2 * stride
            return {"sh": off, "br": bo, "br_base": base, "br_stride": stride, "bytes": a16k(max(total, 16384)) if pad16 else max(total, 64)}

        toff, at = {}, 0
        for n in temps:  # 段内临时：两路共用同一份
            toff[n] = at
            at += g.bufs[n]["bytes"]
        wkeys = []
        for o in ops:
            for x in o["args"]:
                if x[0] == "w" and x[1] not in wkeys:
                    wkeys.append(x[1])
        woff, wat = {}, 0
        for k in wkeys:
            woff[k] = [wat, g.wsize[k]]
            wat += a4k(g.wsize[k])
        out.append({"feature": f"s{s + 1}", "units": [g.units[u]["name"] for u in us],
                    "in": region(ins), "out": region(outs, pad16=True), "act": {"off": toff, "bytes": a16k(max(at, 16384))},
                    "w": {"bytes": max(wat, 4096), "entries": woff}})
    return out


def loc(L: dict, name: str, br: int) -> tuple[str, int]:
    """张量在这一段里的位置（声明, 字节偏移）。"""
    for decl, key in (("in_x", "in"), ("out_y", "out")):
        r = L[key]
        if name in r["sh"]:
            return decl, r["sh"][name]
        if name in r["br"]:
            return decl, r["br_base"] + br * r["br_stride"] + r["br"][name]
    return "act", L["act"]["off"][name]


def emit_stage(L: dict, g: Graph, unit_ids: list[int]) -> list[str]:
    f = L["feature"]
    us = set(unit_ids)
    ops = [o for o in g.ops if o["unit"] in us]
    zo, za = L["out"]["bytes"] // 16384, L["act"]["bytes"] // 16384
    lines = [f"/// {f}：{'、'.join(L['units'])}", f'#[cfg(feature = "{f}")]', f"mod {f} {{", "    use super::*;", "    npu::decls! {",
             f"        in_x: input, bytes = {L['in']['bytes']};", f"        out_y: output, bytes = {L['out']['bytes']};",
             f"        w: weights, bytes = {L['w']['bytes']};", f"        act: workspace, bytes = {L['act']['bytes']};", "    }",
             "    #[npu::program(chiplets = 1, cores = 2)]", "    pub fn step() {",
             "        // 入口第一步：@act 与 out_y 整块清零（两核各清一半）：边框、token 的补齐行、余量都靠它是 0",
             "        if core_rank() == 0 {",
             f"            un_zero(act().cast(), {za // 2});",
             f"            un_zero(out_y().cast(), {zo // 2});",
             "        } else {",
             f"            un_zero(act().at({za // 2 * 16384}).cast(), {za - za // 2});",
             f"            un_zero(out_y().at({zo // 2 * 16384}).cast(), {zo - zo // 2});",
             "        }",
             "        sync_cores();"]

    def call(o, br):
        args = []
        for x in o["args"]:
            if x[0] == "w":
                args.append(f"w().at({L['w']['entries'][x[1]][0]}).cast()")
            else:
                d, off = loc(L, x[1], br)
                args.append(f"{d}().at({off + x[2]}).cast()")
        c = f"{o['kernel']}({', '.join(args)});"
        return ["        if core_rank() == 0 {", f"            {c}", "        }"] if o["kind"] in CORE0 else [f"        {c}"]

    shared = [o for o in ops if g.units[o["unit"]]["shared"]]
    if shared:
        lines.append("        // 时间步那一路（两路共用，算一次）")
        for o in shared:
            lines += call(o, 0) + ["        sync_cores();"]
    for br in (0, 1):
        lines.append(f"        // ── 第 {br} 路（{'有提示词' if br == 0 else '空提示词'}）──")
        last = None
        for o in ops:
            if g.units[o["unit"]]["shared"]:
                continue
            if o["unit"] != last:
                lines.append(f"        // {g.units[o['unit']]['name']}")
                last = o["unit"]
            lines += call(o, br) + ["        sync_cores();"]
    lines += ["    }", "}", ""]
    return lines


def cmd_program(a):
    K, g, info = build(load_unet())
    sizes = json.loads((UNET / "kernel_sizes.json").read_text())
    miss = [k for k in K.k if k not in sizes and K.k[k]["kind"] != "fill16k"]
    if miss:
        raise SystemExit(f"kernel_sizes.json 里缺 {miss}（先 tools/unet_gen.py probe）")
    stages = split_stages(g, sizes, a.budget * 1024, a.call_code)
    Ls = layout(g, info, stages)
    write_cargo([L["feature"] for L in Ls])
    (UNET / "src" / "lib.rs").write_text(LIB_RS)
    body = ["//! 各段入口（**`tools/unet_gen.py program` 生成，不要手改**）：偏移都是字面量，和 `crates/unet/plan.json` 出自同一次计算。",
            "//! 接线的规矩同 VAE（crates/vae/src/step.rs）：`cores = 2` 与按 num_cores() 分行的 kernel 两核都调、不分行的单核 kernel 只让 0 号核调，",
            "//! 每次调用之后 `sync_cores()`；输入、输出互不重叠（不原地）。CFG 的两路先后跑同一串算子：权重同一份，段内临时图同一份，",
            "//! 段间的张量（in_x、out_y）每路一份。", "use crate::kernels::*;", "use npu::prelude::*;", ""]
    for L, us in zip(Ls, stages):
        body += emit_stage(L, g, us)
    (UNET / "src" / "step.rs").write_text("\n".join(body))
    plan = {"generated": time.strftime("%F %T"), "nb": info["nb"], "toffs": info["toffs"], "inputs": info["inputs"], "outputs": info["outputs"],
            "consts": {"B": B, "MARGIN_PX": MARGIN_PX, "HEADS": HEADS, "DP": DP, "CTX_ROWS": CTX_ROWS, "CTX_DIM": CTX_DIM, "CTX_REAL": CTX_REAL,
                       "TEMB_K": TEMB_K, "TROWS": TROWS, "NEG_BF16": NEG_BF16},
            "bufs": g.bufs, "wsize": g.wsize, "wmeta": g.wmeta, "stages": Ls}
    (UNET / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    tot_w = 0
    for L, us in zip(Ls, stages):
        uu = set(us)
        ks = {o["kernel"] for o in g.ops if o["unit"] in uu}
        n = sum(1 if g.units[o["unit"]]["shared"] else 2 for o in g.ops if o["unit"] in uu)
        est = BASE_CODE + sum(sizes[k] for k in ks) + n * a.call_code
        tot_w += L["w"]["bytes"]
        print(f"{L['feature']}：{len(us)} 个单元（{L['units'][0]} … {L['units'][-1]}），{len(ks)} 个 kernel，{n} 次调用，代码估 {est / 1024:.0f} KB；"
              f"in {L['in']['bytes'] / 2**20:.1f} MB、out {L['out']['bytes'] / 2**20:.1f} MB、act {L['act']['bytes'] / 2**20:.0f} MB、w {L['w']['bytes'] / 2**20:.0f} MB")
    print(f"{len(Ls)} 段，权重合计 {tot_w / 2**30:.2f} GiB ⇒ crates/unet/src/step.rs、Cargo.toml、plan.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("kernels").set_defaults(f=cmd_kernels)
    p = sub.add_parser("probe")
    p.add_argument("--only", help="只编这几个探针（逗号分隔，z1,z2…）")
    p.set_defaults(f=cmd_probe)
    p = sub.add_parser("program")
    p.add_argument("--budget", type=int, default=BUDGET // 1024, help="每核代码上限（KB）")
    p.add_argument("--call-code", type=int, default=CALL_CODE, help="入口里每次调用约多少字节")
    p.set_defaults(f=cmd_program)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
