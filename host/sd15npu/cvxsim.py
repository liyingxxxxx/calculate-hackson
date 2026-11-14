"""整步程序的 numpy 模拟：读设备遍编出来的 `.cvx`，照入口函数（`@step`）的调用序列，逐个 kernel 在 numpy 里重放。

为什么读 `.cvx` 而不是另写一张作业表：要查的正是 Rust 入口里的接线（偏移、屏障、哪颗核调），作业表另写一份就查不到它了。
这里拿到的是编译器真正传给每个 kernel 的地址。

它查什么（任何一条不过就抛 `SimError`，不打个警告就算）：
- 指针参数 64 B 对齐（docc 的规矩）；读写不越出声明；
- `cores = 2` 的 kernel 两核都调；按 `num_cores()` 分行的单核 kernel 也要两核都调（只让一颗核调时可能只算一半的行，见 step.rs 的文档）；
- 不原地：一个作业的输出区不和它的输入区重叠；
- 屏障：一个作业读（或写）了前一个作业写的区，中间要有 `barrier.core.sync`（除非两个作业都只在同一颗核上）；
- 读没写过的内存：workspace / output 开头填 bf16 的 NaN（0xFFC1），漏清零的边框会把 NaN 带进结果。

它不查什么：kernel 自己内部算得对不对（kernel 的语义照模板的文档写在下面，算在 f32、每个 kernel 出口舍 bf16）；
exp / rsqrt / tanh 在板上是近似指令，这里用精确值 ⇒ 不是板子的逐位参照（calBLAS AGENTS.md：模拟器不是金标准）。

kernel 的语义按「模板 + 宏参数」认：从程序 crate 的 `kernels.rs` 里用正则读 `模板!(pub 名字: 键 = 值, …)`，
值里的常量（LAT、C……）取自 layout-dump 的 `consts`。认得的模板：conv2d、upsample2、tanh_scale、timestep_embed、fill16k；
正式 VAE 用到的 group_norm、nhwc_to_nchw、nchw_to_nhwc、transpose2d、linear（多行、`w = hf`）、attn_wide、gn_stats、gn_apply；
UNet 用到的 attention（calblas，`mask = keys`）、ln_nhwc、geglu、copy_win、unary（silu）、bias_fold。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bf16

B = 2  # 边框宽，同 calblas kernels::BORDER
POISON = 0xFFC1  # bf16 的一个 NaN：没写过的内存


class SimError(Exception):
    pass


# ───────────────────────── 解析 .cvx ─────────────────────────


@dataclass
class Func:
    name: str
    params: list  # [(寄存器, 类型)]
    kernel: bool = False
    cores: int = 1
    reads_num_cores: bool = False
    body: list = field(default_factory=list)


@dataclass
class Program:
    decls: dict  # 名字 ⇒ (类别, 字节)
    funcs: dict  # 名字 ⇒ Func
    entry: str
    chiplets: int
    cores: int


def parse_cvx(text: str) -> Program:
    decls, funcs = {}, {}
    entry = re.search(r"entry = @(\w+)", text).group(1)
    chiplets = int(re.search(r"chiplets = (\d+)", text).group(1))
    cores = int(re.search(r"invocation\{[^}]*cores = (\d+)", text).group(1))
    cur = None
    # 参数多、行长的 `.func 名字(` 会被打印器折成多行（一行一个参数，最后一行是 `)`）：先拼回一行
    lines, buf = [], None
    for line in text.splitlines():
        s = line.strip()
        if buf is not None:
            buf += (" " if not buf.endswith("(") else "") + s
            if s.startswith(")"):
                lines.append(buf.replace(" )", ")").replace(",)", ")"))
                buf = None
            continue
        if re.match(r"\.func \w+\($", s):
            buf = s
            continue
        lines.append(s)
    for s in lines:
        if m := re.match(r"mem\.decl\s+(\w+), (\w+), (\d+)", s):
            decls[m.group(1)] = (m.group(2), int(m.group(3)))
        elif m := re.match(r"\.func (\w+)\((.*)\)$", s):
            params = [tuple(p.split(": ", 1)) for p in m.group(2).split(", ")] if m.group(2) else []
            cur = funcs[m.group(1)] = Func(m.group(1), params)
        elif cur is None:
            continue
        elif s == ".endfunc":
            cur = None
        elif s == ".attr npu.kernel = unit":
            cur.kernel = True
        elif m := re.match(r"\.attr npu\.cores = (\d+)", s):
            cur.cores = int(m.group(1))
        elif s.startswith(".reg") or s.startswith(".attr"):
            continue
        else:
            if re.search(r"sys\.query\s+\w+ = num_cores", s):
                cur.reads_num_cores = True
            cur.body.append(s)
    return Program(decls, funcs, entry, chiplets, cores)


def _tree(lines: list, i: int = 0):
    """入口函数体 ⇒ 语句树：`if rX` 后跟 then 区（到 `.end`）与 else 区（到下一个 `.end`）。"""
    out = []
    while i < len(lines):
        s = lines[i]
        if s == ".end":
            return out, i + 1
        if m := re.match(r"if\s+(\w+)$", s):
            then, i = _tree(lines, i + 1)
            other, i = _tree(lines, i)
            out.append(("if", m.group(1), then, other))
            continue
        if re.match(r"if\s+\w+ = ", s) or s.startswith("for "):
            raise SimError(f"入口里有带结果的 if 或循环（模拟器只认直线代码 + core_rank 分支）：{s}")
        out.append(("op", s))
        i += 1
    return out, i


# ───────────────────────── kernel 的宏参数 ─────────────────────────


def parse_specs(kernels_rs: str, consts: dict) -> dict:
    """`模板!(pub 名字: 键 = 值, …);` ⇒ {名字: (模板, {键: 值})}。值按 Python 表达式算（常量取 consts，`/` 当整除，去掉 f32 后缀）。"""
    specs = {}
    ns = dict(consts)
    for m in re.finditer(r"(?:\w+::)?(\w+)!\(\s*pub (\w+): (.*?)\);", kernels_rs, re.S):
        kind, name, body = m.groups()
        kv = {}
        for k, v in re.findall(r"(\w+) = (\([^)]*\)|[^,]+)", body):
            v = v.strip()
            if k in ("act", "weights"):
                kv[k] = v
                continue
            v = re.sub(r"(\d)f32\b", r"\1", v).replace(" / ", " // ")
            try:
                kv[k] = eval(v, {"__builtins__": {}}, ns)  # noqa: S307（本仓自己的源码，不是外部输入）
            except (NameError, SyntaxError):
                kv[k] = v  # 标识符形式的选项（linear! 的 `w = hf`、`bias = f32`）：原样留字符串
        if kind == "unary":  # 第一个位置参数是函数名（`unary!(pub 名字: silu, n = …)`）
            kv["f"] = body.split(",")[0].strip()
        if m2 := re.search(r"ks = \[(\d+), (\d+)\], (\w+)", body):  # linear! 的 K 分段：`ks = [k0, k1], first | mid | last`
            kv["ks"] = [int(m2.group(1)), int(m2.group(2))]
            kv["stage_name"] = m2.group(3)
        specs[name] = (kind, kv)
    return specs


# ───────────────────────── 模拟 ─────────────────────────


@dataclass
class Ptr:
    decl: str
    off: int

    def __add__(self, n: int):
        return Ptr(self.decl, self.off + n)


class _Core:
    """`sys.query core_rank` 的值：各核不同，只拿来和常数比。"""


@dataclass
class _Pred:
    k: int
    eq: bool


@dataclass
class Job:
    idx: int
    kernel: str
    kind: str
    cores: frozenset
    reads: list  # [(decl, 起, 止)]
    writes: list
    seconds: float = 0.0
    rmw: list = field(default_factory=list)  # 按设计就地读改写的区（calblas linear! 的 K 分段 mid：部分和 p 每块读、加、写回同一块）


class Sim:
    def __init__(self, cvx_text: str, kernels_rs: str, consts: dict, inputs: dict, verbose: bool = False):
        self.prog = parse_cvx(cvx_text)
        if self.prog.chiplets != 1:
            raise SimError(f"模拟器只认单芯粒程序（这份是 {self.prog.chiplets} 芯粒）")
        self.specs = parse_specs(kernels_rs, consts)
        self.verbose = verbose
        self.mem = {}
        for name, (kind, n) in self.prog.decls.items():
            if kind in ("input", "weights"):
                data = inputs.get(name)
                if data is None:
                    raise SimError(f"没给 {kind} 声明 @{name} 的内容")
                if len(data) != n:
                    raise SimError(f"@{name} 是 {n} 字节，给的是 {len(data)} 字节")
                self.mem[name] = np.frombuffer(bytes(data), np.uint8).copy()
            else:  # workspace / output：填 NaN，读了没写过的地方就看得出来
                self.mem[name] = np.frombuffer(np.full(n // 2, POISON, "<u2").tobytes() + b"\xff" * (n % 2), np.uint8).copy()
        self.jobs: list[Job] = []
        self.pending: list[tuple] = []  # 上一条屏障之后写过的区：(decl, 起, 止, 作业)
        self.barriers = 0

    # ── 地址与区间 ──
    def _check(self, kernel: str, what: str, p: Ptr, n: int):
        size = self.prog.decls[p.decl][1]
        if p.off % 64:
            raise SimError(f"{kernel}：{what} 的指针 @{p.decl}+{p.off} 不在 64 B 上")
        if p.off < 0 or p.off + n > size:
            raise SimError(f"{kernel}：{what} [{p.off}, {p.off + n}) 越出 @{p.decl}（{size} 字节）")
        return (p.decl, p.off, p.off + n)

    def _u16(self, p: Ptr, n_elems: int):
        return self.mem[p.decl][p.off:p.off + 2 * n_elems].view("<u2")

    def _f32(self, p: Ptr, n: int):
        return self.mem[p.decl][p.off:p.off + 4 * n].view("<f4")

    def _img(self, p: Ptr, h: int, w: int, ch: int):
        """[h + 4][w + 4][ch] 的 bf16 图（uint16 视图，读写都走它）。"""
        return self._u16(p, (h + 2 * B) * (w + 2 * B) * ch).reshape(h + 2 * B, w + 2 * B, ch)

    # ── 执行 ──
    def run(self):
        f = self.prog.funcs[self.prog.entry]
        tree, _ = _tree(f.body)
        regs: dict = {}
        self._exec(tree, regs, frozenset(range(self.prog.cores)))
        return self

    def _exec(self, tree, regs, active):
        for node in tree:
            if node[0] == "if":
                cond = regs[node[1]]
                if isinstance(cond, _Pred):
                    hit = frozenset(c for c in active if (c == cond.k) == cond.eq)
                    if hit:
                        self._exec(node[2], dict(regs), hit)
                    if active - hit:
                        self._exec(node[3], dict(regs), active - hit)
                elif isinstance(cond, bool):
                    self._exec(node[2] if cond else node[3], dict(regs), active)
                else:
                    raise SimError(f"if 的条件不是 core_rank 的比较：{node[1]} = {cond!r}")
                continue
            self._op(node[1], regs, active)

    def _op(self, s, regs, active):
        if m := re.match(r"mem\.addr\s+(\w+) = @(\w+)$", s):
            regs[m.group(1)] = Ptr(m.group(2), 0)
        elif m := re.match(r"s\.constant\s+(\w+) = (-?[\d.]+)(?::(\w+))?$", s):
            v = m.group(2)
            regs[m.group(1)] = float(v) if "." in v else int(v)
        elif m := re.match(r"s\.ptradd\s+(\w+) = (\w+), (\w+)$", s):
            regs[m.group(1)] = regs[m.group(2)] + regs[m.group(3)]
        elif m := re.match(r"sys\.query\s+(\w+) = (\w+)$", s):
            regs[m.group(1)] = _Core() if m.group(2) == "core_rank" else self.prog.cores if m.group(2) == "num_cores" else None
        elif m := re.match(r"s\.cmp\s+(\w+) = (\w+), (\w+), (eq|ne)$", s):
            a, b = regs[m.group(2)], regs[m.group(3)]
            if isinstance(a, _Core) and isinstance(b, int):
                regs[m.group(1)] = _Pred(b, m.group(4) == "eq")
            elif isinstance(a, int) and isinstance(b, int):
                regs[m.group(1)] = (a == b) == (m.group(4) == "eq")
            else:
                raise SimError(f"认不得的比较：{s}")
        elif m := re.match(r"s\.(add|sub|mul|div)\s+(\w+) = (\w+), (\w+)$", s):
            a, b = regs[m.group(3)], regs[m.group(4)]
            if not (isinstance(a, int) and isinstance(b, int)):
                raise SimError(f"入口里的标量运算碰到了非常量：{s}")
            regs[m.group(2)] = {"add": a + b, "sub": a - b, "mul": a * b, "div": a // b if b else 0}[m.group(1)]
        elif m := re.match(r"call\s+(.*), @(\w+)$", s):
            self._call(m.group(2), [regs[r.strip()] for r in m.group(1).split(",")], active)
        elif s == "barrier.core.sync":
            if active != frozenset(range(self.prog.cores)):
                raise SimError(f"barrier.core.sync 只在核 {sorted(active)} 上执行：另一颗核不到场就挂死")
            self.pending.clear()
            self.barriers += 1
        elif s in ("yield", "return"):
            pass
        else:
            raise SimError(f"入口里有模拟器不认的指令：{s}")

    def _call(self, name, args, active):
        f = self.prog.funcs.get(name)
        if f is None or not f.kernel:
            raise SimError(f"调了不是 kernel 的 @{name}")
        if name not in self.specs:
            raise SimError(f"kernel @{name} 在 kernels.rs 里找不到宏参数（模拟器认不出它算什么）")
        kind, c = self.specs[name]
        all_cores = frozenset(range(self.prog.cores))
        if f.cores > 1 and active != all_cores:
            raise SimError(f"@{name} 是 cores = {f.cores} 的 kernel，只在核 {sorted(active)} 上调了")
        if f.cores == 1 and f.reads_num_cores and active != all_cores:
            raise SimError(f"@{name} 按 num_cores() 分行，却只在核 {sorted(active)} 上调：num_cores() 是 2 时只算了一半的行（两核都调才稳）")
        job = Job(len(self.jobs), name, kind, active, [], [])
        t = time.time()
        getattr(self, f"_k_{kind}")(job, c, args)
        job.seconds = time.time() - t
        # 原地
        for d, a0, a1 in job.writes:
            if (d, a0, a1) in job.rmw:
                continue
            for e, b0, b1 in job.reads:
                if d == e and a0 < b1 and b0 < a1:
                    raise SimError(f"作业 {job.idx} @{name}：输出 @{d}[{a0}, {a1}) 和输入 [{b0}, {b1}) 重叠（不能原地）")
        # 屏障：上一条屏障之后别的作业写过的区，这里再读 / 写
        for d, a0, a1, prev in self.pending:
            same_core = len(prev.cores) == 1 and prev.cores == active
            if same_core:
                continue
            for e, b0, b1 in job.reads + job.writes:
                if d == e and a0 < b1 and b0 < a1:
                    raise SimError(f"作业 {job.idx} @{name} 碰了作业 {prev.idx} @{prev.kernel} 刚写的 @{d}[{max(a0, b0)}, {min(a1, b1)})，中间少一条 sync_cores()")
        self.pending += [(d, a0, a1, job) for d, a0, a1 in job.writes]
        self.jobs.append(job)
        if self.verbose:
            print(f"  [{job.idx:3d}] {name:<16} 核{sorted(active)} {job.seconds:6.2f} s")

    # ── kernel 的语义（照模板文档：f32 里算、出口舍 bf16）──
    def _k_conv2d(self, job, c, args):
        h, w, ci, co, k, s = c["h"], c["w"], c["cin"], c["cout"], c["k"], c["s"]
        (pi, xo), (po, yo) = c["pin"], c["pout"]
        act = c["act"]
        res = c.get("res")
        if c.get("weights"):
            raise SimError("模拟器没写 weights = ci_major")
        x, wp, bp = args[0], args[1], args[2]
        y = args[-1]
        hi, wi = h * s, w * s
        job.reads.append(self._check(job.kernel, "x", x, (hi + 2 * B) * (wi + 2 * B) * pi * 2))
        job.reads.append(self._check(job.kernel, "w", wp, k * k * ci * co * 2))
        job.reads.append(self._check(job.kernel, "b", bp, co * 4))
        wo = w + 2 * B
        y0 = ((B * wo + B) * po + yo) * 2
        y1 = (((B + h - 1) * wo + B + w - 1) * po + yo + co) * 2
        yr = self._check(job.kernel, "y", y, (h + 2 * B) * wo * po * 2)
        job.writes.append((yr[0], yr[1] + y0, yr[1] + y1))
        xin = bf16.from_bits(self._img(x, hi, wi, pi)[:, :, xo:xo + ci])
        wt = bf16.from_bits(self._u16(wp, k * k * ci * co)).reshape(k, k, ci, co)
        bias = self._f32(bp, co).copy()
        acc = np.zeros((h, w, co), np.float32)
        off = B - k // 2
        for ky in range(k):
            for kx in range(k):
                patch = np.ascontiguousarray(xin[off + ky:off + ky + hi:s, off + kx:off + kx + wi:s])
                acc += (patch.reshape(-1, ci) @ wt[ky, kx]).reshape(h, w, co)
        out = bf16.rnd(acc + bias)
        if act == "silu":
            out = bf16.rnd(out / (1 + np.exp(-out)))
        elif act == "relu":
            out = np.maximum(out, 0)
        elif act not in ("none", "res_relu"):
            raise SimError(f"conv2d：认不得 act = {act}")
        if res is not None:
            pr, ro = res
            rp = args[3]
            job.reads.append(self._check(job.kernel, "r", rp, (h + 2 * B) * wo * pr * 2))
            out = bf16.rnd(out + bf16.from_bits(self._img(rp, h, w, pr)[B:B + h, B:B + w, ro:ro + co]))
        if act == "res_relu":
            out = np.maximum(out, 0)
        self._img(y, h, w, po)[B:B + h, B:B + w, yo:yo + co] = bf16.bits(out)

    def _k_upsample2(self, job, c, args):
        h, w, ch, pi, po = c["h"], c["w"], c["c"], c["pin"], c["pout"]
        x, y = args
        job.reads.append(self._check(job.kernel, "x", x, (h + 2 * B) * (w + 2 * B) * pi * 2))
        wo = 2 * w + 2 * B
        yr = self._check(job.kernel, "y", y, (2 * h + 2 * B) * wo * po * 2)
        job.writes.append((yr[0], yr[1] + (B * wo + B) * po * 2, yr[1] + (((B + 2 * h - 1) * wo + B + 2 * w - 1) * po + ch) * 2))
        v = self._img(x, h, w, pi)[B:B + h, B:B + w, :ch]
        self._img(y, 2 * h, 2 * w, po)[B:B + 2 * h, B:B + 2 * w, :ch] = v.repeat(2, 0).repeat(2, 1)

    def _k_tanh_scale(self, job, c, args):
        n, sc = c["n"], np.float32(c["scale"])
        x, y = args
        job.reads.append(self._check(job.kernel, "x", x, n * 2))
        job.writes.append(self._check(job.kernel, "y", y, n * 2))
        f = bf16.from_bits(self._u16(x, n))
        out = np.tanh(f * (np.float32(1) / sc)) * sc
        self._u16(y, n)[:] = bf16.bits(out.astype(np.float32))

    def _k_timestep_embed(self, job, c, args):
        half = c["half"]
        n = -(-2 * half // 128) * 128
        t, fp, y = args
        job.reads.append(self._check(job.kernel, "t", t, 32))
        job.reads.append(self._check(job.kernel, "fp", fp, 2 * n * 4))
        job.writes.append(self._check(job.kernel, "y", y, n * 2))
        tv = np.float32(self._f32(t, 8).max())  # 行内取最大（8 份同一个 t）
        F, P = self._f32(fp, n), self._f32(fp + n * 4, n)
        a = (tv * F).astype(np.float32) + P  # f32 乘、加（同 kernel）；规约与 cos 用精确值（板上是近似指令）
        self._u16(y, n)[:] = bf16.bits(np.cos(a.astype(np.float64)).astype(np.float32))

    def _k_fill16k(self, job, c, args):
        p, n = args
        if not isinstance(n, int):
            raise SimError(f"{job.kernel}：块数不是常量")
        job.writes.append(self._check(job.kernel, "p", p, n * 16384))
        self._u16(p, n * 8192)[:] = bf16.bits(np.float32(c["value"]))

    # ── 正式 VAE 用到的模板（calblas 的 group_norm / 排布转换 / 转置 / linear，本仓的 attn_wide）──
    @staticmethod
    def _norm_cols(n: int, hw: int) -> int:
        """calblas `kernels::Cols::norm_cols`：group_norm! 一行几个像素（w、b 按行展开，读的时候要知道一行多长）。"""
        cc = hw if hw < 128 else 128
        while cc > 16 and (hw % cc != 0 or (n // cc) % 8 != 0):
            cc //= 2
        return cc

    def _k_group_norm(self, job, c, args):
        hw, ch, g, eps, act = c["hw"], c["c"], c["groups"], np.float32(c["eps"]), c["act"]
        cc = self._norm_cols(ch // g * hw, hw)
        rows = ch * hw // cc
        x, wp, bp, y = args
        job.reads.append(self._check(job.kernel, "x", x, ch * hw * 2))
        job.reads.append(self._check(job.kernel, "w", wp, rows * 8 * 4))
        job.reads.append(self._check(job.kernel, "b", bp, rows * 8 * 4))
        job.writes.append(self._check(job.kernel, "y", y, ch * hw * 2))
        xv = bf16.from_bits(self._u16(x, ch * hw)).reshape(g, -1).astype(np.float64)
        mean = xv.mean(1, keepdims=True)
        var = ((xv - mean) ** 2).mean(1, keepdims=True)  # 两遍法（同模板）
        f = ((xv - mean) / np.sqrt(var + eps)).astype(np.float32).reshape(rows, cc)
        # 逐行的 w、b：每行 8 份同一个值，模板取行内最大
        wv = self._f32(wp, rows * 8).reshape(rows, 8).max(1)[:, None]
        bv = self._f32(bp, rows * 8).reshape(rows, 8).max(1)[:, None]
        out = bf16.rnd(f * wv + bv)
        if act == "silu":
            out = bf16.rnd(out / (1 + np.exp(-out)))
        elif act != "none":
            raise SimError(f"group_norm：认不得 act = {act}")
        self._u16(y, ch * hw)[:] = bf16.bits(out.ravel())

    def _k_gn_stats(self, job, c, args):
        """本仓 gn_stats!：两颗核各自的半边行 ⇒ 每通道 Σx、Σx²（f32）⇒ st 的第 2·core、2·core + 1 行。"""
        h, w, ch, p = c["h"], c["w"], c["c"], c["p"]
        x, st = args
        job.reads.append(self._check(job.kernel, "x", x, (h + 2 * B) * (w + 2 * B) * p * 2))
        job.writes.append(self._check(job.kernel, "st", st, 4 * ch * 4))
        v = bf16.from_bits(self._img(x, h, w, p)[B:B + h, B:B + w, :ch]).astype(np.float64)
        out = np.zeros((4, ch), np.float32)
        h0 = h // 2  # 同 calblas halves!：0 号核前一半，1 号核其余
        for core, rows in enumerate((v[:h0], v[h0:])):
            out[2 * core] = rows.sum((0, 1))
            out[2 * core + 1] = (rows * rows).sum((0, 1))
        self._f32(st, 4 * ch)[:] = out.ravel()

    def _k_gn_apply(self, job, c, args):
        """本仓 gn_apply!：两核的部分和 ⇒ 每组 mean、var = E[x²] − mean² ⇒ (x − mean) · rsqrt(var + eps) · w + b ⇒ 舍 bf16 ⇒（silu）⇒ 只写内部。"""
        h, w, ch, p, g, eps, act = c["h"], c["w"], c["c"], c["p"], c["groups"], np.float64(c["eps"]), c["act"]
        x, st, wp, bp, mk, y = args
        job.reads.append(self._check(job.kernel, "x", x, (h + 2 * B) * (w + 2 * B) * p * 2))
        job.reads.append(self._check(job.kernel, "st", st, 4 * ch * 4))
        job.reads.append(self._check(job.kernel, "w", wp, ch * 4))
        job.reads.append(self._check(job.kernel, "b", bp, ch * 4))
        job.reads.append(self._check(job.kernel, "mk", mk, g * ch * 4))
        wo = w + 2 * B
        yr = self._check(job.kernel, "y", y, (h + 2 * B) * wo * p * 2)
        job.writes.append((yr[0], yr[1] + (B * wo + B) * p * 2, yr[1] + (((B + h - 1) * wo + B + w - 1) * p + ch) * 2))
        sv = self._f32(st, 4 * ch).reshape(4, ch).astype(np.float64)
        mask = self._f32(mk, g * ch).reshape(g, ch).astype(np.float64)
        n = h * w * (ch // g)
        mean, ex2 = mask @ (sv[0] + sv[2]) / n, mask @ (sv[1] + sv[3]) / n
        rstd = 1 / np.sqrt(np.maximum(ex2 - mean * mean, 0) + eps)
        a = (mask.T @ rstd) * self._f32(wp, ch)
        bb = self._f32(bp, ch) - (mask.T @ mean) * a
        v = bf16.from_bits(self._img(x, h, w, p)[B:B + h, B:B + w, :ch])
        out = bf16.rnd((v * a + bb).astype(np.float32))
        if act == "silu":
            out = bf16.rnd(out / (1 + np.exp(-out)))
        elif act != "none":
            raise SimError(f"gn_apply：认不得 act = {act}")
        self._img(y, h, w, p)[B:B + h, B:B + w, :ch] = bf16.bits(out)

    def _k_nhwc_to_nchw(self, job, c, args):
        h, w, ch, p = c["h"], c["w"], c["c"], c["p"]
        x, y = args
        job.reads.append(self._check(job.kernel, "x", x, (h + 2 * B) * (w + 2 * B) * p * 2))
        job.writes.append(self._check(job.kernel, "y", y, ch * h * w * 2))
        v = self._img(x, h, w, p)[B:B + h, B:B + w, :ch]
        self._u16(y, ch * h * w)[:] = np.ascontiguousarray(v.transpose(2, 0, 1)).ravel()

    def _k_nchw_to_nhwc(self, job, c, args):
        h, w, ch, p = c["h"], c["w"], c["c"], c["p"]
        x, y = args
        job.reads.append(self._check(job.kernel, "x", x, ch * h * w * 2))
        wo = w + 2 * B
        yr = self._check(job.kernel, "y", y, (h + 2 * B) * wo * p * 2)
        job.writes.append((yr[0], yr[1] + (B * wo + B) * p * 2, yr[1] + (((B + h - 1) * wo + B + w - 1) * p + ch) * 2))
        v = self._u16(x, ch * h * w).reshape(ch, h, w)
        self._img(y, h, w, p)[B:B + h, B:B + w, :ch] = v.transpose(1, 2, 0)

    def _k_transpose2d(self, job, c, args):
        r, cl = c["rows"], c["cols"]
        x, y = args
        job.reads.append(self._check(job.kernel, "x", x, r * cl * 2))
        job.writes.append(self._check(job.kernel, "y", y, r * cl * 2))
        self._u16(y, r * cl)[:] = np.ascontiguousarray(self._u16(x, r * cl).reshape(r, cl).T).ravel()

    def _k_linear(self, job, c, args):
        if c.get("w") != "hf" or c.get("bias") not in (None, "f32") or c.get("act") or c.get("parts"):
            raise SimError(f"{job.kernel}：模拟器只写了 linear! 的多行 w = hf（可带 f32 bias、K 分段）")
        r, k, n = c["rows"], c["k"], c["n"]
        stage = None
        if c.get("ks"):  # `ks = [k0, k1], first | mid | last`：parse_specs 把它读成 "[0, K]" 与阶段名两个值之一，这里从宏参数重新认
            stage = c["stage_name"]
        if stage in ("first", "mid"):
            x, wp, pp_ = args
            if c["ks"] != [0, k]:
                raise SimError(f"{job.kernel}：模拟器只写了 ks = [0, K]")
            job.reads.append(self._check(job.kernel, "x", x, r * k * 2))
            job.reads.append(self._check(job.kernel, "w", wp, n * k * 2))
            part = bf16.from_bits(self._u16(x, r * k)).reshape(r, k) @ bf16.from_bits(self._u16(wp, n * k)).reshape(n, k).T
            if stage == "mid":
                job.reads.append(self._check(job.kernel, "p", pp_, r * n * 4))
                part = part + self._f32(pp_, r * n).reshape(r, n)
            job.writes.append(self._check(job.kernel, "p", pp_, r * n * 4))
            if stage == "mid":  # 就地加进 p 是 calblas 的设计（每块读自己那块、加、写回），不算「原地」违规
                job.rmw.append(job.writes[-1])
            self._f32(pp_, r * n)[:] = part.astype(np.float32).ravel()
            return
        if stage == "last":
            if c.get("bias"):
                x, wp, bp, pp_, y = args
            else:
                (x, wp, pp_, y), bp = args, None
        elif c.get("bias"):
            (x, wp, bp, y), pp_ = args, None
        else:
            (x, wp, y), bp, pp_ = args, None, None
        job.reads.append(self._check(job.kernel, "x", x, r * k * 2))
        job.reads.append(self._check(job.kernel, "w", wp, n * k * 2))
        xv = bf16.from_bits(self._u16(x, r * k)).reshape(r, k)
        wv = bf16.from_bits(self._u16(wp, n * k)).reshape(n, k)
        out = xv @ wv.T
        if pp_ is not None:
            job.reads.append(self._check(job.kernel, "p", pp_, r * n * 4))
            out = out + self._f32(pp_, r * n).reshape(r, n)
        if bp is not None:
            job.reads.append(self._check(job.kernel, "b", bp, n * 4))
            out = out + self._f32(bp, n)
        job.writes.append(self._check(job.kernel, "y", y, r * n * 2))
        self._u16(y, r * n)[:] = bf16.bits(out.astype(np.float32).ravel())

    def _k_attn_wide(self, job, c, args):
        t, hh, d = c["rows"], c["heads"], c["head"]
        q, k, v, o = args
        for name, p in (("q", q), ("k", k), ("v", v)):
            job.reads.append(self._check(job.kernel, name, p, t * hh * d * 2))
        job.writes.append(self._check(job.kernel, "o", o, t * hh * d * 2))
        qv, kv, vv = (bf16.from_bits(self._u16(p, t * hh * d)).reshape(t, hh, d) for p in (q, k, v))
        out = np.empty((t, hh, d), np.float32)
        for h in range(hh):
            s_ = (qv[:, h] @ kv[:, h].T) * np.float32(1 / np.sqrt(d))
            s_ = np.exp(s_ - s_.max(1, keepdims=True))
            out[:, h] = (s_ / s_.sum(1, keepdims=True)) @ vv[:, h]
        self._u16(o, t * hh * d)[:] = bf16.bits(out.ravel())

    # ── UNet 用到的模板（calblas 的 attention / unary，本仓的 ln_nhwc / geglu / copy_win / bias_fold）──
    def _k_attention(self, job, c, args):
        """calblas attention!（`mask = keys`）：q / o [T, H·D]、k / v [TK, H·D]、mask bf16 [TK]（0 或 −MAX）；分数 · 1/√scale + 掩码 ⇒ softmax ⇒ · v。
        板上分数与 p 舍 bf16、exp 是近似指令；这里在 f32 里算（模拟器不是金标准，见模块文档）。"""
        if c.get("mask") not in ("keys", None) or "kv_heads" in c or c.get("k", "transposed") != "transposed":
            raise SimError(f"{job.kernel}：模拟器只写了 attention! 的 mask = keys（不分组；K 可以是 `k = transposed`）")
        t, hh, d = c["rows"], c["heads"], c["head"]
        tk, sc = c.get("keys", t), c.get("scale", d)
        q, k, v, mk, o = args
        job.reads.append(self._check(job.kernel, "q", q, t * hh * d * 2))
        job.reads.append(self._check(job.kernel, "k", k, tk * hh * d * 2))
        job.reads.append(self._check(job.kernel, "v", v, tk * hh * d * 2))
        if mk is not None:
            job.reads.append(self._check(job.kernel, "mask", mk, tk * 2))
        job.writes.append(self._check(job.kernel, "o", o, t * hh * d * 2))
        qv = bf16.from_bits(self._u16(q, t * hh * d)).reshape(t, hh, d)
        kv = bf16.from_bits(self._u16(k, tk * hh * d))
        # `k = transposed`：K 是 [头, d, key]（transpose2d! 转好的）
        kv = kv.reshape(hh, d, tk).transpose(2, 0, 1) if c.get("k") == "transposed" else kv.reshape(tk, hh, d)
        vv = bf16.from_bits(self._u16(v, tk * hh * d)).reshape(tk, hh, d)
        m = bf16.from_bits(self._u16(mk, tk)).astype(np.float32) if mk is not None else np.zeros(tk, np.float32)
        out = np.empty((t, hh, d), np.float32)
        for h in range(hh):
            s_ = (qv[:, h] @ kv[:, h].T) * np.float32(1 / np.sqrt(sc)) + m[None, :]
            s_ = np.exp(s_ - s_.max(1, keepdims=True))
            out[:, h] = (s_ / s_.sum(1, keepdims=True)) @ vv[:, h]
        self._u16(o, t * hh * d)[:] = bf16.bits(out.ravel())

    def _k_flash_attn(self, job, c, args):
        """本仓 flash_attn!：语义同 attention!（mask = keys，K 按原样 [key, 头 · d]）；板上分数与 p 舍 bf16、exp 近似，这里在 f32 里算。
        `mask = none`：没有 mask 参数；`q = prescaled`：1/√scale 已经乘进 q，kernel 不再乘。"""
        c2 = {**c, "mask": "keys"}
        if c.get("q") == "prescaled":
            c2["scale"] = 1
        if c.get("mask") == "none":
            q, k, v, o = args
            args = [q, k, v, None, o]
        self._k_attention(job, c2, args)

    def _k_ln_nhwc(self, job, c, args):
        """本仓 ln_nhwc!：每个内部像素沿前 C 个通道归一（两遍法）⇒ · w + b ⇒ 舍 bf16，只写内部。"""
        h, w, ch, p, eps = c["h"], c["w"], c["c"], c["p"], np.float64(c["eps"])
        x, wp, bp, y = args
        job.reads.append(self._check(job.kernel, "x", x, (h + 2 * B) * (w + 2 * B) * p * 2))
        job.reads.append(self._check(job.kernel, "w", wp, ch * 4))
        job.reads.append(self._check(job.kernel, "b", bp, ch * 4))
        wo = w + 2 * B
        yr = self._check(job.kernel, "y", y, (h + 2 * B) * wo * p * 2)
        job.writes.append((yr[0], yr[1] + (B * wo + B) * p * 2, yr[1] + (((B + h - 1) * wo + B + w - 1) * p + ch) * 2))
        v = bf16.from_bits(self._img(x, h, w, p)[B:B + h, B:B + w, :ch]).astype(np.float64)
        mean = v.mean(2, keepdims=True)
        var = ((v - mean) ** 2).mean(2, keepdims=True)
        f = ((v - mean) / np.sqrt(var + eps)).astype(np.float32)
        self._img(y, h, w, p)[B:B + h, B:B + w, :ch] = bf16.bits(f * self._f32(wp, ch) + self._f32(bp, ch))

    def _k_geglu(self, job, c, args):
        """本仓 geglu!：y = 舍(h · gelu(g))，gelu 是精确版 x · Φ(x)（Φ 用 Abramowitz–Stegun 7.1.26，同 kernel）。"""
        n = c["n"]
        hp, gp, y = args
        job.reads.append(self._check(job.kernel, "h", hp, n * 2))
        job.reads.append(self._check(job.kernel, "g", gp, n * 2))
        job.writes.append(self._check(job.kernel, "y", y, n * 2))
        hv = bf16.from_bits(self._u16(hp, n)).astype(np.float64)
        gv = bf16.from_bits(self._u16(gp, n)).astype(np.float64)
        z = np.abs(gv) / np.sqrt(2)
        t_ = 1 / (1 + 0.3275911 * z)
        erf = 1 - (((((1.061405429 * t_ - 1.453152027) * t_) + 1.421413741) * t_ - 0.284496736) * t_ + 0.254829592) * t_ * np.exp(-z * z)
        phi = 0.5 * (1 + np.sign(gv) * erf)
        self._u16(y, n)[:] = bf16.bits((hv * gv * phi).astype(np.float32))

    def _win(self, p: Ptr, h: int, w: int, ch: int, pp: int):
        """指针 p 指到通道窗口首通道、行距 pp 的 [h + 4][w + 4][pp] 图：内部 [h][w][ch] 的视图（可写，写回内存）与它覆盖的字节数。"""
        wo = w + 2 * B
        n = ((B + h - 1) * wo + B + w - 1) * pp + ch
        flat = self._u16(p, n)
        return np.lib.stride_tricks.as_strided(flat[(B * wo + B) * pp:], (h, w, ch), (wo * pp * 2, pp * 2, 2)), n * 2

    def _k_copy_win(self, job, c, args):
        """本仓 copy_win!：x 的内部（窗口的 c 个通道）⇒ y 的通道窗口（指针已偏到窗口首通道），只写内部。"""
        h, w, ch, pi, po = c["h"], c["w"], c["c"], c["pin"], c["pout"]
        x, y = args
        wo = w + 2 * B
        src, nx = self._win(x, h, w, ch, pi)
        dst, ny = self._win(y, h, w, ch, po)
        job.reads.append(self._check(job.kernel, "x", x, nx))
        yr = self._check(job.kernel, "y", y, ny)
        job.writes.append((yr[0], yr[1] + (B * wo + B) * po * 2, yr[2]))
        dst[:] = src

    def _k_res_add(self, job, c, args):
        """本仓 res_add!：y 的内部 = 舍(a + r)（三张带边框的图，只读写内部的 c 个通道）。"""
        h, w, ch = c["h"], c["w"], c["c"]
        a, r, y = args
        wo = w + 2 * B
        va, na = self._win(a, h, w, ch, c["pa"])
        vr, nr = self._win(r, h, w, ch, c["pr"])
        vy, ny = self._win(y, h, w, ch, c["py"])
        job.reads.append(self._check(job.kernel, "a", a, na))
        job.reads.append(self._check(job.kernel, "r", r, nr))
        yr = self._check(job.kernel, "y", y, ny)
        job.writes.append((yr[0], yr[1] + (B * wo + B) * c["py"] * 2, yr[2]))
        vy[:] = bf16.bits(bf16.from_bits(va) + bf16.from_bits(vr))

    def _k_pack_int(self, job, c, args):
        """本仓 pack_int!：x 的内部（前 c 个通道）⇒ 稠密的 y [h · w][c]。"""
        h, w, ch, p = c["h"], c["w"], c["c"], c["p"]
        x, y = args
        src, nx = self._win(x, h, w, ch, p)
        job.reads.append(self._check(job.kernel, "x", x, nx))
        job.writes.append(self._check(job.kernel, "y", y, h * w * ch * 2))
        self._u16(y, h * w * ch)[:] = np.ascontiguousarray(src).ravel()

    def _k_unpack_int(self, job, c, args):
        """本仓 unpack_int!：稠密的 x [h · w][c] ⇒ y 的内部（边框不动）。"""
        h, w, ch, p = c["h"], c["w"], c["c"], c["p"]
        x, y = args
        wo = w + 2 * B
        dst, ny = self._win(y, h, w, ch, p)
        job.reads.append(self._check(job.kernel, "x", x, h * w * ch * 2))
        yr = self._check(job.kernel, "y", y, ny)
        job.writes.append((yr[0], yr[1] + (B * wo + B) * p * 2, yr[2]))
        dst[:] = self._u16(x, h * w * ch).reshape(h, w, ch)

    def _k_unary(self, job, c, args):
        if c.get("f") != "silu":
            raise SimError(f"{job.kernel}：模拟器只写了 unary! 的 silu")
        n = c["n"]
        x, y = args
        job.reads.append(self._check(job.kernel, "x", x, n * 2))
        job.writes.append(self._check(job.kernel, "y", y, n * 2))
        v = bf16.from_bits(self._u16(x, n))
        self._u16(y, n)[:] = bf16.bits((v / (1 + np.exp(-v))).astype(np.float32))

    def _k_bias_fold(self, job, c, args):
        n = c["n"]
        a, bp, y = args
        job.reads.append(self._check(job.kernel, "a", a, n * 4))
        job.reads.append(self._check(job.kernel, "b", bp, n * 2))
        job.writes.append(self._check(job.kernel, "y", y, n * 4))
        self._f32(y, n)[:] = self._f32(a, n) + bf16.from_bits(self._u16(bp, n))

    def report(self) -> str:
        by = {}
        for j in self.jobs:
            a = by.setdefault(j.kind, [0, 0.0])
            a[0] += 1
            a[1] += j.seconds
        parts = "、".join(f"{k} {n} 次" for k, (n, _) in by.items())
        return f"{len(self.jobs)} 个作业（{parts}），{self.barriers} 条屏障；接线检查全过；numpy 用时 {sum(j.seconds for j in self.jobs):.1f} s"


def simulate(cvx: Path, kernels_rs: Path, consts: dict, inputs: dict, verbose: bool = False) -> Sim:
    return Sim(Path(cvx).read_text(), Path(kernels_rs).read_text(), consts, inputs, verbose).run()
