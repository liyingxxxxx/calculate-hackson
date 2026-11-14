"""真卡上逐个 kernel 计时（性能优化用）：每个 kernel 一个小程序（`crates/prof`，feature = kernel 名），在卡上把它就地连调 REPS 遍。

为什么这么量：dort 0.4.1 的 `calbin run` 只给整次的墙钟（含部署、读写），没有卡上每个 kernel 的时间。这里让输入、输出都只有 64 B，
权重与特征图全在卡上（@w 是全零的参数块，@act 不清零：只量时间、不看数），于是
    这个 kernel 一遍的时间 ≈ [（整次 − 只部署）−（base 的整次 − base 的只部署）] / REPS
base 是一个只有屏障、不调 kernel 的程序，扣掉一次调用的固定开销（建进程、configure、写读 64 B、起程序）。
kernel 的形状直接从 `crates/vae/src/kernels.rs` 读（同模拟器：`cvxsim.parse_specs`），测的就是分段程序里跑的那一份。

用法（先 source env.sh；Python 用参照环境 `$CALCULET_HOME/venv/bin/python -I tools/prof.py …`）：
  gen                         生成 crates/prof（Cargo.toml 的 feature 与 src/lib.rs）
  build [k1,k2,…]             每个 kernel 出一个 calbin（tools/calbin.sh，参数块全零：只验结构、只用来计时）⇒ out/prof/<k>/cal
  board [k1,k2,…] [--runs N]  🚨 上板：要 SD15NPU_ALLOW_BOARD=1。先只读查卡 ⇒ 每个程序只部署一次 + 整次 N 遍 ⇒ out/prof/board.json 与一张表
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "host")]

from sd15npu import cvxsim  # noqa: E402

REPS = 4
B = 2
MARGIN_PX = 128
KRS = ROOT / "crates" / "vae" / "src" / "kernels.rs"
OUT = ROOT / "out" / "prof"


def a4k(n: int) -> int:
    return (n + 4095) // 4096 * 4096


def img(h: int, p: int) -> int:
    return (h + 2 * B) ** 2 * p * 2


def specs() -> dict:
    sp = cvxsim.parse_specs(KRS.read_text(), {})
    return {k: v for k, v in sp.items() if v[0] != "fill16k"}


def plan(kind: str, c: dict) -> dict:
    """一个 kernel 的计时程序：@act 里的各块（名字 ⇒ 字节）、@w 的各块、调用的参数次序、计算量 / 搬运量。"""
    if kind == "conv2d":
        h, s, k, ci, co = c["h"], c["s"], c["k"], c["cin"], c["cout"]
        pin, pout = c["pin"][0], c["pout"][0]
        act = {"x": img(h * s, pin) + MARGIN_PX * pin * 2, "y": img(h, pout)}
        args = ["act:x", "w:w", "w:b"]
        if "res" in c:
            act["r"] = img(h, c["res"][0])
            args.append("act:r")
        args.append("act:y")
        return {"act": act, "w": {"w": k * k * ci * co * 2, "b": co * 4}, "args": args,
                "flop": h * c["w"] * ci * co * k * k * 2, "bytes": None}
    if kind == "group_norm":
        hw, ch, g = c["hw"], c["c"], c["groups"]
        cc = cvxsim.Sim._norm_cols(ch // g * hw, hw)
        rows = ch * hw // cc
        return {"act": {"x": ch * hw * 2, "y": ch * hw * 2}, "w": {"w": rows * 32, "b": rows * 32},
                "args": ["act:x", "w:w", "w:b", "act:y"], "flop": None, "bytes": ch * hw * 2 * 4}  # 读三遍、写一遍
    if kind == "gn_stats":
        h, ch, p = c["h"], c["c"], c["p"]
        return {"act": {"x": img(h, p), "st": 4 * ch * 4}, "w": {}, "args": ["act:x", "act:st"], "flop": None, "bytes": h * c["w"] * ch * 2}
    if kind == "gn_apply":
        h, ch, p = c["h"], c["c"], c["p"]
        return {"act": {"x": img(h, p), "st": 4 * ch * 4, "y": img(h, p)}, "w": {"w": ch * 4, "b": ch * 4, "m": c["groups"] * ch * 4},
                "args": ["act:x", "act:st", "w:w", "w:b", "w:m", "act:y"], "flop": None, "bytes": 2 * h * c["w"] * ch * 2}  # 读一遍、写一遍
    if kind in ("nhwc_to_nchw", "nchw_to_nhwc"):
        h, ch, p = c["h"], c["c"], c["p"]
        dense, bord = ch * h * c["w"] * 2, img(h, p)
        act = {"x": bord, "y": dense} if kind == "nhwc_to_nchw" else {"x": dense, "y": bord}
        return {"act": act, "w": {}, "args": ["act:x", "act:y"], "flop": None, "bytes": 2 * dense}
    if kind == "transpose2d":
        n = c["rows"] * c["cols"] * 2
        return {"act": {"x": n, "y": n}, "w": {}, "args": ["act:x", "act:y"], "flop": None, "bytes": 2 * n}
    if kind == "linear":
        r, k, n = c["rows"], c["k"], c["n"]
        return {"act": {"x": r * k * 2, "y": r * n * 2}, "w": {"w": n * k * 2, "b": n * 4}, "args": ["act:x", "w:w", "w:b", "act:y"],
                "flop": 2 * r * k * n, "bytes": None}
    if kind == "attn_wide":
        n = c["rows"] * c["heads"] * c["head"] * 2
        return {"act": {"q": n, "k": n, "v": n, "o": n}, "w": {}, "args": ["act:q", "act:k", "act:v", "act:o"],
                "flop": 4 * c["rows"] ** 2 * c["head"] * c["heads"], "bytes": None}
    if kind == "upsample2":
        h, pin, pout = c["h"], c["pin"], c["pout"]
        return {"act": {"x": img(h, pin) + MARGIN_PX * pin * 2, "y": img(2 * h, pout) + MARGIN_PX * pout * 2}, "w": {},
                "args": ["act:x", "act:y"], "flop": None, "bytes": (h * c["w"] + 4 * h * c["w"]) * c["c"] * 2}
    raise SystemExit(f"prof：没写 {kind} 的计时程序")


def offsets(blocks: dict) -> tuple[dict, int]:
    off, at = {}, 0
    for name, n in blocks.items():
        off[name] = at
        at += a4k(n)
    return off, at


def cmd_gen(a):
    sp = specs()
    feats = ["base"] + list(sp)
    d = ROOT / "crates" / "prof"
    (d / "src").mkdir(parents=True, exist_ok=True)
    (d / "Cargo.toml").write_text(
        '[package]\nname = "prof"\nversion = "0.1.0"\nedition = "2024"\npublish = false\n'
        'description = "真卡上逐个 kernel 计时：每个 feature 一个程序，把正式 VAE 的一个 kernel 就地连调几遍（tools/prof.py 生成，不要手改）"\n\n'
        '[lib]\ncrate-type = ["rlib"]\n\n# 一次只开一个（src/lib.rs 里有断言守着）\n[features]\n'
        + "".join(f"{f} = []\n" for f in feats)
        + '\n[dependencies]\nnpu.workspace = true\ncalblas.workspace = true\nsd-kernels = { path = "../kernels" }\n')
    lines = ["//! 真卡上逐个 kernel 计时（**`tools/prof.py gen` 生成，不要手改**）：每个 feature 一个程序，把正式 VAE 的一个 kernel 就地连调",
             f"//! {REPS} 遍，输入输出各 64 B、权重全零、@act 不清零（只量时间、不看数）。`base` 只有屏障，扣固定开销用。",
             "#![no_std]", "#![allow(dead_code)]", "",
             '#[path = "../../vae/src/kernels.rs"]', "mod vae_kernels;", "",
             "const N_FEATURES: usize = " + " + ".join(f'cfg!(feature = "{f}") as usize' for f in feats) + ";",
             'const _: () = assert!(N_FEATURES <= 1, "prof：一次只开一个 feature");', ""]
    lines += ['#[cfg(feature = "base")]', "mod prog {", "    use npu::prelude::*;",
              "    npu::decls! {", "        in_x: input, bytes = 64;", "        out_y: output, bytes = 64;", "    }",
              "    #[npu::program(chiplets = 1, cores = 2)]", "    pub fn step() {"] + ["        sync_cores();"] * REPS + ["    }", "}", ""]
    for name, (kind, c) in sp.items():
        p = plan(kind, c)
        ao, at = offsets(p["act"])
        wo, wt = offsets(p["w"])
        decl = ["        in_x: input, bytes = 64;", "        out_y: output, bytes = 64;"]
        if wt:
            decl.append(f"        w: weights, bytes = {wt};")
        decl.append(f"        act: workspace, bytes = {at};")
        consts = [f"    const A_{k.upper()}: u64 = {v};" for k, v in ao.items()] + [f"    const W_{k.upper()}: u64 = {v};" for k, v in wo.items()]
        call_args = ", ".join(f"{s.split(':')[0]}().at({s.split(':')[0].upper()[0]}_{s.split(':')[1].upper()}).cast()" for s in p["args"])
        lines += [f'/// {name}：{kind} {c}', f'#[cfg(feature = "{name}")]', "mod prog {", "    use npu::prelude::*;",
                  "    npu::decls! {", *decl, "    }", *consts,
                  "    #[npu::program(chiplets = 1, cores = 2)]", "    pub fn step() {"]
        for _ in range(REPS):
            lines += [f"        crate::vae_kernels::{name}({call_args});", "        sync_cores();"]
        lines += ["    }", "}", ""]
    (d / "src" / "lib.rs").write_text("\n".join(lines))
    print(f"crates/prof：{len(feats)} 个程序（base + {len(sp)} 个 kernel）")


def _names(arg: str | None) -> list[str]:
    sp = specs()
    return ["base"] + (list(sp) if not arg else [k for k in arg.split(",") if k != "base"])


def cmd_build(a):
    bad = []
    t0 = time.time()
    for k in _names(a.kernels):
        r = subprocess.run([str(ROOT / "tools/calbin.sh"), "prof", "--features", k, "-o", str(OUT / k)], cwd=ROOT, capture_output=True, text=True)
        ok = r.returncode == 0 and "⛔ 拒收 0 条" in (OUT / k / "doctor.txt").read_text()
        print(f"{'✓' if ok else '✗'} {k}：{(r.stdout.strip().splitlines() or [''])[-3:][0] if ok else (r.stdout + r.stderr)[-300:]}", flush=True)
        if not ok:
            bad.append(k)
    print(f"{'全部出来了' if not bad else '出不来：' + '、'.join(bad)}（{time.time() - t0:.0f} s）")
    sys.exit(1 if bad else 0)


def cmd_board(a):
    from sd15npu import dort

    if not dort.allowed():
        sys.exit("没开上板开关 SD15NPU_ALLOW_BOARD=1：负责人确认这块卡可以用才设（AGENTS.md 硬规矩 3）")
    ok, why, out = dort.device_status(log=OUT / "device.log")
    if not ok:
        sys.exit("✗ 这块卡现在不能用：" + "；".join(why) + "（什么都没往卡上发）")
    sp = specs()
    names = _names(a.kernels)
    log = OUT / "dort.log"
    res = {}
    for k in names:
        cal = OUT / k / "cal"
        if not cal.exists():
            sys.exit(f"没有 {cal}（先 tools/prof.py build）")
        dep = dort.deploy_only(cal, "prof", log=log)["deploy_wall_s"]
        walls = [dort.run_raw(cal, "prof", bytes(64), "in_x", "out_y", log=log)[1]["dort_wall_s"] for _ in range(a.runs)]
        res[k] = {"deploy_only_s": dep, "run_walls_s": walls, "net_s": statistics.median(walls) - dep}
        print(f"  {k}：只部署 {dep * 1e3:.1f} ms，整次 {', '.join(f'{w * 1e3:.1f}' for w in walls)} ms", flush=True)
    base = res["base"]["net_s"]
    rows = []
    for k in names[1:]:
        kind, c = sp[k]
        p = plan(kind, c)
        t = (res[k]["net_s"] - base) / REPS
        r = {"kernel": k, "kind": kind, "spec": c, "per_call_s": t, **res[k]}
        if p["flop"]:
            r["gflop"] = p["flop"] / 1e9
            r["tflops"] = p["flop"] / t / 1e12 if t > 0 else None
        if p["bytes"]:
            r["mbytes"] = p["bytes"] / 1e6
            r["gbps"] = p["bytes"] / t / 1e9 if t > 0 else None
        rows.append(r)
    (OUT / "board.json").write_text(json.dumps({"measured": dort.measured_label(), "reps": REPS, "base_net_s": base, "kernels": rows,
                                                "date": time.strftime("%F %T")}, ensure_ascii=False, indent=1))
    print(f"\n{dort.measured_label()}：一遍的时间 = [（整次 − 只部署）−（base 的）] / {REPS}；base 的净时间 {base * 1e3:.2f} ms")
    for r in rows:
        extra = f"{r['tflops']:.2f} TFLOPS（{r['gflop']:.1f} GFLOP）" if r.get("tflops") else (f"{r['gbps']:.1f} GB/s（{r['mbytes']:.0f} MB）" if r.get("gbps") else "")
        print(f"  {r['kernel']:<11} {r['kind']:<13} {r['per_call_s'] * 1e3:8.2f} ms  {extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen").set_defaults(f=cmd_gen)
    p = sub.add_parser("build")
    p.add_argument("kernels", nargs="?")
    p.set_defaults(f=cmd_build)
    p = sub.add_parser("board")
    p.add_argument("kernels", nargs="?")
    p.add_argument("--runs", type=int, default=3)
    p.set_defaults(f=cmd_board)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
