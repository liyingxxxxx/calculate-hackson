"""SD1.5 UNet（B3，分段程序，DESIGN D18–D20）的命令行：打包、出 calbin、numpy 模拟、对参照、（获准后）上板。

用法（先 `source ../env.sh`；Python 用 CPU 参照环境：`$CALCULET_HOME/venv/bin/python -I tools/unet.py …`）：

  params                 权重 ⇒ out/unet/params/<段>/param_blk0.bin（按 crates/unet/plan.json 的偏移，每段一份）
  structure              每段的设备遍 + docc（不带参数块，只核结构）：每核代码大小、doctor ⇒ out/unet-struct/<段>
  calbin                 params + 每段的设备遍 + docc ⇒ out/unet-real/<段>/cal（带真权重），doctor 都过才算
  sim [--t T]            一次 UNet 前向（CFG 两路：有提示词、空提示词）各段依次在 numpy 里照编出来的 .cvx 重放（查接线）
                         ⇒ 对两份独立参照（ref/unet_ref.py：bf16 口径的 torch、diffusers fp32）⇒ out/unet-sim/metrics.json
  board [--t T] [--runs N]   🚨 上板：要 SD15NPU_ALLOW_BOARD=1（负责人确认这块卡可以用才设）。每段只部署一次（计时）
                         ⇒ 整次前向跑 N 遍（逐字节比是否一致）⇒ 对两份参照 ⇒ 过门槛（BOARD_TOL）退出 0，否则 1

输入（ref/unet_ref.make_inputs）：latent 标准正态（种子 0），上下文是本仓 SD1.5 的 CLIP 对一句提示词 / 空串编出来的。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "host"), str(ROOT / "ref")]

import numpy as np  # noqa: E402

import unet_ref  # noqa: E402
from sd15npu import runner, unet  # noqa: E402

PARAMS, CALBINS = unet.PARAMS, unet.CALBINS
PROMPT = "a photo of an astronaut riding a horse on mars"
# 容差（sim 对 bf16 口径的 torch 参照）：两边都是 bf16 激活、f32 累加；参照里 softmax 在 f32、kernel 内部分数与 p 舍 bf16，
# 首轮按 bf16 参照对 fp32 的信噪比（41–52 dB，ref/unet_ref.py 自检）留余量
TOL = {"cos_bf16": 0.999, "snr_bf16_db": 30.0}
BOARD_TOL = {"cos_bf16": 0.999, "snr_bf16_db": 30.0}


def sd_dir() -> Path:
    return runner.models_dir() / "stable-diffusion-v1-5"


def load():
    return unet_ref.load_unet(sd_dir() / "unet")


def cmd_params(a=None, model=None):
    P = unet.plan()
    t = time.time()
    ps = unet.write_params(model or load(), P, PARAMS)
    print(f"{len(ps)} 段参数块共 {sum(p.stat().st_size for p in ps) / 2**30:.2f} GiB ⇒ {PARAMS}（{time.time() - t:.0f} s）")


def build(feature: str, out: Path, params: Path | None) -> tuple[bool, str]:
    cmd = [str(ROOT / "tools/calbin.sh"), "unet", "--features", feature, "-o", str(out)] + (["--params", str(params)] if params else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    doc = out / "doctor.txt"
    ok = r.returncode == 0 and doc.exists() and "⛔ 拒收 0 条" in doc.read_text()
    return ok, (r.stdout + r.stderr)[-2000:]


def cmd_structure(a):
    """每段：设备遍 + docc（只核结构）⇒ 每核 .text 大小。"""
    sys.path.insert(0, str(ROOT / "tools"))
    import unet_gen

    P = unet.plan()
    bad = []
    for st in P["stages"]:
        f = st["feature"]
        if a.only and f not in a.only.split(","):
            continue
        out = ROOT / "out" / "unet-struct" / f
        t = time.time()
        ok, log = build(f, out, None)
        if ok:
            ts = [unet_gen.text_size(out / "cal" / "unet" / f"ccu{c}_chip0.elf") for c in (0, 1)]
            print(f"✓ {f}：.text {max(ts) / 1024:.0f} KB（{', '.join(st['units'][:1] + st['units'][-1:])}；{time.time() - t:.0f} s）", flush=True)
        else:
            bad.append(f)
            print(f"✗ {f}\n{log}", flush=True)
    sys.exit(1 if bad else 0)


def build_calbins(model=None) -> bool:
    if not unet.params_fresh():
        cmd_params(model=model)
    ok = True
    for st in unet.plan()["stages"]:
        f = st["feature"]
        good, log = build(f, CALBINS / f, PARAMS / f)
        print(f"{'✓' if good else '✗'} {f}" + ("" if good else "\n" + log), flush=True)
        ok = ok and good
    return ok


def cmd_calbin(a):
    sys.exit(0 if build_calbins() else 1)


def inputs(model=None):
    x, ctx = unet_ref.make_inputs(sd_dir(), prompt=PROMPT)
    _, ctx0 = unet_ref.make_inputs(sd_dir(), prompt="")
    return x, [ctx, ctx0]


def refs(model, x, t, ctxs):
    out = {}
    for name, fn in (("bf16", unet_ref.forward_bf16), ("fp32", unet_ref.forward_fp32)):
        out[name] = [fn(model, x, t, c) for c in ctxs]
    return out


def compare(got: list, r: dict) -> dict:
    m = {}
    for br, name in enumerate(("cond", "uncond")):
        m[name] = {"vs_bf16": unet_ref.metrics(got[br], r["bf16"][br]), "vs_fp32": unet_ref.metrics(got[br], r["fp32"][br]),
                   "ref_bf16_vs_fp32": unet_ref.metrics(r["bf16"][br], r["fp32"][br])}
    # CFG 之后（guidance 7.5）的噪声预测：采样器真正用的量
    g = 7.5
    cfg = lambda e: e[1] + g * (e[0] - e[1])  # noqa: E731
    m["cfg"] = {"vs_bf16": unet_ref.metrics(cfg(got), cfg(r["bf16"])), "vs_fp32": unet_ref.metrics(cfg(got), cfg(r["fp32"]))}
    return m


def _fmt(m: dict) -> str:
    s = []
    for k in ("cond", "uncond", "cfg"):
        a, b = m[k]["vs_bf16"], m[k]["vs_fp32"]
        s.append(f"{k}：对 bf16 参照余弦 {a['cos']:.6f}、信噪比 {a['snr_db']:.1f} dB；对 fp32 {b['cos']:.6f}、{b['snr_db']:.1f} dB")
    return "\n".join(s)


def _ok(m: dict, tol: dict) -> list[str]:
    bad = []
    for k in ("cond", "uncond"):
        a = m[k]["vs_bf16"]
        if not a["finite"]:
            bad.append(f"{k}：有 NaN / Inf")
        if a["cos"] < tol["cos_bf16"] or a["snr_db"] < tol["snr_bf16_db"]:
            bad.append(f"{k}：对 bf16 参照余弦 {a['cos']:.6f}、{a['snr_db']:.1f} dB（门槛 {tol['cos_bf16']} / {tol['snr_bf16_db']} dB）")
    return bad


def cmd_sim(a):
    model = load()
    if not unet.params_fresh():
        cmd_params(model=model)
    P = unet.plan()
    for st in P["stages"]:
        runner_build(st["feature"])
    x, ctxs = inputs()
    params = [PARAMS / st["feature"] / "param_blk0.bin" for st in P["stages"]]
    t0 = time.time()
    got, info = unet.run(P, x, a.t, ctxs, "npu-sim", params, verbose=a.verbose)
    for f, rep in info["stages"].items():
        print(f"{f}：{rep}", flush=True)
    t_sim = time.time() - t0
    r = refs(model, x, a.t, ctxs)
    m = {"t": a.t, "sim_s": round(t_sim, 1), "stages": info["stages"], **compare(got, r)}
    out = ROOT / "out" / "unet-sim"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    np.save(out / "eps_sim.npy", np.stack(got))
    print(_fmt(m))
    bad = _ok(m, TOL)
    print(("✗ " + "\n✗ ".join(bad)) if bad else f"✓ UNet 分段程序的模拟（{len(P['stages'])} 段、CFG 两路）：接线检查全过，对 bf16 参照在容差内（numpy 模拟，不是 NPU）")
    sys.exit(1 if bad else 0)


def runner_build(feature: str):
    """设备遍（cargo）：出第几段的 unet.cvx（模拟器读它）。"""
    subprocess.run(["cargo", "build", "-q", "-p", "unet", "--lib", "--target", "dopamine-unknown-none-cvx", "--features", feature,
                    "--target-dir", str(ROOT / "target" / "features" / f"unet-{feature}")], cwd=ROOT, check=True)


def cmd_board(a):
    """🚨 上板：每段只部署一次（计时）⇒ 整次前向跑 N 遍（计时、逐字节比是否一致）⇒ 对两份参照 ⇒ 过门槛退出 0，否则 1。"""
    from sd15npu import dort

    if not dort.allowed():
        sys.exit("没开上板开关 SD15NPU_ALLOW_BOARD=1：负责人确认这块卡可以用才设（AGENTS.md 硬规矩 3）")
    model = load()
    P = unet.plan()
    have = all((CALBINS / st["feature"] / "cal" / "param_blk0.bin").exists() for st in P["stages"]) and unet.params_fresh()
    if not (a.no_build and have) and not build_calbins(model):
        sys.exit("UNet 的 calbin 出不来（见上面 calbin.sh 的输出）")
    ok, why, _ = dort.device_status(log=ROOT / "out" / "unet-board" / "device.log")
    if not ok:
        sys.exit("✗ 这块卡现在不能用：" + "；".join(why) + "（什么都没往卡上发）")
    label, suffix = dort.measured_label(), dort.result_suffix()
    out = ROOT / "out" / f"unet-board{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "dort.log"
    deploy = {}
    for st in P["stages"]:
        deploy[st["feature"]] = round(dort.deploy_only(CALBINS / st["feature"] / "cal", "unet", log=log)["deploy_wall_s"], 3)
        print(f"  只部署 {st['feature']}：{deploy[st['feature']]:.3f} s（参数块 {st['w']['bytes'] / 2**20:.0f} MB）", flush=True)
    x, ctxs = inputs()
    outs, walls = [], []
    for i in range(a.runs):
        got, info = unet.run(P, x, a.t, ctxs, "npu", [], log_dir=out)
        outs.append(np.stack(got))
        walls.append(info["stages"])
        print(f"  第 {i + 1} 遍：各段合计 {sum(info['stages'].values()):.2f} s（主机墙钟 {info['seconds']:.2f} s）", flush=True)
    same = all(np.array_equal(o, outs[0]) for o in outs[1:])
    r = refs(model, x, a.t, ctxs)
    m = compare(list(outs[0]), r)
    tot = sorted(sum(w.values()) for w in walls)
    med = tot[len(tot) // 2]
    bad = _ok(m, BOARD_TOL) + ([] if same else ["几遍结果不一致（多半是少屏障）"])
    res = {"t": a.t, "measured": label, "runs": a.runs, "stages": len(P["stages"]), "deploy_only_wall_s": deploy,
           "deploy_only_total_s": round(sum(deploy.values()), 3), "run_wall_s": walls, "total_median_s": round(med, 3),
           "inference_est_s": round(med - sum(deploy.values()), 3),
           "inference_note": "一次 UNet 前向（CFG 两路）各段整次（各自部署 + 推理 + 读写）之和的中位数减各段「只部署」之和，是估算",
           "deterministic": same, **m, "tolerance": BOARD_TOL, "ok": not bad}
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    np.save(out / "eps_board.npy", outs[0])
    print(_fmt(m))
    print(f"{'✓' if not bad else '✗'} UNet 上板（{label}）：{a.runs} 次{'逐字节一致' if same else '⚠ 不一致'}；{len(P['stages'])} 段整次中位 {med:.2f} s"
          f"（只部署合计 {sum(deploy.values()):.2f} s）；{out}" + ("" if not bad else "\n✗ " + "\n✗ ".join(bad)))
    sys.exit(0 if not bad else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("params").set_defaults(f=cmd_params)
    p = sub.add_parser("structure")
    p.add_argument("--only")
    p.set_defaults(f=cmd_structure)
    sub.add_parser("calbin").set_defaults(f=cmd_calbin)
    p = sub.add_parser("sim")
    p.add_argument("--t", type=float, default=500.0)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(f=cmd_sim)
    p = sub.add_parser("board")
    p.add_argument("--t", type=float, default=500.0)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--no-build", action="store_true")
    p.set_defaults(f=cmd_board)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
