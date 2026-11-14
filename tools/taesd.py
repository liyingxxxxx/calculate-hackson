"""TAESD 解码器（B1）的命令行：打包、出 calbin、numpy 模拟、对参照、（获准后）上板。

用法（先 `source ../env.sh`；Python 用 CPU 参照环境：`$CALCULET_HOME/venv/bin/python -I tools/taesd.py …`）：

  params                 权重 ⇒ out/taesd/params/param_blk0.bin（+ sha256）
  calbin                 params + 设备遍 + docc ⇒ out/taesd-real/cal（带真权重），doctor 过才算
  sim [--latent K]       设备遍 ⇒ numpy 照 .cvx 重放（查接线）⇒ 对两份独立参照（bf16 口径的 torch、diffusers fp32）
                         ⇒ out/taesd-sim/<K>/{sim,ref_bf16,ref_fp32,source}.png 与 metrics.json
  check                  sim 跑一组 latent（image、randn:0、zeros、big），超出容差就退出 1（tools/verify_all.sh 用）
  board [--latent K] [--runs N]   🚨 上板：要 SD15NPU_ALLOW_BOARD=1（负责人确认这块卡可以用才设）。重出 calbin ⇒ 只部署一次（计时）
                         ⇒ 整次跑 N 遍（缺省 3；逐字节比是否一致）⇒ 对两份参照 ⇒ 过门槛（BOARD_TOL）退出 0，否则 1

latent K：image（合成测试图经 TAESD 编码器 ⇒ latent，默认）、randn:<种子>、zeros、big（±8，过 tanh 饱和段）。
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

import taesd_ref  # noqa: E402
from sd15npu import cvxsim, runner, taesd  # noqa: E402

# 容差（sim 对 bf16 口径的 torch 参照）：两边都是 bf16 激活、f32 累加，只差累加次序 ⇒ 个别像素差 1–2 个 bf16 ULP，
# 经后面的层放大后整张图仍应贴得很近。对 fp32 的 PSNR 门槛按 2026-11-14 实测值留余量（见 PROGRESS）。
TOL = {"cos_bf16": 0.9999, "max_abs_bf16": 0.05, "psnr_fp32_db": 35.0}


def cmd_params(a):
    L = taesd.layout()
    p = taesd.write_params(L, runner.taesd_dir() / "diffusion_pytorch_model.safetensors", runner.PARAMS)
    print(f"{p}（{p.stat().st_size} 字节）sha256 {(runner.PARAMS / 'param_blk0.sha256').read_text().split()[0]}")


def build_calbin() -> bool:
    """打参数块 + 设备遍 + docc ⇒ out/taesd-real/cal；doctor 拒收 0 条才算（参数块和程序是同一次构建的产物）。"""
    cmd_params(None)
    r = subprocess.run([str(ROOT / "tools/calbin.sh"), "taesd", "--params", str(runner.PARAMS), "-o", str(runner.CALBIN.parent)], cwd=ROOT)
    return r.returncode == 0 and "⛔ 拒收 0 条" in (runner.CALBIN.parent / "doctor.txt").read_text()


def cmd_calbin(a):
    sys.exit(0 if build_calbin() else 1)


def _refs(lat):
    vae = taesd_ref.load_tiny(runner.taesd_dir())
    t = time.time()
    r16 = taesd_ref.decode_bf16(vae, lat)
    t16 = time.time() - t
    t = time.time()
    r32 = taesd_ref.decode_fp32(vae, lat)
    t32 = time.time() - t
    return vae, r16, r32, {"ref_bf16_s": round(t16, 2), "ref_fp32_s": round(t32, 2)}


def run_sim(kind: str, out_dir: Path, verbose: bool = False) -> dict:
    runner.build_device_pass()
    L = taesd.layout()
    vae = taesd_ref.load_tiny(runner.taesd_dir())
    lat = taesd_ref.make_latent(vae, kind)
    data = taesd.pack_latent(L, lat)
    params = runner.ensure_params(L).read_bytes()
    t = time.time()
    sim = cvxsim.simulate(runner.CVX, runner.KERNELS_RS, L["consts"], {"in_x": data, "w": params}, verbose=verbose)
    t_sim = time.time() - t
    img = taesd.unpack_image(L, sim.mem["out_y"].tobytes())
    _, r16, r32, times = _refs(lat)
    m = {"latent": kind, "sim": sim.report(), "sim_s": round(t_sim, 2), **times,
         "sim_vs_ref_bf16": taesd_ref.metrics(img, r16), "sim_vs_ref_fp32": taesd_ref.metrics(img, r32),
         "ref_bf16_vs_ref_fp32": taesd_ref.metrics(r16, r32)}
    if kind == "image":
        src = taesd_ref.test_image()
        m["ref_fp32_vs_source"] = taesd_ref.metrics(r32, src)
        m["sim_vs_source"] = taesd_ref.metrics(img, src)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, x in [("sim", img), ("ref_bf16", r16), ("ref_fp32", r32)] + ([("source", taesd_ref.test_image())] if kind == "image" else []):
        taesd_ref.to_png(x, out_dir / f"{name}.png")
    (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    return m


def _fmt(m: dict) -> str:
    a, b = m["sim_vs_ref_bf16"], m["sim_vs_ref_fp32"]
    return (f"{m['latent']}：sim 对 bf16 参照 余弦 {a['cos']:.6f} 最大差 {a['max_abs']:.4f} PSNR {a['psnr_db']:.1f} dB；"
            f"对 fp32 余弦 {b['cos']:.6f} PSNR {b['psnr_db']:.1f} dB；sim {m['sim_s']} s")


def cmd_sim(a):
    m = run_sim(a.latent, ROOT / "out" / "taesd-sim" / a.latent.replace(":", "-"), verbose=a.verbose)
    print(m["sim"])
    print(_fmt(m))
    print(json.dumps(m, ensure_ascii=False, indent=1))


def cmd_check(a):
    bad = []
    for kind in ["image", "randn:0", "zeros", "big"]:
        m = run_sim(kind, ROOT / "out" / "taesd-sim" / kind.replace(":", "-"))
        print(_fmt(m))
        s16, s32 = m["sim_vs_ref_bf16"], m["sim_vs_ref_fp32"]
        if not s16["finite"]:
            bad.append(f"{kind}：sim 输出有 NaN / Inf")
        if s16["cos"] < TOL["cos_bf16"] and kind != "zeros":
            bad.append(f"{kind}：对 bf16 参照的余弦 {s16['cos']:.6f} < {TOL['cos_bf16']}")
        if s16["max_abs"] > TOL["max_abs_bf16"]:
            bad.append(f"{kind}：对 bf16 参照的最大差 {s16['max_abs']:.4f} > {TOL['max_abs_bf16']}")
        if kind in ("image", "randn:0") and s32["psnr_db"] < TOL["psnr_fp32_db"]:
            bad.append(f"{kind}：对 fp32 的 PSNR {s32['psnr_db']:.1f} dB < {TOL['psnr_fp32_db']}")
    if bad:
        print("✗ " + "\n✗ ".join(bad))
        sys.exit(1)
    print("✓ TAESD 整步程序的模拟：接线检查全过，对两份参照都在容差内")


# 上板门槛。首轮按 40 / 35 dB 收；2026-11-14 npu3 真机实测四组 latent，都是 3 遍逐字节一致，和模拟最多差 1.5 dB：
#   image 余弦 0.9999987、60.3 / 55.9 dB；randn:0 0.9999903、52.7 / 46.8；zeros 0.9999993、65.8 / 59.4；big 0.9999943、56.1 / 50.7
#   （对 bf16 参照 / 对 fp32 的 PSNR，test/results/npu3-20261114/）。
# 门槛对每一组都用 ⇒ 按最差的 randn:0 定：余弦 1 − cos 留一倍（0.99998），PSNR 留约 2 dB（50 / 45 dB）。
BOARD_TOL = {"cos_bf16": 0.99998, "psnr_bf16_db": 50.0, "psnr_fp32_db": 45.0}


def cmd_board(a):
    """🚨 上板：重出 calbin ⇒ 只部署一次（计时）⇒ 整次跑 N 遍（计时、逐字节比是否一致）⇒ 对两份参照 ⇒ 过门槛退出 0，否则 1。"""
    from sd15npu import dort

    if not dort.allowed():
        sys.exit("没开上板开关 SD15NPU_ALLOW_BOARD=1：负责人确认这块卡可以用才设（AGENTS.md 硬规矩 3）")
    if not build_calbin():
        sys.exit("TAESD 的 calbin 出不来（见上面 calbin.sh 的输出）")
    label, suffix = dort.measured_label(), dort.result_suffix()
    out = ROOT / "out" / f"taesd-board{suffix}" / a.latent.replace(":", "-")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "dort.log"
    L = taesd.layout()
    vae = taesd_ref.load_tiny(runner.taesd_dir())
    lat = taesd_ref.make_latent(vae, a.latent)
    data = taesd.pack_latent(L, lat)
    dep = dort.deploy_only(runner.CALBIN, "taesd", log=log)["deploy_wall_s"]
    outs, walls = [], []
    for _ in range(a.runs):
        o, info = dort.run_raw(runner.CALBIN, "taesd", data, "in_x", "out_y", log=log)
        outs.append(o)
        walls.append(info["dort_wall_s"])
    same = all(o == outs[0] for o in outs[1:])
    img = taesd.unpack_image(L, outs[0])
    _, r16, r32, _ = _refs(lat)
    m16, m32 = taesd_ref.metrics(img, r16), taesd_ref.metrics(img, r32)
    med = sorted(walls)[len(walls) // 2]
    ok = bool(same and m16["finite"] and m16["cos"] >= BOARD_TOL["cos_bf16"] and m16["psnr_db"] >= BOARD_TOL["psnr_bf16_db"]
              and m32["psnr_db"] >= BOARD_TOL["psnr_fp32_db"])
    m = {"latent": a.latent, "measured": label, "runs": a.runs, "deploy_only_wall_s": round(dep, 3), "run_wall_s": [round(w, 3) for w in walls],
         "inference_est_s": round(med - dep, 3), "inference_note": "整次（部署 + 推理 + 读回）的中位数减「只部署」，是估算",
         "deterministic": same, "board_vs_ref_bf16": m16, "board_vs_ref_fp32": m32, "tolerance": BOARD_TOL, "ok": ok}
    taesd_ref.to_png(img, out / "board.png")
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(f"{'✓' if ok else '✗'} TAESD 上板（{label}）：{a.runs} 次{'逐字节一致' if same else '⚠ 结果不一致（多半是少屏障）'}；"
          f"对 bf16 参照余弦 {m16['cos']:.6f}、PSNR {m16['psnr_db']:.1f} dB，对 fp32 PSNR {m32['psnr_db']:.1f} dB；"
          f"只部署 {dep:.2f} s，整次中位 {med:.2f} s ⇒ 推理约 {med - dep:.2f} s（估算）；{out}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("params").set_defaults(f=cmd_params)
    sub.add_parser("calbin").set_defaults(f=cmd_calbin)
    p = sub.add_parser("sim")
    p.add_argument("--latent", default="image")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(f=cmd_sim)
    sub.add_parser("check").set_defaults(f=cmd_check)
    p = sub.add_parser("board")
    p.add_argument("--latent", default="image")
    p.add_argument("--runs", type=int, default=3, help="整次跑几遍（查确定性、取时间的中位数）")
    p.set_defaults(f=cmd_board)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
