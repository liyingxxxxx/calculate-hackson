"""正式 VAE 解码器（B1b，6 段程序，DESIGN D15）的命令行：打包、出 calbin、numpy 模拟、对参照、（获准后）上板。

用法（先 `source ../env.sh`；Python 用 CPU 参照环境：`$CALCULET_HOME/venv/bin/python -I tools/vae.py …`）：

  params                 权重 ⇒ out/vae/params/<段>/param_blk0.bin（+ sha256），6 段
  calbin                 params + 每段的设备遍（`--features s1` …）+ docc ⇒ out/vae-real/<段>/cal（带真权重），doctor 都过才算
  sim [--latent K]       6 段依次在 numpy 里照编出来的 .cvx 重放（查接线；上一段的 out_y 原样当下一段的 in_x）
                         ⇒ 对两份独立参照（ref/vae_ref.py：bf16 口径的 torch、diffusers fp32）⇒ out/vae-sim/<K>/{sim,ref_bf16,ref_fp32}.png 与 metrics.json
  check                  sim 跑一组 latent（image、randn:0），超出容差就退出 1（tools/verify_all.sh 用）
  board [--latent K] [--runs N]   🚨 上板：要 SD15NPU_ALLOW_BOARD=1（负责人确认这块卡可以用才设）。重出 calbin ⇒ 每段只部署一次（计时）
                         ⇒ 整条（6 段）跑 N 遍（逐字节比是否一致）⇒ 对两份参照 ⇒ 过门槛（BOARD_TOL）退出 0，否则 1

latent K（VAE 原空间）：image（合成测试图经本 VAE 的编码器）、randn:<种子>、zeros、big。
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

import vae_ref  # noqa: E402
from sd15npu import cvxsim, runner, vae  # noqa: E402

PARAMS = runner.VAE_PARAMS
CALBINS = runner.VAE_CALBINS
KERNELS_RS = runner.VAE_KERNELS_RS
# 容差（sim 对 bf16 口径的 torch 参照）：两边都是 bf16 激活、f32 累加，只差累加次序与 GroupNorm、softmax 里的求和次序。
# 首轮按 TAESD 的口径定（sim 对 bf16 参照 TAESD 是 52–67 dB），实测后再收（PROGRESS「B1b」）。
TOL = {"cos_bf16": 0.9999, "psnr_bf16_db": 40.0, "psnr_fp32_db": 35.0}


vae_dir, stage_cvx, build_stage = runner.vae_dir, runner.vae_stage_cvx, runner.build_vae_stage


def cmd_params(a=None):
    L = vae.layout()
    t = time.time()
    ps = vae.write_params(L, vae_dir() / "diffusion_pytorch_model.safetensors", PARAMS)
    for p in ps:
        print(f"{p}（{p.stat().st_size} 字节）sha256 {(p.parent / 'param_blk0.sha256').read_text().split()[0][:16]}…")
    print(f"6 段参数块共 {sum(p.stat().st_size for p in ps) / 2**20:.1f} MB，{time.time() - t:.1f} s")


def build_calbins() -> bool:
    """打参数块 + 每段的设备遍 + docc ⇒ out/vae-real/<段>/cal；每段 doctor 拒收 0 条才算（参数块和程序是同一次构建的产物）。"""
    cmd_params()
    ok = True
    for st in vae.layout()["stages"]:
        out = CALBINS / st["name"]
        r = subprocess.run([str(ROOT / "tools/calbin.sh"), "vae", "--features", st["name"], "--params", str(PARAMS / st["name"]), "-o", str(out)], cwd=ROOT)
        ok = ok and r.returncode == 0 and "⛔ 拒收 0 条" in (out / "doctor.txt").read_text()
    return ok


def cmd_calbin(a):
    sys.exit(0 if build_calbins() else 1)


def _refs(lat):
    m = vae_ref.load_vae(vae_dir())
    t = time.time()
    r16 = vae_ref.decode_bf16(m, lat)
    t16 = time.time() - t
    t = time.time()
    r32 = vae_ref.decode_fp32(m, lat)
    return r16, r32, {"ref_bf16_s": round(t16, 2), "ref_fp32_s": round(time.time() - t, 2)}


def run_sim(kind: str, out_dir: Path, verbose: bool = False) -> dict:
    L = vae.layout()
    if not all((PARAMS / st["name"] / "param_blk0.bin").exists() for st in L["stages"]):
        cmd_params()
    lat = vae_ref.make_latent(vae_ref.load_vae(vae_dir()), kind)
    data = vae.pack_latent(L, lat)
    reports, t_all = [], time.time()
    for st in L["stages"]:
        build_stage(st["name"])
        w = (PARAMS / st["name"] / "param_blk0.bin").read_bytes()
        t = time.time()
        sim = cvxsim.simulate(stage_cvx(st["name"]), KERNELS_RS, L["consts"], {"in_x": data, "w": w}, verbose=verbose)
        reports.append(f"{st['name']}：{sim.report()}（{time.time() - t:.1f} s）")
        if verbose:
            print(reports[-1], flush=True)
        data = sim.mem["out_y"].tobytes()  # 上一段的 out_y 原样当下一段的 in_x（DESIGN D15）
        del sim
    img = vae.unpack_image(L, data)
    t_sim = time.time() - t_all
    r16, r32, times = _refs(lat)
    m = {"latent": kind, "stages": reports, "sim_s": round(t_sim, 1), **times,
         "sim_vs_ref_bf16": vae_ref.metrics(img, r16), "sim_vs_ref_fp32": vae_ref.metrics(img, r32),
         "ref_bf16_vs_ref_fp32": vae_ref.metrics(r16, r32)}
    if kind == "image":
        src = vae_ref.test_image()
        m["ref_fp32_vs_source"] = vae_ref.metrics(r32, src)
        m["sim_vs_source"] = vae_ref.metrics(img, src)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, x in [("sim", img), ("ref_bf16", r16), ("ref_fp32", r32)] + ([("source", vae_ref.test_image())] if kind == "image" else []):
        vae_ref.to_png(x, out_dir / f"{name}.png")
    (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    return m


def _fmt(m: dict) -> str:
    a, b = m["sim_vs_ref_bf16"], m["sim_vs_ref_fp32"]
    return (f"{m['latent']}：sim 对 bf16 参照 余弦 {a['cos']:.6f} 最大差 {a['max_abs']:.4f} PSNR {a['psnr_db']:.1f} dB；"
            f"对 fp32 余弦 {b['cos']:.6f} PSNR {b['psnr_db']:.1f} dB；sim {m['sim_s']} s")


def cmd_sim(a):
    m = run_sim(a.latent, ROOT / "out" / "vae-sim" / a.latent.replace(":", "-"), verbose=a.verbose)
    print("\n".join(m["stages"]))
    print(_fmt(m))


def cmd_check(a):
    bad = []
    for kind in ["image", "randn:0"]:
        m = run_sim(kind, ROOT / "out" / "vae-sim" / kind.replace(":", "-"))
        print(_fmt(m))
        s16, s32 = m["sim_vs_ref_bf16"], m["sim_vs_ref_fp32"]
        if not s16["finite"]:
            bad.append(f"{kind}：sim 输出有 NaN / Inf")
        if s16["cos"] < TOL["cos_bf16"]:
            bad.append(f"{kind}：对 bf16 参照的余弦 {s16['cos']:.6f} < {TOL['cos_bf16']}")
        if s16["psnr_db"] < TOL["psnr_bf16_db"]:
            bad.append(f"{kind}：对 bf16 参照的 PSNR {s16['psnr_db']:.1f} dB < {TOL['psnr_bf16_db']}")
        if s32["psnr_db"] < TOL["psnr_fp32_db"]:
            bad.append(f"{kind}：对 fp32 的 PSNR {s32['psnr_db']:.1f} dB < {TOL['psnr_fp32_db']}")
    if bad:
        print("✗ " + "\n✗ ".join(bad))
        sys.exit(1)
    print("✓ 正式 VAE 6 段程序的模拟：接线检查全过，对两份参照都在容差内")


# 上板门槛。首轮按 35 / 30 dB 收；2026-11-14 npu3 真机实测（latent = image）：3 遍逐字节一致，对 bf16 参照余弦 0.999998、PSNR 58.3 dB、
# 对 fp32 56.9 dB，比模拟（59.8 / 57.1）低不到 1.5 dB ⇒ 收紧。门槛对 --latent 的每一组都用，而板上只测过 image：按模拟里最差的 randn:0
# （54.2 / 53.0 dB、余弦 0.999993）再减板上多掉的 1.5 dB、留约 2 dB ⇒ 50 / 48 dB；余弦 1 − cos 留一倍多 ⇒ 0.99998。
BOARD_TOL = {"cos_bf16": 0.99998, "psnr_bf16_db": 50.0, "psnr_fp32_db": 48.0}


def cmd_board(a):
    """🚨 上板：重出 calbin ⇒ 每段只部署一次（计时）⇒ 整条（6 段）跑 N 遍（计时、逐字节比是否一致）⇒ 对两份参照 ⇒ 过门槛退出 0，否则 1。"""
    from sd15npu import dort

    if not dort.allowed():
        sys.exit("没开上板开关 SD15NPU_ALLOW_BOARD=1：负责人确认这块卡可以用才设（AGENTS.md 硬规矩 3）")
    have = all((CALBINS / st["name"] / "cal" / "param_blk0.bin").exists() for st in vae.layout()["stages"])
    if not (a.no_build and have) and not build_calbins():
        sys.exit("VAE 的 calbin 出不来（见上面 calbin.sh 的输出）")
    L = vae.layout()
    label, suffix = dort.measured_label(), dort.result_suffix()
    out = ROOT / "out" / f"vae-board{suffix}" / a.latent.replace(":", "-")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "dort.log"
    lat = vae_ref.make_latent(vae_ref.load_vae(vae_dir()), a.latent)
    deploy = {st["name"]: round(dort.deploy_only(CALBINS / st["name"] / "cal", "vae", log=log)["deploy_wall_s"], 3) for st in L["stages"]}
    outs, walls = [], []
    for _ in range(a.runs):
        data, per = vae.pack_latent(L, lat), {}
        for st in L["stages"]:
            data, info = dort.run_raw(CALBINS / st["name"] / "cal", "vae", data, "in_x", "out_y", log=log)
            per[st["name"]] = round(info["dort_wall_s"], 3)
        outs.append(data)
        walls.append(per)
    same = all(o == outs[0] for o in outs[1:])
    img = vae.unpack_image(L, outs[0])
    r16, r32, _ = _refs(lat)
    m16, m32 = vae_ref.metrics(img, r16), vae_ref.metrics(img, r32)
    tot = sorted(sum(w.values()) for w in walls)
    med = tot[len(tot) // 2]
    ok = bool(same and m16["finite"] and m16["cos"] >= BOARD_TOL["cos_bf16"] and m16["psnr_db"] >= BOARD_TOL["psnr_bf16_db"]
              and m32["psnr_db"] >= BOARD_TOL["psnr_fp32_db"])
    m = {"latent": a.latent, "measured": label, "runs": a.runs, "deploy_only_wall_s": deploy, "run_wall_s": walls,
         "total_median_s": round(med, 3), "inference_est_s": round(med - sum(deploy.values()), 3),
         "inference_note": "6 段整次（各自部署 + 推理 + 读写）之和的中位数减各段「只部署」之和，是估算；段间的特征图经主机转手，也算在里面",
         "deterministic": same, "board_vs_ref_bf16": m16, "board_vs_ref_fp32": m32, "tolerance": BOARD_TOL, "ok": ok}
    vae_ref.to_png(img, out / "board.png")
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(f"{'✓' if ok else '✗'} 正式 VAE 上板（{label}）：{a.runs} 次{'逐字节一致' if same else '⚠ 结果不一致（多半是少屏障）'}；"
          f"对 bf16 参照余弦 {m16['cos']:.6f}、PSNR {m16['psnr_db']:.1f} dB，对 fp32 PSNR {m32['psnr_db']:.1f} dB；"
          f"6 段整次中位 {med:.2f} s（只部署合计 {sum(deploy.values()):.2f} s）；{out}")
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
    p.add_argument("--runs", type=int, default=3, help="整条跑几遍（查确定性、取时间的中位数）")
    p.add_argument("--no-build", action="store_true", help="已有 6 段 calbin 就不重出（刚出过时；docc 约 40 s）")
    p.set_defaults(f=cmd_board)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
