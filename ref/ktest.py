"""单 kernel 测试台的主机侧（配 `crates/ktest`）：出输入、按 torch 语义另算期望、模拟器自检、出 calbin、（获准后）上板比对。

用法（`$CALCULET_HOME/venv/bin/python -I ref/ktest.py …`，先 source env.sh）：
  gen   <测试> [--case C] [--seed S]   ⇒ out/ktest/<测试>/<C>/{in.bin, param_blk0.bin, expect.npy, meta.json}
  sim   <测试> [--case C]              模拟器跑同一份程序、对期望（查测试台自己：偏移、打包、期望写对没有）
  calbin <测试> [--case C]             tools/calbin.sh ktest --features <测试> --params <那个目录> ⇒ …/cal，doctor 过
  board <测试> [--case C|all]          🚨 要 SD15NPU_ALLOW_BOARD=1：每个用例重出 calbin、dort 跑、读回、对期望；缺省跑全部用例，
                                       必过的用例有一个不过就停、退出 1（门槛见 BOARD_TOL）
  all                                  全部测试 × 全部用例：gen + sim + calbin（verify_all 用；不上板）

测试（kernel 都是 TAESD 整步程序里那一份）与用例（边界）：

| 测试 | kernel | 查什么 | 用例 |
| --- | --- | --- | --- |
| conv_in | td_conv_in（64²，32→64，relu） | 本仓加的 relu 分支、Cin = 32 | random、zeros（= relu(bias)）、neg（bias 很负，relu 全砍）、big（输入 ±30） |
| convr_64 | td_convr_64（64²，res_relu） | 本仓加的 res_relu（残差之后再 relu） | random、zeros、neg、big |
| conv_out | td_conv_out（512²，64→32） | 512 宽的快路（每行 8 组 × 4 块交错）、Cout = 32 | random |
| convr_512 | td_convr_512（512²，res_relu） | 512 宽 + 残差 | random |
| tanh | td_tanh | VU 的 tanh_approx：板上标定离精确值多少 ULP | random、big（±1000，饱和到 ±3）、zeros、tiny（1e-30 量级） |
| up_256 | td_up_256（两核都调） | 「单核 kernel 按 num_cores() 分行」的假设：板上若只算一半的行就露馅 | random |
| temb | kt_temb（时间步嵌入，320 维） | 2π 规约 + cos_approx（sin 走 cos(a − π/2)）；期望是 diffusers 的 f32 | t999、t0、t1、frac（451.37）、small（0.25） |

期望怎么算（独立于 kernel 与模拟器）：卷积用 torch 的 F.conv2d（f32，权重与输入先舍 bf16）+ bias ⇒ 舍 bf16 ⇒ relu；
res_relu：舍(舍(conv) + r) ⇒ relu；tanh：舍(tanh(x / 3) · 3)，torch 的 tanh 是精确的 ⇒ 板上的差就是近似指令的差，单独报 ULP 分布。
比的口径：只比 kernel 写的那部分（输出图的内部、kernel 的通道）；边框没人写，板上是什么都行，只报个数。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sd15npu import bf16, cvxsim, layout, temb  # noqa: E402

B = 2


def npu_dir(test):
    """每个测试自己的 target 目录（同 tools/calbin.sh）：cargo 切回编过的 feature 时不重编，共用目录里的 ktest.cvx 会是上一个测试的。"""
    return ROOT / "target" / "features" / f"ktest-{test}" / "dopamine-unknown-none-cvx" / "npu"

KRS = [ROOT / "crates" / "taesd" / "src" / "kernels.rs", ROOT / "crates" / "ktest" / "src" / "extra.rs"]

TESTS = {
    "conv_in": dict(kind="conv", kernel="td_conv_in", res=64, ci=32, co=64, act="relu", resid=False, cases=["random", "zeros", "neg", "big"]),
    "convr_64": dict(kind="conv", kernel="td_convr_64", res=64, ci=64, co=64, act="res_relu", resid=True, cases=["random", "zeros", "neg", "big"]),
    "conv_out": dict(kind="conv", kernel="td_conv_out", res=512, ci=64, co=32, act="none", resid=False, cases=["random"]),
    "convr_512": dict(kind="conv", kernel="td_convr_512", res=512, ci=64, co=64, act="res_relu", resid=True, cases=["random"]),
    "tanh": dict(kind="tanh", kernel="td_tanh", cases=["random", "big", "zeros", "tiny"]),
    "up_256": dict(kind="up", kernel="td_up_256", res=256, c=64, cases=["random"]),
    "temb": dict(kind="temb", kernel="kt_temb", half=160, cases=["t999", "t0", "t1", "frac", "small"]),
}
TEMB_T = {"t999": 999.0, "t0": 0.0, "t1": 1.0, "frac": 451.37, "small": 0.25}


def _a4k(n):
    return (n + 4095) // 4096 * 4096


def _img(res, p):
    return (res + 2 * B) ** 2 * p * 2


def _rb(a):
    """舍 bf16（torch 的 .to(bfloat16) 是 rne），回 f32 的 numpy。"""
    return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(torch.bfloat16).to(torch.float32).numpy()


def _decls(test):
    """编出来的 .cvx 里的声明（字节数以它为准，不另写一份）。"""
    build(test)
    return cvxsim.parse_cvx((npu_dir(test) / "ktest.cvx").read_text()).decls


def build(test):
    subprocess.run(["cargo", "build", "-q", "-p", "ktest", "--lib", "--target", "dopamine-unknown-none-cvx", "--features", test,
                    "--target-dir", str(ROOT / "target" / "features" / f"ktest-{test}")], cwd=ROOT, check=True)


def gen(test, case, seed=0):
    t = TESTS[test]
    rng = np.random.default_rng(seed)
    decls = _decls(test)
    out = {}
    if t["kind"] == "conv":
        res, ci, co = t["res"], t["ci"], t["co"]
        x = np.zeros((res + 2 * B, res + 2 * B, ci), np.float32)
        scale = {"random": 1.0, "zeros": 0.0, "neg": 1.0, "big": 30.0}[case]
        x[B:B + res, B:B + res] = _rb(rng.standard_normal((res, res, ci)) * scale)
        w = _rb(rng.standard_normal((3, 3, ci, co)) / np.sqrt(9 * ci))
        bias = (rng.standard_normal(co) * (0.5 if case != "neg" else 0.1) - (20.0 if case == "neg" else 0.0)).astype(np.float32)
        xin = bf16.bits(x).tobytes()
        rin = b""
        if t["resid"]:
            r = np.zeros((res + 2 * B, res + 2 * B, co), np.float32)
            r[B:B + res, B:B + res] = _rb(rng.standard_normal((res, res, co)) * max(scale, 1.0))
            rin = bf16.bits(r).tobytes()
        ro = _a4k(len(xin) + 128 * ci * 2)
        in_x = xin + bytes(ro - len(xin)) + (rin + bytes(_a4k(len(rin)) - len(rin)) if rin else b"")
        bo = _a4k(9 * ci * co * 2)
        wblob = bf16.bits(w).tobytes() + bytes(bo - 9 * ci * co * 2) + bias.astype("<f4").tobytes()
        wblob += bytes(decls["w"][1] - len(wblob))
        # 期望：torch 的 conv2d（NCHW），输入取带边框的整张图、padding = 0 ⇒ 正好是内部的 padding = 1
        xt = torch.from_numpy(x).permute(2, 0, 1)[None]
        wt = torch.from_numpy(w).permute(3, 2, 0, 1).contiguous()  # [ky][kx][ci][co] ⇒ [co][ci][ky][kx]
        y = F.conv2d(xt, wt, torch.from_numpy(bias))[0, :, 1:-1, 1:-1].permute(1, 2, 0).numpy()
        y = _rb(y)
        if t["act"] == "relu":
            y = np.maximum(y, 0)
        if t["resid"]:
            y = _rb(y + r[B:B + res, B:B + res])
        if t["act"] == "res_relu":
            y = np.maximum(y, 0)
        out.update(in_x=in_x, w=wblob, expect=y)
    elif t["kind"] == "tanh":
        n = decls["in_x"][1] // 2
        v = {"random": rng.standard_normal(n) * 2, "big": np.sign(rng.standard_normal(n)) * rng.uniform(5, 1000, n),
             "zeros": np.zeros(n), "tiny": rng.standard_normal(n) * 1e-30}[case]
        x = _rb(v)
        y = _rb(torch.tanh(torch.from_numpy(x) * np.float32(1 / 3)).numpy() * np.float32(3))
        out.update(in_x=bf16.bits(x).tobytes(), expect=y)
    elif t["kind"] == "temb":
        tv = TEMB_T[case]
        out.update(in_x=temb.t_row(tv) + bytes(decls["in_x"][1] - 32),
                   w=temb.tables(t["half"]) + bytes(decls["w"][1] - 2 * temb.n_padded(t["half"]) * 4),
                   expect=_rb(temb.reference(tv, t["half"])))
    elif t["kind"] == "up":
        res, c = t["res"], t["c"]
        x = np.zeros((res + 2 * B, res + 2 * B, c), np.float32)
        x[B:B + res, B:B + res] = _rb(rng.standard_normal((res, res, c)))
        xin = bf16.bits(x).tobytes()
        out.update(in_x=xin + bytes(decls["in_x"][1] - len(xin)), expect=x[B:B + res, B:B + res].repeat(2, 0).repeat(2, 1))
    if len(out["in_x"]) != decls["in_x"][1]:
        raise SystemExit(f"{test}：打的 in_x {len(out['in_x'])} 字节，程序声明 {decls['in_x'][1]}（测试台和 crates/ktest 对不上）")
    d = ROOT / "out" / "ktest" / test / case
    d.mkdir(parents=True, exist_ok=True)
    (d / "in.bin").write_bytes(out["in_x"])
    (d / "param_blk0.bin").write_bytes(out.get("w", b""))
    np.save(d / "expect.npy", out["expect"])
    (d / "meta.json").write_text(json.dumps({"test": test, "case": case, "seed": seed, "kernel": t["kernel"], "decls": decls}, ensure_ascii=False))
    return d, out


def compare(test, got_bytes, expect) -> dict:
    t = TESTS[test]
    y = bf16.from_bits(np.frombuffer(got_bytes, "<u2"))
    if t["kind"] in ("tanh", "temb"):
        got = y[:expect.size]
        border = 0
    else:
        res = 2 * t["res"] if t["kind"] == "up" else t["res"]
        ch = t["c"] if t["kind"] == "up" else t["co"]
        p = ch
        im = y[:(res + 2 * B) ** 2 * p].reshape(res + 2 * B, res + 2 * B, p)
        got = im[B:B + res, B:B + res, :ch]
        border = int(np.count_nonzero(np.nan_to_num(np.concatenate([im[:B].ravel(), im[-B:].ravel(), im[:, :B].ravel(), im[:, -B:].ravel()]), nan=1)))
    got, want = got.astype(np.float64).ravel(), expect.astype(np.float64).ravel()
    d = np.abs(got - want)
    finite = bool(np.isfinite(got).all())
    if not finite:
        cos = float("nan")
    elif np.linalg.norm(want) == 0 and np.linalg.norm(got) == 0:
        cos = 1.0  # 两边全零（zeros / neg 用例）
    else:
        cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-300))
    # 输出最大量级处的一个 bf16 ULP（7 位显式尾数）：「最大差 ≤ 2 ULP(max|want|)」是累加次序 + 残差相加再舍一次能差出的上界
    top = float(np.max(np.abs(want))) if want.size else 0.0
    ulp_top = 2.0 ** (np.floor(np.log2(top)) - 7) if top > 0 else 0.0
    ulp = np.abs(bf16.bits(got.astype(np.float32)).astype(np.int32) - bf16.bits(want.astype(np.float32)).astype(np.int32))
    return {"finite": finite, "bit_exact_pct": float(np.mean(got == want) * 100), "max_abs": float(np.nanmax(d)) if finite else float("nan"),
            "max_abs_in_top_ulp": float(float(np.nanmax(d)) / ulp_top if ulp_top else (0.0 if np.nanmax(d) == 0 else float("inf"))) if finite else float("nan"),
            "cos": cos, "ulp_max": int(ulp.max()), "ulp_le1_pct": float(np.mean(ulp <= 1) * 100), "border_nonzero": border}


def cmd_sim(test, case, quiet=False):
    d, out = gen(test, case)
    inputs = {"in_x": out["in_x"]}
    if "w" in out:
        inputs["w"] = out["w"]
    sim = cvxsim.Sim((npu_dir(test) / "ktest.cvx").read_text(), "\n".join(p.read_text() for p in KRS), layout.load("taesd")["consts"], inputs).run()
    m = compare(test, sim.mem["out_y"].tobytes(), out["expect"])
    # 模拟器与期望是同一个数学、两份实现：只差 f32 累加的次序 ⇒ 个别输出差 1–2 个 bf16 ULP；接近 0 的输出（相消）ULP 数会很大但绝对差极小，
    # 所以按「最大差 ≤ 2 个 ULP(输出最大的那个值)」判，逐元素 ULP 只报不判
    ok = m["finite"] and m["cos"] >= 0.99999 and m["max_abs_in_top_ulp"] <= 2
    if not quiet:
        print(f"{'✓' if ok else '✗'} sim {test}/{case}：{sim.report()}；逐位相同 {m['bit_exact_pct']:.2f}%，最大差 {m['max_abs_in_top_ulp']:.2f} ULP(max)，"
              f"最大 {m['ulp_max']} ULP，余弦 {m['cos']:.7f}")
    return ok, m


def cmd_calbin(test, case):
    d, _ = gen(test, case)
    r = subprocess.run([str(ROOT / "tools/calbin.sh"), "ktest", "--features", test, "--params", str(d), "-o", str(d / "calbin")], cwd=ROOT,
                       capture_output=True, text=True)
    ok = r.returncode == 0 and "⛔ 拒收 0 条" in (d / "calbin" / "doctor.txt").read_text()
    print(f"{'✓' if ok else '✗'} calbin {test}/{case}：{(d / 'calbin' / 'timing.txt').read_text().splitlines()[-1] if ok else r.stdout[-800:] + r.stderr[-800:]}")
    return ok


# 上板门槛（按 kernel 类型）。首轮是拍的（卷积 8 ULP、tanh / temb 余弦 0.999 与 4 ULP），2026-11-14 npu3 真机实测后按数收紧，
# 各留约一倍余量（实测明细在 PROGRESS「上板准备」与 out/ktest/<测试>/<用例>/board.json）：
# - 卷积：TU 的 f32 累加次序与 torch 不同。实测最差 convr_512 1.00 ULP(max)、余弦 ≥ 0.9999999 ⇒ 2 ULP、0.99999（与模拟的验收口径一致）。
# - tanh：VU 的 tanh_approx（f32）。实测最大差 2.4e-7（≈ 0.00003 ULP(max)）、最多 2 个 bf16 ULP ⇒ 1 ULP(max)、0.99999。
# - 时间步嵌入：cos_approx 的输出在板上**全部落在 1/128 的绝对网格上**（多半往 0 截），最大差 2.5/128（t999）、余弦最差 0.99996
#   ⇒ 3 ULP(max)、0.9999。这是指令本身的分辨率，不是接线错；换成多项式之前门槛不能再紧（DESIGN F19）。
# - 上采样是纯搬运 ⇒ 必须逐位相同（实测 100%）。
BOARD_TOL = {"conv": {"cos": 0.99999, "top_ulp": 2.0}, "tanh": {"cos": 0.99999, "top_ulp": 1.0},
             "temb": {"cos": 0.9999, "top_ulp": 3.0}, "up": {"bit_exact": 100.0}}
# 只记录、不拦的用例：tiny（1e-30 量级）只看近似指令在极小值上的行为，不影响 SD
BOARD_INFO_CASES = {"tiny"}


def board_ok(test, m) -> bool:
    tol = BOARD_TOL[TESTS[test]["kind"]]
    if not m["finite"]:
        return False
    if "bit_exact" in tol:
        return bool(m["bit_exact_pct"] >= tol["bit_exact"])
    return bool(m["cos"] >= tol["cos"] and m["max_abs_in_top_ulp"] <= tol["top_ulp"])


def cmd_board(test, cases) -> bool:
    """逐个用例上板：每次都重出输入、期望、参数块与 calbin（确定的、同一次构建），跑、读回、对期望。任何一个必过的用例不过就返回 False。"""
    from sd15npu import dort

    if not dort.allowed():
        raise SystemExit("没开上板开关 SD15NPU_ALLOW_BOARD=1：负责人确认这块卡可以用才设（AGENTS.md 硬规矩 3）")
    label, suffix = dort.measured_label(), dort.result_suffix()
    all_ok = True
    for case in cases:
        d = ROOT / "out" / "ktest" / test / case
        if not cmd_calbin(test, case):
            return False
        data, info = dort.run_raw(d / "calbin" / "cal", "ktest", (d / "in.bin").read_bytes(), "in_x", "out_y", log=d / f"dort{suffix}.log")
        (d / f"board_out{suffix}.bin").write_bytes(data)
        m = compare(test, data, np.load(d / "expect.npy"))
        ok = board_ok(test, m)
        m.update(measured=label, dort_wall_s=info["dort_wall_s"], tolerance=BOARD_TOL[TESTS[test]["kind"]], ok=ok,
                 blocking=case not in BOARD_INFO_CASES)
        (d / f"board{suffix}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
        print(f"{'✓' if ok else ('✗' if case not in BOARD_INFO_CASES else '·')} 上板 {test}/{case}（{label}）：逐位相同 {m['bit_exact_pct']:.2f}%，"
              f"最大差 {m['max_abs_in_top_ulp']:.2f} ULP(max)，最大 {m['ulp_max']} ULP，余弦 {m['cos']:.7f}，dort {info['dort_wall_s']:.2f} s")
        if not ok and case not in BOARD_INFO_CASES:
            all_ok = False
            break
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen", "sim", "calbin", "board", "all"])
    ap.add_argument("test", nargs="?", choices=list(TESTS))
    ap.add_argument("--case", default=None, help="用例；缺省：gen / sim / calbin 用第一个，board 跑全部（也可写 all）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "all":
        bad = 0
        for test, t in TESTS.items():
            for case in t["cases"]:
                ok, _ = cmd_sim(test, case)
                bad += not ok
            bad += not cmd_calbin(test, t["cases"][0])
        print("单 kernel 测试台：全部用例模拟通过、calbin 都出得来" if not bad else f"{bad} 项不过")
        sys.exit(1 if bad else 0)
    if not a.test:
        ap.error("要给测试名")
    if a.cmd == "board":
        cases = TESTS[a.test]["cases"] if a.case in (None, "all") else [a.case]
        sys.exit(0 if cmd_board(a.test, cases) else 1)
    a.case = a.case or TESTS[a.test]["cases"][0]
    if a.cmd == "gen":
        print(gen(a.test, a.case, a.seed)[0])
    elif a.cmd == "sim":
        sys.exit(0 if cmd_sim(a.test, a.case)[0] else 1)
    elif a.cmd == "calbin":
        sys.exit(0 if cmd_calbin(a.test, a.case) else 1)


if __name__ == "__main__":
    main()
