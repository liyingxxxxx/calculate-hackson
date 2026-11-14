"""上板前的静态核对（⛔ 不开设备）：保证拿到一块正常的真卡时，我们的包 dort 认、主机发的字节对得上。

对每个要上板的 calbin（TAESD 带真权重的那份、正式 VAE 的各段、单 kernel 测试台的每个测试）：
1. `dort calbin parse`（dort 自己说「纯静态，⛔ 不开设备」）能解析 —— 真跑时读包的就是它；
2. `docc calbin doctor` 拒收 0 条；
3. 子模型清单（`<子模型>/submodel_memory_reserved_info.txt`）里的输入 / 输出张量：名字是我们 dort 命令行里写的 in_x / out_y，
   字节数和主机侧打的输入（`--raw-in` 的文件）、解包时要读的输出一样；子模型名就是命令行里的 `-m`；
4. calbin 里的参数块和打包出来的那份逐字节相同（calBLAS AGENTS.md §3：参数块和程序是同一次构建的产物，要一起出、一起用）。

5. 正式 VAE：上一段的 out_y 和下一段的 in_x 字节数相同（主机原样转手，DESIGN D15）。

用法：$CALCULET_HOME/venv/bin/python -I tools/preflight.py（先 source env.sh；要先出过 calbin：tools/taesd.py calbin、tools/vae.py calbin、ref/ktest.py all）。
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "host"), str(ROOT / "ref")]

from sd15npu import layout  # noqa: E402


def manifest_tensors(cal: Path, model: str) -> dict:
    """子模型清单里的 ibuf / obuf：{名字: (方向, 字节)}。"""
    text = (cal / model / "submodel_memory_reserved_info.txt").read_text()
    out = {}
    for m in re.finditer(r"^(ibuf|obuf)\s+(\S+)\s+shape.*?size\s+(\d+)\s*$", text, re.M):
        out[m.group(2)] = (m.group(1), int(m.group(3)))
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(name: str, cal: Path, model: str, in_bytes: int, out_bytes: int, params: Path) -> list[str]:
    bad = []
    if not cal.is_dir():
        return [f"{name}：没有 calbin {cal}（先出包）"]
    r = subprocess.run(["dort", "calbin", "parse", "-f", str(cal)], capture_output=True, text=True)
    if r.returncode != 0:
        bad.append(f"{name}：dort calbin parse 失败：{(r.stdout + r.stderr)[-500:]}")
    elif not re.search(rf"^\s+{re.escape(model)}\s*$", r.stdout, re.M):
        bad.append(f"{name}：dort 解析出来的子模型里没有 {model}（命令行的 -m 对不上）")
    r = subprocess.run(["docc", "calbin", "doctor", str(cal)], capture_output=True, text=True)
    if "⛔ 拒收 0 条" not in r.stdout + r.stderr:
        bad.append(f"{name}：docc doctor 有拒收")
    if not (cal / model / "submodel_memory_reserved_info.txt").exists():
        return bad + [f"{name}：calbin 里没有子模型 {model} 的清单（命令行的 -m 对不上）"]
    t = manifest_tensors(cal, model)
    if t.get("in_x") != ("ibuf", in_bytes):
        bad.append(f"{name}：清单里的输入是 {t}，主机发 in_x {in_bytes} 字节")
    if t.get("out_y") != ("obuf", out_bytes):
        bad.append(f"{name}：清单里的输出是 {t}，主机读 out_y {out_bytes} 字节")
    if set(t) != {"in_x", "out_y"}:
        bad.append(f"{name}：裸字节模式只认一个输入、一个输出（F1），清单里有 {sorted(t)}")
    pin = cal / "param_blk0.bin"
    if params.exists() and params.stat().st_size and sha(pin) != sha(params):
        bad.append(f"{name}：calbin 里的 param_blk0.bin 和打包出来的 {params} 不是同一份（要一起出、一起用）")
    return bad


def main():
    import ktest

    bad, n = [], 0
    k = layout.load("taesd")["consts"]
    bad += check("taesd", ROOT / "out/taesd-real/cal", "taesd", k["IN_BYTES"], k["OUT_BYTES"], ROOT / "out/taesd/params/param_blk0.bin")
    n += 1
    stages = layout.load("vae")["stages"]
    for st in stages:
        bad += check(f"vae/{st['name']}", ROOT / "out/vae-real" / st["name"] / "cal", "vae", st["in_bytes"], st["out_bytes"],
                     ROOT / "out/vae/params" / st["name"] / "param_blk0.bin")
        n += 1
    for a, b in zip(stages, stages[1:]):
        if a["out_bytes"] != b["in_bytes"]:
            bad.append(f"vae：{a['name']} 的 out_y {a['out_bytes']} 字节，{b['name']} 的 in_x {b['in_bytes']} 字节（主机原样转手对不上）")
    for test, t in ktest.TESTS.items():
        case = t["cases"][0]
        d = ROOT / "out" / "ktest" / test / case
        cal = d / "calbin" / "cal"
        decls = ktest.cvxsim.parse_cvx((ktest.npu_dir(test) / "ktest.cvx").read_text()).decls if ktest.npu_dir(test).exists() else {}
        in_bytes = (d / "in.bin").stat().st_size if (d / "in.bin").exists() else -1
        bad += check(f"ktest/{test}", cal, "ktest", in_bytes, decls.get("out_y", (None, -1))[1], d / "param_blk0.bin")
        n += 1
    if bad:
        print("✗ " + "\n✗ ".join(bad))
        sys.exit(1)
    print(f"✓ 上板前静态核对：{n} 个 calbin 都被 dort 解析、doctor 拒收 0 条；张量名 / 字节数 / 子模型名与主机侧一致；参数块是同一份")


if __name__ == "__main__":
    main()
