"""真卡上一次跑完（🚨 要 SD15NPU_ALLOW_BOARD=1：负责人确认这块卡可以用才设）。任何一步不对就停，退出 1：

  0. 重出全部 calbin（TAESD 带真权重、单 kernel 测试台）—— 参数块和程序是同一次构建的产物
  1. 上板前静态核对（tools/preflight.py，⛔ 不开设备）
  2. 卡的状态（只读 `dort device`）：OS up / RT up、dort 没被板上运行时拒（要 DeviceRT）、没有别的进程在用
  3. 逐个 kernel（从小到大：up_256 ⇒ tanh ⇒ temb ⇒ conv_in ⇒ convr_64 ⇒ conv_out ⇒ convr_512），每个跑全部用例
  4. 整个 TAESD 解码器：只部署一次 + 整次 N 遍（确定性、时间）、对两份参照
  4b.（--vae）正式 VAE 解码器：6 段程序依次跑（tools/vae.py board），每段只部署一次 + 整条 N 遍、对两份参照
  5. （--comfy）ComfyUI 原生工作流，解码节点走 npu（加 --vae 时再跑一条正式 VAE 解码的工作流）

用法：SD15NPU_ALLOW_BOARD=1 [SD15NPU_DORT_INDEX=0] [SD15NPU_BOARD_LOCK=…] $CALCULET_HOME/venv/bin/python -I tools/board_check.py [--tests a,b] [--runs 3] [--no-build] [--vae] [--comfy]
结果：out/board[-dryrun]/summary-<时间>.json（每一步的结果与用时）；各步自己的结果在 out/ktest/<测试>/<用例>/board*.json、out/taesd-board*/。
挂死就停、找工程师：⛔ kill 在飞的 dort、⛔ 复位、⛔ 刷写（AGENTS.md 硬规矩 3）。

没有卡时可以用测试替身把整套空跑一遍（`ref/test_board_kit.py`，设 SD15NPU_DORT 指向 `ref/fake_dort.py`）：结果另存 *-dryrun，标「测试替身」。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "host"), str(ROOT / "ref")]

from sd15npu import dort  # noqa: E402

ORDER = ["up_256", "tanh", "temb", "conv_in", "convr_64", "conv_out", "convr_512"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default=",".join(ORDER), help="逐个 kernel 那一步跑哪些（逗号分隔，按给的次序）；空串 = 跳过")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-build", action="store_true", help="跳过第 0 步（刚出过 calbin 时）")
    ap.add_argument("--no-taesd", action="store_true")
    ap.add_argument("--comfy", action="store_true")
    ap.add_argument("--vae", action="store_true", help="再上正式 VAE 的 6 段程序（B1b）")
    a = ap.parse_args()

    if not dort.allowed():
        print("没开上板开关 SD15NPU_ALLOW_BOARD=1：负责人确认这块卡可以用才设（AGENTS.md 硬规矩 3）。什么都没做。")
        sys.exit(2)
    py = sys.executable
    label, suffix = dort.measured_label(), dort.result_suffix()
    out = ROOT / "out" / f"board{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summary = {"measured": label, "started": stamp, "dort_index": os.environ.get("SD15NPU_DORT_INDEX", "0"), "steps": []}

    def step(name, fn):
        t = time.time()
        print(f"━━ {name}", flush=True)
        ok, detail = fn()
        rec = {"step": name, "ok": ok, "seconds": round(time.time() - t, 1), "detail": detail}
        summary["steps"].append(rec)
        (out / f"summary-{stamp}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        if not ok:
            print(f"✗ 停在「{name}」：{detail}\n  挂死就停、找工程师：⛔ kill 在飞的 dort、⛔ 复位、⛔ 刷写。记录在 {out}/summary-{stamp}.json")
            sys.exit(1)
        print(f"✓ {name}（{rec['seconds']} s）", flush=True)

    def sh(*cmd):
        r = subprocess.run(list(cmd), cwd=ROOT)
        return r.returncode == 0, f"退出 {r.returncode}：{' '.join(map(str, cmd[1:]))}"

    if not a.no_build:
        step("0. 重出 TAESD 的 calbin（带真权重）", lambda: sh(py, "-I", "tools/taesd.py", "calbin"))
        step("0. 重出单 kernel 测试台的 calbin（含模拟自检）", lambda: sh(py, "-I", "ref/ktest.py", "all"))
    step("1. 上板前静态核对（不开设备）", lambda: sh(py, "-I", "tools/preflight.py"))

    def status():
        ok, why, _ = dort.device_status(log=out / f"device-{stamp}.log")
        return ok, "；".join(why) if why else "OS up / RT up、dort 没被拒、没有别的进程"
    step("2. 卡的状态（只读）", status)

    for t in [x for x in a.tests.split(",") if x]:
        step(f"3. 单 kernel 上板：{t}", lambda t=t: sh(py, "-I", "ref/ktest.py", "board", t, "--case", "all"))
    if not a.no_taesd:
        step(f"4. TAESD 整个解码器上板（{a.runs} 遍）", lambda: sh(py, "-I", "tools/taesd.py", "board", "--runs", str(a.runs)))
    if a.vae:
        vargs = ["--runs", str(a.runs)] + (["--no-build"] if a.no_build else [])
        step(f"4b. 正式 VAE 6 段上板（{a.runs} 遍）", lambda: sh(py, "-I", "tools/vae.py", "board", *vargs))
    if a.comfy:
        cpy = Path(os.environ.get("CALCULET_HOME", ROOT.parent)) / "comfy-venv" / "bin" / "python"
        step("5. ComfyUI 原生工作流（解码走 npu）", lambda: sh(str(cpy), "-I", "comfy/run_workflow.py", "--backend", "npu", "--steps", "20"))
        if a.vae:
            step("5b. ComfyUI 原生工作流（正式 VAE 解码走 npu）", lambda: sh(str(cpy), "-I", "comfy/run_workflow.py", "comfy/workflows/sd15_vae_npu.api.json",
                                                                  "--backend", "npu", "--steps", "20"))
    print(f"全部通过（{label}）。记录：{out}/summary-{stamp}.json")


if __name__ == "__main__":
    main()
