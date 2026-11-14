"""dort 的测试替身（⛔ 不是真卡，结果一律标「测试替身」）：没有卡时把整套上板流程（tools/board_check.py）空跑一遍。

它查的是**流程本身**：命令行写得对不对（照真 dort 的 --help：`-i` `-j` 全局选项、`calbin run` 的 -f / -m / --deploy-only / --raw-in / --raw-out /
--in-tensor / --out-tensor / --board-lock …，多一个不认得的选项就报错）、张量名与字节数是否和 calbin 的清单对得上、门槛与停下来的路、结果怎么记。
「跑」的时候：先确认 calbin 里的程序（暂存目录的 step.cvxo）和本地编出来的 `.cvxo` 是同一次构建，再用 host/sd15npu/cvxsim.py 算出输出。
它不能说明 kernel 在真卡上算得对（那要真卡）。

环境变量（只给 ref/test_board_kit.py 用）：
- FAKE_DORT_SCENARIO：`dort device` 的样子：healthy（缺省）/ refused（板上不是 DeviceRT，dort 被拒）/ busy（别的进程在用）/ rt_down
- FAKE_DORT_FAULT：corrupt（输出改坏）/ nondet（同一个模型第二次起结果差一位）/ fail_run（calbin run 报错退出）
- FAKE_DORT_LOG：每次调用的参数追加进这个文件（JSON 一行）；FAKE_DORT_STATE：nondet 计次用的目录
用法：做一个可执行的包装（`exec <venv>/bin/python -I ref/fake_dort.py "$@"`），把 SD15NPU_DORT 指向它。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))


def _status(scenario: str) -> str:
    me = os.getpid()
    dev = "OS up   RT DOWN" if scenario == "rt_down" else "OS up   RT up  "
    drt = ["|     DRT  : DeviceRT（测试替身）                                       |"]
    if scenario == "refused":
        drt = ["|     DRT  : 厂商运行时，不是 DeviceRT（测试替身）                      |",
               "|            本 dortcli：拒（dort 开设备回 31）                          |"]
    procs = [f"|     0  {me:>8}  {os.getppid():>8}  {os.getuid():>6}  dort              yes     |"]
    if scenario == "busy":
        procs.insert(0, "|     0    424242    424240     949  other-service     yes     |")
    return "\n".join([
        "+------------------------------------------------------------------------------+",
        "| CALC-SMI（测试替身 fake_dort，不是真卡）                     Devices: 1    |",
        "+------------------------------------------------------------------------------+",
        "| [0] /dev/calculet0（测试替身）                                               |",
        "|     State: active     Health: good (0)     Mode: irq                         |",
        f"|     Dev  : {dev} chips: 2                                          |",
        *drt,
        "+------------------------------------------------------------------------------+",
        "| Processes:                                                                   |",
        "|   DEV       PID      PPID     UID  COMMAND           ACTIVE                  |",
        *procs,
        "+------------------------------------------------------------------------------+",
    ])


def _parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="dort")
    top.add_argument("-j", "--json", action="store_true")
    top.add_argument("-i", "--index")
    sub = top.add_subparsers(dest="cmd", required=True)
    dev = sub.add_parser("device")
    dev.add_argument("-j", "--json", dest="json_sub", action="store_true")
    dev.add_argument("-i", "--index", dest="index_sub")
    dev.add_argument("action", nargs="?")
    cal = sub.add_parser("calbin")
    cs = cal.add_subparsers(dest="ccmd", required=True)
    run = cs.add_parser("run")
    run.add_argument("-f", "--calbin", required=True)
    run.add_argument("-m", "--model")
    run.add_argument("--deploy-only", action="store_true")
    run.add_argument("--dump-layout", action="store_true")
    run.add_argument("--raw-in")
    run.add_argument("--raw-out", default="/tmp/raw_out.bin")
    run.add_argument("--in-tensor", default="in")
    run.add_argument("--out-tensor", default="out")
    run.add_argument("--board-lock")
    run.add_argument("--lock-wait")
    run.add_argument("-j", "--json", dest="json_sub", action="store_true")
    run.add_argument("-i", "--index", dest="index_sub")
    return top


def _die(code: int, msg: str):
    print(f"[fake_dort] {msg}", file=sys.stderr)
    sys.exit(code)


def _run(a):
    from sd15npu import bf16, cvxsim, layout

    cal = Path(a.calbin)
    man = cal / (a.model or "") / "submodel_memory_reserved_info.txt"
    if not a.model or not man.exists():
        _die(3, f"calbin {cal} 里没有模型 {a.model}")
    bufs = {m.group(2): (m.group(1), int(m.group(3)))
            for m in re.finditer(r"^(ibuf|obuf)\s+(\S+)\s+shape.*?size\s+(\d+)\s*$", man.read_text(), re.M)}
    if a.deploy_only:
        print(json.dumps({"fake": True, "deploy_only": True, "model": a.model}) if a.json else "[fake_dort] 只部署（测试替身）")
        return
    if os.environ.get("FAKE_DORT_FAULT") == "fail_run":
        _die(1, "注入的故障：calbin run 报错")
    if not a.raw_in:
        _die(4, "测试替身只做裸字节模式（--raw-in / --raw-out）")
    data = Path(a.raw_in).read_bytes()
    if bufs.get(a.in_tensor, ("", 0))[0] != "ibuf" or bufs.get(a.out_tensor, ("", 0))[0] != "obuf":
        _die(5, f"张量名对不上：清单里是 {bufs}，给的是 --in-tensor {a.in_tensor} --out-tensor {a.out_tensor}")
    if len(data) != bufs[a.in_tensor][1]:
        _die(6, f"--raw-in 是 {len(data)} 字节，张量 {a.in_tensor} 是 {bufs[a.in_tensor][1]} 字节")
    staged = (cal.parent / a.model / "step.cvxo").read_bytes()
    cands = [ROOT / "target/dopamine-unknown-none-cvx/npu" / f"{a.model}.cvxo",
             *ROOT.glob(f"target/features/*/dopamine-unknown-none-cvx/npu/{a.model}.cvxo")]
    match = next((c for c in cands if c.exists() and c.read_bytes() == staged), None)
    if match is None:
        _die(7, "找不到和 calbin 同一次构建的 .cvxo（测试替身没法知道它算什么）")
    text = match.with_suffix(".cvx").read_text()
    krs = "\n".join(p.read_text() for p in [ROOT / "crates/taesd/src/kernels.rs", ROOT / "crates/ktest/src/extra.rs", ROOT / "crates/vae/src/kernels.rs"])
    decls = cvxsim.parse_cvx(text).decls
    inputs = {a.in_tensor: data}
    if "w" in decls:
        inputs["w"] = (cal / "param_blk0.bin").read_bytes()
    sim = cvxsim.Sim(text, krs, layout.load("taesd")["consts"], inputs).run()
    out = bytearray(sim.mem[a.out_tensor].tobytes())
    fault = os.environ.get("FAKE_DORT_FAULT")
    u = memoryview(out).cast("H")
    if fault == "corrupt":
        for i in range(0, len(u), 97):
            u[i] ^= 0x4000
    elif fault == "nondet":
        st = Path(os.environ.get("FAKE_DORT_STATE", "/tmp")) / f"nondet-{a.model}"
        n = int(st.read_text()) if st.exists() else 0
        st.write_text(str(n + 1))
        if n >= 1:
            i = len(u) // 2
            u[i] ^= 0x0001
    Path(a.raw_out).write_bytes(bytes(out))
    _ = bf16  # 留着：以后要按 bf16 解读输出时用
    print(json.dumps({"fake": True, "model": a.model, "out_bytes": len(out)}) if a.json else f"[fake_dort] 跑完（测试替身）：{len(out)} 字节")


def main():
    if os.environ.get("FAKE_DORT_LOG"):
        with open(os.environ["FAKE_DORT_LOG"], "a") as f:
            f.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\n")
    a = _parser().parse_args()
    if a.cmd == "device":
        if a.action == "reprobe":
            _die(99, "⛔ device reprobe：上板流程绝不该调到复位")
        if a.action:
            _die(2, f"认不得的 device 子命令 {a.action}")
        print(_status(os.environ.get("FAKE_DORT_SCENARIO", "healthy")))
    elif a.ccmd == "run":
        _run(a)


if __name__ == "__main__":
    main()
