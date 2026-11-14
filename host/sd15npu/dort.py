"""dort 的包装：卡的状态（只读）、只部署、裸字节模式跑一次（一个输入张量、一个输出张量，见 DESIGN F1 / D1）。

🚨 缺省**不碰板子**：环境变量 `SD15NPU_ALLOW_BOARD=1` 才真调 dort（AGENTS.md 硬规矩 3：负责人确认这块卡可以用才设）。
- 不设超时去杀 dort：⛔ SIGKILL 在飞的作业（挂死就停、找工程师）；等板锁由 dort 自己的 `--lock-wait` 管。
- `SD15NPU_DORT_INDEX`：设备下标（`-i`，缺省 0）；`SD15NPU_BOARD_LOCK`：板锁文件（`--board-lock`，按现场的排队规矩给；不给就用 dort 的缺省）。
- `SD15NPU_DORT`：dort 可执行文件（缺省 PATH 上的 `dort`）。**只给测试替身用**（`ref/fake_dort.py`：没有卡时把整套上板流程空跑一遍），
  真上板不设。
- 计时：每次调用的墙钟。部署与推理分开：`deploy_only` 量「只部署」，整次（部署 + 推理）减去它就是推理（DESIGN：编译、加载、预热、推理分开记）。

卡能不能用（`device_status`，只读，同 calBLAS AGENTS.md「上板之前先查板子健康」）：
1. 设备端 `OS up` 且 `RT up`；
2. dort 没说「拒」：板上跑的不是 DeviceRT（比如厂商运行时）时，这一版 dort 开设备会被拒（DESIGN F18）—— 要重刷 DeviceRT，那是管卡的人的事，⛔ 自己刷；
3. 除了这次查状态的 dort 自己，卡上没有别的进程在用（共享卡按排队来）。
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


class BoardNotAllowed(RuntimeError):
    pass


def allowed() -> bool:
    return os.environ.get("SD15NPU_ALLOW_BOARD") == "1"


def _gate():
    if not allowed():
        raise BoardNotAllowed("没开上板开关：负责人确认这块卡可以用，再设 SD15NPU_ALLOW_BOARD=1")


def measured_label() -> str:
    """结果文件里标的口径：设了测试替身就不是真卡（真机实测与仿真要分清：赛题规则）。"""
    return "测试替身空跑（fake_dort，不是真卡）" if os.environ.get("SD15NPU_DORT") else "真机实测"


def result_suffix() -> str:
    """测试替身的结果另存（文件名带 -dryrun），免得和真卡的结果混在一起。"""
    return "-dryrun" if os.environ.get("SD15NPU_DORT") else ""


def _dort() -> str:
    return os.environ.get("SD15NPU_DORT", "dort")


def _index() -> str:
    return os.environ.get("SD15NPU_DORT_INDEX", "0")


def _lock_args() -> list[str]:
    return ["--board-lock", os.environ["SD15NPU_BOARD_LOCK"]] if os.environ.get("SD15NPU_BOARD_LOCK") else []


def _run(cmd: list[str], log: Path | None) -> tuple[subprocess.CompletedProcess, float]:
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"$ {' '.join(cmd)}\n# 退出 {p.returncode}，墙钟 {wall:.3f} s\n{p.stdout}\n{p.stderr}\n")
    return p, wall


def device_status(log: Path | None = None) -> tuple[bool, list[str], str]:
    """（能用吗, 不能用的理由, dort 的原样输出）。只读：`dort device -i <下标>`，⛔ `device reprobe`。"""
    _gate()
    p = subprocess.Popen([_dort(), "device", "-i", _index()], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(f"$ {_dort()} device -i {_index()}\n{out}\n")
    why = []
    if p.returncode != 0:
        why.append(f"dort device 退出 {p.returncode}")
    if not re.search(r"OS up\s+RT up", out):
        dev = [l.strip(" |") for l in out.splitlines() if "Dev " in l]
        why.append(f"设备端 OS / RT 没起来：{dev[0] if dev else '（没读到 Dev 行）'}")
    if "拒" in out:
        why.append("这一版 dort 被板上运行时拒了（多半不是 DeviceRT）：找管卡的人重刷，⛔ 自己刷")
    others, in_procs = [], False
    for line in out.splitlines():
        if "Processes:" in line:
            in_procs = True
            continue
        if in_procs:
            tok = line.strip().strip("|").split()
            if len(tok) >= 6 and tok[0].isdigit() and tok[1].isdigit() and int(tok[1]) != p.pid:
                others.append(f"PID {tok[1]}（UID {tok[3]}，{tok[4]}）")
    if others:
        why.append("卡上有别的进程在用：" + "、".join(others) + "（共享卡按排队来）")
    return not why, why, out


def deploy_only(calbin: Path, model: str, log: Path | None = None) -> dict:
    _gate()
    cmd = [_dort(), "-i", _index(), "-j", "calbin", "run", "-f", str(calbin), "-m", model, "--deploy-only", *_lock_args()]
    p, wall = _run(cmd, log)
    if p.returncode != 0:
        raise RuntimeError(f"dort --deploy-only 退出 {p.returncode}：{(p.stdout + p.stderr).strip()[-2000:]}")
    return {"deploy_wall_s": wall, "dort_stdout": p.stdout}


def run_raw(calbin: Path, model: str, data: bytes, in_tensor: str, out_tensor: str, log: Path | None = None) -> tuple[bytes, dict]:
    _gate()
    with tempfile.TemporaryDirectory(prefix="sd15npu-") as td:
        fin, fout = Path(td) / "in.bin", Path(td) / "out.bin"
        fin.write_bytes(data)
        cmd = [_dort(), "-i", _index(), "-j", "calbin", "run", "-f", str(calbin), "-m", model, "--raw-in", str(fin), "--raw-out", str(fout),
               "--in-tensor", in_tensor, "--out-tensor", out_tensor, *_lock_args()]
        p, wall = _run(cmd, log)
        if p.returncode != 0:
            raise RuntimeError(f"dort 退出 {p.returncode}：{(p.stdout + p.stderr).strip()[-2000:]}")
        return fout.read_bytes(), {"dort_wall_s": wall, "dort_stdout": p.stdout}
