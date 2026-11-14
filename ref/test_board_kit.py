"""上板流程（tools/board_check.py）自己的测试：拿测试替身 ref/fake_dort.py 当 dort，把整套空跑一遍，再把每条该停下来的路走一遍。

为什么要它：真卡难排、一次机会贵；流程本身的错（命令行写错、张量名对不上、门槛没生效、该停没停、结果没记）不该等到卡上才发现。
它不说明 kernel 在真卡上算得对（那要真卡）。结果都写进 *-dryrun，标「测试替身」，不会和真卡的结果混。

| 情形 | 期望 |
| --- | --- |
| 没开上板开关 | 退出 2，一次 dort 都没调 |
| 板上不是 DeviceRT（dort 被拒）/ 有别的进程在用 / RT 没起来 | 停在「卡的状态」，退出 1，只调过一次只读的 `dort device` |
| 一切正常（全套：重出 calbin、静态核对、7 个单 kernel 全部用例、TAESD 2 遍） | 退出 0；结果标「测试替身」；从没调过 `device reprobe` |
| 输出被改坏 | 停在第一个单 kernel 测试，退出 1 |
| 两次结果差一位 | 停在 TAESD 那一步（确定性），退出 1 |
| dort 报错 | 停在那一步，退出 1 |
| 正式 VAE（`--vae`，6 段） | 退出 0；6 次只部署、6 次裸字节运行，`-m vae`，上一段的输出接下一段的输入；结果标「测试替身」 |

用法：$CALCULET_HOME/venv/bin/python -I ref/test_board_kit.py（先 source env.sh；tools/verify_all.sh 会跑它，约 3 分钟）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(env_extra: dict, args: list[str], tmp: Path) -> tuple[int, list, str]:
    calls = tmp / "calls.jsonl"
    calls.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SD15NPU_", "FAKE_DORT_"))}
    env.update({"SD15NPU_DORT": str(tmp / "dort"), "FAKE_DORT_LOG": str(calls), "FAKE_DORT_STATE": str(tmp)})
    env.update(env_extra)
    for f in tmp.glob("nondet-*"):
        f.unlink()
    r = subprocess.run([sys.executable, "-I", "tools/board_check.py", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    got = [json.loads(l) for l in calls.read_text().splitlines()] if calls.exists() else []
    return r.returncode, got, r.stdout + r.stderr


def expect(name: str, cond: bool, out: str):
    if not cond:
        print(out[-3000:])
        raise SystemExit(f"✗ {name}")
    print(f"✓ {name}")


def main():
    with tempfile.TemporaryDirectory(prefix="boardkit-") as td:
        tmp = Path(td)
        w = tmp / "dort"
        w.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{ROOT / "ref/fake_dort.py"}" "$@"\n')
        w.chmod(0o755)
        allow = {"SD15NPU_ALLOW_BOARD": "1"}

        rc, calls, out = run({}, ["--no-build"], tmp)
        expect("没开上板开关：退出 2、一次 dort 都没调", rc == 2 and not calls, out)

        for sc, word in [("refused", "拒"), ("busy", "别的进程"), ("rt_down", "OS / RT")]:
            rc, calls, out = run({**allow, "FAKE_DORT_SCENARIO": sc}, ["--no-build"], tmp)
            only_status = len(calls) == 1 and calls[0][0] == "device"
            expect(f"卡的状态 {sc}：停在第 2 步、只调过一次只读的 dort device", rc == 1 and only_status and word in out and "停在「2. 卡的状态" in out, out)

        rc, calls, out = run(allow, ["--runs", "2"], tmp)
        flat = [x for c in calls for x in c]
        runs = [c for c in calls if "--raw-in" in c]
        deploys = [c for c in calls if "--deploy-only" in c]
        expect(f"全套空跑：退出 0、{len(runs)} 次裸字节运行、{len(deploys)} 次只部署、从没调过 reprobe", rc == 0 and "reprobe" not in flat and runs and deploys, out)
        m = json.loads((ROOT / "out/taesd-board-dryrun/image/metrics.json").read_text())
        expect("TAESD 空跑的结果标了「测试替身」、确定、过门槛", "测试替身" in m["measured"] and m["deterministic"] and m["ok"], json.dumps(m, ensure_ascii=False))
        kj = json.loads((ROOT / "out/ktest/conv_in/random/board-dryrun.json").read_text())
        expect("单 kernel 空跑的结果标了「测试替身」", "测试替身" in kj["measured"] and kj["ok"], json.dumps(kj, ensure_ascii=False))
        for c in runs:
            i = c.index("-i")
            assert c[i + 1] == "0" and c[c.index("--in-tensor") + 1] == "in_x" and c[c.index("--out-tensor") + 1] == "out_y", c
        expect("每次运行的命令行：-i 0、in_x / out_y（真 dort 的选项名，替身照 --help 严格解析）", True, "")

        rc, calls, out = run({**allow, "FAKE_DORT_FAULT": "corrupt"}, ["--no-build", "--tests", "conv_in", "--no-taesd"], tmp)
        expect("输出被改坏：停在单 kernel conv_in，退出 1", rc == 1 and "停在「3. 单 kernel 上板：conv_in" in out, out)

        rc, calls, out = run({**allow, "FAKE_DORT_FAULT": "nondet"}, ["--no-build", "--tests", "", "--runs", "2"], tmp)
        m = json.loads((ROOT / "out/taesd-board-dryrun/image/metrics.json").read_text())
        expect("两次结果差一位：停在 TAESD、记成不确定，退出 1", rc == 1 and "停在「4. TAESD" in out and not m["deterministic"], out)

        rc, calls, out = run({**allow, "FAKE_DORT_FAULT": "fail_run"}, ["--no-build", "--tests", "up_256", "--no-taesd"], tmp)
        expect("dort 报错：停在那一步，退出 1", rc == 1 and "停在「3. 单 kernel 上板：up_256" in out, out)

        rc, calls, out = run(allow, ["--no-build", "--tests", "", "--no-taesd", "--vae", "--runs", "1"], tmp)
        vruns = [c for c in calls if "--raw-in" in c]
        vdep = [c for c in calls if "--deploy-only" in c]
        m = json.loads((ROOT / "out/vae-board-dryrun/image/metrics.json").read_text()) if rc == 0 else {}
        expect(f"正式 VAE 空跑：退出 0、{len(vdep)} 次只部署、{len(vruns)} 次裸字节运行（6 段）、-m vae、结果标「测试替身」并过门槛",
               rc == 0 and len(vruns) == 6 and len(vdep) == 6 and all(c[c.index("-m") + 1] == "vae" for c in vruns)
               and "测试替身" in m.get("measured", "") and m.get("ok"), out)
    print("上板流程的自测全过（测试替身空跑；真卡上算得对不对要真卡）")


if __name__ == "__main__":
    main()
