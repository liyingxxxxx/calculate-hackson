"""读 Rust 的偏移表：`cargo run -q --release -p layout-dump -- <程序>` 出 JSON，缓存在 `out/layout/<程序>.json`。

偏移只有 Rust（`crates/<程序>/src/layout.rs`）这一份：设备程序用它，主机打包、模拟都经这里读，不在 Python 里另算。
缓存比 layout.rs 旧就重出（要 `source tools/env.sh` 过的环境：cargo 走隔离的工具链）。
"""
import json
import os
import subprocess

from . import ROOT


def load(program: str = "taesd") -> dict:
    src = ROOT / "crates" / program / "src" / "layout.rs"
    cache = ROOT / "out" / "layout" / f"{program}.json"
    if not cache.exists() or cache.stat().st_mtime < src.stat().st_mtime:
        out = subprocess.run(["cargo", "run", "-q", "--release", "-p", "layout-dump", "--", program],
                             cwd=ROOT, check=True, capture_output=True, text=True, env=os.environ).stdout
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(out)
    return json.loads(cache.read_text())
