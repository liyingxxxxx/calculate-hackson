"""出交接材料：`<CALCULET_HOME>/交接提示词.md`（正文 docs/HANDOFF.md + 附录 A 赛题原文 + 附录 B 仓库文件全文）与
`<CALCULET_HOME>/handoff/sd15-npu-handoff-<日期>.tar.gz`（仓库 + env.sh + 交接提示词）。仓库本身在 GitHub（docs/HANDOFF.md 开头），tar 包给连不上 GitHub 的时候用。

用法：python3 -I tools/make_handoff.py（只用标准库）。改了仓库之后重跑；tar 包里不带 target/、out/、__pycache__（都能重出）。
附录 B 只收文本文件，跳过脚本生成的大 JSON（docs/inventory/*.json，`ref/op_inventory.py` 重出）。
"""
from __future__ import annotations

import os
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HACK = Path(os.environ.get("CALCULET_HOME", ROOT.parent))
SKIP_DIRS = {"target", "out", "__pycache__", ".git"}
LANG = {".rs": "rust", ".toml": "toml", ".sh": "bash", ".py": "python", ".md": "markdown", ".json": "json", ".lock": "text"}


def repo_files() -> list[Path]:
    # 有 git 就按 git 的视角列（尊重 .gitignore）；没有就自己走目录
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True, check=True).stdout
        files = [ROOT / l for l in out.splitlines() if l]
    except Exception:
        files = [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)]
    return sorted(p for p in files if p.is_file())


def appendix_b(files: list[Path]) -> str:
    parts = ["## 附录 B：仓库文件全文（拿不到 tar 包时照此重建；路径相对 `<CALCULET_HOME>/`）", "",
             "首选 `git clone https://github.com/liyingxxxxx/calculate-hackson.git sd15-npu`；连不上 GitHub 时照下面重建：",
             "`chmod +x sd15-npu/tools/*.sh sd15-npu/ref/*.sh sd15-npu/comfy/*.sh sd15-npu/test/*.sh`、`cd sd15-npu && git init -b main`，再跑 `bash tools/bootstrap.sh`。", ""]
    for p in files:
        rel = p.relative_to(ROOT)
        if rel.as_posix() == "docs/赛题原文.md":
            parts += [f"### `sd15-npu/{rel}`", "", "（见附录 A。）", ""]
            continue
        if rel.parts[:2] == ("docs", "inventory") and p.suffix == ".json":
            parts += [f"### `sd15-npu/{rel}`", "", "（脚本生成的大 JSON，不收：`ref/op_inventory.py` 重出。）", ""]
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            parts += [f"### `sd15-npu/{rel}`", "", "（二进制，不收）", ""]
            continue
        fence = "~~~~~" if "~~~~" in text else "~~~~"
        parts += [f"### `sd15-npu/{rel}`", "", f"{fence}{LANG.get(p.suffix, 'text')}", text.rstrip("\n"), fence, ""]
    return "\n".join(parts)


def main():
    files = repo_files()
    body = (ROOT / "docs" / "HANDOFF.md").read_text(encoding="utf-8")
    a = (ROOT / "docs" / "赛题原文.md").read_text(encoding="utf-8")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    doc = (f"{body.rstrip()}\n\n---\n\n> 本文由 `sd15-npu/tools/make_handoff.py` 于 {stamp} 生成。\n\n"
           f"## 附录 A：赛题原文（仓里的 docs/赛题原文.md，与赛题页一致）\n\n{a.rstrip()}\n\n---\n\n{appendix_b(files)}\n")
    out_doc = HACK / "交接提示词.md"
    out_doc.write_text(doc, encoding="utf-8")

    (HACK / "handoff").mkdir(exist_ok=True)
    tgz = HACK / "handoff" / f"sd15-npu-handoff-{time.strftime('%Y%m%d-%H%M')}.tar.gz"
    base = HACK.name
    with tarfile.open(tgz, "w:gz") as t:
        for p in files:
            t.add(p, arcname=f"{base}/sd15-npu/{p.relative_to(ROOT)}", recursive=False)
        if (ROOT / ".git").is_dir():
            t.add(ROOT / ".git", arcname=f"{base}/sd15-npu/.git")
        for extra in ["env.sh", "交接提示词.md"]:
            if (HACK / extra).exists():
                t.add(HACK / extra, arcname=f"{base}/{extra}")
    print(f"{out_doc}（{len(doc) // 1024} KB，附录 B 收 {len(files)} 个文件）")
    print(f"{tgz}（{tgz.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    main()
