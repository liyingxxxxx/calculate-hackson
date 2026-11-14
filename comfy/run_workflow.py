"""无界面跑一份 ComfyUI 工作流（API 格式 JSON）：起服务 ⇒ POST /prompt ⇒ 等 /history ⇒ 收输出图 ⇒ 停服务。
自检与演示用（tools/verify_all.sh 用 --steps 4 跑一遍）；平时在浏览器里开 http://127.0.0.1:8188、把 workflows/ 下的 JSON 拖进去即可。

用法：<comfy-venv>/bin/python -I comfy/run_workflow.py [工作流.json] [--steps N] [--backend npu-sim|npu|cpu] [--seed S] [--port P]
输出：ComfyUI/output/ 下的图；本次的记录（各节点耗时、图路径）写进 out/comfy/<时间>.json。
只用标准库（urllib），不用 websocket：轮询 /history 就够。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow", nargs="?", default=str(ROOT / "comfy/workflows/sd15_taesd_npu.api.json"))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--backend")
    ap.add_argument("--port", type=int, default=int(os.environ.get("COMFY_PORT", "8189")))
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    wf = json.loads(Path(a.workflow).read_text())
    for node in wf.values():
        ins = node["inputs"]
        if a.steps is not None and "steps" in ins:
            ins["steps"] = a.steps
        if a.seed is not None and "seed" in ins:
            ins["seed"] = a.seed
        if a.backend and node["class_type"] == "SD15NPU_TAESDDecode":
            ins["backend"] = a.backend

    log = ROOT / "out" / "comfy" / f"server-{a.port}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, COMFY_PORT=str(a.port))
    srv = subprocess.Popen(["bash", str(ROOT / "comfy/run_comfyui.sh")], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                           start_new_session=True)
    base = f"http://127.0.0.1:{a.port}"
    try:
        t0 = time.time()
        while True:
            try:
                _get(base + "/system_stats")
                break
            except Exception:
                if srv.poll() is not None or time.time() - t0 > 300:
                    sys.exit(f"ComfyUI 没起来，见 {log}")
                time.sleep(1)
        t_up = time.time() - t0
        req = urllib.request.Request(base + "/prompt", data=json.dumps({"prompt": wf}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            pid = json.loads(r.read())["prompt_id"]
        t1 = time.time()
        while True:
            h = _get(f"{base}/history/{pid}")
            if pid in h and h[pid].get("status", {}).get("completed") is not None:
                break
            if time.time() - t1 > a.timeout:
                sys.exit("工作流超时")
            time.sleep(2)
        rec = h[pid]
        st = rec["status"]
        images = [str(Path(os.environ.get("COMFY_DIR", ROOT.parent / "ComfyUI")) / "output" / i.get("subfolder", "") / i["filename"])
                  for o in rec.get("outputs", {}).values() for i in o.get("images", [])]
        res = {"workflow": a.workflow, "steps": a.steps, "backend": a.backend, "server_start_s": round(t_up, 1),
               "run_s": round(time.time() - t1, 1), "status": st.get("status_str"), "completed": st.get("completed"),
               "messages": st.get("messages"), "images": images}
        out = ROOT / "out" / "comfy" / time.strftime("run-%Y%m%d-%H%M%S.json")
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "messages"}, ensure_ascii=False, indent=1))
        if not st.get("completed"):
            errs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            sys.exit(f"工作流失败：{json.dumps(errs, ensure_ascii=False)[:3000]}")
    finally:
        os.killpg(srv.pid, signal.SIGTERM)  # 我们自己起的 ComfyUI 服务（不是板上的作业）
        srv.wait(timeout=30)


if __name__ == "__main__":
    main()
