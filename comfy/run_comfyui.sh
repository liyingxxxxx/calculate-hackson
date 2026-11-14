#!/usr/bin/env bash
# 一键拉起 ComfyUI（CPU 模式，本仓的节点已链进 custom_nodes）。参数原样传给 ComfyUI 的 main.py，例如：
#   bash comfy/run_comfyui.sh                       # 起服务，浏览器开 http://127.0.0.1:8188
#   bash comfy/run_comfyui.sh --quick-test-for-ci   # 只加载一遍（含自定义节点）就退出：自检用
# NPU 执行器要不要真去调 dort 由环境变量 SD15NPU_BACKEND 定（见 comfy/sd15_npu/README.md）；缺省 cpu，不碰板子。
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HACK=${CALCULET_HOME:-$(cd "$ROOT/.." && pwd)}
COMFY=${COMFY_DIR:-$HACK/ComfyUI}
VENV=${COMFY_VENV:-$HACK/comfy-venv}
[ -x "$VENV/bin/python" ] && [ -f "$COMFY/main.py" ] || { echo "先 bash comfy/setup_comfyui.sh" >&2; exit 1; }
# 本仓的工具（dort、docc、calbin.sh）要的环境：隔离的工具链目录、PATH 最前面是工具链自带的 docc / dort
source "$ROOT/tools/env.sh"
export SD15NPU_ROOT=$ROOT
cd "$COMFY"
exec "$VENV/bin/python" main.py --cpu --listen 127.0.0.1 --port "${COMFY_PORT:-8188}" "$@"
