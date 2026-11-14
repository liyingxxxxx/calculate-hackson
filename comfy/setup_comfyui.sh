#!/usr/bin/env bash
# ComfyUI（钉版本）+ 它自己的 venv + 把本仓的自定义节点链进去。全部放仓外（$CALCULET_HOME 下），不改 ComfyUI 源码。
#   源码  ：$CALCULET_HOME/ComfyUI，钉 tag v0.45.3（提交 71096d1a…），换版本改下面两行并在 PROGRESS 记一笔
#   venv  ：$CALCULET_HOME/comfy-venv（uv 建，Python 3.12）。和 CPU 参照环境（venv/）分开：ComfyUI 的依赖多，
#           混进去会顶掉 ref/requirements.lock 里钉好的版本，逐算子参照就不是同一套了
#   版本  ：装完冻结进 comfy/requirements.lock；有 lock 就照它装（换机器复现同一套），COMFY_UPGRADE=1 才按 ComfyUI 的
#           requirements.txt 重新解析并重写 lock
#   torch ：从 PyTorch 的 CPU 索引装，与参照环境同一版（ref/requirements.lock 里的 torch==…）；torchvision / torchaudio 由 uv 配对
#   节点  ：$CALCULET_HOME/ComfyUI/custom_nodes/sd15_npu → 本仓 comfy/sd15_npu（符号链接）
#   模型  ：$MODELS/stable-diffusion-v1-5（diffusers 目录）链进 ComfyUI/models/diffusers/，给 DiffusersLoader 用；
#           $MODELS/taesd 链进 ComfyUI/models/vae_approx/（我们的节点从这里找 TAESD）
# 用法：bash comfy/setup_comfyui.sh；装完 bash comfy/run_comfyui.sh --quick-test-for-ci 自检。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HACK=${CALCULET_HOME:-$(cd "$ROOT/.." && pwd)}
COMFY=${COMFY_DIR:-$HACK/ComfyUI}
VENV=${COMFY_VENV:-$HACK/comfy-venv}
MODELS=${MODELS:-$(cd "$HACK/.." && pwd)/models}
COMFY_URL=https://github.com/Comfy-Org/ComfyUI.git
COMFY_TAG=v0.45.3
COMFY_REV=71096d1af6c9d1c34fc42094ec4ec59768104786
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
LOCK=$ROOT/comfy/requirements.lock
TORCH=$(grep '^torch==' "$ROOT/ref/requirements.lock")

unset PYTHONPATH VIRTUAL_ENV DOPE_TOOLCHAIN_DIR LLVM_BINARY_DIR

# 1. 源码：钉到提交（tag 可能被挪，提交号才是准的）
if [ ! -d "$COMFY/.git" ]; then
  git clone --filter=blob:none "$COMFY_URL" "$COMFY"
fi
git -C "$COMFY" fetch -q --tags origin "$COMFY_TAG" 2>/dev/null || git -C "$COMFY" fetch -q origin
git -C "$COMFY" -c advice.detachedHead=false checkout -q "$COMFY_REV"
[ "$(git -C "$COMFY" rev-parse HEAD)" = "$COMFY_REV" ] || { echo "ComfyUI 没钉到 $COMFY_REV" >&2; exit 1; }

# 2. venv 与依赖
[ -x "$VENV/bin/python" ] || "$UV" venv --python 3.12 "$VENV"
PY=$VENV/bin/python
if [ -f "$LOCK" ] && [ "${COMFY_UPGRADE:-0}" != 1 ]; then
  grep -E '^(torch|torchvision|torchaudio)==' "$LOCK" > "$VENV/torch.txt"
  "$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu -r "$VENV/torch.txt"
  grep -vE '^(torch|torchvision|torchaudio)==' "$LOCK" > "$VENV/rest.txt"
  "$UV" pip install --python "$PY" -r "$VENV/rest.txt"
else
  "$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "$TORCH" torchvision torchaudio
  # ComfyUI 的 requirements.txt 里 torch / torchvision 不带版本：上一步装好了就不会再动（uv 不升已满足的）
  "$UV" pip install --python "$PY" -r "$COMFY/requirements.txt"
  # 我们的节点要 diffusers（CPU 执行器与参照用）；版本跟参照环境走
  "$UV" pip install --python "$PY" "$(grep '^diffusers==' "$ROOT/ref/requirements.lock")"
  "$UV" pip freeze --python "$PY" > "$LOCK"
fi

# 3. 节点与模型链接
mkdir -p "$COMFY/custom_nodes" "$COMFY/models/diffusers" "$COMFY/models/vae_approx"
ln -sfn "$ROOT/comfy/sd15_npu" "$COMFY/custom_nodes/sd15_npu"
[ -d "$MODELS/stable-diffusion-v1-5" ] && ln -sfn "$MODELS/stable-diffusion-v1-5" "$COMFY/models/diffusers/stable-diffusion-v1-5"
[ -d "$MODELS/taesd" ] && ln -sfn "$MODELS/taesd" "$COMFY/models/vae_approx/taesd-diffusers"

echo "ComfyUI $COMFY_TAG（$(git -C "$COMFY" rev-parse --short HEAD)）就绪：$COMFY；venv $VENV；$("$PY" -c 'import torch; print("torch", torch.__version__)')"
