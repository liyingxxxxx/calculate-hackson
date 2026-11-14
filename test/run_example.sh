#!/usr/bin/env bash
# 一个简单示例：一张 512² 合成测试图的 latent ⇒ TAESD 解码器（35 个卷积 + 3 次上采样 + tanh，整个是一个设备程序、一次 dort 调用）⇒ 图，
# 再和两份独立参照（torch 按 bf16 口径另写的、diffusers fp32 原样的）比余弦与 PSNR。
#
#   没有卡：            bash test/run_example.sh          ⇒ npu-sim（CPU 上照编出来的 .cvx 重放，只查接线；⛔ 不是 NPU 的数）
#   有卡（确认可以用）：SD15NPU_ALLOW_BOARD=1 [SD15NPU_BOARD_LOCK=<现场的板锁文件>] bash test/run_example.sh
#                                                         ⇒ 先只读查卡（OS / RT 起着、板上是 DeviceRT、没别人在用），满足才在真卡上跑
#   加 --vae：解码器换成正式 VAE（6 段程序，画质同 ComfyUI 自带的 VAEDecode；真卡上约 5 s，模拟约 30 s）
#   加 --comfy：再跑一遍 ComfyUI 原生工作流（提示词 → CLIP → KSampler 20 步 → SD15 NPU · TAESD / VAE Decode → 存图），解码走同一个后端
#
# 前提：已经 bash tools/bootstrap.sh（README「在一台新机器上复现」）。
# 结果：真卡 out/taesd-board/image/{board.png, metrics.json}；模拟 out/taesd-sim/image/{sim.png, metrics.json}；ComfyUI 的图在 $CALCULET_HOME/ComfyUI/output/。
# test/results/npu3-20261114/ 存着 2026-11-14 在 npu3 真卡上跑出来的那一份（test/README.md），可以对照。
# 挂死就停、找管卡的人：⛔ kill 在飞的 dort、⛔ 复位、⛔ 刷写（AGENTS.md 硬规矩 3）。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=/dev/null
source "$ROOT/tools/env.sh"
PY=$CALCULET_HOME/venv/bin/python
CPY=$CALCULET_HOME/comfy-venv/bin/python
cd "$ROOT"
comfy=0
vae=0
for arg in "$@"; do
  case $arg in
    --comfy) comfy=1 ;;
    --vae) vae=1 ;;
    *) echo "不认得的参数：$arg（只有 --vae、--comfy）" >&2; exit 2 ;;
  esac
done
dec=taesd
wf=comfy/workflows/sd15_taesd_npu.api.json
if [ $vae = 1 ]; then
  dec=vae
  wf=comfy/workflows/sd15_vae_npu.api.json
fi
tools/check_env.sh > /dev/null || { tools/check_env.sh; exit 1; }

if [ "${SD15NPU_ALLOW_BOARD:-}" = 1 ]; then
  echo "━━ 只读查卡（dort device，不开设备跑作业）"
  "$PY" -I - <<'EOF'
import sys
sys.path.insert(0, "host")
from sd15npu import dort
ok, why, out = dort.device_status()
print(out)
if not ok:
    sys.exit("✗ 这块卡现在不能用：" + "；".join(why) + "\n  什么都没往卡上发。找管卡的人（⛔ 自己刷写 / 复位）。")
print("✓ 卡可以用")
EOF
  echo "━━ 真卡：$dec 解码测试图（重出 calbin ⇒ 只部署一次 ⇒ 整次 3 遍，查逐字节一致、对两份参照）"
  "$PY" -I tools/$dec.py board --latent image --runs 3
  backend=npu
else
  echo "━━ 没开上板开关 SD15NPU_ALLOW_BOARD=1：走 npu-sim（CPU 上模拟设备程序，⛔ 不是 NPU 的数）"
  "$PY" -I tools/$dec.py sim --latent image | grep -E "^[0-9]+ 个作业|^s[1-6]：|^image："
  backend=npu-sim
fi

if [ $comfy = 1 ]; then
  echo "━━ ComfyUI 原生工作流（解码走 $backend）"
  "$CPY" -I comfy/run_workflow.py "$wf" --backend $backend --steps 20
fi
echo "完成（后端 $backend）。对照：test/README.md"
