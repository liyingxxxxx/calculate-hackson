#!/usr/bin/env bash
# 不上板能做的全部检查，一条命令跑完（换机器之后、改了东西之后、交之前都跑一遍）。
# 每项的结果与用时汇总在末尾；完整输出在 $CALCULET_HOME/logs/verify-<时间>.log。任何一项不过，退出码就是 1。
#   VERIFY_FULL=1：再加 ComfyUI 整条工作流（SD1.5 在 CPU 上 4 步 + 我们的 TAESD / 正式 VAE 节点走 npu-sim，约 1.5 分钟）
# 不跑任何会开设备的 dort 子命令（AGENTS.md 硬规矩 3）。
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=/dev/null
source "$ROOT/tools/env.sh"
HACK=$CALCULET_HOME
PY=$HACK/venv/bin/python
CPY=$HACK/comfy-venv/bin/python
mkdir -p "$HACK/logs"
LOG=$HACK/logs/verify-$(date +%Y%m%d-%H%M%S).log
cd "$ROOT"

names=(); results=(); fails=0
run() {
  local name=$1; shift
  local t0 t1 rc
  echo "━━ $name" | tee -a "$LOG"
  t0=$(date +%s.%N)
  "$@" >>"$LOG" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  local dt
  dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')
  if [ $rc = 0 ]; then results+=("✓ $name（${dt} s）"); else results+=("✗ $name（退出 $rc，${dt} s，见日志）"); fails=$((fails + 1)); tail -15 "$LOG"; fi
  names+=("$name")
}

params_deterministic() {
  local a b
  "$PY" -I tools/taesd.py params || return 1
  a=$(cat out/taesd/params/param_blk0.sha256)
  "$PY" -I tools/taesd.py params
  b=$(cat out/taesd/params/param_blk0.sha256)
  echo "第一次 $a；第二次 $b"
  [ "$a" = "$b" ]
}

comfy_loads() {
  timeout 600 bash comfy/run_comfyui.sh --quick-test-for-ci --port 8199 > "$HACK/logs/comfy-quicktest.log" 2>&1 || { cat "$HACK/logs/comfy-quicktest.log"; return 1; }
  grep -q "custom_nodes/sd15_npu" "$HACK/logs/comfy-quicktest.log" && ! grep -q "IMPORT FAILED" "$HACK/logs/comfy-quicktest.log"
}

run "环境自检（隔离目录、工具链自带的 docc / dort、Cargo.lock 来源）" tools/check_env.sh
run "设备遍：smoke、taesd（dopamine-unknown-none-cvx）" cargo build -q -p smoke -p taesd --lib --target dopamine-unknown-none-cvx
run "主机程序：layout-dump" cargo build -q --release -p layout-dump
run "冒烟 calbin + docc doctor" bash -c 'tools/calbin.sh smoke && grep -q "⛔ 拒收 0 条" out/smoke/doctor.txt'
run "TAESD 参数块逐字节可复现（打两次比 sha256）" params_deterministic
run "TAESD calbin（真权重）+ docc doctor" bash -c '"$0" -I tools/taesd.py calbin && grep -q "⛔ 拒收 0 条" out/taesd-real/doctor.txt' "$PY"
run "模拟器自测（7 种坏法都要抓到）" "$PY" -I ref/test_cvxsim.py
run "TAESD 整步程序的 numpy 模拟对两份独立参照（4 组 latent）" "$PY" -I tools/taesd.py check
run "单 kernel 测试台（7 个测试 × 边界用例：模拟对期望、calbin 出得来）" "$PY" -I ref/ktest.py all
run "正式 VAE：6 段 calbin（真权重）+ docc doctor" "$PY" -I tools/vae.py calbin
run "正式 VAE 6 段程序的 numpy 模拟对两份独立参照（2 组 latent）" "$PY" -I tools/vae.py check
run "上板前静态核对（dort 认得我们的包，张量名 / 字节数 / 参数块与主机侧一致）" "$PY" -I tools/preflight.py
run "上板流程自测（测试替身空跑全套 + 6 种该停的情形）" "$PY" -I ref/test_board_kit.py
run "ComfyUI 加载本仓节点" comfy_loads
if [ "${VERIFY_FULL:-0}" = 1 ]; then
  run "ComfyUI 原生工作流（SD1.5 4 步 + TAESD npu-sim）" "$CPY" -I comfy/run_workflow.py --steps 4 --port 8198
  run "ComfyUI 原生工作流（SD1.5 4 步 + 正式 VAE npu-sim）" "$CPY" -I comfy/run_workflow.py comfy/workflows/sd15_vae_npu.api.json --steps 4 --port 8197
fi

echo
echo "══ 汇总（$(date '+%F %T')，日志 $LOG）"
printf '%s\n' "${results[@]}" | tee -a "$LOG"
echo "· $(rustc +calculet --version)；$(docc --version)；$(dort --version)" | tee -a "$LOG"
[ $fails = 0 ] && echo "全部通过（没有上板：数值都是 CPU 上的模拟与参照）" | tee -a "$LOG" || echo "$fails 项不过" | tee -a "$LOG"
exit $((fails > 0))
