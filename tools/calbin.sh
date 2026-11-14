#!/usr/bin/env bash
# 把一个整步程序 crate 编成 calbin，并用 docc 自检（不上板）：
#   cargo 设备遍（dopamine-unknown-none-cvx）⇒ 暂存目录（step.cvxo + param_blk{芯粒}.bin）⇒ docc calbin from-calvx ⇒ docc calbin doctor
# 编译与 docc 各自计时，写进 <输出目录>/timing.txt（规则要求编译、加载、预热、推理分开记，这里是「编译」那一段）。
#
# 用法：tools/calbin.sh <crate> [--features <f>] [--params <目录>] [-o <输出目录>]
#   --params 给了：从那个目录取打包好的 param_blk{c}.bin（真权重）；
#   不给：按 .cvx 里 weights 声明的字节数造同长的稀疏文件，docc 加 --no-params 只核结构（calBLAS 部署说明里的做法）。
# 输出：<输出目录>/<crate>/（暂存，目录名就是 calbin 里的子模型名）、<输出目录>/cal/（calbin）、doctor.txt、timing.txt。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
crate=${1:?用法：tools/calbin.sh <crate> [--features <f>] [--params <目录>] [-o <输出目录>]}; shift
features=; params=; out=
while [ $# -gt 0 ]; do
  case $1 in
    --features) features=$2; shift 2 ;;
    --params) params=$(cd "$2" && pwd); shift 2 ;;
    -o) out=$2; shift 2 ;;
    *) echo "不认得的参数：$1" >&2; exit 2 ;;
  esac
done
out=${out:-$ROOT/out/$crate}

"$ROOT/tools/check_env.sh" >/dev/null || { "$ROOT/tools/check_env.sh"; exit 1; }

cd "$ROOT"
# 带 feature 的用自己的 target 目录：cargo 按 feature 组合各记一份指纹，切回编过的组合时不重编，
# 而设备遍的产物 npu/<crate>.cvx 只有一个文件名 ⇒ 留着的是上一次编的那个组合（2026-11-14 在 ktest 上踩到）。
tdir=target
[ -n "$features" ] && tdir=target/features/$crate-${features//[, ]/_}
t0=$(date +%s.%N)
cargo build -q -p "$crate" --lib --target dopamine-unknown-none-cvx ${features:+--features "$features"} --target-dir "$tdir"
t1=$(date +%s.%N)

name=${crate//-/_}
cvx=$tdir/dopamine-unknown-none-cvx/npu/$name.cvx
cvxo=$tdir/dopamine-unknown-none-cvx/npu/$name.cvxo
[ -f "$cvxo" ] || { echo "设备遍没出 $cvxo" >&2; exit 1; }

rm -rf "$out" && mkdir -p "$out/$name"
cp "$cvxo" "$out/$name/step.cvxo"
chiplets=$(sed -n 's/^ *chiplets = \([0-9]*\),.*/\1/p' "$cvx" | head -1)
wbytes=$(awk '$1 == "mem.decl" && $3 == "weights," { print $4 }' "$cvx" | head -1)
: "${chiplets:?.cvx 里找不到 chiplets}"

docc_args=()
for ((c = 0; c < chiplets; c++)); do
  if [ -n "$params" ]; then
    ln -s "$params/param_blk$c.bin" "$out/$name/param_blk$c.bin"
  else
    truncate -s "${wbytes:-0}" "$out/$name/param_blk$c.bin"
    docc_args=(--no-params)
  fi
done

t2=$(date +%s.%N)
docc calbin from-calvx "$out/$name" -o "$out/cal" "${docc_args[@]}"
t3=$(date +%s.%N)
# --no-params 时 docc 不往 calbin 里写参数块，doctor 会拒「清单点名了 param_blk 但盘上没有」。
# 补同长的全零稀疏文件让结构完整；这样的 calbin 只能验结构，上板算出来的数没有意义（timing.txt 里写明）。
if [ -z "$params" ]; then
  for ((c = 0; c < chiplets; c++)); do truncate -s "${wbytes:-0}" "$out/cal/param_blk$c.bin"; done
fi

status=0
docc calbin doctor "$out/cal" > "$out/doctor.txt" 2>&1 || status=$?
{
  echo "crate=$crate features=${features:-无} chiplets=$chiplets weights=${wbytes:-0} 字节 params=${params:-全零稀疏文件（--no-params，只验结构，不能拿去上板算数）}"
  echo "$(rustc +calculet --version)；$(docc --version)"
  awk -v a="$t0" -v b="$t1" -v c="$t2" -v d="$t3" 'BEGIN { printf "设备遍 %.2f s；docc from-calvx %.2f s\n", b - a, d - c }'
} > "$out/timing.txt"
cat "$out/timing.txt"
grep -E '检查|拒收|⛔' "$out/doctor.txt" | head -20
exit $status
