#!/usr/bin/env bash
# 环境自检：本工程只用赛方的公开栈（calculet 工具链、赛方 cargo 源）。逐条查，任何一条不过就退出 1（响亮地失败，不打个警告就算）：
#   1. rustup / cargo 在隔离目录（$CALCULET_HOME/{rustup,cargo}），里面没有凭据文件；
#   2. PATH 上的 docc / dort 是 calculet 工具链自带的那两个（机器上可能另有老版本，比如 /usr/local/bin 里的 dort 0.2.8、docc 0.1.4）；
#   3. Cargo.lock 里每个包都来自 crates.io 或赛方源（115.159.82.89），没有别的源。
# 用法：source $CALCULET_HOME/env.sh && tools/check_env.sh
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HACK=${CALCULET_HOME:-$(cd "$ROOT/.." && pwd)}
fail=0
bad() { echo "✗ $*"; fail=1; }
ok() { echo "✓ $*"; }

# 1. 隔离目录
[ "${RUSTUP_HOME:-}" = "$HACK/rustup" ] && ok "RUSTUP_HOME=$RUSTUP_HOME" || bad "RUSTUP_HOME 不是 $HACK/rustup（先 source tools/env.sh）"
[ "${CARGO_HOME:-}" = "$HACK/cargo" ] && ok "CARGO_HOME=$CARGO_HOME" || bad "CARGO_HOME 不是 $HACK/cargo（先 source tools/env.sh）"
if [ -n "${CARGO_HOME:-}" ] && [ -e "$CARGO_HOME/credentials.toml" -o -e "$CARGO_HOME/credentials" ]; then
  bad "隔离的 CARGO_HOME 里出现了凭据文件：本工程只匿名读赛方源，不该有 token"
fi

# 2. 工具链：docc / dort 必须是 calculet 工具链自带的
sysroot=$(rustc +calculet --print sysroot 2>/dev/null) || { bad "没有 calculet 工具链（见 README「在一台新机器上复现」）"; sysroot=; }
for t in docc dort; do
  p=$(command -v $t 2>/dev/null) || { bad "PATH 上没有 $t"; continue; }
  real=$(readlink -f "$p")
  if [ -n "$sysroot" ] && [ "$real" = "$(readlink -f "$sysroot/bin/$t")" ]; then ok "$t → $real"; else bad "$t 不是 calculet 工具链自带的那个：$real"; fi
done

# 3. Cargo.lock 的来源：白名单（crates.io、赛方源），出现别的源就不过
if [ -f "$ROOT/Cargo.lock" ]; then
  others=$(grep -o 'source = "[^"]*"' "$ROOT/Cargo.lock" | sort -u \
    | grep -v -e '"registry+https://github.com/rust-lang/crates.io-index"' -e '"sparse+http://115.159.82.89/git/api/packages/calculet/cargo/"')
  n=$(grep -c 'source = "sparse+http://115.159.82.89/' "$ROOT/Cargo.lock")
  if [ -n "$others" ]; then bad "Cargo.lock 里有赛方源、crates.io 以外的源：$others"; else ok "Cargo.lock：$n 个包来自赛方源，其余来自 crates.io，没有别的源"; fi
fi

# 版本（统一复测要写清用的哪一版）
echo "· $(rustc +calculet --version 2>/dev/null)；$(docc --version 2>/dev/null)；$(dort --version 2>/dev/null)"
[ $fail = 0 ] && echo "环境自检通过" || echo "环境自检不通过"
exit $fail
