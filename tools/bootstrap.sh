#!/usr/bin/env bash
# 新机器从零重建（不碰板子）：只要把本仓放到 <CALCULET_HOME>/sd15-npu（缺省 CALCULET_HOME = 本仓的上一层），然后
#   bash tools/bootstrap.sh
# 每一步都是幂等的：做过的跳过，所以中途断了重跑即可。依次：
#   0. 前提：Linux x86_64，有 git、curl、tar、sha256sum
#   1. 入口 $CALCULET_HOME/env.sh（相对路径，source 本仓的 tools/env.sh：隔离的 RUSTUP_HOME / CARGO_HOME、PATH 最前面是工具链自带的 docc / dort）
#   2. uv（没有就装进 $CALCULET_HOME/uv，不改 shell 配置）
#   3. rustup（没有就装进隔离目录，不装缺省工具链）
#   4. 赛方工具链：下载 install.sh，**核 sha256**（和 2026-11-13 读过的那份一致才执行），装出 calculet 工具链与 docc / dort；
#      再装文档里的可选组件 rustfmt、clippy、rust-src
#   5. 赛方三个公开仓库：克隆到 $CALCULET_HOME/work，切到钉住的提交（读模板、对照用；本仓经 cargo 源依赖，不靠它们编译）
#   6. calBLAS 的 tiny 例子编一遍设备遍（确认工具链好使）
#   7. CPU 参照环境与权重（ref/setup_ref_env.sh：venv、SD1.5 钉 revision、TAESD；约 4 GiB 下载）
#   8. ComfyUI（comfy/setup_comfyui.sh：钉 v0.45.3、单独的 venv、链入本仓节点与模型）
#   9. tools/verify_all.sh：不上板能做的全部检查
# 网络要能到：115.159.82.89（http）、腾讯云 COS（工具链包）、huggingface.co、download.pytorch.org、pypi.org、github.com、astral.sh、sh.rustup.rs。
# 跳过某几步：BOOTSTRAP_SKIP="7 8"（比如只想先验编译链）。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export CALCULET_HOME=${CALCULET_HOME:-$(cd "$ROOT/.." && pwd)}
HACK=$CALCULET_HOME
mkdir -p "$HACK/logs" "$HACK/work"
SKIP=" ${BOOTSTRAP_SKIP:-} "
INSTALL_SH_SHA=3c175657a77a5f40afee8d96c79af785db59a8f87868381b1b2257a93bd92950
declare -A PIN=(
  [calBLAS]=32d2bb83d7160a593eaab67aa02e5dddbc8706cf
  [calDNN]=1e06c41c557a469a9acc7566acb9f7e2c7f7ca62
  [npu]=172aad2611c86ead4071356fbaca3cf6facd2917
)
step() { echo; echo "━━ $1. $2"; }
skip() { case "$SKIP" in *" $1 "*) echo "（按 BOOTSTRAP_SKIP 跳过）"; return 0 ;; esac; return 1; }
die() { echo "✗ $*" >&2; exit 1; }

step 0 "前提"
[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || die "要 Linux x86_64（赛方工具链只出了这一种主机）"
for t in git curl tar sha256sum; do command -v $t >/dev/null || die "缺 $t"; done
echo "✓ $(uname -sm)；本仓 $ROOT；CALCULET_HOME=$HACK"

step 1 "隔离环境的入口 $HACK/env.sh"
cat > "$HACK/env.sh" <<'EOF'
# 比赛专用环境的入口（tools/bootstrap.sh 写的）：内容在仓里的 tools/env.sh，随仓走。
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sd15-npu/tools/env.sh"
EOF
[ "$(basename "$ROOT")" = sd15-npu ] || die "本仓的目录名要是 sd15-npu（env.sh、脚本按 <CALCULET_HOME>/sd15-npu 找它），现在是 $ROOT"
# shellcheck source=/dev/null
source "$ROOT/tools/env.sh"
mkdir -p "$CARGO_HOME/bin"   # 赛方 install.sh 最后一步往这里链 docc / dort，目录不在会半途退出
echo "✓ RUSTUP_HOME=$RUSTUP_HOME CARGO_HOME=$CARGO_HOME"

step 2 "uv"
if UVB=$(command -v uv 2>/dev/null) || { [ -x "$HOME/.local/bin/uv" ] && UVB=$HOME/.local/bin/uv; } || { [ -x "$HACK/uv/uv" ] && UVB=$HACK/uv/uv; }; then
  echo "✓ 已有 $UVB"
else
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HACK/uv" INSTALLER_NO_MODIFY_PATH=1 sh
  UVB=$HACK/uv/uv
fi
export UV=$UVB
echo "· $("$UV" --version)"

step 3 "rustup"
if command -v rustup >/dev/null; then
  echo "✓ 已有 $(command -v rustup)（只当代理用，工具链都在隔离的 RUSTUP_HOME 里）"
else
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none
  echo "✓ rustup 装进 $CARGO_HOME/bin"
fi

step 4 "赛方工具链（calculet：rustc 1.98.1 + 设备 target + docc + dort）"
if rustc +calculet --version >/dev/null 2>&1 && [ -e "$CARGO_HOME/bin/docc" ] && [ -e "$CARGO_HOME/bin/dort" ]; then
  echo "✓ 已装：$(rustc +calculet --version)"
else
  curl -sSf http://115.159.82.89/install.sh -o "$HACK/logs/install.sh"
  if ! echo "$INSTALL_SH_SHA  $HACK/logs/install.sh" | sha256sum -c - >/dev/null; then
    die "赛方 install.sh 和 2026-11-13 读过的那份不一样了（$HACK/logs/install.sh）：先读一遍、确认没问题，再改本脚本里的 INSTALL_SH_SHA"
  fi
  sh "$HACK/logs/install.sh" 2>&1 | tee "$HACK/logs/install-toolchain.log"
fi
# 可选组件（赛题文档「工具链」一节）：rustfmt、clippy、rust-src，从同一个源装。install.sh 用的是 --profile minimal，不带它们
for comp in rustfmt clippy rust-src; do
  if ! rustup component list --toolchain nightly-2026-11-13 --installed | grep -q "^$comp"; then
    RUSTUP_DIST_SERVER=http://115.159.82.89 rustup component add --toolchain nightly-2026-11-13 rustfmt clippy rust-src
    break
  fi
done
echo "· $(rustc +calculet --version)；$(docc --version)；$(dort --version)；$(cargo +calculet fmt --version)；$(cargo +calculet clippy --version)"

step 5 "赛方三个公开仓库（$HACK/work）"
for r in calBLAS calDNN npu; do
  d=$HACK/work/$r
  [ -d "$d/.git" ] || git clone -q "http://115.159.82.89/git/calculet/$r.git" "$d"
  git -C "$d" fetch -q origin || echo "⚠ $r：fetch 失败（离线？）用本地已有的"
  if git -C "$d" cat-file -e "${PIN[$r]}^{commit}" 2>/dev/null; then
    git -C "$d" -c advice.detachedHead=false checkout -q "${PIN[$r]}"
    echo "✓ $r @ ${PIN[$r]:0:7}（钉住的那版）"
  else
    echo "⚠ $r：钉住的 ${PIN[$r]:0:7} 不在了，用 $(git -C "$d" rev-parse --short HEAD) —— 在 PROGRESS.md 记一笔、核对 calblas 的模板有没有变"
  fi
done

step 6 "calBLAS 的 tiny 设备遍（工具链自检）"
(cd "$HACK/work/calBLAS" && cargo build -q -p tiny --lib --target dopamine-unknown-none-elf) && echo "✓ tiny.cvxo 出来了"

step 7 "CPU 参照环境与权重"
skip 7 || bash "$ROOT/ref/setup_ref_env.sh" 2>&1 | tee "$HACK/logs/setup-ref-env.log"

step 8 "ComfyUI"
skip 8 || bash "$ROOT/comfy/setup_comfyui.sh" 2>&1 | tee "$HACK/logs/setup-comfyui.log"

step 9 "不上板能做的全部检查"
skip 9 || bash "$ROOT/tools/verify_all.sh"
echo
echo "重建完成。每个新 shell 先：source $HACK/env.sh"
