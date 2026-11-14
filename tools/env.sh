# 比赛专用环境（source 本文件，不要直接执行）：
#   1. RUSTUP_HOME / CARGO_HOME 指到隔离目录 $CALCULET_HOME/{rustup,cargo}：和平时的 ~/.rustup、~/.cargo 完全隔开
#      （平时的 ~/.cargo 里可能配着别的 cargo 源与凭据，不让它们进作用域；本工程只匿名读赛方源）；
#   2. $CARGO_HOME/bin 放进 PATH 最前面（赛方 install.sh 把 docc / dort 链在那里）：机器上若另装着老版本的 docc / dort，
#      用的也是工具链自带的这一版（tools/check_env.sh 核）；
#   3. 清掉 VIRTUAL_ENV、PYTHONPATH：本仓的 Python 一律用指定的 venv、带 -I 跑，免得混进别处的包。
# CALCULET_HOME 缺省是本仓的上一层目录。
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CALCULET_HOME=${CALCULET_HOME:-$(dirname "$_here")}
export H=$CALCULET_HOME
export RUSTUP_HOME=$CALCULET_HOME/rustup
export CARGO_HOME=$CALCULET_HOME/cargo
case ":$PATH:" in *":$CARGO_HOME/bin:"*) ;; *) export PATH=$CARGO_HOME/bin:$PATH ;; esac
unset VIRTUAL_ENV PYTHONPATH _here
