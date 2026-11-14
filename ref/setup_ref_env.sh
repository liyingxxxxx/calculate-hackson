#!/usr/bin/env bash
# CPU 参照环境与权重：只装公开的包（CPU 版 torch、diffusers、transformers），只下公开的权重，全部放仓外（CALCULET_HOME 及其上一层）。
#   venv  ：$CALCULET_HOME/venv（uv 建，Python 3.12）；装完把版本冻结进 ref/requirements.lock
#   权重  ：<CALCULET_HOME 的上一层>/models/stable-diffusion-v1-5（钉 revision，只下 fp32 的 safetensors 与配置；fp16 / .bin / non_ema 不要）
#           <CALCULET_HOME 的上一层>/models/taesd（diffusers 格式的 TAESD，B1 先用它）
#   核对  ：每个权重文件的 sha256 与大小写进 <模型目录>/download-verification.json（复现时对照）
# 为什么钉 revision：runwayml 的原仓已下架，这个镜像仓以后再改，参照就不是同一份了。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
HACK=${CALCULET_HOME:-$(cd "$ROOT/.." && pwd)}
VENV=$HACK/venv
# 权重放仓外、和 CALCULET_HOME 并排（即 $CALCULET_HOME/../models）；runner.py 按同一条规矩找
MODELS=${MODELS:-$(cd "$HACK/.." && pwd)/models}
SD_REPO=stable-diffusion-v1-5/stable-diffusion-v1-5
SD_REV=451f4fe16113bff5a5d2269ed5ad43b0592e9a14
TAESD_REPO=madebyollin/taesd
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
# Python 3.12：给版本号即可，uv 找不到就自己下一个受管的解释器（换机器不用改路径）
PY312=${PY312:-3.12}
LOCK=$ROOT/ref/requirements.lock

# 平时的 shell 可能带着别的设备栈的变量：这里一律不用
unset PYTHONPATH VIRTUAL_ENV DOPE_TOOLCHAIN_DIR LLVM_BINARY_DIR

if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv --python "$PY312" "$VENV"
fi
# torch 从 PyTorch 的 CPU 索引装（PyPI 上的默认构建带 CUDA，几个 GB 白下）。
# 有 requirements.lock 就照它装（换机器复现同一套版本）；REF_UPGRADE=1 才装最新版并重写 lock。
if [ -f "$LOCK" ] && [ "${REF_UPGRADE:-0}" != 1 ]; then
  "$UV" pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' "$LOCK")"
  grep -v '^torch==' "$LOCK" > "$VENV/requirements-no-torch.txt"
  "$UV" pip install --python "$VENV/bin/python" -r "$VENV/requirements-no-torch.txt"
else
  "$UV" pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu torch
  "$UV" pip install --python "$VENV/bin/python" diffusers transformers safetensors huggingface_hub numpy pillow
  "$UV" pip freeze --python "$VENV/bin/python" > "$LOCK"
fi

"$VENV/bin/python" -I - "$MODELS" "$SD_REPO" "$SD_REV" "$TAESD_REPO" <<'PY'
import hashlib, json, os, sys, time
from huggingface_hub import snapshot_download, model_info

models, sd_repo, sd_rev, taesd_repo = sys.argv[1:5]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def fetch(repo, rev, dest, patterns):
    t = time.time()
    snapshot_download(repo, revision=rev, local_dir=dest, allow_patterns=patterns)
    files = {}
    for dp, _, fs in os.walk(dest):
        if "/.cache" in dp or dp.endswith(".cache"):
            continue
        for f in fs:
            if f == "download-verification.json":  # 核对文件自己不算（重跑时它已经在了）
                continue
            p = os.path.join(dp, f)
            files[os.path.relpath(p, dest)] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    rec = {"repo": repo, "revision": rev, "allow_patterns": patterns, "files": files,
           "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t, 1)}
    with open(os.path.join(dest, "download-verification.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"{repo}@{rev[:12]} → {dest}：{len(files)} 个文件，{sum(v['bytes'] for v in files.values()) / 2**30:.2f} GiB，{rec['seconds']} s")

fetch(sd_repo, sd_rev, os.path.join(models, "stable-diffusion-v1-5"), [
    "model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json", "text_encoder/model.safetensors",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"])
taesd_rev = model_info(taesd_repo).sha
fetch(taesd_repo, taesd_rev, os.path.join(models, "taesd"), ["config.json", "diffusion_pytorch_model.safetensors"])
PY
echo "CPU 参照环境就绪：$VENV"
