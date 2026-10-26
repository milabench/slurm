#!/usr/bin/env bash
# Standalone: clone milabench (branch `megatron`), install it, and install + prepare the megatron (GLM-5, frozen,
# torchrun) benches: venv, slime / Megatron-LM checkouts, TE wheels, and the model's config.json under
# $MILABENCH_BASE/data/megatron/hf (random init: no weights to download). Run it from anywhere, outside any repo.
# Same install as download_slime_data.sh (shared `torch` venv): run this after it and it only adds what is missing.
#
#   MILABENCH_BASE=/home/delaunay/links/scratch/shared ./download_megatron_data.sh   (run by schedule_megatron_glm5.sh)
#
# env:  MILABENCH_BASE  (default /home/delaunay/links/scratch/shared) where data/ venv/ runs/ go; put it on the shared filesystem
#       WORKDIR         checkout + milabench's own venv (default: $MILABENCH_BASE/src)
#       REPO / BRANCH   default git@github.com:milabench/milabench.git / megatron
#       SELECT          benches to install (default: the 2-node smoke entries; any megatron-glm5-* works, they share the venv)
#       HF_TOKEN        for gated repos
#       TORCH_VERSION / CUDA_VERSION   torch + CUDA the venv is built with (default 2.13.0 / 130)
#       WHEELS_TAG      release of milabench/wheels with the TE / TMS wheels (default torch2.13-cu130)
#       INSTALL_WHEELS  0 to skip the TE / torch_memory_saver wheel install (default 1)
# TE and torch_memory_saver are installed from the prebuilt wheels in github.com/milabench/wheels (release
# torch2.13-cu130: transformer_engine_torch 2.18.0 + the zhuzilin torch_memory_saver slime pins), so nothing is
# compiled and no GPU is needed. The venv is built with torch 2.13 / CUDA 13.0 to match those wheels.
# torch_dist conversion is NOT done here (needs GPUs): run `milabench install` + `prepare` on a GPU node afterwards
# (GLM: multi-node job with SLIME_CONVERT=1).
# Re-running is cheap: the checkout is fast-forwarded, downloads resume, finished files are skipped.
set -euo pipefail

MILABENCH_BASE=${MILABENCH_BASE:-/home/delaunay/links/scratch/shared}
REPO=${REPO:-git@github.com:milabench/milabench.git}
BRANCH=${BRANCH:-megatron}
WORKDIR=${WORKDIR:-$MILABENCH_BASE/src}
TORCH_VERSION=${TORCH_VERSION:-2.13.0}
CUDA_VERSION=${CUDA_VERSION:-130}
WHEELS_TAG=${WHEELS_TAG:-torch2.13-cu130}
WHEELS_URL=${WHEELS_URL:-https://github.com/milabench/wheels/releases/download/$WHEELS_TAG}
TE_VERSION=${TE_VERSION:-2.18.0}
TMS_WHEEL=${TMS_WHEEL:-torch_memory_saver-0.0.9.post1%2Bslime-cp312-cp312-linux_x86_64.whl}
INSTALL_WHEELS=${INSTALL_WHEELS:-1}
export MILABENCH_BASE

select=${SELECT:-megatron-glm5-frozen-smoke-2n,megatron-glm5-frozen-smoke-2n-topo}

mkdir -p "$MILABENCH_BASE" "$(dirname "$WORKDIR")"
echo "[megatron-data] base=$MILABENCH_BASE branch=$BRANCH select=$select"

# --- checkout ---
if [ -d "$WORKDIR/.git" ]; then
    git -C "$WORKDIR" fetch origin "$BRANCH"
    git -C "$WORKDIR" checkout "$BRANCH"
    git -C "$WORKDIR" merge --ff-only "origin/$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO" "$WORKDIR"
fi

# --- milabench's own venv ---
venv=$WORKDIR/.venv
pip_install() {
    rm -rf "$venv"
    python3 -m venv "$venv"
    "$venv/bin/pip" install -U pip
    "$venv/bin/pip" install -e "$WORKDIR"
}
if [ ! -x "$venv/bin/milabench" ]; then
    # uv ignores the pip config (e.g. a cluster wheelhouse/mirror), so fall back to plain pip if it fails
    if command -v uv >/dev/null && [ "${USE_UV:-1}" = 1 ]; then
        { uv venv --python 3.12 "$venv" && uv pip install --python "$venv/bin/python" -e "$WORKDIR"; } \
            || { echo "[megatron-data] uv failed, retrying with pip" >&2; pip_install; }
    else
        pip_install
    fi
fi
# voir's CUDA backend (nvidia-ml-py = milabench's `cuda` extra): `milabench run` fails with "cuda is not installed"
# without it, and the compute nodes cannot install it. Also fixes a venv created before this step existed.
if ! "$venv/bin/python" -c 'import pynvml' 2>/dev/null; then
    if command -v uv >/dev/null; then uv pip install --python "$venv/bin/python" nvidia-ml-py \
        || "$venv/bin/pip" install nvidia-ml-py
    else "$venv/bin/pip" install nvidia-ml-py; fi
fi
export PATH=$venv/bin:$PATH

if "$venv/bin/python" -c 'import hf_transfer' 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi

# Login nodes have no GPU: without this milabench picks the CPU torch wheel and TE's build then fails with
# "Invalid version: None" (torch.version.cuda is None). TMS_CUDA_MAJOR is what benchfile.py only sets when the
# config has version_overrides.cuda (the toolkit wheels default to CUDA 13.0).
# slow/flaky egress from the login node: uv's default timeout is 30s (finished downloads stay in uv's cache)
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-900}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10}
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-300}
# This script only downloads: skip what needs a GPU node (TE / torch_memory_saver builds, torch_dist conversion).
# Needs a milabench branch that has SLIME_DOWNLOAD_ONLY (benchfile.py / prepare.py). SLIME_DOWNLOAD_ONLY=0 to disable.
export SLIME_DOWNLOAD_ONLY=${SLIME_DOWNLOAD_ONLY:-1}
[ "$SLIME_DOWNLOAD_ONLY" = 0 ] && unset SLIME_DOWNLOAD_ONLY
export MILABENCH_GPU_ARCH=${MILABENCH_GPU_ARCH:-cuda}
export TMS_CUDA_MAJOR=${TMS_CUDA_MAJOR:-13}

# a venv built by an earlier run without the above holds a CPU torch: drop it (the marker means "install done")
torch_venv=$MILABENCH_BASE/venv/torch
# same for a venv built with another torch than the one the prebuilt TE / torch_memory_saver wheels target
if [ -x "$torch_venv/bin/python" ] && [ "$INSTALL_WHEELS" = 1 ]; then
    have=$("$torch_venv/bin/python" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || true)
    if [ -n "$have" ] && [ "$have" != "$TORCH_VERSION" ]; then
        echo "[megatron-data] $torch_venv has torch $have, need $TORCH_VERSION: removing it" >&2
        rm -rf "$torch_venv"
    fi
fi
if [ -x "$torch_venv/bin/python" ] && [ "$("$torch_venv/bin/python" -c 'import torch; print(torch.version.cuda)' 2>/dev/null)" = "None" ]; then
    echo "[megatron-data] $torch_venv has a CPU-only torch, removing it" >&2
    rm -rf "$torch_venv"
fi

config=$WORKDIR/benchmarks/megatron/dev.yaml
# --set pins the slime venv to the torch / CUDA the prebuilt TE + torch_memory_saver wheels were built for
milabench install --config "$config" --select "$select" "$@" --set "cuda=$CUDA_VERSION" "torch=$TORCH_VERSION"

# --- TE + torch_memory_saver from the prebuilt wheels (no compile, no GPU) ---
# benchfile.py skips both under SLIME_DOWNLOAD_ONLY. The wheels are cp312 / x86_64 / torch 2.13 / cu13.
if [ "$INSTALL_WHEELS" = 1 ]; then
    py=$torch_venv/bin/python
    pyver=$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    torchver=$("$py" -c 'import torch; print(torch.__version__)')
    if [ "$(uname -m)" != x86_64 ] || [ "$pyver" != 3.12 ] || [ "${torchver%%+*}" != "$TORCH_VERSION" ]; then
        echo "[megatron-data] WARNING: skipping the TE / torch_memory_saver wheels: they need x86_64, python 3.12 and torch $TORCH_VERSION" \
             "(found $(uname -m), python $pyver, torch $torchver)" >&2
    else
        echo "[megatron-data] installing transformer_engine_torch $TE_VERSION and torch_memory_saver from $WHEELS_URL"
        if command -v uv >/dev/null; then pipi=(uv pip install --python "$py"); else pipi=("$py" -m pip install); fi
        # transformer-engine (pure python) + our transformer_engine_torch; the core transformer_engine_cu13 and the
        # other deps come from PyPI. torch is already 2.13, so nothing replaces it.
        "${pipi[@]}" "transformer-engine==$TE_VERSION" \
            "transformer_engine_torch @ $WHEELS_URL/transformer_engine_torch-$TE_VERSION-cp312-cp312-linux_x86_64.whl"
        "${pipi[@]}" --no-deps "torch_memory_saver @ $WHEELS_URL/$TMS_WHEEL"
        "$py" -c 'import transformer_engine.pytorch, torch_memory_saver; print("[megatron-data] TE + torch_memory_saver import OK")' \
            || echo "[megatron-data] WARNING: TE / torch_memory_saver import failed here (a login node without libcuda may be the cause)" >&2
    fi
fi

# GLM-5.x (glm5_next plugin): flash-linear-attention (`fla`) + tilelang (versions of slime's Dockerfile) for the linear
# attention layers, and the transformers version sglang 0.5.21 pins (5.12.1: it accepts the `deepseek_sparse_attention` layer type of the GLM
# config. Without `fla`, slime swallows the plugin's ImportError and Megatron fails with "specialize for NoneType".
# Pure python wheels, installed here because the compute nodes have no internet.
GLM_PIP=${GLM_PIP:-"flash-linear-attention==0.5.2 tilelang==0.1.12 sgl-deep-gemm==0.2.0 transformers==5.12.1"}
if command -v uv >/dev/null; then uv pip install --python "$torch_venv/bin/python" $GLM_PIP; else "$torch_venv/bin/python" -m pip install $GLM_PIP; fi

milabench prepare --config "$config" --select "$select" "$@"

echo "[megatron-data] done: $MILABENCH_BASE/data/megatron"
du -sh "$MILABENCH_BASE"/data/megatron/* 2>/dev/null || true
