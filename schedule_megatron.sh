#!/usr/bin/env bash
# Submit the megatron (GLM-5, frozen parameters, torchrun) benches on N nodes of the MILABENCH2026 reservation
# (Rorqual, 4 GPUs per node).
#
#   ./schedule_megatron.sh [N=2] [extra sbatch args]
#
# Default SELECT: megatron-glm5-frozen-smoke-2n (2 nodes, 4 layers) and its topology-ordered twin: same layout, the
# nodes packed per leaf switch (compare `rate`, and the placement report in the job log).
# Full size: SELECT=megatron-glm5-frozen-16n,megatron-glm5-frozen-16n-topo,megatron-glm5-frozen-16n-topo-pp ./schedule_megatron.sh 16
#
# Setup, once, from a login node (the compute nodes have no internet; the job runs offline against $MILABENCH_BASE):
#   - the venv and the slime / Megatron-LM checkouts: `SLIME_DOWNLOAD_ONLY=1 milabench install --config benchmarks/megatron/dev.yaml
#     --select megatron-glm5-frozen-smoke-2n` (or download_slime_data.sh, which does the same for slime)
#   - this script downloads the model's config.json (no weights: random init) when it is missing
# Job script: megatron.sbatch (next to this script).
# Results: $MILABENCH_BASE/runs/megatron.<jobid>   logs: $MILABENCH_BASE/slurm/megatron-<jobid>.out
#
# env:  MILABENCH_BASE (default /home/delaunay/links/scratch/shared)   WORKDIR (default $MILABENCH_BASE/src)
#       SELECT   benches to run (comma separated)
#       RESERVATION / ACCOUNT / TIME / GPUS   default MILABENCH2026 / def-bengioy_gpu / 01:00:00 / 4 (GPUs per node)
#       UPDATE=1 fast-forward the checkout if it is behind origin/$BRANCH (default: fail)   BRANCH (default megatron)
#       TEST_ONLY=1   sbatch --test-only: where/when it would start, nothing is submitted
set -euo pipefail

MILABENCH_BASE=${MILABENCH_BASE:-/home/delaunay/links/scratch/shared}
NODES=${1:-2}
[ $# -gt 0 ] && shift
WORKDIR=${WORKDIR:-$MILABENCH_BASE/src}
RESERVATION=${RESERVATION:-MILABENCH2026}
ACCOUNT=${ACCOUNT:-def-bengioy_gpu}
TIME=${TIME:-01:00:00}
GPUS=${GPUS:-4}
SELECT=${SELECT:-megatron-glm5-frozen-smoke-2n,megatron-glm5-frozen-smoke-2n-topo}

case $SELECT in *-16n*) [ "$NODES" -ge 16 ] || { echo "the 16n benches need 16 nodes (N=$NODES)" >&2; exit 1; } ;; esac
case $SELECT in *-2n*) [ "$NODES" -ge 2 ] || { echo "the 2n benches need 2 nodes (N=$NODES)" >&2; exit 1; } ;; esac

# the job runs the checkout as it is: make sure it is the branch's latest code (login node: has internet)
BRANCH=${BRANCH:-megatron}
if git -C "$WORKDIR" fetch --quiet origin "$BRANCH"; then
    behind=$(git -C "$WORKDIR" rev-list --count "HEAD..origin/$BRANCH")
    if [ "$behind" != 0 ]; then
        if [ "${UPDATE:-0}" = 1 ]; then
            git -C "$WORKDIR" merge --ff-only "origin/$BRANCH"
        else
            echo "$WORKDIR is $behind commit(s) behind origin/$BRANCH: git pull there, or rerun with UPDATE=1" >&2
            exit 1
        fi
    fi
    [ -z "$(git -C "$WORKDIR" status --porcelain --untracked-files=no)" ] \
        || echo "WARNING: $WORKDIR has local modifications, the job runs them" >&2
    echo "milabench $(git -C "$WORKDIR" rev-parse --short HEAD) ($BRANCH, up to date)"
else
    echo "WARNING: could not fetch origin/$BRANCH, running $(git -C "$WORKDIR" rev-parse --short HEAD) unchecked" >&2
fi

code=$WORKDIR/benchmarks/megatron
venv_py=$MILABENCH_BASE/venv/torch/bin/python
[ -x "$WORKDIR/.venv/bin/milabench" ] || { echo "no milabench in $WORKDIR/.venv" >&2; exit 1; }
[ -x "$venv_py" ] || { echo "no $venv_py: run the install (see the header)" >&2; exit 1; }
[ -d "$code/Megatron-LM" ] && [ -d "$code/slime_src" ] \
    || { echo "no Megatron-LM / slime_src in $code: run the install (see the header)" >&2; exit 1; }

# same extra packages as the GLM-5 slime benches (sglang is installed --no-deps by the install): wheels, installed
# from the login node when missing
missing=()
for pair in fla=flash-linear-attention==0.5.2 tilelang=tilelang==0.1.12 deep_gemm=sgl-deep-gemm==0.2.0 sgl_kernel=sglang-kernel==0.4.7; do
    "$venv_py" -c "import ${pair%%=*}" 2>/dev/null || missing+=("${pair#*=}")
done
"$venv_py" -c "import sys, transformers; sys.exit(transformers.__version__ != '5.12.1')" 2>/dev/null || missing+=("transformers==5.12.1")
if [ ${#missing[@]} -gt 0 ]; then
    echo "installing ${missing[*]}"
    if command -v uv >/dev/null; then uv pip install --python "$venv_py" "${missing[@]}"; else "$venv_py" -m pip install "${missing[@]}"; fi
fi
# a login node has no GPU: a failing import is only a warning
PYTHONPATH=$code:$code/slime_src:$code/Megatron-LM "$venv_py" -c "import slime_plugins.models.glm5.glm5" 2>&1 | tail -3 \
    || echo "WARNING: the GLM-5 plugin does not import on this node, see above" >&2

# the GLM-5 spec reads the indexer sizes from the HF config: download it here (no weights needed)
hf=$MILABENCH_BASE/data/megatron/hf
if [ ! -f "$hf/config.json" ]; then
    echo "downloading the GLM-5.3 config to $hf"
    "$venv_py" -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('zai-org/GLM-5.3', local_dir='$hf', token=os.environ.get('MILABENCH_HF_TOKEN') or os.environ.get('HF_TOKEN'),
                  allow_patterns=['config.json', '*.py', 'generation_config.json'])"
fi

mkdir -p "$MILABENCH_BASE/slurm"
# --export=ALL hands these to the job (not through --export=NAME=value: SELECT contains commas)
export MILABENCH_BASE WORKDIR SELECT

scontrol show reservation "$RESERVATION" 2>/dev/null | head -2 || true

jobid=$(sbatch --parsable ${TEST_ONLY:+--test-only} \
    --job-name=megatron \
    --reservation="$RESERVATION" --account="$ACCOUNT" \
    --nodes="$NODES" --ntasks-per-node=1 --exclusive --mem=0 \
    --gpus-per-node="$GPUS" \
    --time="$TIME" \
    --output="$MILABENCH_BASE/slurm/megatron-%j.out" \
    --export=ALL \
    "$@" "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/megatron.sbatch")
[ -n "${TEST_ONLY:-}" ] && exit 0   # --test-only submits nothing
echo "submitted job $jobid"

# follow the log (the file only exists once the job starts, so create it first); ctrl-c stops tailing, not the job
log=$MILABENCH_BASE/slurm/megatron-${jobid%%;*}.out
touch "$log"
tail -f "$log"
