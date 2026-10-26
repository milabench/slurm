#!/usr/bin/env bash
# Submit the multi-node slime benches on N nodes of the MILABENCH2026 reservation (Rorqual, 4 GPUs per node).
#
#   ./schedule_slime_multinode.sh [N=8] [extra sbatch args]
#
# Run download_slime_data.sh first (login node): the compute nodes have no internet, so the job runs offline
# against $MILABENCH_BASE (venv, HF weights, datasets, all on the shared filesystem).
#
# Default SELECT, one `milabench run` over an N-node allocation (each bench takes the first num_machines nodes):
#   slime-qwen2.5-0.5b-sft-2n4g, -grpo-colocate-2n4g   2 nodes   Ray, data parallel, SGLang colocated
#   slime-qwen3-30b-a3b-sft-2n4g, -sft-routing-2n4g    2 nodes   TP4 x EP8 MoE (conversion on 4 GPUs first)
#   slime-glm5.3-flash-sft-frozen                      8 nodes   32 GPUs, PP4 x DP8, EP8, experts frozen
# so N must be >= 8 (the job refuses less). Benches larger than N nodes are skipped by requires_capabilities.
# Job script: slime_multinode.sbatch (next to this script).
# Results: $MILABENCH_BASE/runs/slime-multinode.<jobid>   logs: $MILABENCH_BASE/slurm/slime-multinode-<jobid>.out
#
# env:  MILABENCH_BASE (default /home/delaunay/links/scratch/shared)   WORKDIR (default $MILABENCH_BASE/src)
#       SELECT   benches to run (comma separated, see above)
#       RESERVATION / ACCOUNT / TIME / GPUS   default MILABENCH2026 / def-bengioy_gpu / 04:00:00 / 4 (GPUs per node)
#       UPDATE=1 fast-forward the checkout if it is behind origin/$BRANCH (default: fail)   BRANCH (default megatron)
#       PACK (default slime) / CONFIG (default $WORKDIR/benchmarks/$PACK/dev.yaml): the megatron pack runs the same way,
#       see schedule_megatron_glm5.sh
#       TEST_ONLY=1   sbatch --test-only: where/when it would start, nothing is submitted
set -euo pipefail

MILABENCH_BASE=${MILABENCH_BASE:-/home/delaunay/links/scratch/shared}
NODES=${1:-8}
[ $# -gt 0 ] && shift
WORKDIR=${WORKDIR:-$MILABENCH_BASE/src}
RESERVATION=${RESERVATION:-MILABENCH2026}
ACCOUNT=${ACCOUNT:-def-bengioy_gpu}
TIME=${TIME:-04:00:00}
GPUS=${GPUS:-4}
SELECT=${SELECT:-slime-qwen2.5-0.5b-sft-2n4g,slime-qwen2.5-0.5b-grpo-colocate-2n4g,slime-qwen3-30b-a3b-sft-2n4g,slime-qwen3-30b-a3b-sft-routing-2n4g,slime-glm5.3-flash-sft-frozen}

PACK=${PACK:-slime}
CONFIG=${CONFIG:-$WORKDIR/benchmarks/$PACK/dev.yaml}

case $SELECT in *-16n*) [ "$NODES" -ge 16 ] || { echo "the 744B bench needs 16 nodes (N=$NODES)" >&2; exit 1; } ;; esac
case $SELECT in *flash*) [ "$NODES" -ge 8 ] || { echo "the Flash bench needs 8 nodes (N=$NODES)" >&2; exit 1; } ;; esac
# the job runs the checkout as it is: make sure it is the branch's latest code (this runs on the login node, which
# has internet; the compute nodes do not). BRANCH=megatron by default; UPDATE=1 fast-forwards instead of failing.
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

# GLM-5.x needs a few packages that benchfile.py skips (sglang is installed --no-deps) or that are not in
# requirements: install what is missing from the login node (the compute nodes have no internet), pins from slime's
# Dockerfile / sglang 0.5.21. module=package pairs; all are wheels, nothing is compiled.
case $SELECT in *glm*)
    venv_py=$MILABENCH_BASE/venv/torch/bin/python
    missing=()
    for pair in fla=flash-linear-attention==0.5.2 tilelang=tilelang==0.1.12 deep_gemm=sgl-deep-gemm==0.2.0 sgl_kernel=sglang-kernel==0.4.7; do
        "$venv_py" -c "import ${pair%%=*}" 2>/dev/null || missing+=("${pair#*=}")
    done
    # transformers: exactly sglang 0.5.21's pin. Two other releases seen failing on the GLM plugin: the one first installed
    # (rejected the deepseek_sparse_attention layer_types) and 5.20.x (GlmOcr vision rope "supports only axial rope")
    "$venv_py" -c "import sys, transformers; sys.exit(transformers.__version__ != '5.12.1')" 2>/dev/null || missing+=("transformers==5.12.1")
    if [ ${#missing[@]} -gt 0 ]; then
        echo "installing ${missing[*]}"
        if command -v uv >/dev/null; then uv pip install --python "$venv_py" "${missing[@]}"; else "$venv_py" -m pip install "${missing[@]}"; fi
    fi
    # import the model plugin the way slime does: a missing module is reported here instead of after the job started
    # (slime swallows the ImportError and fails later with "specialize for NoneType"). A login node has no GPU, so only warn.
    code=$WORKDIR/benchmarks/$PACK
    for plugin in slime_plugins.models.glm5_next.model slime_plugins.models.glm5.glm5; do
        PYTHONPATH=$code:$code/slime_src:$code/Megatron-LM "$venv_py" -c "import $plugin" 2>&1 | tail -3 \
            || echo "WARNING: $plugin does not import on this node, see above" >&2
    done
    ;;
esac

[ -x "$WORKDIR/.venv/bin/milabench" ] || { echo "no milabench in $WORKDIR/.venv: run download_slime_data.sh first" >&2; exit 1; }
[ -d "$MILABENCH_BASE/data/$PACK" ] || { echo "no $MILABENCH_BASE/data/$PACK: run download_${PACK}_data.sh first" >&2; exit 1; }
[ -d "$WORKDIR/benchmarks/$PACK/Megatron-LM" ] || { echo "no Megatron-LM in $WORKDIR/benchmarks/$PACK: run download_${PACK}_data.sh first" >&2; exit 1; }
mkdir -p "$MILABENCH_BASE/slurm"
# --export=ALL hands these to the job (not through --export=NAME=value: SELECT contains commas)
export MILABENCH_BASE WORKDIR SELECT CONFIG

# the reservation may not have started yet: show it so a pending job is not a surprise
scontrol show reservation "$RESERVATION" 2>/dev/null | head -2 || true

jobid=$(sbatch --parsable ${TEST_ONLY:+--test-only} \
    --job-name=slime-multinode \
    --reservation="$RESERVATION" --account="$ACCOUNT" \
    --nodes="$NODES" --ntasks-per-node=1 --exclusive --mem=0 \
    --gpus-per-node="$GPUS" \
    --time="$TIME" \
    --output="$MILABENCH_BASE/slurm/slime-multinode-%j.out" \
    --export=ALL \
    "$@" "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/slime_multinode.sbatch")
[ -n "${TEST_ONLY:-}" ] && exit 0   # --test-only submits nothing
echo "submitted job $jobid"

# follow the log (the file only exists once the job starts, so create it first); ctrl-c stops tailing, not the job
log=$MILABENCH_BASE/slurm/slime-multinode-${jobid%%;*}.out
touch "$log"
tail -f "$log"
