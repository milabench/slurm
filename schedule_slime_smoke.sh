#!/usr/bin/env bash
# Submit the slime smoke tests on N nodes of the MILABENCH2026 reservation (Rorqual).
#
#   ./schedule_slime_smoke.sh [N] [extra sbatch args]
#
# Run download_slime_data.sh first (login node): the compute nodes have no internet, so the job runs offline
# against $MILABENCH_BASE (venv, HF weights, datasets, all on the shared filesystem).
#
# The job (slime_smoke.sbatch, next to this script):
#   1. on the first node: `milabench prepare` for the smoke benches. It only has to do the torch_dist conversion of
#      Qwen2.5-0.5B (needs a GPU; skipped when already converted). SLIME_DOWNLOAD_ONLY is NOT set here.
#   2. on every node at once: `milabench run` of the smoke benches, one after the other (they are 1-GPU benches, so
#      each node is an independent sample: node-to-node spread is part of the result).
# Results: $MILABENCH_BASE/runs/slime-smoke.<jobid>.<node>   logs: $MILABENCH_BASE/slurm/slime-smoke-<jobid>.out
#
# env:  MILABENCH_BASE (default /home/delaunay/links/scratch/shared)   WORKDIR (default $MILABENCH_BASE/src, the checkout of download_slime_data.sh)
#       SELECT   benches to run (default: the 3 Qwen2.5-0.5B smoke entries)
#       RESERVATION / ACCOUNT / TIME / GPUS   default MILABENCH2026 / def-bengioy_gpu / 03:00:00 / 4 (GPUs per node)
#       UPDATE=1 fast-forward the checkout if it is behind origin/$BRANCH (default: fail)   BRANCH (default megatron)
#       TEST_ONLY=1   sbatch --test-only: where/when it would start, nothing is submitted
set -euo pipefail

MILABENCH_BASE=${MILABENCH_BASE:-/home/delaunay/links/scratch/shared}
NODES=${1:-1}
[ $# -gt 0 ] && shift
WORKDIR=${WORKDIR:-$MILABENCH_BASE/src}
RESERVATION=${RESERVATION:-MILABENCH2026}
ACCOUNT=${ACCOUNT:-def-bengioy_gpu}
TIME=${TIME:-03:00:00}
GPUS=${GPUS:-4}
SELECT=${SELECT:-slime-qwen2.5-0.5b-sft-smoke,slime-qwen2.5-0.5b-fineweb-smoke,slime-qwen2.5-0.5b-grpo-smoke}

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

[ -x "$WORKDIR/.venv/bin/milabench" ] || { echo "no milabench in $WORKDIR/.venv: run download_slime_data.sh first" >&2; exit 1; }
[ -d "$MILABENCH_BASE/data/slime/models" ] || { echo "no $MILABENCH_BASE/data/slime/models: run download_slime_data.sh first" >&2; exit 1; }
mkdir -p "$MILABENCH_BASE/slurm"
# --export=ALL hands these to the job (not through --export=NAME=value: SELECT contains commas)
export MILABENCH_BASE WORKDIR SELECT

# the reservation may not have started yet: show it so a pending job is not a surprise
scontrol show reservation "$RESERVATION" 2>/dev/null | head -2 || true

jobid=$(sbatch --parsable ${TEST_ONLY:+--test-only} \
    --job-name=slime-smoke \
    --reservation="$RESERVATION" --account="$ACCOUNT" \
    --nodes="$NODES" --ntasks-per-node=1 --exclusive --mem=0 \
    --gpus-per-node="$GPUS" \
    --time="$TIME" \
    --output="$MILABENCH_BASE/slurm/slime-smoke-%j.out" \
    --export=ALL \
    "$@" "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/slime_smoke.sbatch")
[ -n "${TEST_ONLY:-}" ] && exit 0   # --test-only submits nothing
echo "submitted job $jobid"

# follow the log (the file only exists once the job starts, so create it first); ctrl-c stops tailing, not the job
log=$MILABENCH_BASE/slurm/slime-smoke-${jobid%%;*}.out
touch "$log"
tail -f "$log"
