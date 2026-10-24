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

[ -x "$WORKDIR/.venv/bin/milabench" ] || { echo "no milabench in $WORKDIR/.venv: run download_slime_data.sh first" >&2; exit 1; }
[ -d "$MILABENCH_BASE/data/slime/models" ] || { echo "no $MILABENCH_BASE/data/slime/models: run download_slime_data.sh first" >&2; exit 1; }
mkdir -p "$MILABENCH_BASE/slurm"
# --export=ALL hands these to the job (not through --export=NAME=value: SELECT contains commas)
export MILABENCH_BASE WORKDIR SELECT

# the reservation may not have started yet: show it so a pending job is not a surprise
scontrol show reservation "$RESERVATION" 2>/dev/null | head -2 || true

sbatch ${TEST_ONLY:+--test-only} \
    --job-name=slime-smoke \
    --reservation="$RESERVATION" --account="$ACCOUNT" \
    --nodes="$NODES" --ntasks-per-node=1 --exclusive --mem=0 \
    --gpus-per-node="$GPUS" \
    --time="$TIME" \
    --output="$MILABENCH_BASE/slurm/slime-smoke-%j.out" \
    --export=ALL \
    "$@" "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/slime_smoke.sbatch"
