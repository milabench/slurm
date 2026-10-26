#!/usr/bin/env bash
# GLM-5 expert placement runs G1-G6 (scripts/analysis/GLM_RUNS.md), Megatron-LM, frozen experts, Rorqual.
#   G1 EP8 aligned      52 nodes   0 / 26 expert groups cross a switch
#   G2 EP8 crossed      52 nodes   26 / 26
#   G3 EP8 mixed        52 nodes   6 / 26 (3 swaps)
#   G4 EP8 every node   66 nodes   allocation order, PP3 (PP13 does not divide 264 GPUs): record the crossing count
#   G5 FP8              52 nodes   = G1
#   G6 EP16             52 nodes   = G1's set, default order
# The job takes every healthy node it is given (NODES, default 66); the entry picks and orders its own set from the
# allocation and `scontrol show topology` (benchmarks/megatron/placement.py), so G1-G3 and G5-G6 run on the same set
# only if the allocation is the same: submit them with the same --nodelist / exclusions, and never mix sets.
# The 14 nodes outside the 52-node set sit idle in the job.
#
#   ./schedule_megatron_glm_groups.sh G1 [G2 ...] [-- extra sbatch args]      one job per run, same arguments
#   ./schedule_megatron_glm_groups.sh plan HOSTLIST [--group-nodes 2]         node lists + crossing counts (login node)
#   env: NODES (default 66)  WORKDIR (default $MILABENCH_BASE/src)  TIME (default 00:45:00)  UPDATE (default 1)  MILABENCH_BASE
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
MILABENCH_BASE=${MILABENCH_BASE:-/home/delaunay/links/scratch/shared}

declare -A entry=(
    [G1]=megatron-glm5-g1-ep8-aligned [G2]=megatron-glm5-g2-ep8-crossed [G3]=megatron-glm5-g3-ep8-mixed
    [G4]=megatron-glm5-g4-ep8-all-nodes [G5]=megatron-glm5-g5-fp8 [G6]=megatron-glm5-g6-ep16
)

if [ "${1:-}" = plan ]; then
    shift
    hosts=${1:?usage: plan HOSTLIST [--group-nodes N]}
    shift
    exec "$MILABENCH_BASE/venv/torch/bin/python" "${WORKDIR:-$MILABENCH_BASE/src}/benchmarks/megatron/placement.py" --hosts "$hosts" "$@"
fi

runs=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do runs+=("$1"); shift; done
[ $# -gt 0 ] && shift
[ ${#runs[@]} -gt 0 ] || { echo "usage: $0 G1 [G2 ...] [-- sbatch args] | plan HOSTLIST" >&2; exit 2; }

export PACK=megatron
export TIME=${TIME:-00:45:00}
export UPDATE=${UPDATE:-1}
[ ${#runs[@]} -gt 1 ] && export NO_TAIL=1
for run in "${runs[@]}"; do
    [ -n "${entry[$run]:-}" ] || { echo "unknown run $run (G1-G6)" >&2; exit 2; }
    echo "== $run: ${entry[$run]}"
    SELECT=${entry[$run]} "$here/schedule_slime_multinode.sh" "${NODES:-66}" "$@"
done
