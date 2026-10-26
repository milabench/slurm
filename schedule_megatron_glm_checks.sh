#!/usr/bin/env bash
# Checks to run BEFORE the GLM-5 placement runs G1-G6 (scripts/analysis/GLM_RUNS.md, "Before the window"), to decide
# what to use afterwards. Megatron-LM, frozen experts, Rorqual.
#
#   plan HOSTLIST   login node, no GPU: the aligned / crossed / mixed / leaf node lists of the healthy nodes and how many
#                   expert groups of each cross a switch (does the generator give the counts of the table?)
#   small           one 4-node job, 7 benches one after the other (4 layers, PP1, experts frozen):
#                     check-mapping-4n   expert-group-to-host mapping: the log lists every group, its hosts and switches
#                     check-ep{4,8,16}-{bf16,fp8}-4n   EP size and precision: pipeline end to end, Fig. 10 small points,
#                                        `time_to_first_step` (run length), `params_trainable_fraction` (< 1: another model)
#   fit             one job on NODES (default 66) nodes: G1's configuration for 4 steps: does the full model fit on 208 GPUs
#                   at EP8 (peak memory in the report), and how long does a run take
#   all             small, then fit
#
# What to read in the job log / report afterwards:
#   [megatron-glm5] hosts in rank order / ep group N ... CROSSES   mapping: groups are the consecutive hosts we think
#   WARNING: the hosts run in another order than the placement      -> the node-order layouts mean nothing, fix first
#   expert groups crossing a switch: X / Y                         must match the table (G1 0, G2 all, G3 ~1 in 4)
#   a2a_time_crossing vs a2a_time_aligned                          Fig. 12 mechanism, if crossing groups exist
#   rate, peak_memory (report), time_to_first_step, params_trainable_fraction
#
#   ./schedule_megatron_glm_checks.sh small|fit|all [-- extra sbatch args]
#   ./schedule_megatron_glm_checks.sh plan HOSTLIST [--group-nodes 2]
#   env: SMALL_NODES (default 4)  NODES (fit, default 66)  TIME_SMALL (default 02:00:00)  TIME_FIT (default 01:00:00)
#        UPDATE (default 1)  MILABENCH_BASE  WORKDIR
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
MILABENCH_BASE=${MILABENCH_BASE:-/home/delaunay/links/scratch/shared}
WORKDIR=${WORKDIR:-$MILABENCH_BASE/src}

what=${1:?usage: $0 small|fit|all [-- sbatch args] | plan HOSTLIST}
shift

if [ "$what" = plan ]; then
    hosts=${1:?usage: plan HOSTLIST [--group-nodes N]}
    shift
    exec "$MILABENCH_BASE/venv/torch/bin/python" "$WORKDIR/benchmarks/megatron/placement.py" --hosts "$hosts" "$@"
fi
[ $# -gt 0 ] && [ "$1" = -- ] && shift

export PACK=megatron
export UPDATE=${UPDATE:-1}

small() {
    SELECT=megatron-glm5-check-mapping-4n,megatron-glm5-check-ep4-bf16-4n,megatron-glm5-check-ep8-bf16-4n,megatron-glm5-check-ep16-bf16-4n,megatron-glm5-check-ep4-fp8-4n,megatron-glm5-check-ep8-fp8-4n,megatron-glm5-check-ep16-fp8-4n \
    TIME=${TIME_SMALL:-02:00:00} "$here/schedule_slime_multinode.sh" "${SMALL_NODES:-4}" "$@"
}
fit() {
    SELECT=megatron-glm5-check-fit-52n \
    TIME=${TIME_FIT:-01:00:00} "$here/schedule_slime_multinode.sh" "${NODES:-66}" "$@"
}

case $what in
    small) small "$@" ;;
    fit)   fit "$@" ;;
    all)   NO_TAIL=1 small "$@"; fit "$@" ;;
    *) echo "unknown '$what' (plan|small|fit|all)" >&2; exit 2 ;;
esac
