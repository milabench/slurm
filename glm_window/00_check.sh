#!/usr/bin/env bash
# BEFORE the window (GLM_RUNS.md, "Before the window"): the checks that decide what to use afterwards.
#   1. login node, no GPU: do the node lists give the table's crossing counts (aligned 0, crossed all, mixed ~1 in 4)?
#        ./00_check.sh plan "<healthy full-GPU hostlist, main node first>"
#   2. 4-node job: expert-group-to-host mapping, EP4/8/16 x BF16/FP8 on a truncated model (small-scale points of Fig. 10,
#      run length, params_trainable_fraction)
#   3. 52-node set: does the full model fit on 208 GPUs at EP8, and how long does a run take?
#        ./00_check.sh small | fit | all
# Read the log: no "hosts run in another order" warning, "expert groups crossing a switch: X / Y" as in the table,
# peak memory / torchmem max_allocated of the fit run, time_to_first_step. If the model does not fit: ONE truncated
# depth for all the G runs (benchmarks/megatron/dev.yaml, _megatron_glm5_g), written in the summary's deviations.
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
export ACCOUNT=${ACCOUNT:-def-bengioy}
export RESERVATION=${RESERVATION:-MILABENCH2026}
what=${1:-all}
[ $# -gt 0 ] && shift
if [ "$what" = plan ]; then exec "$here/../schedule_megatron_glm_checks.sh" plan "$@"; fi
extra=()
[ -n "${EXCLUDE:-}" ]  && extra+=(--exclude="$EXCLUDE")
[ -n "${NODELIST:-}" ] && extra+=(--nodelist="$NODELIST")
exec "$here/../schedule_megatron_glm_checks.sh" "$what" -- "${extra[@]}" "$@"
