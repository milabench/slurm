#!/usr/bin/env bash
# Shared by the numbered scripts: submit one GLM-5 block (one run of scripts/analysis/GLM_RUNS.md) on the reservation.
#   _block.sh <G1..G6> <tag>        tag = what the numbered script is, e.g. pass1 (goes in the job name)
#
# env (all optional):
#   ACCOUNT      Slurm account                         (default def-bengioy)
#   RESERVATION  reservation                           (default MILABENCH2026)
#   NODES        nodes of the allocation               (default 66: every healthy full-GPU node; the entry picks and
#                orders its own set from it, 52 for G1-G3 / G5 / G6)
#   TIME         wall time of the block                (default 00:30:00; the document assumes ~15 min of run, plus
#                startup: measure it with the checks and set it here)
#   EXCLUDE      hostlist to leave out (unhealthy nodes after the window health check)  -> sbatch --exclude
#   NODELIST     hostlist the job must use          -> sbatch --nodelist. Use the SAME one for G1, G2, G3, G5, G6:
#                the comparison is only valid on one set of nodes
#   AFTER        job id this block waits for (afterany: a failed block does not stop the next)
#   NO_TAIL      1 = do not follow the log
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
run=${1:?usage: _block.sh G1..G6 tag}
tag=${2:?usage: _block.sh G1..G6 tag}

export ACCOUNT=${ACCOUNT:-def-bengioy}
export RESERVATION=${RESERVATION:-MILABENCH2026}
export TIME=${TIME:-00:30:00}
export NODES=${NODES:-66}

args=(--job-name="glm5-$run-$tag")
[ -n "${EXCLUDE:-}" ]  && args+=(--exclude="$EXCLUDE")
[ -n "${NODELIST:-}" ] && args+=(--nodelist="$NODELIST")
[ -n "${AFTER:-}" ]    && args+=(--dependency="afterany:$AFTER")

echo "[window] $run ($tag): account=$ACCOUNT reservation=$RESERVATION nodes=$NODES time=$TIME ${EXCLUDE:+exclude=$EXCLUDE }${NODELIST:+nodelist=$NODELIST }${AFTER:+after=$AFTER}"
exec "$here/../schedule_megatron_glm_groups.sh" "$run" -- "${args[@]}"
