#!/usr/bin/env bash
# Job script of schedule_megatron.sh: multi-node Megatron GLM-5 benches on 4-GPU nodes (torchrun over srun, no Ray).
# Needs MILABENCH_BASE, WORKDIR and SELECT in the environment (--export=ALL).
#   `milabench slurm system` describes the allocation, then ONE `milabench run` over all of SELECT: each bench takes
#   the first num_machines nodes of the system. Nothing to prepare on the nodes: the HF config was staged from the
#   login node, the weights and the data are random.
set -euo pipefail
: "${MILABENCH_BASE:?}" "${WORKDIR:?}" "${SELECT:?}"

export PATH=$WORKDIR/.venv/bin:$PATH
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
config=$WORKDIR/benchmarks/megatron/dev.yaml
system=$MILABENCH_BASE/slurm/system.$SLURM_JOB_ID.yaml
echo "[megatron] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST select=$SELECT"

milabench slurm system > "$system"
cat "$system"
# the switches the placement logic will see (a node without an entry there is "unknown": no topology ordering)
scontrol show topology 2>/dev/null | head -20 || echo "[megatron] scontrol show topology is not available" >&2

milabench run --config "$config" --system "$system" --select "$SELECT" \
    --run-name "megatron.$SLURM_JOB_ID" \
    || echo "[megatron] run exited with $?" >&2

echo "[megatron] done: $MILABENCH_BASE/runs/megatron.$SLURM_JOB_ID"
