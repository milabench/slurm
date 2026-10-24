#!/usr/bin/env bash
# Job script of schedule_slime_smoke.sh (resources, reservation and output are given on its sbatch command line).
# Needs MILABENCH_BASE, WORKDIR and SELECT in the environment (--export=ALL).
#   1. first node: `milabench prepare` (torch_dist conversion of Qwen2.5-0.5B, needs a GPU; skipped when done)
#   2. every node: `milabench run` of the smoke benches, one after the other
set -euo pipefail
: "${MILABENCH_BASE:?}" "${WORKDIR:?}" "${SELECT:?}"

export PATH=$WORKDIR/.venv/bin:$PATH
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
# no internet on the compute nodes: use what the login-node download left on the shared filesystem
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SLIME_DOWNLOAD_ONLY
export CONFIG=$WORKDIR/benchmarks/slime/dev.yaml
echo "[smoke] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST select=$SELECT"

# 1. torch_dist conversion (GPU), once
srun --nodes=1 --ntasks=1 --nodelist="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" \
    milabench prepare --config "$CONFIG" --select "$SELECT"

# 2. the smoke benches on every node. A failing node does not stop the others, the job reports it at the end.
srun --ntasks-per-node=1 bash -c '
    node=$(hostname -s)
    milabench run --config "$CONFIG" --select "$SELECT" --run-name "slime-smoke.$SLURM_JOB_ID.$node"
    echo "[smoke] $node exit=$?"
' || echo "[smoke] some node failed, see above" >&2

echo "[smoke] done: $MILABENCH_BASE/runs/slime-smoke.$SLURM_JOB_ID.*"
