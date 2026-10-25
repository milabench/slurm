#!/usr/bin/env bash
# Job script of schedule_slime_multinode.sh: multi-node slime benches on 4-GPU nodes (Ray through RayCluster).
# Needs MILABENCH_BASE, WORKDIR and SELECT in the environment (--export=ALL).
#   1. first node: `milabench prepare` (Qwen3-30B torch_dist conversion on 4 GPUs; Flash is loaded from HF, no
#      conversion). Offline: the weights were downloaded from a login node.
#   2. `milabench slurm system` describes the allocation, then ONE `milabench run` over all of SELECT: each bench
#      takes the first num_machines nodes of the system (2 for the Qwen ones, 8 for GLM-5.3-Flash).
set -euo pipefail
: "${MILABENCH_BASE:?}" "${WORKDIR:?}" "${SELECT:?}"

export PATH=$WORKDIR/.venv/bin:$PATH
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SLIME_DOWNLOAD_ONLY
# Ray (2.5x+) authenticates its RPCs with a token that every node must share (head, workers, raywait, the driver):
# one random token per job, exported here so srun hands it to every task. Not written to disk.
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN=${RAY_AUTH_TOKEN:-$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')}
export SLIME_CONVERT_NPROC=${SLIME_CONVERT_NPROC:-4}   # one node: 4 GPUs (dev.yaml says 8)
config=$WORKDIR/benchmarks/slime/dev.yaml
system=$MILABENCH_BASE/slurm/system.$SLURM_JOB_ID.yaml
echo "[multinode] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST select=$SELECT"

milabench slurm system > "$system"
cat "$system"

# 1. conversion (GPU), once. A failure here is not fatal: the benches that do not need it can still run.
srun --nodes=1 --ntasks=1 --nodelist="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" \
    milabench prepare --config "$config" --select "$SELECT" \
    || echo "[multinode] prepare failed, see above" >&2

# 2. the benches
milabench run --config "$config" --system "$system" --select "$SELECT" \
    --run-name "slime-multinode.$SLURM_JOB_ID" \
    || echo "[multinode] run exited with $?" >&2

echo "[multinode] done: $MILABENCH_BASE/runs/slime-multinode.$SLURM_JOB_ID"
