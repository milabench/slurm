#!/usr/bin/env bash
# One 16-node job: GLM-5.3 744B (frozen experts, 64 GPUs) + the GLM-5.3-Flash throughput-scaling checks + a second sample
# of the earlier Flash sweep. Each bench takes the first num_machines nodes of the allocation, a failing one does not
# stop the next. In this order (most informative first):
#   slime-glm5-sft-frozen-16n                   16 nodes  744B, PP8 x DP8, EP8, raw FP8 HF load (untested), 6 rollouts
#   slime-glm5.3-flash-sft-frozen-gbs64/-gbs128  8 nodes  global batch 64 / 128 (is 8 nodes batch-starved at 32?)
#   slime-glm5.3-flash-sft-frozen-4n-gbs64/-gbs128 4 nodes  same on 16 GPUs
#   slime-glm5.3-flash-sft-frozen, -4n, -routing  second sample of the first sweep (compare with job 25153928)
# Every Flash bench reloads the 328 GB checkpoint (about 25 min of start-up each); the 744B one reads 755 GB of FP8.
# The default time is long: trim SELECT (comma separated) or TIME to what the reservation has left
# (scontrol show reservation MILABENCH2026, it ends 2026-10-27 04:00).
#
#   ./schedule_slime_glm_all.sh [extra sbatch args]
#   env: TIME (default 12:00:00)  UPDATE (default 1: fast-forward the checkout, push dev.yaml first)  SELECT
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

export SELECT=${SELECT:-slime-glm5-sft-frozen-16n,slime-glm5.3-flash-sft-frozen-gbs64,slime-glm5.3-flash-sft-frozen-gbs128,slime-glm5.3-flash-sft-frozen-4n-gbs64,slime-glm5.3-flash-sft-frozen-4n-gbs128,slime-glm5.3-flash-sft-frozen,slime-glm5.3-flash-sft-frozen-4n,slime-glm5.3-flash-sft-frozen-routing}
export TIME=${TIME:-12:00:00}
export UPDATE=${UPDATE:-1}
exec "$here/schedule_slime_multinode.sh" 16 "$@"
