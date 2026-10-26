#!/usr/bin/env bash
# One 16-node job for GLM-5.3 744B (frozen experts), after the first 64-GPU run (614 tokens/s, batch 32, 6 rollouts):
#   slime-glm5-sft-frozen-12n-gbs64    12 nodes  48 GPUs, PP6 x DP8, EP8   fewer GPUs: does it still fit?
#   slime-glm5-sft-frozen-10n-gbs64    10 nodes  40 GPUs, PP5 x DP8, EP8   (every stage <= 16 layers, like the 16-node peak stage)
#   slime-glm5-sft-frozen-16n-gbs64    16 nodes  64 GPUs, batch 64, 10 rollouts: the better throughput number
#   slime-glm5-sft-frozen-16n-gbs128   16 nodes  64 GPUs, batch 128
# Each bench reloads the 755 GB FP8 checkpoint (about 80 minutes before the first step), a failing one does not stop the
# next, so the whole list is about 6-7 hours. Trim SELECT or TIME to what the reservation has left
# (scontrol show reservation MILABENCH2026, it ends 2026-10-27 04:00); the order is by information per hour.
#
#   ./schedule_slime_glm744.sh [extra sbatch args]
#   env: TIME (default 08:00:00)  UPDATE (default 1: fast-forward the checkout, push dev.yaml first)  SELECT
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

export SELECT=${SELECT:-slime-glm5-sft-frozen-12n-gbs64,slime-glm5-sft-frozen-10n-gbs64,slime-glm5-sft-frozen-16n-gbs64,slime-glm5-sft-frozen-16n-gbs128}
export TIME=${TIME:-08:00:00}
export UPDATE=${UPDATE:-1}
exec "$here/schedule_slime_multinode.sh" 16 "$@"
