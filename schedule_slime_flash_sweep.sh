#!/usr/bin/env bash
# One 8-node job running the GLM-5.3-Flash SFT sweep (see the comment above _glm5_flash_sft in dev.yaml), in this order:
#   slime-glm5.3-flash-sft-frozen           8 nodes  32 GPUs, experts frozen, 10 rollouts: the reference number
#   slime-glm5.3-flash-sft-frozen-4n        4 nodes  16 GPUs  } how few GPUs the frozen pipeline needs
#   slime-glm5.3-flash-sft-frozen-2n        2 nodes   8 GPUs  } (2n may not fit, which is the answer)
#   slime-glm5.3-flash-sft-frozen-routing   8 nodes  router forced to balance for the first 4 rollouts, then real routing
#   slime-glm5.3-flash-sft-unfrozen         8 nodes  same as the first, nothing frozen: what freezing saves (may OOM)
# Each bench takes the first num_machines nodes of the allocation. A bench that fails does not stop the next one.
# Every bench loads the 328 GB HF checkpoint again (about 25 minutes of start-up in the first run), hence the long default.
#
#   ./schedule_slime_flash_sweep.sh [extra sbatch args]
#   env: TIME (default 07:00:00)  UPDATE (default 1: fast-forward the checkout, push dev.yaml first)  SELECT (override the list)
# Everything else (preflight, reservation, log following) is schedule_slime_multinode.sh.
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

export SELECT=${SELECT:-slime-glm5.3-flash-sft-frozen,slime-glm5.3-flash-sft-frozen-4n,slime-glm5.3-flash-sft-frozen-2n,slime-glm5.3-flash-sft-frozen-routing,slime-glm5.3-flash-sft-unfrozen}
export TIME=${TIME:-07:00:00}
export UPDATE=${UPDATE:-1}
exec "$here/schedule_slime_multinode.sh" 8 "$@"
