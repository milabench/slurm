#!/usr/bin/env bash
# Megatron-LM GLM-5 (frozen experts, torchrun, random init) on Rorqual: the 2-node smoke entry and its topology-ordered
# twin (same layout, nodes packed per leaf switch: compare `rate` and the placement report in the job log).
#   megatron-glm5-frozen-smoke-2n        2 nodes  8 GPUs, 4 layers, PP2 x EP4 x DP4, nodes as the scheduler gave them
#   megatron-glm5-frozen-smoke-2n-topo   2 nodes  same, nodes ordered by leaf switch
# Full size (744B, 16 nodes): SELECT=megatron-glm5-frozen-16n,megatron-glm5-frozen-16n-topo,megatron-glm5-frozen-16n-topo-pp \
#   ./schedule_megatron_glm5.sh   (N=16 then: NODES=16)
# One command, from a login node (internet): first download_megatron_data.sh (checkout, venv, slime / Megatron-LM
# clones, TE wheels, model config.json; cheap when already done), then the job is submitted.
#
#   ./schedule_megatron_glm5.sh [extra sbatch args]
#   env: SKIP_DOWNLOAD=1 (submit only)  NODES (default 2)  TIME (default 01:00:00)  UPDATE (default 1: fast-forward the checkout)  SELECT
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

export PACK=megatron
export SELECT=${SELECT:-megatron-glm5-frozen-smoke-2n,megatron-glm5-frozen-smoke-2n-topo}
export TIME=${TIME:-01:00:00}
export UPDATE=${UPDATE:-1}
[ "${SKIP_DOWNLOAD:-0}" = 1 ] || "$here/download_megatron_data.sh"
exec "$here/schedule_slime_multinode.sh" "${NODES:-2}" "$@"
