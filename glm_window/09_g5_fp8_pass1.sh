#!/usr/bin/env bash
# Window block 22: G5, FP8, otherwise = G1 (52-node set, aligned).
# See _block.sh for the environment (ACCOUNT, RESERVATION, NODES, TIME, EXCLUDE, NODELIST, AFTER).
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
exec "$here/_block.sh" G5 pass1 "$@"
