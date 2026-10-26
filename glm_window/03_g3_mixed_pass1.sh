#!/usr/bin/env bash
# Window block 5: G3, EP8 mixed (3 swaps), same 52 nodes, 6 / 26 groups cross (~1 in 4).
# See _block.sh for the environment (ACCOUNT, RESERVATION, NODES, TIME, EXCLUDE, NODELIST, AFTER).
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
exec "$here/_block.sh" G3 pass1 "$@"
