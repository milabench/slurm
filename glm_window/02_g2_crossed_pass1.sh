#!/usr/bin/env bash
# Window block 3: G2, EP8 crossed, same 52 nodes as G1, 26 / 26 groups cross.
# Same node set as G1: use the same NODELIST / EXCLUDE.
# See _block.sh for the environment (ACCOUNT, RESERVATION, NODES, TIME, EXCLUDE, NODELIST, AFTER).
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
exec "$here/_block.sh" G2 pass1 "$@"
