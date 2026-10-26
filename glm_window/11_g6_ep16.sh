#!/usr/bin/env bash
# Window block 35 (P3, only if time remains): G6, EP16 (expert group = 4 nodes), G1's set, default order.
# Record the crossing count.
# See _block.sh for the environment (ACCOUNT, RESERVATION, NODES, TIME, EXCLUDE, NODELIST, AFTER).
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
exec "$here/_block.sh" G6 pass1 "$@"
