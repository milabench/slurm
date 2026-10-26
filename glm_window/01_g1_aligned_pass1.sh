#!/usr/bin/env bash
# Window block 1: G1, EP8 aligned, 52-node set, 0 / 26 expert groups cross a switch.
# GLM blocks come first in the window (complex, more likely to fail, more time to fix).
# See _block.sh for the environment (ACCOUNT, RESERVATION, NODES, TIME, EXCLUDE, NODELIST, AFTER).
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
exec "$here/_block.sh" G1 pass1 "$@"
