#!/usr/bin/env bash
# Window block 7: G4, EP8, every healthy node (an even count), PP3 (PP13 does not divide 264 GPUs).
# Runs alone: it needs every node. Record the node count and the crossing count.
# See _block.sh for the environment (ACCOUNT, RESERVATION, NODES, TIME, EXCLUDE, NODELIST, AFTER).
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
exec "$here/_block.sh" G4 pass1 "$@"
