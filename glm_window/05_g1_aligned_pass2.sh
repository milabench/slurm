#!/usr/bin/env bash
# Window block 11: G1 again (pass 2). Same set as pass 1 and as G2 / G3: rerun all of them if it changed.
# See _block.sh for the environment (ACCOUNT, RESERVATION, NODES, TIME, EXCLUDE, NODELIST, AFTER).
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
exec "$here/_block.sh" G1 pass2 "$@"
