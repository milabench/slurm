#!/usr/bin/env bash
# Window block 15: G3 again (pass 2).
# See _block.sh for the environment (ACCOUNT, RESERVATION, NODES, TIME, EXCLUDE, NODELIST, AFTER).
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
exec "$here/_block.sh" G3 pass2 "$@"
