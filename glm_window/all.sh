#!/usr/bin/env bash
# Submit the blocks 01..11 in the order of the document, each waiting for the previous one (afterany: a failed block does
# not stop the next). Do the checks first (00_check.sh). Use the same NODELIST / EXCLUDE for all of them.
#
#   ./all.sh [first [last]]        numbers of the scripts, default 01 11  (e.g. ./all.sh 01 04 = pass 1 of G1-G4)
#   ./all.sh --list                what would be submitted
#
# NOTE: the window interleaves other experiments between these blocks (the document's block numbers are 1, 3, 5, 7, 11,
# 13, ...): chained like this the GLM blocks run back to back. To follow the queue, submit the numbered scripts one at a
# time at their turn instead.
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
first=${1:-01}; last=${2:-11}
if [ "${1:-}" = --list ]; then first=01; last=11; fi
scripts=()
for f in "$here"/[0-9][0-9]_*.sh; do
    n=$(basename "$f" | cut -c1-2)
    [ "$n" = 00 ] && continue
    [ "$((10#$n))" -ge "$((10#$first))" ] && [ "$((10#$n))" -le "$((10#$last))" ] && scripts+=("$f")
done
if [ "${1:-}" = --list ]; then printf '%s\n' "${scripts[@]##*/}"; exit 0; fi

after=${AFTER:-}
for f in "${scripts[@]}"; do
    echo "== ${f##*/}"
    out=$(AFTER=$after NO_TAIL=1 "$f" 2>&1 | tee /dev/stderr)
    id=$(printf '%s\n' "$out" | sed -n 's/^submitted job \([0-9]*\).*/\1/p' | tail -1)
    [ -n "$id" ] || { echo "no job id from ${f##*/}, stopping" >&2; exit 1; }
    after=$id
done
