#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_precondition_wait.sh
#   S2 on-card completion (kernels/resident_precondition.h): a descriptor issued a window early
#   carries "producer P has completed C", and its READER must check that before it reads a
#   single input. A reader that skips the call runs on its producer's unfinished output and
#   returns a plausible wrong answer with every kernel completing — nothing on the card says
#   so. This proves, from the sources, for every resident reader any program ships:
#     1. it includes resident_precondition.h;
#     2. kernel_main calls ftq_precondition_wait(&ctx) exactly once, inside the descriptor
#        loop: after ftq_wait_for_work(&ctx, kFieldHead) and before the first line that reads
#        the descriptor's operands or issues a NoC read (ftq_parameters, read_one_*,
#        execute_simple_opcode, ft_tensix_noc_read, a descriptor word past the opcode);
#     3. a refusal leaves the loop (`running = false;` follows the call);
#   and that the helper itself issues no NoC write (reads only).
#
#   The reader list is every DM0 kernel named `resident_*_reader.c` in the programs header,
#   so a new reader cannot be missed; the puller (the issuer) and resident_queue.c (the
#   BRISC-only queue kernel, no inputs) are the only resident DM0 kernels exempt.
#
#   bash tt/tools/check_precondition_wait.sh [kernels directory] [programs header]
#     (defaults: tt/kernels, tt/ft_tt_programs.h)
set -euo pipefail
TT=$(cd "$(dirname "$0")/.." && pwd)
K=${1:-$TT/kernels}
H=${2:-$TT/ft_tt_programs.h}

fail=0
refuse() { echo "check_precondition_wait: REFUSED — $*"; fail=1; }

readers=$(grep -oE '\{"resident_[a-z_0-9]+_reader\.c", FT_TT_PROC_DM0' "$H" | sed -E 's/\{"([^"]+)".*/\1/' | sort -u)
[ -n "$readers" ] || { echo "check_precondition_wait: no resident readers in $H"; exit 1; }

# The helper: reads only.
if grep -nE 'noc_write|inline_dword' "$K/resident_precondition.h" "$K/resident_precondition_protocol.h" |
    grep -vE '^[^:]+:[0-9]+:[[:space:]]*(\*|//|/\*)' | grep -q .; then
    refuse "resident_precondition*.h issues a NoC write (the wait must read only)"
fi

for r in $readers; do
    f=$K/$r
    [ -f "$f" ] || { refuse "$r is in the programs header but not in $K"; continue; }
    grep -q '^#include "resident_precondition.h"' "$f" || refuse "$r does not include resident_precondition.h"
    # kernel_main to the end of the file, comments dropped, numbered by original line.
    body=$(awk '/^void kernel_main\(void\)/ { on = 1 } on { print NR": "$0 }' "$f" |
        grep -vE '^[0-9]+:[[:space:]]*(\*|//|/\*)')
    calls=$(printf '%s\n' "$body" | grep -c 'ftq_precondition_wait(&ctx)' || true)
    [ "$calls" = 1 ] || { refuse "$r: kernel_main calls ftq_precondition_wait $calls times (want 1)"; continue; }
    line() { printf '%s\n' "$body" | grep -E "$1" | head -1 | cut -d: -f1; }
    wait_at=$(line 'ftq_wait_for_work\(&ctx, kFieldHead\)')
    call_at=$(line 'ftq_precondition_wait\(&ctx\)')
    first_read=$(printf '%s\n' "$body" | awk -F: -v from="$wait_at" '$1 > from' |
        grep -E 'ftq_parameters\(|read_one_|execute_simple_opcode\(|ft_tensix_noc_read|noc_issue_read|descriptor\[kDescA|descriptor\[kDescToken' |
        head -1 | cut -d: -f1)
    if [ -z "$wait_at" ] || [ -z "$first_read" ]; then
        refuse "$r: no descriptor loop found (wait ${wait_at:-none}, first read ${first_read:-none})"
        continue
    fi
    if [ "$call_at" -le "$wait_at" ] || [ "$call_at" -ge "$first_read" ]; then
        refuse "$r: the wait is at line $call_at, not between the head wait (line $wait_at) and the first input read (line $first_read)"
    fi
    after=$(awk -v at="$call_at" 'NR > at && NR <= at + 3' "$f")
    printf '%s\n' "$after" | grep -q 'running = false;' || refuse "$r: a refused precondition does not leave the loop"
    echo "check_precondition_wait: $r waits at line $call_at (head wait $wait_at, first input read $first_read)"
done

[ "$fail" = 0 ] || exit 1
echo "check_precondition_wait: every resident reader ($(echo "$readers" | wc -l | tr -d ' ')) checks its precondition before its first input read"
