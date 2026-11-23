#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_group_done_word.sh
#   Control word 56 (kFieldGroupDone, control byte 0xE0) is the done-sequence a member 0 group
#   pull reads (kernels/resident_group_pull.h). It must have ONE writer on every core: the
#   queue's writer kernel, storing the descriptor's a5 just before it publishes completion.
#   Any other store, add or clock in that word can satisfy the pull before the member has
#   finished — member 0 then posts early, the host reuses the member's parameter slot under a
#   running descriptor, and the member wedges (S1 card run: the gdn reader's CONV barrier clock
#   lived in word 56).
#
#   Refuses, in the kernel sources (comments ignored):
#     1. word 56 named by number: an index, enum or define of 56, `56 * 4`, or `+ 0xE0`,
#        anywhere but kFieldGroupDone's own definition;
#     2. kFieldGroupDone used outside the writers and the pull itself.
#
#   bash tt/tools/check_group_done_word.sh [kernels directory]   (default: tt/kernels)
set -euo pipefail
K=${1:-$(cd "$(dirname "$0")/../kernels" && pwd)}
WRITERS='resident_queue_protocol\.h|resident_group_pull\.h|resident_matmul_writer\.c|resident_vector_writer\.c|resident_attention_writer\.c|resident_gdn_writer\.c|resident_place_writer\.c|resident_norm_role_writer\.c'

# file:line:text, minus lines that are comments.
code() { grep -nE "$1" "$K"/*.c "$K"/*.h | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(\*|//|/\*)' || true; }

numbered=$(code '(\[|=)[[:space:]]*56u?[[:space:]]*(\]|[,;}])|define[[:space:]]+[A-Za-z_0-9]+[[:space:]]+56u?\b|\+[[:space:]]*0xE0u?\b|\b56u?[[:space:]]*\*[[:space:]]*4' |
    grep -vE '/resident_queue_protocol\.h:[0-9]+:enum \{ kFieldGroupDone = 56 \};' || true)
foreign=$(code 'kFieldGroupDone' | grep -vE "/($WRITERS):" || true)

if [ -n "$numbered$foreign" ]; then
    echo "check_group_done_word: REFUSED — control word 56 is the group-pull done-sequence:"
    [ -n "$numbered" ] && printf '  names word 56 by number: %s\n' "$numbered"
    [ -n "$foreign" ] && printf '  kFieldGroupDone outside a writer: %s\n' "$foreign"
    exit 1
fi
echo "check_group_done_word: word 56 has one writer per core in $K"
