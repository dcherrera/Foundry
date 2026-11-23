#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# The broadcast partition's MUTATION BATTERY.
#
# Every check in test_group_broadcast_partition.c is re-run against a build in
# which exactly ONE property of the code under test has been broken, and each
# damaged build MUST go red. A gate never seen failing is not evidence — eleven
# vacuous gates have been caught on this project, two of them inside one round.
#
# What each arm is really asking: could this suite have told the difference
# between a partition that keeps every packet inside the four-destination hazard
# limit and inside the group, and one that does not? Both failure modes are
# SILENT on silicon — a piece one column too wide writes into another residency's
# core, a piece one column too narrow leaves a member computing from the previous
# block with no error anywhere — so the answer has to come from here.
#
# Each mutation works on a SCRATCH COPY of tt/ with its own build directory, so
# the real tree is never edited and GNU make 3.81's one-second mtime granularity
# cannot serve a stale binary (coding-best-practices.md §9). The whole-program
# link means there are no object files to go stale either.
#
# It runs the UNMUTATED build first. Without that control, "every mutation went
# red" would be consistent with a suite that fails unconditionally.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
TT_DIR="$(cd "$HERE/.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}/ft_tt_broadcast_partition_gates.$$"
PASS=0
FAIL=0

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

# $1 = name, $2 = file to patch (relative to tt/), $3 = sed expression, $4 = want,
# $5 = optional make variables (e.g. SANITIZE=1)
run_arm() {
    local name="$1" file="$2" expr="$3" want="$4" makevars="${5:-}"
    local root="$SCRATCH/arm"
    local rc out

    rm -rf "$root"
    mkdir -p "$root"
    cp -R "$TT_DIR/." "$root/" 2>/dev/null
    rm -rf "$root"/tests/build* "$root"/tools/build "$root"/build 2>/dev/null

    if [ -n "$expr" ]; then
        if ! sed -i.bak "$expr" "$root/$file"; then
            echo "  ERROR: could not apply the mutation to $file"
            FAIL=$((FAIL + 1))
            return
        fi
        if cmp -s "$root/$file" "$root/$file.bak"; then
            # A mutation that changed nothing would produce a GREEN run and be
            # scored as "the gate did not catch it". Refuse instead.
            echo "  ERROR: the mutation did not change $file — the pattern has drifted"
            FAIL=$((FAIL + 1))
            return
        fi
    fi

    # shellcheck disable=SC2086
    out=$(cd "$root/tests" && make -f Makefile.broadcastpartition $makevars check 2>&1)
    rc=$?

    if [ "$want" = "green" ]; then
        if [ $rc -eq 0 ]; then
            echo "  ok    CONTROL (unmutated) passes"
            PASS=$((PASS + 1))
        else
            echo "  FAIL  CONTROL (unmutated) did NOT pass — every arm below is meaningless"
            echo "$out" | tail -20
            FAIL=$((FAIL + 1))
        fi
        return
    fi

    if [ $rc -ne 0 ]; then
        echo "  ok    RED as required: $name"
        echo "$out" | grep '^  FAIL' | sed 's/^/          /' | head -4
        PASS=$((PASS + 1))
    else
        echo "  FAIL  the suite stayed GREEN with: $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== broadcast partition mutation battery ==="
run_arm "control" "" "" green

# --- the hazard limit itself ------------------------------------------------
# THE ARM THAT MATTERS MOST. If the chunking is removed the partition emits one
# packet per row whatever its width, which is exactly the 5-destination broadcast
# that wedges a core at its next firmware boot.
run_arm "the row is never chunked — one packet per row, whatever its width" \
    ft_tt_group_broadcast_partition.c \
    's/const uint32_t chunk = remaining < max_destinations ? remaining : max_destinations;/const uint32_t chunk = remaining;/' red

run_arm "a caller may ask for more than four destinations per packet" \
    ft_tt_group_broadcast_partition.c \
    's/max_destinations > FT_TT_GROUP_BROADCAST_HAZARD_LIMIT/max_destinations > 64u/' red

run_arm "the chunk is one destination too wide" \
    ft_tt_group_broadcast_partition.c \
    's/x + chunk - 1u, leader/x + chunk, leader/' red

# --- the leader's own cell --------------------------------------------------
# The leader is inside the group's bounding box. A partition that forgets to
# remove it broadcasts into the leader's own A block while the leader is reading
# it, and the answer is wrong with every kernel completing.
run_arm "the leader's row is emitted whole, leader cell included" \
    ft_tt_group_broadcast_partition.c \
    's/if (y != leader.y)/if (1)/' red

run_arm "only the run LEFT of the leader is emitted" \
    ft_tt_group_broadcast_partition.c \
    's/if (leader.x < group->x_high)/if (0)/' red

run_arm "only the run RIGHT of the leader is emitted" \
    ft_tt_group_broadcast_partition.c \
    's/if (leader.x > group->x_low) {/if (0) {/' red

# --- the coverage and count checks ------------------------------------------
# THE ISOLATING ARM for `destination_total`. Deleting its CHECK is not enough: the
# coverage test catches every shape this suite builds, so the check reads as redundant.
# Breaking the SUM instead leaves coverage and every corner correct and moves only the
# number the leader's acknowledgement accounting is built on — which is the one thing
# nothing else here can see.
run_arm "the destination total counts packets instead of destinations" \
    ft_tt_group_broadcast_partition.c \
    's/partition->destination_total += rectangle.destination_count;/partition->destination_total += 1u;/' red

run_arm "a piece is counted but never appended" \
    ft_tt_group_broadcast_partition.c \
    's/partition->rectangle\[partition->rectangle_count\] = rectangle;/(void)rectangle;/' red

# THE CAPACITY BOUND IS A MEMORY-SAFETY PROPERTY, NOT A BEHAVIOURAL ONE, and saying so is
# the result rather than a workaround. Relaxing it does NOT make the partition claim
# success: the destination-total and coverage checks still refuse, so a plain build cannot
# tell the difference and the arm reads GREEN. The only observable difference is the write
# past the piece array, so the arm is run under AddressSanitizer, where it is a
# stack-buffer-overflow.
#
# So the arm SHRINKS THE CAPACITY instead of removing its check: three pieces cannot hold
# a 4x4 group's four rows, and the suite must then report a refusal rather than a truncated
# plan — a truncated plan leaves members reading the previous block with nothing reporting
# an error. Well defined, no undefined behaviour, and it is the property that actually
# matters: the bound is REPORTED, never silently applied.
#
# Two mutations were tried first and are recorded because each is a trap. Deleting the
# bound outright, and loosening it by one, both let the loop overwrite `rectangle_count` —
# its own counter — with rectangle data, and the arm then SPINS instead of going red. An
# unbounded mutation is a hang, not a result; it is the same rule as "mutate a length to a
# wrong NON-ZERO value".
run_arm "the piece array holds only three, so a 4x4 group does not fit" \
    ft_tt_group_broadcast_partition.h \
    's/#define FT_TT_GROUP_BROADCAST_MAX_RECTANGLES 8u/#define FT_TT_GROUP_BROADCAST_MAX_RECTANGLES 3u/' red

# --- what the partition inherits from the rectangle arithmetic --------------
# NOT AN ARM, and it is worth writing down why. Deleting ft_tt_multicast.c's pigeonhole
# fill check leaves this suite GREEN, because the same non-rectangular member set is then
# refused a second time by this module's own destination-total check — the bounding box of
# a set that does not fill it always holds more receivers than the set has members. That is
# defence in depth working as intended, not a vacuous gate: the fill check itself is driven
# red by tests/test_multicast_rectangle.c, which owns it.

run_arm "the leader is not required to be a member" \
    ft_tt_group_broadcast_partition.c \
    's/if (!contains_coordinate(members, member_count, leader))/if (0)/' red

# --- the vacuity killers: attack the INSTRUMENT, not the mechanism ----------
# If `_covers` answered from a constant, every "covers every receiver exactly
# once" check in the suite would be worthless. These two prove it does not.
run_arm "the coverage check always says yes" \
    ft_tt_group_broadcast_partition.c \
    's/if (hits != (is_leader ? 0u : 1u))/if (0)/' red

run_arm "the coverage check ignores the leader, so covering it twice is fine" \
    ft_tt_group_broadcast_partition.c \
    's/if (hits != (is_leader ? 0u : 1u))/if (hits > 1u)/' red

# --- the resident cooperative plan, the arg block and the reset ---------------
run_arm "a RECEIVER is handed the leader's rectangle" \
    ft_tt_op_matmul_resident_multicast.c \
    's/if (index != plan->leader) {/if (0) {/' red

run_arm "every member is given the LEADER role" \
    ft_tt_op_matmul_resident_multicast.c \
    's/out_args\[kArgCoopRole\] = index == plan->leader ? FT_TT_RESIDENT_COOP_ROLE_LEADER/out_args[kArgCoopRole] = 1 ? FT_TT_RESIDENT_COOP_ROLE_LEADER/' red

run_arm "an unbounded rendezvous wait is accepted" \
    ft_tt_op_matmul_resident_multicast.c \
    's/if (spin_limit == 0u) {/if (0) {/' red

run_arm "the reset stages the leader's ready semaphore FIRST" \
    ft_tt_op_matmul_resident_multicast.c \
    's/for (index = 0u; index < plan->member_count; index++) {/for (index = 1u; index < plan->member_count + 1u; index++) {/' red

run_arm "the packed rectangle word drops the destination count" \
    kernels/resident_matmul_multicast_protocol.h \
    's/(destination_count << FT_TT_RESIDENT_COOP_COUNT_SHIFT)/(1u << FT_TT_RESIDENT_COOP_COUNT_SHIFT)/' red

run_arm "the packed rectangle word accepts five destinations" \
    kernels/resident_matmul_multicast_protocol.h \
    's/destination_count > FT_TT_RESIDENT_COOP_MAX_DESTINATIONS/destination_count > 64u/' red

run_arm "the packed rectangle word swaps x_high and y_high" \
    kernels/resident_matmul_multicast_protocol.h \
    's/(x_high << (2u \* FT_TT_RESIDENT_COOP_COORDINATE_BITS))/(y_high << (2u * FT_TT_RESIDENT_COOP_COORDINATE_BITS))/' red

# THE COUNTER MUST NEVER BE THE VALUE A RECEIVER CLEARS ITS ARRIVAL WORD TO. If it could
# be, a stale clear would read as an arrival and the receiver would compute from the
# previous block with nothing reporting an error.
run_arm "the block counter is allowed to wrap to the INVALID value" \
    kernels/resident_matmul_multicast_protocol.h \
    's/return next == 0u ? FT_TT_RESIDENT_COOP_FIRST_COUNTER : next;/return next;/' red

echo
echo "=== $PASS arms as required, $FAIL wrong ==="
[ "$FAIL" -eq 0 ]
