#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# The transport seam's MUTATION BATTERY.
#
# Every gate in test_runtime_seam.c is re-run against a build in which exactly
# ONE property of the code under test has been broken, and each damaged build
# MUST go red. A gate never seen failing is not evidence — nine vacuous gates
# have been caught on this project, two of them inside the rehearsal itself.
#
# Each mutation works on a SCRATCH COPY of tt/ with its own build directory, so
# the real tree is never edited and GNU make 3.81's one-second mtime granularity
# cannot serve a stale binary (coding-best-practices.md §9). The whole-program
# link means there are no object files to go stale either.
#
# It also runs the UNMUTATED build first. Without that control, "every mutation
# went red" would be consistent with a suite that fails unconditionally.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
TT_DIR="$(cd "$HERE/.." && pwd)"
SCRATCH="$HERE/build-seam-gates/run.$$"   # in the repo (gitignored), never /tmp
PASS=0
FAIL=0

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

# $1 = name, $2 = file to patch (relative to tt/), $3 = sed expression, "" for none
run_arm() {
    local name="$1" file="$2" expr="$3" want="$4"
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

    out=$(cd "$root/tests" && make -f Makefile.runtimeseam check 2>&1)
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

echo "=== transport seam mutation battery ==="
run_arm "control" "" "" green

# --- ft_tt_arc_telemetry.c: the walk both transports share -------------------
run_arm "the walker IGNORES the tag id and takes entries in order" \
    ft_tt_arc_telemetry.c \
    's/if (out_found\[i\] || tags\[i\] != tag)/if (out_found[i])/' red

run_arm "the table pointer is read from the VALUES scratch register" \
    ft_tt_arc_telemetry.c \
    's/user, telemetry->table_pointer_scratch_index,/user, telemetry->values_pointer_scratch_index,/' red

run_arm "the CSM bound on a tag's value address is dropped" \
    ft_tt_arc_telemetry.c \
    's/if (!csm_contains_range(telemetry, value_address, 4u)) {/if (0) {/' red

run_arm "the column count POPCOUNTS the raw word instead of masking first" \
    ft_tt_arc_telemetry.c \
    's/        value &= (1u << bit_count) - 1u;/        (void)0;/' red

run_arm "the entry-count bound is removed" \
    ft_tt_arc_telemetry.c \
    's/if (location.entry_count > FT_TT_ARC_TELEMETRY_MAX_ENTRIES) {/if (0) {/' red

run_arm "an unpublished table (pointer 0) is treated as a valid one" \
    ft_tt_arc_telemetry.c \
    's/if (location.table_address == 0u) {/if (0) {/' red

run_arm "a missing tag is defaulted to zero instead of refused" \
    ft_tt_arc_telemetry.c \
    's/        if (!found\[i\]) {/        if (0) {/' red

# --- ft_tt_runtime.c: the selection and the capacity -------------------------
run_arm "an unrecognised FT_TT_DRIVER silently defaults to kmd" \
    ft_tt_runtime.c \
    's/^        return FT_TT_EINVAL;$/        *out_driver = FT_TT_RUNTIME_DRIVER_KMD;/' red

run_arm "the capacity ignores whether the grid was ever read" \
    ft_tt_runtime.c \
    's/if (!runtime || !runtime->grid.valid || !runtime->grid.arch)/if (!runtime || !runtime->grid.arch)/' red

# --- ft_tt_arch.c: selection with no driver opinion --------------------------
run_arm "the vendor id is no longer checked when there is no driver to ask" \
    ft_tt_arch.c \
    's/if (pci_vendor_id != TENSTORRENT_PCI_VENDOR_ID) {/if (0) {/' red

echo
echo "=== $PASS arms as required, $FAIL wrong ==="
[ "$FAIL" -eq 0 ]
