#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_cooperative_matmul_suite.sh
#   Runs the cooperative-matmul suite the way this project requires it to be run, on the bank.
#
#   THE POINT OF THIS FILE IS THE LAST STEP. Every suite here must end by proving the cores it
#   used still BOOT *and* still COMPUTE, as two separate checks, run UNCONDITIONALLY even when
#   an earlier stage failed -- because a read cannot tell a wedged core from a healthy idle
#   one, and a multicast bug is exactly the kind that wedges a receiver quietly
#   (DEV_NOTES.md 1). The BOOT half is stage 6 inside test_cooperative_matmul; the COMPUTE half
#   is `ft_tt_core probe`, which drives its OWN vector_add triad and therefore cannot be fooled
#   by anything this suite left behind.
#
#   FT_TT_FIRMWARE_DIR IS SET PER COMMAND, NEVER EXPORTED. `ft_tt_core probe` with that
#   variable in the environment boots a FOREIGN program under vector_add's runtime args and
#   wedges the core -- it cost core 26, and the gate that was added for `--program` did not
#   cover the environment route (DEV_NOTES.md 1). The probe line below deliberately has no
#   access to it.
#
#   Usage:
#     bash run_cooperative_matmul_suite.sh <program-directory> [repeats]
#
#   Environment (all optional, passed through):
#     FT_TT_COOP_FIRST_INDEX   live index of the group's first core (default 100)
#     FT_TT_COOP_ROWS          pool rows    (default 2)
#     FT_TT_COOP_COLUMNS       pool columns (default 4 -- DO NOT raise past 4)
#     FT_TT_COOP_PROBE_CORES   what to probe afterwards (default: the cores used)
#
#   Build the program directory first (on the bank, where SFPI lives). The compute and writer
#   thirds are the SHIPPED matmul_compute.c and vector_add_writer.c, unchanged -- a
#   cooperating matmul is a reader-side and host-side change:
#     RELOCATE_KERNEL_TEXT=1 \
#     READER_CTARGS=0,1,2048,2048 COMPUTE_CTARGS=0,1,16 WRITER_CTARGS=16,2048 \
#         bash ../build_tensix_program.sh ../kernels/cooperative_matmul_reader.c \
#              ../kernels/matmul_compute.c ../kernels/vector_add_writer.c /tmp/ttk_coop
#
#   Status: NEW 2026-08-13.
#   Issues: None
#   Todo:   None
#
set -uo pipefail

PROGRAM_DIRECTORY=${1:?usage: run_cooperative_matmul_suite.sh <program-directory> [repeats]}
REPEATS=${2:-5}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUITE=$HERE/build-cooperative/test_cooperative_matmul
PLAN=$HERE/build-cooperative/test_cooperative_matmul_plan
CORE_TOOL=$HERE/../tools/build/ft_tt_core

FIRST_INDEX=${FT_TT_COOP_FIRST_INDEX:-100}
POOL_ROWS=${FT_TT_COOP_ROWS:-2}
POOL_COLUMNS=${FT_TT_COOP_COLUMNS:-4}

for binary in "$SUITE" "$PLAN"; do
    [ -x "$binary" ] || { echo "missing $binary — run: make -f Makefile.cooperative PHASE1B=1 all" >&2; exit 2; }
done

# ---- 0. Host and link, before anything host-involved is timed -------------------------------
# A card-level reset leaves the PCIe link at Gen1 every time so far, host bandwidth halves, and
# every host-involved number is then quietly wrong (DEV_NOTES.md 1). And the bank is not a quiet
# host: a `pantheon serve` on the K80s takes ~58% of two cores, and a run that STRADDLES a
# change in host load is void -- which is why the load is recorded at BOTH ends.
echo "=== 0. host and link state, BEFORE ==="
if command -v lspci >/dev/null 2>&1; then
    # BOTH addresses are DERIVED. The card was 12:00.0 on the p150a bank and is 03:00.0 on
    # the TT host; its upstream bridge was 00:1c.6 and is 00:1c.3. A hardcoded pair reads the wrong
    # device's link and tells you to retrain someone else's.
    CARD=$(lspci -Dn | awk '/1e52:b140/ {print $1; exit}')
    if [ -n "$CARD" ]; then
        BRIDGE=$(basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/$CARD")")")
        sudo lspci -vvs "$CARD" 2>/dev/null | grep -m1 LnkSta: || echo "  (could not read LnkSta)"
        echo "  it must say 5GT/s; if it says 2.5GT/s run:"
        echo "    sudo setpci -s ${BRIDGE#0000:} CAP_EXP+10.w=0020:0020"
    else
        echo "  (no Tenstorrent Blackhole found)"
    fi
fi
uptime
pgrep -af pantheon | head -3 || echo "  (no pantheon serve running)"

# ---- 1. Off-hardware first ------------------------------------------------------------------
# The work split and the refusals need no card, and a failure there makes every on-card result
# uninterpretable, so it runs first and its result is reported separately.
echo
echo "=== 1. off-hardware: the work split and the refusals ==="
"$PLAN"
PLAN_STATUS=$?

# ---- 2. On card ------------------------------------------------------------------------------
echo
echo "=== 2. on card: the measurement ==="
set +e
FT_TT_FIRMWARE_DIR="$PROGRAM_DIRECTORY" \
FT_TT_COOP_FIRST_INDEX="$FIRST_INDEX" \
FT_TT_COOP_ROWS="$POOL_ROWS" \
FT_TT_COOP_COLUMNS="$POOL_COLUMNS" \
    "$SUITE" "$PROGRAM_DIRECTORY" "$REPEATS"
SUITE_STATUS=$?

echo
echo "=== host state, AFTER (a run that straddles a change in it is void) ==="
uptime
pgrep -af pantheon | head -3 || echo "  (no pantheon serve running)"

# ---- 3. The COMPUTE half of the health rule, unconditionally --------------------------------
# Note there is NO FT_TT_FIRMWARE_DIR on this line and none exported above it.
LAST_INDEX=$((FIRST_INDEX + (POOL_ROWS - 1) * 14 + POOL_COLUMNS - 1))
PROBE_CORES=${FT_TT_COOP_PROBE_CORES:-$FIRST_INDEX-$LAST_INDEX}

echo
echo "=== 3. do those cores still COMPUTE? (ft_tt_core probe, its own vector_add triad) ==="
if [ -x "$CORE_TOOL" ]; then
    "$CORE_TOOL" probe "$PROBE_CORES" --write --i-know-this-must-be-a-vector-add-triad
    PROBE_STATUS=$?
else
    echo "  $CORE_TOOL not built — run: cd ../tools && make PHASE1B=1" >&2
    PROBE_STATUS=2
fi

echo
echo "=== summary ==="
echo "  off-hardware plan suite : exit $PLAN_STATUS"
echo "  on-card suite (BOOT check included as its stage 6) : exit $SUITE_STATUS"
echo "  compute probe : exit $PROBE_STATUS"
[ "$PLAN_STATUS" -eq 0 ] && [ "$SUITE_STATUS" -eq 0 ] && [ "$PROBE_STATUS" -eq 0 ]
