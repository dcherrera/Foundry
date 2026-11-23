#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_multicast_suite.sh
#   Runs the NoC multicast suite the way this project requires it to be run, on the bank.
#
#   THE POINT OF THIS FILE IS THE LAST TWO STEPS. Every suite here must end by proving the
#   cores it used still BOOT *and* still COMPUTE, as two separate checks, run
#   UNCONDITIONALLY even when an earlier stage failed -- because a read cannot tell a wedged
#   core from a healthy idle one, and a multicast bug is exactly the kind that wedges a
#   receiver quietly. The BOOT half is stage 4 inside
#   test_multicast_bench; the COMPUTE half is `ft_tt_core probe`, which drives its OWN
#   vector_add triad and therefore cannot be fooled by anything this suite left behind.
#
#   `probe` is run with `|| true` and after `set +e` on purpose: it must run even when the
#   bench exited non-zero, and its result is reported rather than short-circuited.
#
#   FT_TT_FIRMWARE_DIR IS SET PER COMMAND, NEVER EXPORTED. `ft_tt_core probe` with that
#   variable in the environment boots a FOREIGN program under vector_add's runtime args and
#   wedges the core -- it cost core 26, and the gate that was added for `--program` did not
#   cover the environment route. The probe line below deliberately has no
#   access to it.
#
#   Usage:
#     bash run_multicast_suite.sh <program-directory> [iterations] [payload-bytes]
#
#   Environment (all optional, passed through):
#     FT_TT_MULTICAST_FIRST_INDEX    live index of the receiver pool's first core
#     FT_TT_MULTICAST_SENDER_INDEX   live index of the sender
#     FT_TT_MULTICAST_POOL_ROWS      pool shape
#     FT_TT_MULTICAST_POOL_COLUMNS
#     FT_TT_MULTICAST_ARMS           both | multicast | dram -- the last two are CONTROLS
#     FT_TT_MULTICAST_PROBE_CORES    what to probe afterwards (default: the cores used)
#     FT_TT_MULTICAST_EXPERIMENT     none (default) | relaunch | split
#     FT_TT_MULTICAST_SENDER_B_INDEX the second sender, `split` only (default 120)
#     FT_TT_MULTICAST_RELAUNCH_CYCLES broadcast+probe cycles, `relaunch` only (default 2)
#
#   THE TWO EXPERIMENTS, and the exact lines to run them. Each takes its OWN pool default,
#   so neither inherits the sweep's 2x5 -- a ten-destination broadcast nobody asked for:
#
#     # A, the CONTROL first: a 4-destination shape that has never wedged. It proves the
#     # re-launch probe reads CLEAN on a healthy core before the real arm is believed.
#     FT_TT_MULTICAST_EXPERIMENT=relaunch FT_TT_MULTICAST_POOL_ROWS=1 \
#     FT_TT_MULTICAST_POOL_COLUMNS=4 FT_TT_MULTICAST_ARMS=multicast \
#         bash run_multicast_suite.sh /tmp/ttk_multicast 2000 2048
#
#     # A, the real arm: 1x5, the smallest shape that reproduces the wedge. THIS ONE MAY
#     # COST CORES -- that is the hypothesis. Budget a card reset; recover with
#     # `ft_tt_core recover <cores> --rung 3 --write --then-probe`, which resets the WHOLE
#     # card, and RETRAIN THE LINK afterwards or every later timing is invalid.
#     FT_TT_MULTICAST_EXPERIMENT=relaunch FT_TT_MULTICAST_ARMS=multicast \
#         bash run_multicast_suite.sh /tmp/ttk_multicast 2000 2048
#
#     # B: N=8 as two disjoint 4-destination broadcasts. No packet exceeds four.
#     FT_TT_MULTICAST_EXPERIMENT=split FT_TT_MULTICAST_ARMS=multicast \
#         bash run_multicast_suite.sh /tmp/ttk_multicast 2000 2048
#
#   ONE EXPERIMENT PER PROCESS, and probe between arms. A wedge has to be attributable to a
#   shape, and the next arm silently falling back to a damaged core is what turns one wedge
#   into a whole run of void measurements.
#
#   Build the program directory first (on the bank, where SFPI lives):
#     CTARGS=0 NOC=0 bash ../build_tensix_dataflow_kernel.sh \
#         ../kernels/multicast_bench_reader.c brisc /tmp/ttk_multicast
#     CTARGS=0 bash ../build_tensix_kernel.sh \
#         ../kernels/noop_compute.c /tmp/ttk_multicast
#     CTARGS=16,2048 NOC=1 bash ../build_tensix_dataflow_kernel.sh \
#         ../kernels/vector_add_writer.c ncrisc /tmp/ttk_multicast
#   (the NCRISC image is there only because the firmware set is five files; the bench
#   enables DM0 + the three TRISCs and never runs it)
#
#   Status: NEW 2026-08-13.
#   Issues: None
#   Todo:   None
#
set -uo pipefail

PROGRAM_DIRECTORY=${1:?usage: run_multicast_suite.sh <program-directory> [iterations] [payload-bytes]}
ITERATIONS=${2:-20000}
PAYLOAD_BYTES=${3:-2048}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$HERE/build-multicast/test_multicast_bench
RECTANGLE=$HERE/build-multicast/test_multicast_rectangle
CORE_TOOL=$HERE/../tools/build/ft_tt_core

FIRST_INDEX=${FT_TT_MULTICAST_FIRST_INDEX:-100}
SENDER_INDEX=${FT_TT_MULTICAST_SENDER_INDEX:-119}
SENDER_B_INDEX=${FT_TT_MULTICAST_SENDER_B_INDEX:-120}
EXPERIMENT=${FT_TT_MULTICAST_EXPERIMENT:-none}

# The pool default follows the EXPERIMENT, and the bench applies the same rule for a caller
# that runs it directly. Passing the sweep's 2x5 into an experiment would issue a
# ten-destination broadcast without anyone choosing one.
case $EXPERIMENT in
    none)     DEFAULT_ROWS=2; DEFAULT_COLUMNS=5 ;;
    relaunch) DEFAULT_ROWS=1; DEFAULT_COLUMNS=5 ;;
    split)    DEFAULT_ROWS=2; DEFAULT_COLUMNS=4 ;;
    *) echo "FT_TT_MULTICAST_EXPERIMENT must be none, relaunch or split" >&2; exit 2 ;;
esac
POOL_ROWS=${FT_TT_MULTICAST_POOL_ROWS:-$DEFAULT_ROWS}
POOL_COLUMNS=${FT_TT_MULTICAST_POOL_COLUMNS:-$DEFAULT_COLUMNS}

for binary in "$BENCH" "$RECTANGLE"; do
    [ -x "$binary" ] || { echo "missing $binary — run: make -f Makefile.multicast PHASE1B=1 all" >&2; exit 2; }
done

# ---- 0. The PCIe link, before anything host-involved is timed -------------------------
# A card-level reset leaves the link at Gen1 every time so far, host bandwidth halves, and
# every host-involved number is then quietly wrong. The bench's own timing
# is device-side, but its stage-1 CALIBRATION divides device cycles by a HOST interval, so
# it is exactly a host-involved measurement.
echo "=== 0. PCIe link speed ==="
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

# ---- 1. Off-hardware first -------------------------------------------------------------
# The rectangle arithmetic needs no card, and a failure there makes every on-card result
# uninterpretable, so it runs first and its result is reported separately.
echo
echo "=== 1. off-hardware: rectangle arithmetic and refusals ==="
"$RECTANGLE"
RECTANGLE_STATUS=$?

# ---- 2. On card -------------------------------------------------------------------------
echo
echo "=== 2. on card: the measurement ==="
set +e
FT_TT_FIRMWARE_DIR="$PROGRAM_DIRECTORY" \
FT_TT_MULTICAST_FIRST_INDEX="$FIRST_INDEX" \
FT_TT_MULTICAST_SENDER_INDEX="$SENDER_INDEX" \
FT_TT_MULTICAST_POOL_ROWS="$POOL_ROWS" \
FT_TT_MULTICAST_POOL_COLUMNS="$POOL_COLUMNS" \
FT_TT_MULTICAST_ARMS="${FT_TT_MULTICAST_ARMS:-both}" \
FT_TT_MULTICAST_EXPERIMENT="$EXPERIMENT" \
FT_TT_MULTICAST_SENDER_B_INDEX="$SENDER_B_INDEX" \
FT_TT_MULTICAST_RELAUNCH_CYCLES="${FT_TT_MULTICAST_RELAUNCH_CYCLES:-2}" \
    "$BENCH" "$PROGRAM_DIRECTORY" "$ITERATIONS" "$PAYLOAD_BYTES"
BENCH_STATUS=$?

# ---- 3. The COMPUTE half of the health rule, unconditionally ---------------------------
# Note there is NO FT_TT_FIRMWARE_DIR on this line and none exported above it. `probe` with
# that variable set boots a foreign program under vector_add's runtime args and wedges the
# core; it cost core 26.
LAST_POOL_INDEX=$((FIRST_INDEX + (POOL_ROWS - 1) * 14 + POOL_COLUMNS - 1))
DEFAULT_PROBE_CORES=$FIRST_INDEX-$LAST_POOL_INDEX,$SENDER_INDEX
# The split experiment drives a SECOND sender, and a core the run touched but nobody probes
# is exactly how a wedge goes unnoticed until the next arm reports void numbers.
[ "$EXPERIMENT" = "split" ] && DEFAULT_PROBE_CORES=$DEFAULT_PROBE_CORES,$SENDER_B_INDEX
PROBE_CORES=${FT_TT_MULTICAST_PROBE_CORES:-$DEFAULT_PROBE_CORES}

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
echo "  off-hardware rectangle suite : exit $RECTANGLE_STATUS"
echo "  on-card bench (BOOT check included as its stage 4) : exit $BENCH_STATUS"
echo "  compute probe : exit $PROBE_STATUS"
[ "$RECTANGLE_STATUS" -eq 0 ] && [ "$BENCH_STATUS" -eq 0 ] && [ "$PROBE_STATUS" -eq 0 ]
