#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_program_blob_gates.sh
#   The embedded-program-blob mutation battery: prove every gate can FAIL, and fail for the
#   reason it was aimed at, before any of them is trusted green.
#
#   Two halves, because there are two mechanisms:
#
#     A. THE ACCESSORS. Six damaged blobs, each built into its own binary and run. Each must
#        exit non-zero AND print the specific refusal that damage should cause. A red for
#        some other reason proves nothing about the check it was aimed at, so the expected
#        text is checked, not just the exit code.
#
#     B. THE GENERATOR. Eleven program directories that must be REFUSED, and the refusal
#        must leave no output file behind. A generator that half-writes a header on the way
#        to failing has produced exactly the artifact this whole item exists to prevent.
#
#     B2. THE PROVENANCE CHECKER, red and green, on directories AND on a generated blob.
#        Its two red cases are the ones the mtime check calls `current` in the same run,
#        which is asserted rather than assumed -- that blindness is the reason the content
#        record exists.
#
#   Usage:  bash run_program_blob_gates.sh
#   Env:    FIXTURES  where make_program_blob_fixtures.py wrote (default /tmp/ft_tt_program_blob_fixtures)
#
#   NO CARD. Nothing here opens a device, and every binary is built with the NoC fence
#   closed (no -DFT_TT_PHASE1B_NOC_ENABLED).
#
#   EVERY BUILD IS FROM SCRATCH. Apple ships GNU make 3.81 and its one-second mtime
#   granularity silently skips a rebuild of a source edited in the same second as its object
#   (coding-best-practices.md 9). These are whole-program links with the binary deleted
#   first, so there is no object file and no binary that can survive into the next arm.
#
#   Status: 6 accessor mutations + 11 generator refusals + 5 provenance-checker arms, all
#           seen red on the bank 2026-08-19. On the Mac section A does not link (the udrv
#           sources are Linux-only), which is why the battery runs there.
#   Issues: None
#   Todo:   None
set -uo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
FIXTURES=${FIXTURES:-/tmp/ft_tt_program_blob_fixtures}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ftpb_gates.XXXXXX")
EMIT="$TT/tools/emit_tensix_program_header.py"
BASE_HEADER="$FIXTURES/ft_tt_programs_fixture.h"

CFLAGS="-std=c99 -O1 -g -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion \
        -Wstrict-prototypes -Wformat=2 -fstack-protector-strong -I$TT -I$HERE"

# ft_tt_runtime_udrv.c and the four files it needs are here because ft_tt_runtime.c calls
# them unconditionally. Without them the BASELINE does not link, and a battery whose
# baseline does not build reports "nothing below means anything" and exits 1 — which is what
# it did from the day it was written until 2026-08-19. Every gate in it was unproven for
# that whole time; none of them had ever been seen fire on this machine.
SOURCES="$TT/ft_tt_kmd.c $TT/ft_tt_arch.c $TT/ft_tt_arch_blackhole.c $TT/ft_tt_device.c \
         $TT/ft_tt_tlb.c $TT/ft_tt_coord.c $TT/ft_tt_arc_telemetry.c $TT/ft_tt_noc.c \
         $TT/ft_tt_buffer.c $TT/ft_tt_elf.c $TT/ft_tt_kernel_blob.c $TT/ft_tt_udrv_claim.c \
         $TT/ft_tt_udrv_reset.c $TT/ft_tt_udrv_pci_sysfs.c $TT/ft_tt_udrv_tlb_encode.c \
         $TT/ft_tt_udrv_noc.c $TT/ft_tt_udrv_arc_msg.c $TT/ft_tt_udrv_arc_probe.c \
         $TT/ft_tt_runtime_udrv.c \
         $TT/ft_tt_tensix_claim.c $TT/ft_tt_launch.c $TT/ft_tt_runtime.c \
         $TT/ft_tt_program_blob.c $TT/ft_tt_program_resolve.c $HERE/ft_tt_test_report.c \
         $HERE/test_program_blob.c"

passes=0
failures=0

pass() { printf '  PROVEN RED  %-22s %s\n' "$1" "$2"; passes=$((passes + 1)); }
fail() { printf '  NOT RED     %-22s %s\n' "$1" "$2"; failures=$((failures + 1)); }

[ -f "$BASE_HEADER" ] || {
    echo "no fixture header at $BASE_HEADER -- run: make -f Makefile.programblob fixtures" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# 0. The baseline. If the UNDAMAGED blob does not pass, every red below is
#    meaningless -- it would be red before the mutation was applied.
# ---------------------------------------------------------------------------
echo "=== baseline: the undamaged fixture blob must be GREEN"
rm -f "$WORK/baseline"
if ! cc $CFLAGS -I"$FIXTURES" -DFT_TT_PROGRAMS_HEADER='"ft_tt_programs_fixture.h"' \
        $SOURCES -o "$WORK/baseline" > "$WORK/baseline.build" 2>&1; then
    echo "  BASELINE DOES NOT BUILD -- nothing below means anything"; cat "$WORK/baseline.build"
    exit 1
fi
if "$WORK/baseline" "$FIXTURES" > "$WORK/baseline.out" 2>&1; then
    echo "  baseline green"
else
    echo "  BASELINE IS RED -- nothing below means anything"; tail -30 "$WORK/baseline.out"
    exit 1
fi

# ---------------------------------------------------------------------------
# A. The accessors, one damaged blob at a time.
# ---------------------------------------------------------------------------
echo
echo "=== A. every accessor gate, watched failing"

run_mutation() {
    local mode=$1 expected=$2
    local header="$WORK/$mode.h" binary="$WORK/$mode"

    if ! python3 "$HERE/mutate_program_blob_header.py" "$mode" "$BASE_HEADER" "$header" \
            > "$WORK/$mode.mutate" 2>&1; then
        fail "$mode" "the mutation itself did not apply: $(tail -1 "$WORK/$mode.mutate")"
        return
    fi
    rm -f "$binary"
    if ! cc $CFLAGS -I"$WORK" -DFT_TT_PROGRAMS_HEADER="\"$mode.h\"" \
            $SOURCES -o "$binary" > "$WORK/$mode.build" 2>&1; then
        fail "$mode" "the damaged blob did not build; a build error is not the red we want"
        return
    fi
    if "$binary" "$FIXTURES" > "$WORK/$mode.out" 2>&1; then
        fail "$mode" "the suite PASSED on a damaged blob"
        return
    fi
    if grep -q -- "$expected" "$WORK/$mode.out"; then
        pass "$mode" "refused with: $expected"
    else
        fail "$mode" "red, but not for '$expected' -- see $WORK/$mode.out"
    fi
}

run_mutation flip-byte          "is CORRUPT"
run_mutation truncate-image     "is CORRUPT"
run_mutation break-elf-magic    "not a loadable Tensix ELF"
run_mutation swap-kernel-images "FAIL  prog_alpha/brisc.elf embedded == on disk"
run_mutation merge-firmware     "FAIL  alpha and gamma were NOT merged"
run_mutation drop-program       "FAIL  the blob holds the three fixture programs"

# ---------------------------------------------------------------------------
# B. The generator, one bad program directory at a time.
# ---------------------------------------------------------------------------
echo
echo "=== B. every generator refusal, watched firing"

run_refusal() {
    local label=$1
    shift
    local output="$WORK/refused_$label.h"

    rm -f "$output"
    if python3 "$EMIT" -o "$output" "$@" > "$WORK/refused_$label.out" 2>&1; then
        fail "$label" "the generator ACCEPTED it"
        return
    fi
    if [ -e "$output" ]; then
        fail "$label" "refused, but left $output behind -- a half-written header is worse \
than none"
        return
    fi
    pass "$label" "$(grep -m1 'REFUSED' "$WORK/refused_$label.out" | cut -c1-90)"
}

# The PROVENANCE cases get the stricter treatment the accessor mutations get, for a reason
# specific to them: both directories satisfy every older check, and the same run prints
# `current` from check_tensix_program_sources.sh right above the refusal. That is the whole
# point of the content record, so it is asserted rather than assumed -- otherwise one
# refusal could stand in for another's proof.
run_refusal_because() {
    local label=$1 expected=$2
    shift 2
    run_refusal "$label" "$@"
    if grep -q -- "$expected" "$WORK/refused_$label.out"; then
        pass "$label-reason" "refused for '$expected'"
    else
        fail "$label-reason" "red, but not for '$expected' -- see $WORK/refused_$label.out"
    fi
    if grep -q ': current (' "$WORK/refused_$label.out"; then
        pass "$label-mtime-blind" "the mtime check called this directory current"
    else
        fail "$label-mtime-blind" "the mtime check did NOT call it current, so this is not \
the case the content record exists for"
    fi
}

BAD="$FIXTURES/bad"
KERNELS="$FIXTURES/kernels"

run_refusal stale-missing-source --kernels "$KERNELS" \
    --program "$BAD/stale_missing_source/prog_stale"
run_refusal stale-newer-source   --kernels "$KERNELS" \
    --program "$BAD/stale_newer_source/prog_newer"
run_refusal no-sidecars          --kernels "$KERNELS" --program "$BAD/no_sidecars/prog_mute"
run_refusal orphan-sidecar       --kernels "$KERNELS" \
    --program "$BAD/orphan_sidecar/prog_orphan"
run_refusal missing-firmware     --kernels "$KERNELS" \
    --program "$BAD/missing_firmware/prog_partial_fw"
run_refusal empty-kernel-text    --kernels "$KERNELS" --program "$BAD/tiny_text/prog_tiny"
run_refusal duplicate-name       --kernels "$KERNELS" \
    --program "$BAD/dup_left/prog_dup" --program "$BAD/dup_right/prog_dup"
run_refusal no-programs          --kernels "$KERNELS"
run_refusal absent-directory     --kernels "$KERNELS" --program "$FIXTURES/programs/nope"
run_refusal_because no-provenance    "has no provenance record" --kernels "$KERNELS" \
    --program "$BAD/no_provenance/prog_unvouched"
run_refusal_because stale-provenance "STALE -- brisc.elf was built from" --kernels "$KERNELS" \
    --program "$BAD/stale_provenance/prog_drifted"

# ---------------------------------------------------------------------------
# B2. The standalone provenance checker, on the same two directories and on a good one.
#     It is what `make check-programs` and a bank session run BEFORE a regeneration, so it
#     has to be red on the same evidence the generator is red on -- and green on the rest.
# ---------------------------------------------------------------------------
echo
echo "=== B2. check_tensix_program_provenance.py, red and green"

CHECKER="$TT/tools/check_tensix_program_provenance.py"
check_provenance() {
    local label=$1 want=$2
    shift 2
    if python3 "$CHECKER" --tt-root "$FIXTURES" "$@" > "$WORK/prov_$label.out" 2>&1; then
        got=green
    else
        got=red
    fi
    if [ "$got" = "$want" ]; then
        pass "checker-$label" "$got, as it must be"
    else
        fail "checker-$label" "$got, wanted $want -- see $WORK/prov_$label.out"
    fi
}

check_provenance good           green "$FIXTURES/programs/prog_alpha"
check_provenance unvouched      red   "$BAD/no_provenance/prog_unvouched"
check_provenance drifted        red   "$BAD/stale_provenance/prog_drifted"
check_provenance header         green --header "$BASE_HEADER"

# The --header arm's RED, and it is the one that matters on the bank: the blob is fine, the
# TREE moved. Checked against a copied tree so the fixtures the rest of this script needs
# are untouched -- editing the real prog_alpha source here would make section C's "the
# generator still accepts the good directories" arm red for this reason instead.
cp -R "$FIXTURES" "$WORK/drifted_tree"
printf '/* edited under the blob */\n' > "$WORK/drifted_tree/kernels/prog_alpha_brisc.c"
if python3 "$CHECKER" --tt-root "$WORK/drifted_tree" --header "$BASE_HEADER" \
        > "$WORK/prov_header_red.out" 2>&1; then
    fail "checker-header-red" "the blob passed against a tree whose source had changed"
elif grep -q "prog_alpha_brisc.c" "$WORK/prov_header_red.out"; then
    pass "checker-header-red" "named kernels/prog_alpha_brisc.c as the input that moved"
else
    fail "checker-header-red" "red, but it did not name the file that changed"
fi

# ---------------------------------------------------------------------------
# C. The verification TOOL. It is what will be used on the bank to prove a real
#    regeneration, so it needs the same treatment: green on matching inputs, red on a
#    difference, and red for the right reason.
# ---------------------------------------------------------------------------
echo
echo "=== C. verify_tensix_program_blob, green and red"

VERIFY_SOURCES="$TT/ft_tt_kmd.c $TT/ft_tt_arch.c $TT/ft_tt_arch_blackhole.c \
                $TT/ft_tt_device.c $TT/ft_tt_elf.c $TT/ft_tt_program_blob.c \
                $TT/tools/verify_tensix_program_blob.c"

rm -f "$WORK/verify"
if cc $CFLAGS -I"$FIXTURES" -DFT_TT_PROGRAMS_HEADER='"ft_tt_programs_fixture.h"' \
        $VERIFY_SOURCES -o "$WORK/verify" > "$WORK/verify.build" 2>&1 &&
   "$WORK/verify" "$FIXTURES/programs" > "$WORK/verify.out" 2>&1; then
    printf '  MATCHES     %-22s %s\n' "blob-vs-disk" "$(tail -1 "$WORK/verify.out")"
    passes=$((passes + 1))
else
    fail "verify-baseline" "the tool cannot even agree with the directories it was built from"
fi

# A single flipped byte on DISK -- the direction the bank cares about, where the header is
# right and the tree has moved.
cp -R "$FIXTURES/programs" "$WORK/tampered"
python3 - "$WORK/tampered/prog_alpha/trisc1.elf" <<'PY'
import sys
path = sys.argv[1]
with open(path, "rb") as handle:
    payload = bytearray(handle.read())
payload[len(payload) // 2] ^= 0xff
with open(path, "wb") as handle:
    handle.write(payload)
PY
if "$WORK/verify" "$WORK/tampered" > "$WORK/verify_bad.out" 2>&1; then
    fail "verify-tampered" "the tool passed a directory with a flipped byte in it"
elif grep -q "trisc1.elf.*DIFFERS" "$WORK/verify_bad.out"; then
    pass "verify-tampered" "named prog_alpha/trisc1.elf as the difference"
else
    fail "verify-tampered" "red, but it did not name the file that changed"
fi

# The empty blob must REFUSE to compare rather than report zero differences: an empty blob
# matches every directory tree in existence.
#
# THE EMPTY BLOB IS A FIXTURE, NOT THE DEFAULT HEADER. This arm used to compile with no
# -DFT_TT_PROGRAMS_HEADER at all, which meant it tested the SHIPPING blob and only passed
# while that blob was still the placeholder — so it has been silently green-then-wrong since
# the first real regeneration on 2026-08-17, exactly the backwards empty-state gate
# Makefile.programblob's own comment describes for test_program_blob_empty. Caught
# 2026-08-19, when a battery that finally linked reported it as the one arm not proven.
printf '%s\n' \
    '#ifndef FT_TT_PROGRAMS_H' '#define FT_TT_PROGRAMS_H' \
    'static const ft_tt_program_blob_entry *const ft_tt_program_blob_table_storage[1] = { 0 };' \
    '#define ft_tt_program_blob_table ft_tt_program_blob_table_storage' \
    '#define FT_TT_PROGRAM_BLOB_TABLE_COUNT 0u' '#endif' > "$WORK/ft_tt_programs_empty.h"
rm -f "$WORK/verify_empty"
if cc $CFLAGS -I"$WORK" -DFT_TT_PROGRAMS_HEADER='"ft_tt_programs_empty.h"' \
        $VERIFY_SOURCES -o "$WORK/verify_empty" > "$WORK/verify_empty.build" 2>&1 &&
   ! "$WORK/verify_empty" "$FIXTURES/programs" > "$WORK/verify_empty.out" 2>&1 &&
   grep -q "nothing to compare" "$WORK/verify_empty.out"; then
    pass "verify-empty-blob" "an empty blob refuses instead of reporting 0 differences"
else
    fail "verify-empty-blob" "an empty blob compared clean, which is a statement about nothing"
fi

# The other half of a refusal gate: the SAME generator must still accept the good
# directories, or "it refuses everything" would read as eight passes.
echo
echo "=== the generator still accepts the good directories"
if python3 "$EMIT" -o "$WORK/accepted.h" --kernels "$KERNELS" \
        --program "$FIXTURES/programs/prog_alpha" \
        --program "$FIXTURES/programs/prog_beta" \
        --program "$FIXTURES/programs/prog_gamma" > "$WORK/accepted.out" 2>&1; then
    printf '  ACCEPTED    %-22s %s\n' "good-directories" "$(tail -1 "$WORK/accepted.out")"
    passes=$((passes + 1))
else
    printf '  REFUSED     %-22s %s\n' "good-directories" \
           "the generator refuses everything -- every red above is vacuous"
    failures=$((failures + 1))
fi

echo
echo "$passes proven, $failures not proven   (artifacts in $WORK)"
[ "$failures" -eq 0 ] || exit 1
rm -rf "$WORK"
exit 0
