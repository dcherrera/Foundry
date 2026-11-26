#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_gdn_bf16_in_program_gates.sh
#   The compile-side gates of G2a: the gdn reader takes PHASE CONV's stream from the bf16
#   projection strip and widens it itself (kGdnFlagConvBf16In). Needs the SFPI toolchain (7.68.0, the blob's),
#   no card:
#
#     1. explode the committed blob (ft_tt_programs.h) into $WORK/blob: the shipped images
#     2. rebuild EVERY program from its own provenance recipe and THIS tree into ../programs;
#        every image LOAD-IDENTICAL to the shipped one EXCEPT gdn/brisc.elf (the reader), which
#        must differ (else the change never reached the compiler) and keep text + data inside
#        the 16 KiB BRISC slab (ft_tt_op_l1_layout.h, F0)
#     3. regenerate the blob from ../programs: CHECK=1 (diff only) unless REGEN=1, which writes
#        ../ft_tt_programs.h and then runs the provenance check and the verifier
#
#   Usage (from tt/):  bash tools/check_gdn_bf16_in_program_gates.sh
#   Env:  SFPI   toolchain root (default /opt/tenstorrent/sfpi)
#         WORK   scratch (default ../build/gdn-bf16-gates, gitignored)
#         REGEN  1 = write the regenerated ft_tt_programs.h
#   On macOS: export SFPI=<install root> PATH="$(brew --prefix)/bin:$PATH" (bash >= 4)
#
#   Output: $WORK/gates.txt (the verdicts), $WORK/rebuild/ (per-program logs and summary)
#   Exit:   0 every gate green; 1 any gate red; 2 setup error
#
#   Status: added 2026-11-20 with G2a.
#   Issues: None
#   Todo:   None
#
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
WORK=${WORK:-$TT/build/gdn-bf16-gates}
REGEN=${REGEN:-0}
PROGRAMS=$TT/programs
MOVED=gdn/brisc.elf
SLAB_BYTES=16384
GATES=$WORK/gates.txt
red=0

say() { printf '%s\n' "$*" | tee -a "$GATES"; }
fail() { say "RED: $*"; red=1; }
load_bytes() { "$SFPI/compiler/bin/riscv-tt-elf-size" -A "$1" | awk '$1 == ".text" || $1 == ".data" {s += $2} END {print s + 0}'; }

[ -x "$SFPI/compiler/bin/riscv-tt-elf-gcc" ] || { echo "no SFPI compiler under $SFPI" >&2; exit 2; }
export SFPI
mkdir -p "$WORK"
: > "$GATES"
say "=== G2a gdn bf16-in program gates  $(date -u +%FT%TZ)  sfpi $("$SFPI/compiler/bin/riscv-tt-elf-gcc" -dumpversion)"

# 1. The shipped images, out of the committed blob.
make -C "$HERE" explode-programs > "$WORK/explode_build.log" 2>&1 || { echo "explode build failed" >&2; exit 2; }
rm -rf "$WORK/blob"
"$HERE/build/explode_tensix_program_blob" "$WORK/blob" --verify > "$WORK/explode.log" 2>&1 ||
    { echo "explode failed (see $WORK/explode.log)" >&2; exit 2; }
say "exploded $(ls "$WORK/blob" | wc -l | tr -d ' ') programs from ft_tt_programs.h"

# 2. Every program from this tree's sources.
rm -rf "$PROGRAMS" "$WORK/rebuild"
bash "$HERE/rebuild_all_tensix_programs.sh" "$WORK/blob" "$PROGRAMS" $(ls "$WORK/blob") > /dev/null 2>&1
mkdir -p "$WORK/rebuild"
mv "$PROGRAMS"/*.rebuild.log "$PROGRAMS"/*.diff "$PROGRAMS"/summary.txt "$WORK/rebuild/" 2>/dev/null
rm -f "$PROGRAMS/DONE"
S=$WORK/rebuild/summary.txt
images=$(awk 'NF >= 4 && $2 ~ /\.elf$/' "$S" | wc -l | tr -d ' ')
identical=$(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 == "LOAD-IDENTICAL"' "$S" | wc -l | tr -d ' ')
moved=$(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 != "LOAD-IDENTICAL" {print $1"/"$2}' "$S" | tr '\n' ' ')
say "LOAD-IDENTICAL: $identical of $images images; moved: ${moved:-none}"
grep -q "REBUILD FAILED" "$S" && fail "$(grep -c 'REBUILD FAILED' "$S") program(s) failed to rebuild"
[ "$moved" = "$MOVED " ] || fail "expected exactly $MOVED to move"
if [ -f "$PROGRAMS/$MOVED" ]; then
    bytes=$(load_bytes "$PROGRAMS/$MOVED")
    say "$MOVED text + data $bytes B (shipped $(load_bytes "$WORK/blob/$MOVED") B), slab $SLAB_BYTES B"
    [ "$bytes" -lt "$SLAB_BYTES" ] || fail "$MOVED does not fit the $SLAB_BYTES B slab"
else
    fail "$MOVED missing"
fi

# 3. The blob from ../programs.
check=1
[ "$REGEN" = 1 ] && check=0
(cd "$HERE" && PROGRAMS=$PROGRAMS CHECK=$check bash regen_tensix_program_header.sh) > "$WORK/regen.log" 2>&1
regen_rc=$?
say "regen (CHECK=$check) rc=$regen_rc: $(grep -E 'IDENTICAL|DIFFERS|distinct firmware' "$WORK/regen.log" | tr -s ' ' | tr '\n' ' ')"
if [ "$REGEN" = 1 ]; then
    [ "$regen_rc" = 0 ] || fail "regen refused (see $WORK/regen.log)"
    python3 "$HERE/check_tensix_program_provenance.py" --header "$TT/ft_tt_programs.h" > "$WORK/provenance.log" 2>&1 ||
        fail "provenance --header (see $WORK/provenance.log)"
    say "provenance --header: $(tail -1 "$WORK/provenance.log")"
    make -C "$HERE" verify-programs > "$WORK/verify_build.log" 2>&1
    "$HERE/build/verify_tensix_program_blob" "$PROGRAMS" > "$WORK/verify.log" 2>&1 || fail "blob verify (see $WORK/verify.log)"
    say "verify: $(tail -1 "$WORK/verify.log")"
fi

[ "$red" = 0 ] && say "ALL GATES GREEN" || say "GATES RED"
exit "$red"
