#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# check_norm_role_program_gates.sh
#   The compile-side gates of the persistent norm role (P0.2):
#   one NEW program, `norm_role`, and no shipped image moved.
#   Needs the SFPI toolchain (7.68.0, the blob's), no card:
#     1. explode the committed blob (ft_tt_programs.h) into $WORK/blob: the shipped images
#     2. rebuild EVERY shipped program from its own provenance recipe and THIS tree into
#        ../programs; every image must be LOAD-IDENTICAL to the shipped one (the role adds
#        headers and kernels of its own and changes no shared one)
#     3. build norm_role from its recipe into ../programs/norm_role; each of its five images
#        must fit its text slab (ft_tt_op_l1_layout.h, F0)
#     4. regenerate the blob from ../programs: CHECK=1 (diff only) unless REGEN=1, which writes
#        ../ft_tt_programs.h and runs the provenance check and the verifier
#   Usage (from tt/):  bash tools/check_norm_role_program_gates.sh
#   Env:  SFPI   toolchain root (default /opt/tenstorrent/sfpi)
#         WORK   scratch (default ../build/norm-role-gates, gitignored)
#         REGEN  1 = write the regenerated ft_tt_programs.h
#   On macOS: export SFPI=<install root> PATH="$(brew --prefix)/bin:$PATH" (bash >= 4)
#   Exit:   0 every gate green; 1 any gate red; 2 setup error
#   Status: added with P0.
#   Issues: None
#   Todo:   None
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
WORK=${WORK:-$TT/build/norm-role-gates}
REGEN=${REGEN:-0}
PROGRAMS=$TT/programs
GATES=$WORK/gates.txt
red=0

say() { printf '%s\n' "$*" | tee -a "$GATES"; }
fail() { say "RED: $*"; red=1; }
load_bytes() { "$SFPI/compiler/bin/riscv-tt-elf-size" -A "$1" | awk '$1 == ".text" || $1 == ".data" {s += $2} END {print s + 0}'; }

[ -x "$SFPI/compiler/bin/riscv-tt-elf-gcc" ] || { echo "no SFPI compiler under $SFPI" >&2; exit 2; }
export SFPI
mkdir -p "$WORK"
: > "$GATES"
say "=== norm_role program gates  $(date -u +%FT%TZ)  sfpi $("$SFPI/compiler/bin/riscv-tt-elf-gcc" -dumpversion)"

# 1. The shipped images, out of the committed blob.
make -C "$HERE" explode-programs > "$WORK/explode_build.log" 2>&1 || { echo "explode build failed" >&2; exit 2; }
rm -rf "$WORK/blob"
"$HERE/build/explode_tensix_program_blob" "$WORK/blob" --verify > "$WORK/explode.log" 2>&1 ||
    { echo "explode failed (see $WORK/explode.log)" >&2; exit 2; }
say "exploded $(ls "$WORK/blob" | wc -l | tr -d ' ') programs from ft_tt_programs.h"

# 2. Every shipped program from this tree's sources.
rm -rf "$PROGRAMS" "$WORK/rebuild"
bash "$HERE/rebuild_all_tensix_programs.sh" "$WORK/blob" "$PROGRAMS" $(ls "$WORK/blob" | grep -v '^norm_role$') > /dev/null 2>&1
mkdir -p "$WORK/rebuild"
mv "$PROGRAMS"/*.rebuild.log "$PROGRAMS"/*.diff "$PROGRAMS"/summary.txt "$WORK/rebuild/" 2>/dev/null
rm -f "$PROGRAMS/DONE"
S=$WORK/rebuild/summary.txt
images=$(awk 'NF >= 4 && $2 ~ /\.elf$/' "$S" | wc -l | tr -d ' ')
identical=$(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 == "LOAD-IDENTICAL"' "$S" | wc -l | tr -d ' ')
moved=$(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 != "LOAD-IDENTICAL" {print $1"/"$2}' "$S" | tr '\n' ' ')
say "LOAD-IDENTICAL: $identical of $images shipped images; moved: ${moved:-none}"
grep -q "REBUILD FAILED" "$S" && fail "$(grep -c 'REBUILD FAILED' "$S") program(s) failed to rebuild"
[ -z "$moved" ] || fail "a shipped image moved: $moved"

# 3. The new program.
rm -rf "$PROGRAMS/norm_role"
(cd "$TT" && RELOCATE_KERNEL_TEXT=1 READER_CTARGS=2048,1 COMPUTE_CTARGS=1 WRITER_CTARGS=2048,1 \
    CB_TABLE=kernels/cb_tables/norm_role_float16b.cbtable \
    bash build_tensix_program.sh kernels/resident_norm_role_reader.c \
        kernels/resident_norm_role_compute.c kernels/resident_norm_role_writer.c \
        "$PROGRAMS/norm_role") > "$WORK/norm_role.build.log" 2>&1 || fail "norm_role build (see $WORK/norm_role.build.log)"
for pair in brisc:16384 ncrisc:11264 trisc0:14336 trisc1:12288 trisc2:12288; do
    image=${pair%%:*}; slab=${pair##*:}
    if [ -f "$PROGRAMS/norm_role/$image.elf" ]; then
        bytes=$(load_bytes "$PROGRAMS/norm_role/$image.elf")
        say "norm_role/$image.elf text + data $bytes B, slab $slab B"
        [ "$bytes" -lt "$slab" ] || fail "norm_role/$image.elf does not fit its slab"
    else
        fail "norm_role/$image.elf missing"
    fi
done

# 4. The blob from ../programs.
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
