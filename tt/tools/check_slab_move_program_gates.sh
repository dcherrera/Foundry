#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_slab_move_program_gates.sh
#   The compile-side gates of a kernel-text slab move (ft_tt_op_l1_layout.h).
#   A moved slab relinks every image in it, so LOAD-IDENTICAL
#   cannot hold; this proves each relinked image is the shipped program at its new address.
#   Needs the SFPI toolchain (7.68.0, the blob's), no card:
#
#     1. explode the committed blob (ft_tt_programs.h) into $WORK/blob: the shipped images
#     2. rebuild EVERY program from its own provenance recipe and THIS tree into ../programs
#     3. per image: firmware LOAD-IDENTICAL; a kernel in an unmoved slab LOAD-IDENTICAL; a
#        kernel in a moved slab RELOC-IDENTICAL (tools/compare_tensix_relocated_images.py) with
#        delta = its slab's move; every kernel's text + data load image inside its new slab
#     4. regenerate the blob from ../programs: CHECK=1 (diff only) unless REGEN=1, which
#        writes ../ft_tt_programs.h and then runs the provenance check and the verifier
#
#   Usage (from tt/):  bash tools/check_slab_move_program_gates.sh
#   Env:  SFPI   toolchain root (default /opt/tenstorrent/sfpi)
#         WORK   scratch (default ../build/slab-gates, gitignored)
#         OLD_MAP  "brisc ncrisc trisc0 trisc1 trisc2" slab bases the blob was linked at
#                (default the pre-F0 map, 0x15000 0x18000 0x1B000 0x1E000 0x21000)
#         REGEN  1 = write the regenerated ft_tt_programs.h
#   On macOS: export SFPI=<install root> PATH="$(brew --prefix)/bin:$PATH" (bash >= 4)
#
#   Output: $WORK/gates.txt (the verdicts), $WORK/images.txt (one line per image),
#           $WORK/rebuild/ (per-program logs)
#   Exit:   0 every gate green; 1 any gate red; 2 setup error
#
#   Status: added 2026-11-19 with the asymmetric slabs (F0).
#   Issues: None
#   Todo:   None
#
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
WORK=${WORK:-$TT/build/slab-gates}
REGEN=${REGEN:-0}
OLD_MAP=${OLD_MAP:-0x15000 0x18000 0x1B000 0x1E000 0x21000}
PROGRAMS=$TT/programs
LAYOUT=$TT/ft_tt_op_l1_layout.h
GATES=$WORK/gates.txt
IMAGES=$WORK/images.txt
red=0

say() { printf '%s\n' "$*" | tee -a "$GATES"; }
fail() { say "RED: $*"; red=1; }
macro() { awk -v m="$1" '$1=="#define" && $2==m {gsub(/[uU]$/,"",$3); print $3; exit}' "$LAYOUT"; }
# The extent an image occupies in its slab: text plus every load image placed above it.
slab_extent() {
    python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import compare_tensix_relocated_images as c
base = int(sys.argv[3]); loads = c.read_elf_view(sys.argv[2])[1]
print(max(l[1] + l[2] for l in loads if base <= l[1] < base + 0x100000) - base)' "$HERE" "$1" "$2"
}

[ -x "$SFPI/compiler/bin/riscv-tt-elf-gcc" ] || { echo "no SFPI compiler under $SFPI" >&2; exit 2; }
export SFPI
mkdir -p "$WORK"
: > "$GATES"
: > "$IMAGES"
say "=== slab-move program gates  $(date -u +%FT%TZ)  sfpi $("$SFPI/compiler/bin/riscv-tt-elf-gcc" -dumpversion)"

config_base=$(macro FT_TT_OP_KERNEL_CONFIG_BASE)
declare -A new_base slab_bytes old_base
read -r -a old_bases <<< "$OLD_MAP"
index=0
for pair in brisc:DM0 ncrisc:DM1 trisc0:MATH0 trisc1:MATH1 trisc2:MATH2; do
    proc=${pair%%:*} name=${pair#*:}
    new_base[$proc]=$(( config_base + $(macro FT_TT_OP_TEXT_OFFSET_$name) ))
    slab_bytes[$proc]=$(( $(macro FT_TT_OP_TEXT_SLAB_BYTES_$name) ))
    old_base[$proc]=$(( ${old_bases[$index]} ))
    say "$proc slab 0x$(printf %X ${old_base[$proc]}) -> 0x$(printf %X ${new_base[$proc]}), ${slab_bytes[$proc]} B"
    index=$((index + 1))
done

# 1. The shipped images, out of the committed blob.
make -C "$HERE" explode-programs > "$WORK/explode_build.log" 2>&1 || { echo "explode build failed" >&2; exit 2; }
rm -rf "$WORK/blob"
"$HERE/build/explode_tensix_program_blob" "$WORK/blob" --verify > "$WORK/explode.log" 2>&1 ||
    { echo "explode failed (see $WORK/explode.log)" >&2; exit 2; }
say "exploded $(ls "$WORK/blob" | wc -l | tr -d ' ') programs from ft_tt_programs.h"

# 2. Every program from this tree.
rm -rf "$PROGRAMS" "$WORK/rebuild"
bash "$HERE/rebuild_all_tensix_programs.sh" "$WORK/blob" "$PROGRAMS" > /dev/null 2>&1
mkdir -p "$WORK/rebuild"
mv "$PROGRAMS"/*.rebuild.log "$PROGRAMS"/*.diff "$PROGRAMS"/summary.txt "$WORK/rebuild/" 2>/dev/null
rm -f "$PROGRAMS/DONE"
grep -q "REBUILD FAILED" "$WORK/rebuild/summary.txt" &&
    fail "$(grep -c 'REBUILD FAILED' "$WORK/rebuild/summary.txt") program(s) failed to rebuild"

# 3. Per image.
counts_load=0 counts_reloc=0 images=0
largest=""
for shipped in "$WORK"/blob/*/*.elf; do
    program=$(basename "$(dirname "$shipped")") file=$(basename "$shipped")
    proc=${file%.elf}
    rebuilt=$PROGRAMS/$program/$file
    images=$((images + 1))
    [ -f "$rebuilt" ] || { fail "$program/$file missing"; continue; }
    verdict=$(python3 "$HERE/compare_tensix_relocated_images.py" "$shipped" "$rebuilt")
    case $proc in
    fw_*)
        [ "$verdict" = LOAD-IDENTICAL ] || fail "$program/$file firmware moved: $verdict" ;;
    *)
        delta=$(( new_base[$proc] - old_base[$proc] ))
        extent=$(slab_extent "$rebuilt" "${new_base[$proc]}")
        if [ "$delta" = 0 ]; then
            [ "$verdict" = LOAD-IDENTICAL ] || fail "$program/$file in an unmoved slab: $verdict"
        else
            want="delta $( [ "$delta" -gt 0 ] && printf '+%#x' "$delta" || printf -- '-%#x' $(( -delta )) )"
            case $verdict in RELOC-IDENTICAL*"$want"*) ;; *) fail "$program/$file ($want): $verdict" ;; esac
        fi
        [ "$extent" -le "${slab_bytes[$proc]}" ] ||
            fail "$program/$file occupies $extent B of its ${slab_bytes[$proc]} B slab"
        printf '%s %s %s\n' "$extent" "$program/$file" "${slab_bytes[$proc]}" >> "$WORK/extents.txt" ;;
    esac
    case $verdict in LOAD-IDENTICAL) counts_load=$((counts_load + 1)) ;; RELOC-IDENTICAL*) counts_reloc=$((counts_reloc + 1)) ;; esac
    printf '%-28s %-12s %s\n' "$program" "$file" "$verdict" >> "$IMAGES"
done
say "images: $images; LOAD-IDENTICAL $counts_load, RELOC-IDENTICAL $counts_reloc"
for proc in brisc ncrisc trisc0 trisc1 trisc2; do
    say "largest $proc: $(grep " [^ ]*/$proc.elf " "$WORK/extents.txt" | sort -n | tail -1 | awk '{print $1" B ("$2") of "$3}')"
done
rm -f "$WORK/extents.txt"

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
