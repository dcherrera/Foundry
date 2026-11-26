#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_timeline_program_gates.sh
#   The compile-side gates of the per-descriptor card TIMELINE (resident_queue_protocol.h,
#   FT_TIMELINE) and of the IDLE CLOCK fix that moved the gdn reader's DELTA clock off
#   control word 60. Needs the SFPI toolchain (7.68.0, the blob's), no card:
#
#     1. explode the committed blob (ft_tt_programs.h) into $WORK/blob: the shipped images
#     2. rebuild EVERY program from its own recipe and THIS tree's sources into ../programs;
#        every image LOAD-IDENTICAL to the shipped one EXCEPT the declared movers ($MOVERS,
#        none since the gdn DELTA move landed in the blob), which must DIFFER
#     3. the TIMELINE set: every program again with EXTRA_KERNEL_DEFINES=-DFT_TIMELINE=1 into
#        $TIMELINE (what PN_TT_PROGRAM_DIR points at on the card). Against step 2's images:
#        every resident reader (BRISC) and writer (NCRISC) image must DIFFER (else the define
#        never reached it), every TRISC and firmware image must be LOAD-IDENTICAL (the
#        timeline is data-movement only), and every kernel .text must fit the 12,288 B slab
#     4. regenerate the blob from ../programs: CHECK=1 (diff only) unless REGEN=1, which
#        writes ../ft_tt_programs.h and then runs the provenance check and the verifier
#
#   Usage (from tt/):  bash tools/check_timeline_program_gates.sh
#   Env:  SFPI      toolchain root (default /opt/tenstorrent/sfpi)
#         WORK      scratch (default ../build/timeline-gates, gitignored)
#         TIMELINE  the timeline program root (default ../build/timeline/programs)
#         REGEN     1 = write the regenerated ft_tt_programs.h
#         TL_KEEP_SHIPPED  images the timeline set keeps as shipped because the stamps do not
#                   fit their slab (default gdn/brisc.elf: 12,276 B shipped with the decode
#                   table gates, 13,512 B stamped). Its core times from the other processor's
#                   stream (the gdn writer); the host join skips a processor without magic.
#   On macOS: export SFPI=<install root> PATH="$(brew --prefix)/bin:$PATH" (bash >= 4)
#
#   Output: $WORK/gates.txt (the verdicts), $WORK/rebuild/ and $WORK/timeline/ (per-program
#           logs and summaries)
#   Exit:   0 every gate green; 1 any gate red; 2 setup error
#
#   Status: added 2026-11-17 with the card timeline.
#   Issues: None
#   Todo:   None
#
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
WORK=${WORK:-$TT/build/timeline-gates}
TIMELINE=${TIMELINE:-$TT/build/timeline/programs}
REGEN=${REGEN:-0}
PROGRAMS=$TT/programs
SLAB_BYTES=12288
# Declared shipped movers. The gdn DELTA-clock move (gdn/brisc.elf) is in the committed
# blob, so a rebuild must now move nothing.
MOVERS=${MOVERS:-}
TL_KEEP_SHIPPED=${TL_KEEP_SHIPPED-gdn/brisc.elf}
GATES=$WORK/gates.txt
red=0

say() { printf '%s\n' "$*" | tee -a "$GATES"; }
fail() { say "RED: $*"; red=1; }
text_bytes() { "$SFPI/compiler/bin/riscv-tt-elf-size" -A "$1" | awk '$1 == ".text" {print $2}'; }
source_of() { awk '$1 == "source" {print $2}' "$1.prov" 2>/dev/null; }
is_mover() { case " $MOVERS " in *" $1 "*) return 0 ;; esac; return 1; }
is_kept() { case " $TL_KEEP_SHIPPED " in *" $1 "*) return 0 ;; esac; return 1; }

[ -x "$SFPI/compiler/bin/riscv-tt-elf-gcc" ] || { echo "no SFPI compiler under $SFPI" >&2; exit 2; }
export SFPI
mkdir -p "$WORK"
: > "$GATES"
say "=== timeline program gates  $(date -u +%FT%TZ)  sfpi $("$SFPI/compiler/bin/riscv-tt-elf-gcc" -dumpversion)"

# 1. The shipped images, out of the committed blob.
make -C "$HERE" explode-programs > "$WORK/explode_build.log" 2>&1 || { echo "explode build failed" >&2; exit 2; }
rm -rf "$WORK/blob"
"$HERE/build/explode_tensix_program_blob" "$WORK/blob" --verify > "$WORK/explode.log" 2>&1 ||
    { echo "explode failed (see $WORK/explode.log)" >&2; exit 2; }
say "exploded $(ls "$WORK/blob" | wc -l | tr -d ' ') programs from ft_tt_programs.h"

# 2. Every program from this tree's sources: LOAD-IDENTICAL but for the declared movers.
rm -rf "$PROGRAMS" "$WORK/rebuild"
bash "$HERE/rebuild_all_tensix_programs.sh" "$WORK/blob" "$PROGRAMS" > /dev/null 2>&1
mkdir -p "$WORK/rebuild"
mv "$PROGRAMS"/*.rebuild.log "$PROGRAMS"/*.diff "$PROGRAMS"/summary.txt "$WORK/rebuild/" 2>/dev/null
rm -f "$PROGRAMS/DONE"
summary=$WORK/rebuild/summary.txt
grep -q "REBUILD FAILED" "$summary" && fail "$(grep -c 'REBUILD FAILED' "$summary") program(s) failed to rebuild"
images=$(awk 'NF >= 4 && $2 ~ /\.elf$/' "$summary" | wc -l | tr -d ' ')
identical=$(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 == "LOAD-IDENTICAL"' "$summary" | wc -l | tr -d ' ')
say "LOAD-IDENTICAL: $identical of $images images (every shipped program, kernels + firmware)"
[ "$images" -gt 0 ] || fail "no images rebuilt"
while read -r name file _ load _; do
    if is_mover "$name/$file"; then
        [ "$load" != LOAD-IDENTICAL ] || fail "$name/$file is a declared mover and did not move"
    else
        [ "$load" = LOAD-IDENTICAL ] || fail "$name/$file moved ($load) and is not a declared mover"
    fi
done < <(awk 'NF >= 4 && $2 ~ /\.elf$/' "$summary")
for mover in $MOVERS; do
    awk -v n="${mover%/*}" -v f="${mover#*/}" '$1 == n && $2 == f' "$summary" | grep -q . ||
        fail "declared mover $mover was not rebuilt"
done
say "declared movers: ${MOVERS:-none}"

# 3. The timeline set: the same recipes plus FT_TIMELINE, judged against step 2's images.
rm -rf "$TIMELINE" "$WORK/timeline"
mkdir -p "$TIMELINE"
EXTRA_KERNEL_DEFINES=-DFT_TIMELINE=1 \
    bash "$HERE/rebuild_all_tensix_programs.sh" "$WORK/blob" "$TIMELINE" > /dev/null 2>&1
mkdir -p "$WORK/timeline"
mv "$TIMELINE"/*.rebuild.log "$TIMELINE"/*.diff "$TIMELINE"/summary.txt "$WORK/timeline/" 2>/dev/null
rm -f "$TIMELINE/DONE"
grep -q "REBUILD FAILED" "$WORK/timeline/summary.txt" &&
    fail "$(grep -c 'REBUILD FAILED' "$WORK/timeline/summary.txt") timeline program(s) failed to build"
for entry in $TL_KEEP_SHIPPED; do
    [ -f "$PROGRAMS/$entry" ] || { fail "kept image $entry has no off-build twin"; continue; }
    cp -p "$PROGRAMS/$entry" "$TIMELINE/$entry"
    say "kept shipped (stamps do not fit the slab): $entry, $(text_bytes "$PROGRAMS/$entry") B"
done
stamped=0 kept=0 largest=0 largest_at=
for dir in "$TIMELINE"/*/; do
    name=$(basename "$dir")
    for image in "$dir"*.elf; do
        file=$(basename "$image")
        base=$PROGRAMS/$name/$file
        [ -f "$base" ] || { fail "$name/$file has no off-build twin"; continue; }
        same=$(python3 "$HERE/compare_tensix_load_images.py" "$base" "$image" | cut -d: -f1)
        src=$(source_of "$base")
        if is_kept "$name/$file"; then
            [ "$same" = LOAD-IDENTICAL ] || fail "$name/$file is kept shipped and moved"
            kept=$((kept + 1))
            continue
        fi
        case "$file" in
            brisc.elf | ncrisc.elf)
                size=$(text_bytes "$image")
                [ "${size:-0}" -gt "$largest" ] && { largest=$size; largest_at=$name/$file; }
                [ "${size:-0}" -lt "$SLAB_BYTES" ] || fail "$name/$file text $size B does not fit the $SLAB_BYTES B slab"
                case "$src" in
                    resident_queue.c | resident_puller.c | resident_exit_only.c | resident_*_probe.c | "")
                        [ "$same" = LOAD-IDENTICAL ] || fail "$name/$file ($src) moved but holds no timeline"
                        kept=$((kept + 1)) ;;
                    resident_*)
                        [ "$same" != LOAD-IDENTICAL ] || fail "$name/$file ($src) did not take FT_TIMELINE"
                        stamped=$((stamped + 1)) ;;
                    *)
                        [ "$same" = LOAD-IDENTICAL ] || fail "$name/$file ($src) moved but holds no timeline"
                        kept=$((kept + 1)) ;;
                esac ;;
            *)
                [ "$same" = LOAD-IDENTICAL ] || fail "$name/$file moved under FT_TIMELINE (TRISC/firmware must not)"
                kept=$((kept + 1)) ;;
        esac
    done
done
say "timeline set ($TIMELINE): $stamped resident data-movement images stamped, $kept images LOAD-IDENTICAL to the off build"
say "largest timeline data-movement .text: $largest B ($largest_at), slab $SLAB_BYTES B"
[ "$stamped" -gt 0 ] || fail "no image took FT_TIMELINE"

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
