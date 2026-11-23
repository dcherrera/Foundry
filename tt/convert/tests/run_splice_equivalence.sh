#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# The splice EQUIVALENCE gate on a real checkpoint (the unit test runs it on a synthetic one).
#
#   splice(bfp8 rung, the bfp4 rung's own tiles for X and Y)  ==  convert(bfp8 + map X, Y = bfp4_b)
#
# byte for byte up to the directory's end, followed by exactly the splice record
# (2 x 176 + 128 = 480 B). Everything is written under build/splice_equivalence/ (gitignored);
# pass --keep to leave the artifacts there, otherwise they are deleted at the end.
#
#   ./run_splice_equivalence.sh <model.gguf> [X] [Y] [--keep]
#   default X = blk.1.ffn_gate.weight, Y = blk.3.ffn_down.weight

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
TOOLS="$HERE/../tools"
WORK="$HERE/build/splice_equivalence"
GGUF="${1:?usage: $0 <model.gguf> [X] [Y] [--keep]}"
X="${2:-blk.1.ffn_gate.weight}"
Y="${3:-blk.3.ffn_down.weight}"
KEEP="${4:-}"

make -C "$TOOLS" build/ft_tt_convert_gguf build/ft_tt_fttw_splice build/ft_tt_fttw_list >/dev/null
mkdir -p "$WORK"
printf '%s bfp4_b\n%s bfp4_b\n' "$X" "$Y" >"$WORK/xy.map"

run() { echo "+ $*"; "$@"; }
run "$TOOLS/build/ft_tt_convert_gguf" "$GGUF" "$WORK/base.bfp8.fttw" --ledger bfp8 --quiet
run "$TOOLS/build/ft_tt_convert_gguf" "$GGUF" "$WORK/rung.bfp4.fttw" --ledger bfp4 --quiet
run "$TOOLS/build/ft_tt_convert_gguf" "$GGUF" "$WORK/mapped.fttw" --ledger bfp8 --format-map "$WORK/xy.map" --quiet
run "$TOOLS/build/ft_tt_fttw_splice" --extract "$WORK/rung.bfp4.fttw" --tensor "$X" --tensor "$Y" \
    --out "$WORK/xy.mxtl"
echo "+ /usr/bin/time -l ft_tt_fttw_splice base --tile-file xy.mxtl --out spliced"
/usr/bin/time -l "$TOOLS/build/ft_tt_fttw_splice" "$WORK/base.bfp8.fttw" --tile-file "$WORK/xy.mxtl" \
    --out "$WORK/spliced.fttw" 2>"$WORK/splice_time.txt"
grep -E "maximum resident set size|real" "$WORK/splice_time.txt" || true

MAPPED_BYTES=$(stat -f%z "$WORK/mapped.fttw" 2>/dev/null || stat -c%s "$WORK/mapped.fttw")
SPLICED_BYTES=$(stat -f%z "$WORK/spliced.fttw" 2>/dev/null || stat -c%s "$WORK/spliced.fttw")
echo "mapped  $MAPPED_BYTES B"
echo "spliced $SPLICED_BYTES B (difference $((SPLICED_BYTES - MAPPED_BYTES)) B; the record is 480 B)"
"$TOOLS/build/ft_tt_fttw_list" "$WORK/spliced.fttw" | sed -n '1,3p;/census — MATMUL/,/payload total/p'
du -sh "$WORK"

# Negative control: the same comparison must see the unspliced base as different, or a PASS
# below would mean nothing.
# (head -c | cmp, not cmp -n: BSD cmp reports "EOF" when one file ends exactly at the limit.)
same_prefix() { head -c "$MAPPED_BYTES" "$1" | cmp -s - "$WORK/mapped.fttw"; }
if same_prefix "$WORK/base.bfp8.fttw"; then
    echo "control: FAIL — the comparison cannot tell the base from the mapped artifact"
    exit 1
fi
echo "control: the base differs from the mapped artifact"
VERDICT=FAIL
if [ $((SPLICED_BYTES - MAPPED_BYTES)) -eq 480 ] && same_prefix "$WORK/spliced.fttw"; then
    VERDICT=PASS
fi
echo "sha256 mapped              $(shasum -a 256 <"$WORK/mapped.fttw" | cut -d' ' -f1)"
echo "sha256 spliced[0, total)   $(head -c "$MAPPED_BYTES" "$WORK/spliced.fttw" | shasum -a 256 | cut -d' ' -f1)"
echo "equivalence: $VERDICT (bytes [0, $MAPPED_BYTES) identical, then the 480 B record)"
if [ "$KEEP" != "--keep" ]; then
    rm -f "$WORK"/*.fttw "$WORK"/*.mxtl
fi
[ "$VERDICT" = PASS ]
