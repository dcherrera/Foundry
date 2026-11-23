#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# The splice suite's MUTATION BATTERY (coding-best-practices.md rule 8).
#
# test_artifact_splice is rebuilt against copies of tt/ in which exactly ONE property of
# SHA-256, the MXTL reader/writer or the splice has been broken; each damaged build MUST go
# red. The unmutated copy runs first and must be green, or "every arm went red" would also
# describe a suite that always fails. A mutation that changes nothing, or does not build, is
# an ERROR, never a caught arm.
#
# Copies live under build/ (gitignored) and are rebuilt from nothing per arm: whole-program
# links, no objects, so GNU make 3.81's one-second mtime trap cannot serve a stale binary.
#
#   ./run_splice_mutations.sh            # from tt/convert/tests

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
TT_DIR="$(cd "$HERE/../.." && pwd)"
WORK="$HERE/build/splice_mutations"
CAUGHT=0
ERRORS=0

# $1 = name, $2 = file under tt/convert/, $3 = python expression old, $4 = new ("" = control)
run_arm() {
    local name="$1" file="$2" old="$3" new="$4"
    local root="$WORK/arm" out rc failed

    rm -rf "$root"
    mkdir -p "$root"
    rsync -a --exclude build --exclude '*.dSYM' "$TT_DIR/" "$root/"
    if [ -n "$old" ]; then
        if ! python3 - "$root/convert/$file" "$old" "$new" <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read()
if text.count(old) != 1:
    sys.exit(1)
open(path, "w").write(text.replace(old, new))
EOF
        then
            echo "  ERROR  $name: the pattern is not in $file exactly once"
            ERRORS=$((ERRORS + 1))
            return
        fi
    fi
    if ! out=$(cd "$root/convert/tests" && make build/test_artifact_splice 2>&1); then
        echo "  ERROR  $name: the mutated tree does not build"
        echo "$out" | grep -E 'error' | head -2 | sed 's/^/           /'
        ERRORS=$((ERRORS + 1))
        return
    fi
    out=$(cd "$root/convert/tests" && ./build/test_artifact_splice 2>&1)
    rc=$?
    failed=$(echo "$out" | sed -n 's/^[0-9]* checks, \([0-9]*\) failed$/\1/p')
    if [ -z "$old" ]; then
        if [ $rc -eq 0 ]; then
            echo "  green  control: ${failed:-?} failed"
        else
            echo "  ERROR  control is red (${failed:-?} failed) — nothing below means anything"
            ERRORS=$((ERRORS + 1))
        fi
    elif [ $rc -ne 0 ]; then
        echo "  red    $name: ${failed:-crashed} checks failed"
        CAUGHT=$((CAUGHT + 1))
    else
        echo "  MISSED $name: the suite stayed green"
        ERRORS=$((ERRORS + 1))
    fi
}

S=ft_tt_artifact_splice.c
T=ft_tt_tile_file.c
run_arm control "" "" ""
run_arm "sha256: one round constant" ft_sha256.c "0x428a2f98u" "0x428a2f99u"
run_arm "writer: format and layout swapped" $T "put_u32(fixed, entry->format);" "put_u32(fixed, entry->layout);"
run_arm "reader: sha256 not compared" $T "memcmp(got, want, FT_SHA256_BYTES) == 0 ?" "(memcmp(got, want, FT_SHA256_BYTES) == 0 || 1) ?"
run_arm "reader: reserved not checked" $T "get_u32(header + 12) != 0u" "0"
run_arm "reader: trailing bytes accepted" $T "rc == FT_TT_TILE_FILE_OK && cursor != (uint64_t)status.st_size" "0"
run_arm "reader: duplicates accepted" $T "if (strcmp(file->entries[j].name, entry->name) == 0) {" "if (0) {"
run_arm "reader: payload bound not checked" $T "if (entry->n_bytes > size - *cursor)" "if (0)"
run_arm "splice: role rule dropped" $S "    if (refusal) {
        splice_error" "    if (0) {
        splice_error"
run_arm "splice: format rule dropped" $S "tile->format != FT_TT_FORMAT_BFP4_B && tile->format != FT_TT_FORMAT_BFP8_B" "0"
run_arm "splice: layout rule dropped" $S "if (tile->layout != target->layout) {" "if (0) {"
run_arm "splice: size rule dropped" $S "if (want == 0u || tile->n_bytes != want) {" "if (want == 0u) {"
run_arm "splice: size ignores the format" $S "format == FT_TT_FORMAT_BFP4_B ? FT_BFP_BFP4_B : FT_BFP_BFP8_B" "format == FT_TT_FORMAT_BFP4_B ? FT_BFP_BFP4_B : FT_BFP_BFP4_B"
run_arm "splice: cross-file duplicates accepted" $S "if (by_entry[index].file) {" "if (0) {"
run_arm "splice: same file by path only" $S "return stat(a, &sa) == 0" "return 0 && stat(a, &sa) == 0"
run_arm "splice: output may be the source" $S "if (splice_same_file(output_path, source_path)) {" "if (splice_same_file(output_path, source_path) && 0) {"
run_arm "splice: re-splicing accepted" $S "rc == FT_TT_SPLICE_OK && source.has_splice_record" "0"
run_arm "splice: unknown tail accepted" $S "    if (get_u32(magic) != FT_TT_PROVENANCE_MAGIC || tail != SPLICE_PROVENANCE_BYTES)
        goto unknown;" "    if (get_u32(magic) != FT_TT_PROVENANCE_MAGIC || tail != SPLICE_PROVENANCE_BYTES)
        return FT_TT_SPLICE_OK;"
run_arm "splice: base trailer not verified" $S "ft_tt_artifact_provenance_verify_file(path, &record, why, sizeof(why)) != FT_TT_PROVENANCE_OK" "0"
run_arm "splice: no alignment padding" $S "return (value + FT_TT_ARTIFACT_ALIGNMENT - 1u) / FT_TT_ARTIFACT_ALIGNMENT * FT_TT_ARTIFACT_ALIGNMENT;" "return value;"
run_arm "splice: directory format not updated" $S "directory[i].format = tile->format;" ""
run_arm "splice: source payload kept" $S "int rc = replacement->file" "int rc = 0"
run_arm "splice: record not written" $S "rc = splice_write_record(out, source, by_entry, replaced);" "rc = 1 ? FT_TT_SPLICE_OK : splice_write_record(out, source, by_entry, replaced);"
run_arm "splice: base trailer not kept" $S "        memcpy(footer, source->provenance, SPLICE_PROVENANCE_BYTES);" "        ;"
run_arm "splice: record magic wrong" $S "    put_u32(footer + 124, FT_TT_SPLICE_MAGIC);" "    put_u32(footer + 124, FT_TT_PROVENANCE_MAGIC);"

echo
echo "caught $CAUGHT, errors/missed $ERRORS"
[ $ERRORS -eq 0 ]
