#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# test_ingot_cli.sh — CLI and conversion-script checks on the files test_ingot_container makes
# (run it first). Run from ingot/: ./tests/test_ingot_cli.sh
set -uo pipefail
cd "$(dirname "$0")/.."
D=build/test_data
INGOT=build/ingot
pass=0 fail=0
check() {  # description, command...
    local what=$1; shift
    if "$@" >/dev/null 2>&1; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL: $what"; fi
}
refuse() {  # description, command... (must fail)
    local what=$1; shift
    if "$@" >/dev/null 2>&1; then fail=$((fail + 1)); echo "FAIL (should fail): $what"; else pass=$((pass + 1)); fi
}

[ -s "$D/base.safetensors" ] && [ -s "$D/tune.safetensors" ] || { echo "run test_ingot_container first"; exit 1; }
rm -rf "$D/cli_plain" "$D/cli_delta" "$D/cli_out_plain" "$D/cli_out_delta"
mkdir -p "$D/cli_plain" "$D/cli_delta"
cp "$D/base.safetensors" "$D/cli_plain/"
cp "$D/tune.safetensors" "$D/cli_delta/"

check "convert local dir, rebuild sha256" \
    scripts/ingot_convert.sh --rebuild-check -j 2 "$D/cli_plain" "$D/cli_out_plain"
check "convert log says verified + sha256-ok" grep -q "verified	sha256-ok" "$D/cli_out_plain/convert.tsv"
check "convert delta against an .ingot base" \
    scripts/ingot_convert.sh --rebuild-check -j 2 --base "$D/cli_out_plain/base.safetensors.ingot" \
    "$D/cli_delta" "$D/cli_out_delta"
check "delta file has delta tensors" grep -q "delta" <("$INGOT" info "$D/cli_out_delta/tune.safetensors.ingot")
check "verify against source" "$INGOT" verify "$D/cli_out_plain/base.safetensors.ingot" "$D/base.safetensors"
refuse "verify delta without base" "$INGOT" verify "$D/cli_out_delta/tune.safetensors.ingot"
refuse "verify against the wrong source" "$INGOT" verify "$D/cli_out_plain/base.safetensors.ingot" "$D/tune.safetensors"
refuse "verify corrupted file" "$INGOT" verify "$D/bad.safetensors.ingot"
refuse "unknown option" "$INGOT" encode --bogus a b

# Names: encode adds .ingot, decode strips it; files are recognised by magic, not name.
N="$D/cli_names"
rm -rf "$N" && mkdir -p "$N"
cp "$D/base.safetensors" "$N/m.safetensors"
check "encode default name adds .ingot" "$INGOT" encode "$N/m.safetensors"
check "encoded file is m.safetensors.ingot" test -s "$N/m.safetensors.ingot"
refuse "encode refuses an existing default output" "$INGOT" encode "$N/m.safetensors"
refuse "encode refuses an .ingot input" "$INGOT" encode "$N/m.safetensors.ingot" "$N/x.ingot"
refuse "decode refuses to overwrite the original" "$INGOT" decode "$N/m.safetensors.ingot"
mv "$N/m.safetensors" "$N/orig.safetensors"
check "decode default name strips .ingot" "$INGOT" decode "$N/m.safetensors.ingot"
check "decoded file is byte-identical" cmp "$N/m.safetensors" "$N/orig.safetensors"
cp "$N/m.safetensors.ingot" "$N/blob"
check "renamed file still verifies" "$INGOT" verify "$N/blob" "$N/orig.safetensors"
refuse "decode without .ingot suffix needs DST" "$INGOT" decode "$N/blob"
check "renamed file decodes with DST" "$INGOT" decode "$N/blob" "$N/blob.safetensors"
printf '\005' | dd of="$N/blob" bs=1 seek=4 conv=notrunc 2>/dev/null   # version 2 -> 5
refuse "unknown format version refused" "$INGOT" verify "$N/blob"
check "unknown version is named" grep -q "unsupported .ingot format version 5" \
    <("$INGOT" info "$N/blob" 2>&1)
check "non-ingot file is named" grep -q "not an .ingot file" \
    <("$INGOT" info "$N/orig.safetensors" 2>&1)

# GGUF: stdin encode, delta against a base, info, clear errors (files from test_ingot_stream).
G="$D/cli_gguf"
rm -rf "$G" && mkdir -p "$G"
[ -s "$D/s_base.gguf" ] && [ -s "$D/s_tune.gguf" ] || { echo "run test_ingot_stream first"; exit 1; }
check "encode a GGUF from stdin" sh -c "\"$INGOT\" encode -q - \"$G/b.gguf.ingot\" < \"$D/s_base.gguf\""
refuse "encode from stdin needs DST" sh -c "\"$INGOT\" encode -q - < \"$D/s_base.gguf\""
check "decode from stdin" sh -c "\"$INGOT\" decode -q - \"$G/b.gguf\" < \"$G/b.gguf.ingot\""
check "stdin round trip byte-identical" cmp "$G/b.gguf" "$D/s_base.gguf"
check "info shows quant tensors" grep -q "Q8_0	quant" <("$INGOT" info "$G/b.gguf.ingot")
check "info shows the source sha256" grep -q "^#source_sha256" <("$INGOT" info "$G/b.gguf.ingot")
BSIZE=$(wc -c < "$G/b.gguf.ingot" | tr -d " ")
check "measure: same size as the file, nothing written" grep -q "output_bytes $BSIZE	" \
    <("$INGOT" encode -q --measure --self-check - < "$D/s_base.gguf")
check "delta encode against a base GGUF" "$INGOT" encode -q --base "$D/s_base.gguf" "$D/s_tune.gguf" "$G/t.gguf.ingot"
check "delta header names the base" grep -q "^#base_sha256" <("$INGOT" info "$G/t.gguf.ingot")
check "delta verify with base and sha256 check" "$INGOT" verify -q --check-base --base "$D/s_base.gguf" "$G/t.gguf.ingot" "$D/s_tune.gguf"
refuse "delta verify without base" "$INGOT" verify -q "$G/t.gguf.ingot"
check "missing base names its sha256" grep -q "needs its base model: --base FILE with sha256" <("$INGOT" verify -q "$G/t.gguf.ingot" 2>&1)
refuse "delta verify with the wrong base" "$INGOT" verify -q --base "$D/s_tune.gguf" "$G/t.gguf.ingot"
head -c 5000 "$G/b.gguf.ingot" > "$G/cut.ingot"
check "truncated file is named" grep -q "truncated or incomplete download" <("$INGOT" verify -q "$G/cut.ingot" 2>&1)

# Predict (version 3): CLI encode, verify, stdin decode, the tensor log (files from
# test_ingot_predict).
P="$D/cli_pred"
rm -rf "$P" && mkdir -p "$P"
[ -s "$D/pred.safetensors" ] || { echo "run test_ingot_predict first"; exit 1; }
check "encode with prediction + tensor log" "$INGOT" encode -q --self-check --max-version 3 \
    --tensor-log "$P/log.tsv" "$D/pred.safetensors" "$P/p.ingot"
check "info names version 3 and predict tensors" sh -c \
    "\"$INGOT\" info \"$P/p.ingot\" | grep -q 'version 3' && \"$INGOT\" info \"$P/p.ingot\" | grep -q predict"
check "verify a version 3 file against its source" "$INGOT" verify -q "$P/p.ingot" "$D/pred.safetensors"
check "decode a version 3 file from stdin" sh -c "\"$INGOT\" decode -q - \"$P/p.st\" < \"$P/p.ingot\""
check "stdin version 3 round trip byte-identical" cmp "$P/p.st" "$D/pred.safetensors"
check "tensor log: one line per entry, 5 predicted below their v2 size" awk -F'\t' \
    'NR > 1 { n++ } $4 == "predict" && $5 < $6 && $7 > 0 { p++ } END { exit !(n == 9 && p == 5) }' \
    "$P/log.tsv"
check "policy plain predicts nothing" sh -c \
    "! \"$INGOT\" encode -q --measure --policy plain \"$D/pred.safetensors\" | grep -q predict"

# Match (version 4): the same file through the default writer.
check "encode version 4 (match)" "$INGOT" encode -q --self-check --tensor-log "$P/log4.tsv" \
    "$D/pred.safetensors" "$P/p4.ingot"
check "info names version 4 and match tensors" sh -c \
    "\"$INGOT\" info \"$P/p4.ingot\" | grep -q 'version 4' && \"$INGOT\" info \"$P/p4.ingot\" | grep -q match"
check "verify a version 4 file against its source" "$INGOT" verify -q "$P/p4.ingot" "$D/pred.safetensors"
check "decode a version 4 file from stdin" sh -c "\"$INGOT\" decode -q - \"$P/p4.st\" < \"$P/p4.ingot\""
check "stdin version 4 round trip byte-identical" cmp "$P/p4.st" "$D/pred.safetensors"
check "version 4 not larger than version 3" sh -c \
    "[ \$(wc -c < \"$P/p4.ingot\") -le \$(wc -c < \"$P/p.ingot\") ]"

echo "cli: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
