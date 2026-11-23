#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# The artifact cache's MUTATION BATTERY.
#
# Every check in test_artifact_cache.c is re-run against a build in which exactly ONE
# property of the code under test has been broken, and each damaged build MUST go red. A
# gate never seen failing is not evidence — twelve vacuous gates have been caught on this
# project, two of them inside one round.
#
# What each arm is really asking: could this suite have told the difference between a cache
# that serves the right artifact and one that serves whatever file happens to be at the
# path? That failure is SILENT — a stale artifact is the same SIZE, the same shape and the
# same tensor list as a fresh one, and the model it produces still answers — so the answer
# has to come from here.
#
# Each mutation works on a SCRATCH COPY of tt/ with its own build directory, so the real
# tree is never edited and GNU make 3.81's one-second mtime granularity cannot serve a stale
# binary (coding-best-practices.md §9). The whole-program link means there are no object
# files to go stale either.
#
# A MUTATION THAT DOES NOT COMPILE IS NOT A RED ARM. The build and the run are separate
# steps here, and a build failure is reported as an ERROR rather than counted as the suite
# catching something — that mistake has already been made once on this project.
#
# It runs the UNMUTATED build first. Without that control, "every mutation went red" would
# be consistent with a suite that fails unconditionally.

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
CONVERT_DIR="$(cd "$HERE/.." && pwd)"
TT_DIR="$(cd "$CONVERT_DIR/.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}/ft_tt_artifact_cache_gates.$$"
PASS=0
FAIL=0

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

# $1 = name, $2 = file to patch (relative to tt/convert/), $3 = sed expression, $4 = want
run_arm() {
    local name="$1" file="$2" expr="$3" want="$4"
    local root="$SCRATCH/arm"
    local rc out

    rm -rf "$root"
    mkdir -p "$root"
    cp -R "$TT_DIR/." "$root/" 2>/dev/null
    rm -rf "$root"/convert/tests/build "$root"/convert/tools/build "$root"/tests/build* \
           "$root"/tools/build "$root"/build 2>/dev/null

    if [ -n "$expr" ]; then
        if ! sed -i.bak "$expr" "$root/convert/$file"; then
            echo "  ERROR: could not apply the mutation to $file"
            FAIL=$((FAIL + 1))
            return
        fi
        if cmp -s "$root/convert/$file" "$root/convert/$file.bak"; then
            # A mutation that changed nothing would produce a GREEN run and be scored as
            # "the gate did not catch it". Refuse instead.
            echo "  ERROR: the mutation did not change $file — the pattern has drifted"
            FAIL=$((FAIL + 1))
            return
        fi
        rm -f "$root/convert/$file.bak"
    fi

    if ! out=$(cd "$root/convert/tests" && make build/test_artifact_cache 2>&1); then
        echo "  ERROR: the mutated tree did not BUILD — this arm measures nothing: $name"
        echo "$out" | grep -E 'error|Error' | sed 's/^/          /' | head -3
        FAIL=$((FAIL + 1))
        return
    fi

    out=$(cd "$root/convert/tests" && ./build/test_artifact_cache 2>&1)
    rc=$?

    if [ "$want" = "green" ]; then
        if [ $rc -eq 0 ]; then
            echo "  ok    CONTROL (unmutated) passes"
            PASS=$((PASS + 1))
        else
            echo "  FAIL  CONTROL (unmutated) did NOT pass — every arm below is meaningless"
            echo "$out" | tail -20
            FAIL=$((FAIL + 1))
        fi
        return
    fi

    if [ $rc -ne 0 ]; then
        echo "  ok    RED as required: $name"
        echo "$out" | grep '^  FAIL' | sed 's/^/          /' | head -3
        PASS=$((PASS + 1))
    else
        echo "  FAIL  the suite stayed GREEN with: $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== artifact cache, provenance and rung-selection mutation battery ==="
run_arm "control" "" "" green

# --- the cache's freshness decision -----------------------------------------
# THE ARM THAT MATTERS MOST. This is the shape of the two green-measurement-on-a-stale-
# artifact failures already in this project's record: an artifact of the right size at the
# right path, served without being read.
run_arm "any readable trailer is accepted as current" \
    ft_tt_artifact_cache.c \
    's|    if (ft_tt_artifact_provenance_verify_file(path, got, note, note_bytes) != FT_TT_PROVENANCE_OK)|    note[0] = 0; return 1; if (ft_tt_artifact_provenance_verify_file(path, got, note, note_bytes) != FT_TT_PROVENANCE_OK)|' red

run_arm "the trailer is read but its disagreement is never acted on" \
    ft_tt_artifact_cache.c \
    's|    if (ft_tt_artifact_provenance_compare(want, got, note, note_bytes) != FT_TT_PROVENANCE_OK)|    if (ft_tt_artifact_provenance_compare(want, got, note, note_bytes) == 12345)|' red

run_arm "provenance comparison always agrees" \
    ft_tt_artifact_provenance.c \
    's|    PROVENANCE_COMPARE_U64(source_digest, "source tensor directory digest");|    return FT_TT_PROVENANCE_OK;|' red

run_arm "the artifact is never checked against its own length" \
    ft_tt_artifact_provenance.c \
    's|    if ((uint64_t)size != record->payload_bytes + PROVENANCE_ON_DISK_BYTES) {|    if (0) {|' red

run_arm "a trailer that disagrees with itself is accepted" \
    ft_tt_artifact_provenance.c \
    's|    if (out->trailer_bytes != PROVENANCE_ON_DISK_BYTES \|\||    if (0 \&\&|' red

run_arm "the trailer is written with no magic, so nothing is ever cached" \
    ft_tt_artifact_provenance.c \
    's|    record.magic = FT_TT_PROVENANCE_MAGIC;|    record.magic = 0u;|' red

# --- the source digest ------------------------------------------------------
# A digest over the names alone would pass every test that renames a tensor and miss every
# re-quantization, which is the commonest way a checkpoint changes under an artifact.
run_arm "the digest ignores tensor shapes" \
    ft_tt_artifact_provenance.c \
    's|            hash = provenance_fnv64_u64(hash, tensor->dims\[d\]);|            hash = provenance_fnv64_u64(hash, 0u);|' red

run_arm "the digest ignores tensor names" \
    ft_tt_artifact_provenance.c \
    's|        hash = provenance_fnv64(hash, tensor->name, strlen(tensor->name) + 1u);||' red

# --- the rung selector ------------------------------------------------------
run_arm "the selector always takes the first rung, fit or not" \
    ft_tt_capacity_plan.c \
    's|        if (needed > budget->card_bytes)|        if (0)|' red

run_arm "the ladder is walked cheapest-first, so everything gets bfp4" \
    ft_tt_capacity_plan.c \
    's|    FT_TT_LEDGER_BFP8, FT_TT_LEDGER_BFP4|    FT_TT_LEDGER_BFP4, FT_TT_LEDGER_BFP8|' red

run_arm "the reserve is ignored, so context costs nothing" \
    ft_tt_capacity_plan.c \
    's|        budget->runtime_state_bytes + budget->kv_bytes_per_token \* budget->context_tokens;|        0u;|' red

run_arm "the refusal names no rung and shows no per-rung cost" \
    ft_tt_capacity_plan.c \
    's|ft_tt_ledger_mode_name(rung->mode),|ft_tt_ledger_mode_name((ft_tt_ledger_mode)99),|' red

run_arm "an unclassified checkpoint is reported as a capacity problem" \
    ft_tt_capacity_plan.c \
    's|    if (out->rungs\[0\].unclassified_2d > 0u) {|    if (0) {|' red

# --- the fit arithmetic -----------------------------------------------------
# The planner decides whether a model fits WITHOUT converting it. If its arithmetic drifts
# from the converter's, the first sign is a serve that overruns the card after a
# twenty-minute conversion.
# The 1-D branch, NOT the tiled one: every tile size on the ladder is already a multiple of
# 64, so removing the tiled alignment changes no number and an arm on it stays green — which
# is a property of the arithmetic, not a hole in the suite. Said out loud in the source.
run_arm "1-D tensor sizes drop the artifact's alignment padding" \
    ft_tt_capacity_plan.c \
    's|        return capacity_align_up(cols \* sizeof(float), FT_TT_ARTIFACT_ALIGNMENT);|        return cols * sizeof(float);|' red

run_arm "a partial tile is not counted as a whole one" \
    ft_tt_capacity_plan.c \
    's|    return (elements + 31u) / 32u;|    return elements / 32u;|' red

# --- the rotary pairing -----------------------------------------------------
# Two artifacts converted with different pairings are the SAME SIZE and hold the SAME
# TENSORS. Nothing but the flag distinguishes them, and getting it wrong costs 22%
# perplexity on every chat-templated prompt.
run_arm "the pairing is left out of the cached artifact's key" \
    ft_tt_artifact_cache.c \
    's|    want->rope_pairing = (uint32_t)rope_pairing;|    want->rope_pairing = 0u;|' red

run_arm "the header flags are left out of the key" \
    ft_tt_artifact_cache.c \
    's|    want->artifact_flags =$|    want->artifact_flags = 0u; if (0) want->artifact_flags =|' red

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
