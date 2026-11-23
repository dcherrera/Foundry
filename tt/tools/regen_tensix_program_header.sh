#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# regen_tensix_program_header.sh -- the ONE entry point that regenerates
# foundry/tt/ft_tt_programs.h from the program directories under foundry/tt/programs/.
#
# WHY THIS EXISTS, IN ONE SENTENCE: a binary that reads its Tensix programs out of
# one absolute programs directory runs on exactly one machine, and this is the command that
# puts them inside the binary instead.
#
# It is the program-level twin of regen_tensix_kernel_header.sh, and it inherits that
# script's discipline for the same reasons:
#
#   * IT REFUSES A STALE DIRECTORY rather than embedding it, BY CONTENT. Every image
#     carries a `.prov` record hashing the files its compile opened and the recipe that
#     reached code generation, and the emitter re-hashes them against the tree before it
#     writes anything (tools/tensix_image_provenance.py). The `.src` sidecar's mtime
#     comparison is still run and is still useful for a source that has VANISHED, but it is
#     not the staleness check any more: an `explode` resets every mtime, which is how 21
#     directories reported `current` with two of them provably stale. A stale directory on
#     disk costs a re-run. A stale directory BAKED INTO THE BINARY survives a rebuild.
#   * IT WRITES NOTHING WHEN IT REFUSES. The emitter builds the whole header in memory
#     before it opens the output, so a refusal cannot leave a half-written header behind.
#   * IT SAYS WHAT MOVED, ALWAYS, before anything is written.
#
# WHAT IT DOES **NOT** DO, so the guarantee is not overread: it does not build the programs.
# `build_tensix_program.sh` does that, per program, with its own CTARGS and CB table — there
# is no single recipe that reconstructs all of them, and inventing one here would be a
# second source of truth for how each program is built. This script embeds what is there and
# refuses what it cannot vouch for.
#
# Usage:
#   tools/regen_tensix_program_header.sh [program-dir ...]
#
#   With no arguments it takes every directory under $PROGRAMS that holds a brisc.elf.
#
#   PROGRAMS   where the program directories are      (default ../programs)
#   KERNELS    the kernel source tree the sidecars name (default ../kernels)
#   TOOLCHAIN  recorded in the blob for logs          (default 7.68.0)
#   CHECK=1    diff against the committed header only; write nothing
#
# Runs on the card host: the program directories and the SFPI builds that produced them are
# there. THIS HAS NEVER BEEN RUN FOR REAL -- it was written on the Mac and gated against
# synthetic fixtures (tests/Makefile.programblob). The first real run needs a bank session
# and should be followed by `verify_tensix_program_blob ../programs`.
#
set -u

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)

PROGRAMS=${PROGRAMS:-$TT/programs}
KERNELS=${KERNELS:-$TT/kernels}
TOOLCHAIN=${TOOLCHAIN:-7.68.0}
CHECK=${CHECK:-0}

HEADER="$TT/ft_tt_programs.h"
STAGED=$(mktemp "${TMPDIR:-/tmp}/ft_tt_programs.XXXXXX.h")

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\nregen-programs: FAILED -- %s\n' "$*" >&2; rm -f "$STAGED"; exit 1; }

[ -d "$KERNELS" ] || die "no kernel source tree at $KERNELS -- nothing can be checked
        against it, and a program directory nobody can check must not be embedded"

# ---------------------------------------------------------------------------
# 1. Which directories. Named on the command line, or every one that looks like a
#    build_tensix_program.sh output.
# ---------------------------------------------------------------------------
DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
    [ -d "$PROGRAMS" ] || die "no program directory tree at $PROGRAMS"
    while IFS= read -r elf; do
        DIRS+=("$(dirname "$elf")")
    done < <(find "$PROGRAMS" -mindepth 2 -maxdepth 2 -name brisc.elf | sort)
fi
[ ${#DIRS[@]} -gt 0 ] || die "no program directories found under $PROGRAMS. An EMPTY blob is
        a real state, but it is not something a glob should be able to produce by accident:
        ask for it explicitly with
            python3 $HERE/emit_tensix_program_header.py -o $HEADER --allow-empty"

say "embedding ${#DIRS[@]} program director(y|ies)"
for dir in "${DIRS[@]}"; do
    printf '    %s\n' "$dir"
done

# ---------------------------------------------------------------------------
# 2. Emit. The emitter runs check_tensix_program_sources.sh itself and refuses on a
#    stale sidecar, a missing firmware image, a duplicate name or an unloadable ELF.
# ---------------------------------------------------------------------------
say "emitting (the stale-source check runs inside this step)"
ARGS=()
for dir in "${DIRS[@]}"; do
    ARGS+=(--program "$dir")
done
python3 "$HERE/emit_tensix_program_header.py" -o "$STAGED" --recorded-output "$HEADER" \
    --kernels "$KERNELS" --toolchain "$TOOLCHAIN" \
    --recipe "built by build_tensix_program.sh; see the log beside the directory" \
    "${ARGS[@]}" || die "the emitter refused. Nothing was written."

# ---------------------------------------------------------------------------
# 3. The firmware sets, reported ALWAYS. Programs built from one tree share one set; more
#    than one set is legal and is carried correctly, but it means a run that arms two
#    families from different sets will be refused at load, which is worth knowing here
#    rather than on the card.
# ---------------------------------------------------------------------------
say "firmware sets in this blob"
grep -c 'static const ft_tt_program_blob_firmware' "$STAGED" \
    | awk '{printf "    %s distinct firmware set(s)\n", $1}'
grep 'first seen in' "$STAGED" | sed 's/^/    /'
if [ "$(grep -c 'static const ft_tt_program_blob_firmware' "$STAGED")" -gt 1 ]; then
    printf '    NOTE: these programs were NOT all built against one firmware. A kernel is
        linked at its own firmware'"'"'s __fw_export_text_end, so any run that boots one
        set and loads another set'"'"'s kernels is a wedged core -- and is refused. If that
        was not intended, rebuild them together.\n'
fi

# ---------------------------------------------------------------------------
# 4. Against the committed header, before anything is written.
# ---------------------------------------------------------------------------
say "against the committed header"
if cmp -s "$STAGED" "$HEADER"; then
    printf '    IDENTICAL: these directories reproduce the committed ft_tt_programs.h\n'
    header_changed=0
else
    printf '    DIFFERS. %s changed lines (the payload is one line per 12 bytes):\n' \
           "$(diff "$HEADER" "$STAGED" 2>/dev/null | grep -c '^[<>]')"
    diff "$HEADER" "$STAGED" 2>/dev/null | grep '^[<>]' | grep -v '0x' | head -30
    header_changed=1
fi

if [ "$CHECK" != "0" ]; then
    say "CHECK=1: nothing written"
    rm -f "$STAGED"
    exit $header_changed
fi

cp "$STAGED" "$HEADER" || die "cannot write $HEADER"
rm -f "$STAGED"
printf '\nregen-programs: wrote %s\n' "$HEADER"
printf 'regen-programs: NOW PROVE IT, by comparison and not by construction:\n'
printf '    make -f tools/Makefile verify-programs\n'
printf '    tools/build/verify_tensix_program_blob %s\n' "$PROGRAMS"
exit 0
