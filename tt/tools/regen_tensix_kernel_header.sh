#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# regen_tensix_kernel_header.sh -- the ONE entry point that rebuilds every embedded Tensix
# kernel and regenerates foundry/tt/ft_tt_kernels.h from the sources under tt/kernels/.
#
# WHY THIS EXISTS. There was no single command that did this, and that is exactly how the
# committed header went stale: it shipped C++ ELFs from before the Foundry move for weeks
# after the C kernels replaced them, because "regenerate the blob" was four shell commands
# nobody had written down together. A header nobody can regenerate in one step is a header
# that drifts from its sources silently -- and the drift is not a build error, it is the
# card running last month's kernel.
#
# THE FIRMWARE MOVES WITH THE KERNELS, AND THAT IS THE DANGEROUS PART.
# A kernel's .text is linked at its own firmware's __fw_export_text_end (build_tensix_
# kernel.sh 3-4). Rebuild the kernels, leave the firmware in FT_TT_FIRMWARE_DIR alone, and
# the kernel entry addresses no longer match the firmware the tests boot -- which is a
# wedged core, not a failed call. On 2026-08-13 four of five installed images were stale by
# exactly this route (entry 0x4dd0 where the new build wanted 0x5000). So this script
# ALWAYS compares the firmware it just built against the installed set and REFUSES to
# finish quietly when they differ:
#
#   * default              -- regenerate the header, then FAIL loudly if the firmware in
#                             FT_TT_FIRMWARE_DIR is not what these kernels were linked
#                             against, naming every image that moved.
#   * INSTALL_FIRMWARE=1   -- back the installed set up FIRST (to a directory that is
#                             created fresh, never overwriting an existing backup), then
#                             install the new images.
#   * CHECK=1              -- build and diff against the committed header; write nothing.
#
# Usage:
#   tools/regen_tensix_kernel_header.sh [OUT_DIR]
#
#   OUT_DIR              scratch for the built ELFs        (default /tmp/ftk_regen)
#   FT_TT_FIRMWARE_DIR   where the tests boot firmware from (default $HOME/ttprog)
#   CHECK=1              diff only; do not write ft_tt_kernels.h and do not touch firmware
#   INSTALL_FIRMWARE=1   back up and install the rebuilt firmware
#
# Runs on a host with the SFPI toolchain installed.
#
set -u

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)

OUT=${1:-/tmp/ftk_regen}
FIRMWARE_DIR=${FT_TT_FIRMWARE_DIR:-$HOME/ttprog}
CHECK=${CHECK:-0}
INSTALL_FIRMWARE=${INSTALL_FIRMWARE:-0}
TOOLCHAIN=${TOOLCHAIN:-7.68.0}

HEADER="$TT/ft_tt_kernels.h"
STAGED="$OUT/ft_tt_kernels.h.new"

# The five firmware images a resident program boots. All five or none: they are one build.
FIRMWARE_IMAGES="fw_brisc.elf fw_ncrisc.elf fw_trisc0.elf fw_trisc1.elf fw_trisc2.elf"

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\nregen-kernels: FAILED -- %s\n' "$*" >&2; exit 1; }

cd "$TT" || die "cannot cd to $TT"

# ---------------------------------------------------------------------------
# 1. Build every embedded kernel. A fresh OUT, so a stale ELF cannot be picked up
#    and re-embedded -- the exact failure this script exists to end.
# ---------------------------------------------------------------------------
say "building into $OUT (removed first, so nothing stale can be embedded)"
rm -rf "$OUT" || die "cannot clear $OUT"
mkdir -p "$OUT" || die "cannot create $OUT"

log() { printf '%s\n' "--- $1"; }

log "noop_compute.c"
bash build_tensix_kernel.sh kernels/noop_compute.c "$OUT/noop" \
    > "$OUT/noop.log" 2>&1 || die "noop build (see $OUT/noop.log)"
log "delay_compute.c"
bash build_tensix_kernel.sh kernels/delay_compute.c "$OUT/delay" \
    > "$OUT/delay.log" 2>&1 || die "delay build (see $OUT/delay.log)"
log "runtime_arg_probe_compute.c"
bash build_tensix_kernel.sh kernels/runtime_arg_probe_compute.c "$OUT/runtime_arg_probe" \
    > "$OUT/runtime_arg_probe.log" 2>&1 || die "runtime_arg_probe build"
log "vector_add triad"
READER_CTARGS=0,1,2048 COMPUTE_CTARGS=0,1,16 WRITER_CTARGS=16,2048 \
    bash build_tensix_program.sh kernels/vector_add_reader.c kernels/vector_add_compute.c \
         kernels/vector_add_writer.c "$OUT/vector_add" \
    > "$OUT/vector_add.log" 2>&1 || die "vector_add build (see $OUT/vector_add.log)"

for image in noop/trisc0.elf noop/trisc1.elf noop/trisc2.elf \
             delay/trisc0.elf delay/trisc1.elf delay/trisc2.elf \
             runtime_arg_probe/trisc0.elf runtime_arg_probe/trisc1.elf \
             runtime_arg_probe/trisc2.elf \
             vector_add/brisc.elf vector_add/ncrisc.elf vector_add/trisc0.elf \
             vector_add/trisc1.elf vector_add/trisc2.elf; do
    [ -s "$OUT/$image" ] || die "no ELF at $OUT/$image -- the build did not produce it"
done

# ---------------------------------------------------------------------------
# 2. The four builds must have produced ONE firmware, not four. A kernel links
#    against its own tree's firmware, so if the trees disagree the header would
#    carry images with mutually incompatible entry addresses.
# ---------------------------------------------------------------------------
say "the four builds agree on the TRISC firmware"
for id in 0 1 2; do
    reference="$OUT/vector_add/fw_trisc$id.elf"
    for tree in noop delay runtime_arg_probe; do
        cmp -s "$reference" "$OUT/$tree/fw_trisc$id.elf" \
            || die "$tree/fw_trisc$id.elf differs from vector_add's -- the trees were not
        built against one firmware, and the embedded kernels' entry addresses cannot all
        be right"
    done
    printf '    fw_trisc%s.elf identical across all four trees\n' "$id"
done

# ---------------------------------------------------------------------------
# 3. Emit the header.
# ---------------------------------------------------------------------------
say "emitting the header"
python3 tools/emit_tensix_kernel_header.py -o "$STAGED" \
  --program noop --toolchain "$TOOLCHAIN" --ctargs none \
    --image MATH0="$OUT/noop/trisc0.elf" --image MATH1="$OUT/noop/trisc1.elf" \
    --image MATH2="$OUT/noop/trisc2.elf" \
  --program delay --toolchain "$TOOLCHAIN" --ctargs none \
    --image MATH0="$OUT/delay/trisc0.elf" --image MATH1="$OUT/delay/trisc1.elf" \
    --image MATH2="$OUT/delay/trisc2.elf" \
  --program runtime_arg_probe --toolchain "$TOOLCHAIN" --ctargs none \
    --image MATH0="$OUT/runtime_arg_probe/trisc0.elf" \
    --image MATH1="$OUT/runtime_arg_probe/trisc1.elf" \
    --image MATH2="$OUT/runtime_arg_probe/trisc2.elf" \
  --program vector_add --toolchain "$TOOLCHAIN" \
    --ctargs "DM0=0,1,2048 DM1=16,2048 MATH0..2=0,1,16" \
    --image DM0="$OUT/vector_add/brisc.elf" --image DM1="$OUT/vector_add/ncrisc.elf" \
    --image MATH0="$OUT/vector_add/trisc0.elf" --image MATH1="$OUT/vector_add/trisc1.elf" \
    --image MATH2="$OUT/vector_add/trisc2.elf" \
  || die "the emitter refused"

# ---------------------------------------------------------------------------
# 4. Say what moved, ALWAYS -- before anything is written.
# ---------------------------------------------------------------------------
say "against the committed header"
if cmp -s "$STAGED" "$HEADER"; then
    printf '    IDENTICAL: same sources reproduced the committed ft_tt_kernels.h byte for byte\n'
    header_changed=0
else
    printf '    DIFFERS. Changed lines (the ELF payload is one byte-array line per 12 bytes):\n'
    diff "$HEADER" "$STAGED" | head -40
    printf '    ... %s changed lines total\n' "$(diff "$HEADER" "$STAGED" | grep -c '^[<>]')"
    header_changed=1
fi

# ---------------------------------------------------------------------------
# 5. The firmware. Compared ALWAYS, because a regen that silently desyncs the
#    firmware is worse than no regen at all.
# ---------------------------------------------------------------------------
say "against the installed firmware in $FIRMWARE_DIR"
firmware_stale=0
for image in $FIRMWARE_IMAGES; do
    built="$OUT/vector_add/$image"
    installed="$FIRMWARE_DIR/$image"

    if [ ! -f "$installed" ]; then
        printf '    %-16s MISSING from %s\n' "$image" "$FIRMWARE_DIR"
        firmware_stale=1
    elif cmp -s "$built" "$installed"; then
        printf '    %-16s matches\n' "$image"
    else
        printf '    %-16s STALE -- installed differs from what these kernels link against\n' \
               "$image"
        firmware_stale=1
    fi
done

if [ "$CHECK" != "0" ]; then
    say "CHECK=1: nothing written"
    [ "$firmware_stale" = 0 ] || die "the installed firmware does not match this build"
    exit $header_changed
fi

# ---------------------------------------------------------------------------
# 6. Install. Firmware FIRST and only on request, because installing it is the
#    step that can wedge a core.
# ---------------------------------------------------------------------------
if [ "$firmware_stale" != 0 ]; then
    if [ "$INSTALL_FIRMWARE" = "0" ]; then
        cp "$STAGED" "$HEADER" || die "cannot write $HEADER"
        printf '\nregen-kernels: wrote %s\n' "$HEADER"
        die "the firmware in $FIRMWARE_DIR is NOT what these kernels were linked against.
        The header is regenerated but the pair is DESYNCED: booting that firmware with
        these kernels is a wedged core, not a failed call. Re-run with INSTALL_FIRMWARE=1
        to back the old set up and install the new one, or point FT_TT_FIRMWARE_DIR at a
        directory built from this same tree."
    fi

    stamp=$(date +%Y%m%d)
    backup="$FIRMWARE_DIR.bak.$stamp"
    suffix=1
    # NEVER overwrite an existing backup: the previous good set is the only way back from
    # a firmware install that turns out to be wrong.
    while [ -e "$backup" ]; do
        backup="$FIRMWARE_DIR.bak.$stamp.$suffix"
        suffix=$((suffix + 1))
    done
    say "backing the installed firmware up to $backup"
    cp -a "$FIRMWARE_DIR" "$backup" || die "backup failed -- nothing installed"
    for image in $FIRMWARE_IMAGES; do
        cp "$OUT/vector_add/$image" "$FIRMWARE_DIR/$image" || die "installing $image failed"
        printf '    installed %s\n' "$image"
    done
fi

cp "$STAGED" "$HEADER" || die "cannot write $HEADER"
printf '\nregen-kernels: wrote %s\n' "$HEADER"
exit 0
