#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# build_tensix_dataflow_kernel.sh
#   Compiles ONE data-movement kernel -- BRISC (NoC 0) or NCRISC (NoC 1) -- into an ELF,
#   together with the processor's firmware ELF whose exported symbols that link needs, from
#   Foundry sources only (tt/fw/, tt/kernels/) with the SFPI compiler. build_tensix_kernel.sh
#   does the compute (TRISC0/1/2) part of the triad. Both take every architecture- and
#   board-shaped value from tensix_kernel_build_arch.sh.
#
#   Usage:  build_tensix_dataflow_kernel.sh <kernel.c> <brisc|ncrisc> [outdir]
#   Env:    TT_ARCH / TT_NUM_L1_BANKS / TT_NUM_DRAM_BANKS
#                   as build_tensix_kernel.sh; see tensix_kernel_build_arch.sh
#           CTARGS  comma-separated KERNEL_COMPILE_TIME_ARGS (default 0)
#           NOC     NoC index this kernel issues on (default 0 for brisc, 1 for ncrisc)
#           KERNEL_DEFINES / LIGHTWEIGHT_ASSERTS  as build_tensix_kernel.sh
#           KERNEL_TEXT_BASE  link this kernel's text at an explicit L1 address instead of at
#                   __fw_export_text_end (build_tensix_kernel.sh's KERNEL TEXT RELOCATION
#                   block). NCRISC's firmware tail is small
#   Output: <outdir>/<proc>.elf and <outdir>/fw_<proc>.elf, with .src / .prov records.
#
#   Differences from the compute builder: -mcpu tt-bh (no Tensix co-processor
#   instructions); -Os; -fno-tree-loop-distribute-patterns (no loop becomes a library
#   call); COMPILE_FOR_BRISC / COMPILE_FOR_NCRISC and NOC_INDEX instead of the TRISC
#   defines; the entry is fw/ft_dataflow_entry.c; BRISC firmware also links fw/ft_noc.c.
#
#   NOC_INDEX is baked in at compile time, but the firmware takes its NoC at run time from
#   the launch message's brisc_noc_id (fw/ft_brisc_fw.c). Host and kernel must agree.
#
#   The `objcopy --weaken` step below is load-bearing: point --just-symbols at the
#   un-weakened firmware and LTO drops our _start, giving a clean link and an empty
#   kernel. Full write-up: docs/tt_kernel_toolchain.md 4.
#
#   Status: verified on the p150a card host 2026-08-11 and since; staged-tree paths removed
#           2026-10-08 (clean-room Phase 7) with every image byte-identical.
#   Issues: None
#   Todo:   None
#
set -euo pipefail
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
GCC=$SFPI/compiler/bin/riscv-tt-elf-gcc
NM=$SFPI/compiler/bin/riscv-tt-elf-nm
SIZE=$SFPI/compiler/bin/riscv-tt-elf-size
OBJCOPY=$SFPI/compiler/bin/riscv-tt-elf-objcopy

KERNEL=${1:?usage: build_tensix_dataflow_kernel.sh <kernel.c> <brisc|ncrisc> [outdir]}
PROC=${2:?usage: build_tensix_dataflow_kernel.sh <kernel.c> <brisc|ncrisc> [outdir]}
OUT=${3:-/tmp/ttk}
KERNEL=$(cd "$(dirname "$KERNEL")" && pwd)/$(basename "$KERNEL")

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=tensix_kernel_build_arch.sh
. "$HERE/tensix_kernel_build_arch.sh"

case $PROC in
  brisc)  PROC_ID=0; PROC_DEFINE=COMPILE_FOR_BRISC;  DEFAULT_NOC=0 ;;
  ncrisc) PROC_ID=1; PROC_DEFINE=COMPILE_FOR_NCRISC; DEFAULT_NOC=1 ;;
  *) echo "error: processor must be brisc or ncrisc, got '$PROC'" >&2; exit 2 ;;
esac
NOC=${NOC:-$DEFAULT_NOC}

GEN=$OUT/gen_$PROC
mkdir -p "$GEN"

case $KERNEL in
  *.c)   KERNEL_LANGUAGE=c ;;
  *.cpp) echo "error: C++ kernels are not supported -- the kernel entry is Foundry's C" \
             "(fw/ft_*_entry.c, clean-room plan Phase 2); port '$KERNEL' to C" >&2; exit 2 ;;
  *) echo "error: kernel must be .c, got '$KERNEL'" >&2; exit 2 ;;
esac

# ---- Include set, flags and defines ------------------------------------------------
# Kernels include Foundry's headers by relative path; $GEN holds this build's own outputs.
INC="-I$GEN"

# -Os keeps the NoC helpers out of line (O3 inlines them into a much larger .text).
# C11 for _Static_assert (the CB-interface stride pin in ft_tensix_circular_buffer.h).
KERNEL_CFLAGS="-mcpu=$TT_MCPU_DM -fno-tree-loop-distribute-patterns -std=c11 -flto=auto \
-ffast-math -Os -Wall -Wextra -Wno-unused-variable -Wno-unused-function"

# LIGHTWEIGHT_KERNEL_ASSERTS: as build_tensix_kernel.sh (on by default; the entry's
# post-kernel NoC checks and the CB overrun guard compile to `ebreak`). NOC_MODE=0 is the
# dedicated-NoC mode, the only one Foundry's firmware serves.
ASSERT_DEFINE=""
if [ "${LIGHTWEIGHT_ASSERTS:-1}" = 1 ]; then
    ASSERT_DEFINE="-DLIGHTWEIGHT_KERNEL_ASSERTS"
fi

DEF="-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 $ASSERT_DEFINE $TT_ARCH_DEFINE -D$PROC_DEFINE \
-DPROCESSOR_INDEX=$PROC_ID -DPROGRAMMABLE_CORE_TYPE=0 -DDISPATCH_MESSAGE_ADDR=0 \
-DNOC_INDEX=$NOC -DNOC_MODE=0 \
$TT_BOARD_DEFINES"

# Foundry's firmware has no watcher (build_tensix_kernel.sh's WATCHER note).
if [ "${WATCHER:-0}" != 0 ]; then
    echo "error: WATCHER=${WATCHER} -- Foundry's firmware has no watcher (clean-room Phase 7)" >&2
    exit 2
fi

# The data-movement half of build_tensix_kernel.sh's KERNEL_DEFINES, and it has to exist
# here too: the matmul READER reads the same resident_matmul_block_protocol.h the compute
# thread does, so a section size the two disagree about is a reader that refuses every
# descriptor the compute thread would have run. Omitted from the recipe when empty.
KERNEL_DEFINES=${KERNEL_DEFINES:-}
if [ -n "$KERNEL_DEFINES" ]; then
    DEF="$DEF $KERNEL_DEFINES"
    echo "### KERNEL_DEFINES: $KERNEL_DEFINES"
fi

LDFLAGS="-flto=auto -fno-exceptions -fno-rtti -nostdlib -Wl,-z,max-page-size=16 \
-Wl,-z,common-page-size=16 -Wno-stringop-overflow"


# ---- Support objects (fw/ft_libc.c, fw/ft_crt0.S), into $GEN so a shared output
# directory never clobbers the compute path's own copies.
echo "### support objects for $PROC"
rm -f "$GEN/ft_libc.o" "$GEN/ft_crt0.o"
$GCC $KERNEL_CFLAGS -c "$HERE/fw/ft_libc.c" -o "$GEN/ft_libc.o" ||
    { echo "### ABORT: fw/ft_libc.c did not compile"; exit 1; }
$GCC -mcpu=$TT_MCPU_DM -c "$HERE/fw/ft_crt0.S" -o "$GEN/ft_crt0.o" ||
    { echo "### ABORT: fw/ft_crt0.S did not assemble"; exit 1; }
# The runtime ARCHIVE: our memcpy (fw/ft_libc.c -DFT_LIBC_RUNTIME) and binary32 helpers
# (fw/ft_softfloat.c), the only runtime calls GCC can emit here. Every link is -nostdlib, so no
# toolchain library reaches an image; built WITHOUT LTO and linked as an archive so a member
# joins only an image that calls it and every other image is unchanged.
rm -f "$GEN/ft_rt_libc.o" "$GEN/ft_softfloat.o" "$GEN/libft_rt.a"
AR=$SFPI/compiler/bin/riscv-tt-elf-ar
RT_CFLAGS="-mcpu=$TT_MCPU_DM -std=c11 -Os -Wall -Wextra -ffreestanding \
-fno-tree-loop-distribute-patterns"
# RUNTIME_DEFINES: red builds only (e.g. -DFT_SOFTFLOAT_RED_COMPARE_INVERTED); never shipped.
RT_CFLAGS="$RT_CFLAGS ${RUNTIME_DEFINES:-}"
[ -n "${RUNTIME_DEFINES:-}" ] && echo "### RUNTIME_DEFINES (RED BUILD): $RUNTIME_DEFINES"
$GCC $RT_CFLAGS -DFT_LIBC_RUNTIME -c "$HERE/fw/ft_libc.c" -o "$GEN/ft_rt_libc.o" &&
    $GCC $RT_CFLAGS -c "$HERE/fw/ft_softfloat.c" -o "$GEN/ft_softfloat.o" &&
    $AR rcs "$GEN/libft_rt.a" "$GEN/ft_rt_libc.o" "$GEN/ft_softfloat.o" ||
    { echo "### ABORT: runtime archive did not build"; exit 1; }
LINK_OBJS="$GEN/ft_libc.o $GEN/libft_rt.a"
FW_OBJS=""
if [ "$PROC" = brisc ]; then
    # BRISC firmware's NoC setup (fw/ft_noc.c, clean-room Phase 6). Firmware only: no kernel
    # calls into it.
    rm -f "$GEN/ft_noc.o"
    $GCC $KERNEL_CFLAGS $DEF -DFW_BUILD -c "$HERE/fw/ft_noc.c" -o "$GEN/ft_noc.o" ||
        { echo "### ABORT: fw/ft_noc.c did not compile"; exit 1; }
    FW_OBJS="$GEN/ft_noc.o"
fi

# ---- 4. Firmware ELF: the kernel link needs its __fw_export_* symbols ------------------
echo "### firmware $PROC"
rm -f "$GEN/fw_$PROC.o"
# Foundry's firmware: fw/ft_brisc_fw.c (clean-room Phase 6) or fw/ft_ncrisc_fw.c (Phase 5).
$GCC $KERNEL_CFLAGS $DEF -DFW_BUILD -I"$HERE" -I"$HERE/kernels" -c \
    "$HERE/fw/ft_${PROC}_fw.c" -o "$GEN/fw_$PROC.o"
[ -f "$GEN/fw_$PROC.o" ] || { echo "### ABORT: firmware $PROC did not compile"; exit 1; }
python3 "$HERE/tools/emit_tensix_linker_script.py" --cc "$GCC" --memmap "$HERE/ft_tt_memmap.h" \
    --processor $PROC --type firmware --template "$HERE/fw/ft_tensix_firmware.ld" \
    -o "$GEN/firmware_$PROC.ld" || { echo "### ABORT: firmware_$PROC.ld"; exit 1; }
$GCC -mcpu=$TT_MCPU_DM $LDFLAGS -Os -T"$GEN/firmware_$PROC.ld" \
    "$GEN/ft_crt0.o" "$GEN/fw_$PROC.o" $FW_OBJS $LINK_OBJS -o "$OUT/fw_$PROC.elf"
$SIZE "$OUT/fw_$PROC.elf"

# ---- 5. Kernel ELF, linked against a WEAKENED copy of that firmware --------------------
echo "### kernel $PROC (NOC_INDEX=$NOC, CTARGS=${CTARGS:-0})"
# -MMD: the depfile is what tools/record_tensix_image_provenance.py records this image's
# input closure from. It is the COMPILER'S OWN list, not a scan of #include lines, which is
# the difference between a closure that tracks the build and one that drifts from it.
# The entry is Foundry's (fw/ft_dataflow_entry.c), one source for both processors.
rm -f "$GEN/$PROC.o"
$GCC $KERNEL_CFLAGS $DEF -DKERNEL_BUILD -I"$HERE" -MMD -MF "$GEN/$PROC.d" -c \
    "$HERE/fw/ft_dataflow_entry.c" -o "$GEN/$PROC.o" ||
    { echo "### ABORT: fw/ft_dataflow_entry.c did not compile for $PROC"; exit 1; }
KERNEL_OBJS=""
DEPFILES=(--depfile "$GEN/$PROC.d")
if [ "$KERNEL_LANGUAGE" = c ]; then
    # The kernel is its own object, compiled with this processor's $DEF: NOC_INDEX /
    # NUM_NOCS / COMPILE_FOR_* are what select its register block.
    $GCC $KERNEL_CFLAGS $DEF -DKERNEL_BUILD -DKERNEL_COMPILE_TIME_ARGS="${CTARGS:-0}" $INC \
        -MMD -MF "$GEN/kernel_body.d" -c "$KERNEL" -o "$GEN/kernel_body.o"
    KERNEL_OBJS="$GEN/kernel_body.o"
    DEPFILES+=(--depfile "$GEN/kernel_body.d")
fi
python3 "$HERE/tools/emit_tensix_linker_script.py" --cc "$GCC" --memmap "$HERE/ft_tt_memmap.h" \
    --processor $PROC --type kernel --template "$HERE/fw/ft_tensix_kernel.ld" \
    -o "$GEN/kernel_$PROC.ld" || { echo "### ABORT: kernel_$PROC.ld"; exit 1; }
echo "    __fw_export_text_end=0x$($NM "$OUT/fw_$PROC.elf" | awk '/__fw_export_text_end/{print $1}')" \
     " __fw_export_ldm_end=0x$($NM "$OUT/fw_$PROC.elf" | awk '/__fw_export_ldm_end/{print $1}')"
$OBJCOPY --weaken "$OUT/fw_$PROC.elf" "$GEN/fw_${PROC}_weak.elf"
TEXT_DEFSYM=""
if [ -n "${KERNEL_TEXT_BASE:-}" ]; then
    TEXT_DEFSYM="-Wl,--defsym=__fw_export_text_end=$KERNEL_TEXT_BASE"
    echo "    relocating kernel text to $KERNEL_TEXT_BASE"
fi
$GCC -mcpu=$TT_MCPU_DM $LDFLAGS -Os -T"$GEN/kernel_$PROC.ld" $TEXT_DEFSYM \
    -Wl,--just-symbols="$GEN/fw_${PROC}_weak.elf" \
    "$GEN/$PROC.o" $KERNEL_OBJS $LINK_OBJS -o "$OUT/$PROC.elf"
$SIZE "$OUT/$PROC.elf"

# The ELF is named after the processor, so record which kernel it came from for
# tools/emit_tensix_kernel_header.py to put in the generated header's source_name.
basename "$KERNEL" > "$OUT/$PROC.elf.src"

# ...and record WHAT IT WAS BUILT FROM, by content. The `.src` sidecar names one file and
# is checked by mtime, which an `explode` resets; the `.prov` record hashes every in-tree
# file the compile opened plus the recipe that reached code generation, and nothing about
# timestamps can launder that. tools/tensix_image_provenance.py has the incident it is
# written against.
RECIPE_DEFINES=()
# A recipe is space-separated `key=value` pairs, so several defines are RECORDED comma-joined
# (rebuild_tensix_program.sh splits them back). Recorded space-joined, a reader stopped at the
# first define and a regen rebuilt `-DA -DB` as `-DA` alone (qwen36 S4, 2026-11-02).
[ -n "$KERNEL_DEFINES" ] && RECIPE_DEFINES=(--recipe "defines=$(echo $KERNEL_DEFINES | tr ' ' ',')")
python3 "$HERE/tools/record_tensix_image_provenance.py" \
    --tt-root "$HERE" --image "$OUT/$PROC.elf" --source "$KERNEL" "${DEPFILES[@]}" \
    --recipe "toolchain=$($GCC -dumpversion)" \
    --recipe "ctargs=${CTARGS:-0}" \
    --recipe "noc=$NOC" \
    --recipe "watcher=${WATCHER:-0}" \
    --recipe "lightweight_asserts=${LIGHTWEIGHT_ASSERTS:-1}" \
    --recipe "relocate_kernel_text=${RELOCATE_KERNEL_TEXT:-0}" \
    --recipe "kernel_text_base=${KERNEL_TEXT_BASE:-}" "${RECIPE_DEFINES[@]}"
echo "__DATAFLOW_DONE__ $PROC"
