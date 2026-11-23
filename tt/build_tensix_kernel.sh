#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# build_tensix_kernel.sh
#   Compiles one Tensix compute kernel into three TRISC ELFs (unpack, math, pack) plus the
#   three TRISC firmware ELFs they link against, from Foundry sources only (tt/fw/,
#   tt/kernels/) with the SFPI compiler. No Tenstorrent source tree is read.
#   Blackhole is the only implemented target; every architecture- and board-shaped value
#   lives in tensix_kernel_build_arch.sh, which refuses any other part by name.
#
#   Usage:  build_tensix_kernel.sh <compute-kernel.c> [outdir]
#   Env:    TT_ARCH   which part to build for (default blackhole; only blackhole is implemented)
#           TT_NUM_L1_BANKS / TT_NUM_DRAM_BANKS
#                     the BOARD's live counts, baked into every image (default 140 / 8). These
#                     must stay in step with FT_TT_KERNEL_BUILD_* in ft_tt_memmap.h
#           CTARGS    comma-separated KERNEL_COMPILE_TIME_ARGS (default 1,1,1)
#           CB_TABLE  CB table file driving ft_tensix_cb_formats.h. Unset means the legacy
#                     uniform Float16_b / 32x32 / 4-face / 2048 B set. Format and derivation
#                     rules: tools/emit_tensix_cb_formats.py
#           KERNEL_DEFINES  extra -D flags for a compile-time selection the source cannot make
#           LIGHTWEIGHT_ASSERTS  1 (default) keeps the kernels' assert checks as `ebreak`
#           KERNEL_TEXT_BASE_TRISC<n>  link this TRISC's kernel text at an explicit L1 address
#                     instead of at __fw_export_text_end (see KERNEL TEXT RELOCATION below)
#   Output: <outdir>/trisc{0,1,2}.elf (+ the fw_trisc{0,1,2}.elf they link against), each with
#           a .src sidecar and a .prov provenance record.
#   Companion: build_tensix_dataflow_kernel.sh builds BRISC/NCRISC; build_tensix_program.sh
#           drives both for a whole program.
#
#   One compute source becomes three programs: it is compiled once per TRISC with
#   FT_TRISC_{UNPACK,MATH,PACK} and COMPILE_FOR_TRISC=n, which select the statements each
#   thread keeps, and each is LTO-linked with Foundry's entry (fw/ft_trisc_entry.c).
#
#   A kernel ELF is linked against a WEAKENED copy of its firmware ELF through
#   --just-symbols (docs/tt_kernel_toolchain.md 4): an un-weakened one makes LTO drop the
#   kernel's own `_start` and link an empty image.
#
#   Status: verified on the p150a card host 2026-08-11 and since; staged-tree paths removed
#           2026-10-08 (clean-room Phase 7) with every image byte-identical.
#   Issues: Quoted -D values cannot survive $DEF word-splitting.
#   Todo:   None
set -euo pipefail
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
GCC=$SFPI/compiler/bin/riscv-tt-elf-gcc
# THE BRACES IN THIS USAGE MESSAGE USED TO CREATE A FILE. Bash ends a `${1:?word}`
# expansion at the FIRST unescaped `}`, which in `<compute-kernel.{c,cpp}>` is the one
# closing `{c,cpp}` — so the remainder `> [outdir]}` parsed as a REDIRECTION and every
# invocation dropped a zero-byte `[outdir]}` into the caller's cwd, which for a program
# rebuild is the checkout root. Reproduced in one line off-hardware. No braces here now.
KERNEL=${1:?usage: build_tensix_kernel.sh <compute-kernel.c> [outdir]}
OUT=${2:-/tmp/ttk}
KERNEL=$(cd "$(dirname "$KERNEL")" && pwd)/$(basename "$KERNEL")
mkdir -p "$OUT"

# ---- C KERNEL PATH ----------------------------------------------------------
# THE THREE-WAY COMPILATION IS THE WHOLE DIFFICULTY, and it is not the language. One
case $KERNEL in
  *.c)   KERNEL_LANGUAGE=c ;;
  *.cpp) echo "error: C++ kernels are not supported -- the kernel entry is Foundry's C" \
             "(fw/ft_*_entry.c, clean-room plan Phase 2); port '$KERNEL' to C" >&2; exit 2 ;;
  *) echo "error: kernel must be .c, got '$KERNEL'" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Every architecture- and board-shaped value comes from here and nowhere else. A part
# other than blackhole is REFUSED by name, with the list of what a port owes.
# shellcheck source=tensix_kernel_build_arch.sh
. "$HERE/tensix_kernel_build_arch.sh"

# ---- 1. The CB descriptors header the compute kernels include -------------

# ft_tensix_cb_formats.h IS the compile-time half of the CB contract: whatever the host
# writes into the L1 CB config array must match what is baked in here, and there is no
# runtime negotiation. It used to be generated inline as ONE uniform descriptor set, so
# exactly one tile shape had ever been compiled and any other page size or format got
# silently wrong descriptors. It now comes from the real CB table when one is given.
if [ -n "${CB_TABLE:-}" ]; then
  python3 "$HERE/tools/emit_tensix_cb_formats.py" -o "$OUT/ft_tensix_cb_formats.h" \
      --cb-table "$CB_TABLE"
else
  python3 "$HERE/tools/emit_tensix_cb_formats.py" -o "$OUT/ft_tensix_cb_formats.h" \
      --uniform-default
fi


# ---- 2. Include set, flags and defines ------------------------------------------
# Kernel bodies include Foundry's kernels/ft_tensix_*.h by relative path; the only include
# directory they need is $OUT, for the generated ft_tensix_cb_formats.h.
INC="-I$OUT"

# C11 for _Static_assert, which kernels/ft_tensix_*.h use to pin binary contracts.
KERNEL_CFLAGS="-mcpu=$TT_MCPU_COMPUTE -std=c11 -flto=auto -ffast-math -O3 \
-Wall -Wextra -Wno-unused-variable -Wno-unused-function"
# LIGHTWEIGHT_KERNEL_ASSERTS (default on): the kernels' and entries' assert checks compile
# to a bare `ebreak` (kernels/ft_tensix_circular_buffer.h FT_TENSIX_CB_ASSERT, the CB
# overrun guard; fw/ft_dataflow_entry.c, the post-kernel NoC checks). Measured 2026-08-13:
# no wall-clock cost (matmul 464-465 GFLOP/s either way), +296..+404 B of text a kernel.
# LIGHTWEIGHT_ASSERTS=0 builds without them, for an A/B against numbers that predate it.
ASSERT_DEFINE=""
if [ "${LIGHTWEIGHT_ASSERTS:-1}" = 1 ]; then
    ASSERT_DEFINE="-DLIGHTWEIGHT_KERNEL_ASSERTS"
fi

DEF="-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 $ASSERT_DEFINE $TT_ARCH_DEFINE -DKERNEL_BUILD -DPROGRAMMABLE_CORE_TYPE=0 \
-DDISPATCH_MESSAGE_ADDR=0 -DKERNEL_COMPILE_TIME_ARGS=${CTARGS:-1,1,1} $TT_BOARD_DEFINES"

# ---- KERNEL_DEFINES: a compile-time SELECTION the source cannot make for itself -------
# One kernel source with two legal renderings -- resident_matmul_dst_sync.h's SyncHalf and
# SyncFull -- is a program-directory difference, not a source difference, so the choice has
# to reach the compiler and then be RECORDED. It rides in the recipe as `defines=` and is
# OMITTED when empty, so every program built before this existed keeps its recipe line
# byte for byte and nothing goes stale for having gained an unused field.
#
# Unquoted -D values only, per this file's own Issues note: $DEF is word-split.
KERNEL_DEFINES=${KERNEL_DEFINES:-}
if [ -n "$KERNEL_DEFINES" ]; then
  DEF="$DEF $KERNEL_DEFINES"
  echo "### KERNEL_DEFINES: $KERNEL_DEFINES"
fi

# ---- WATCHER: not supported ---------------------------------------------------------
# Foundry's firmware and kernel entries (tt/fw/) have no watcher, so there is nothing for
# WATCHER=1 to turn on; refuse it rather than record a recipe that means nothing.
if [ "${WATCHER:-0}" != 0 ]; then
  echo "error: WATCHER=${WATCHER} -- Foundry's firmware has no watcher (clean-room Phase 7)" >&2
  exit 2
fi

# ---- `| head -40` KILLS THE COMPILER, AND SILENTLY -------------------------
# Every compile below used to be piped straight into `head -40`. Under `set -o pipefail`
# that is a trap rather than a convenience: `head` exits after the 40th line, the compiler
# takes SIGPIPE, the pipeline reports failure, `set -e` ends the script -- and because the
# 41st line onward was never read, the ERROR TEXT IS NEVER PRINTED. A kernel that merely
# emits many WARNINGS is enough to trigger it, so the failure looks like a build that
# stopped for no reason. It cost a round on 2026-08-13.
#
# The fix is to let the compiler write a whole log and to head THE LOG. Nothing about the
# compilation changes -- same flags, same order -- so the instruction-stream gate is
# unaffected; only the reporting is.
run_compile() {
  local log=$1; shift
  if "$@" > "$log" 2>&1; then
    head -40 "$log"
    return 0
  fi
  echo "### COMPILE FAILED -- full log: $log"
  head -80 "$log"
  return 1
}

for T in UNPACK MATH PACK; do
  case $T in UNPACK) id=0;; MATH) id=1;; PACK) id=2;; esac
  low=$(echo $T | tr A-Z a-z)
  echo "### building trisc$id ($T)"
  # -MMD: the depfile is the input closure tools/record_tensix_image_provenance.py records
  # beside the image -- the compiler's own list of what it opened, not a scan of #includes.
  # A FAILED COMPILE MUST NOT LINK THE PREVIOUS OBJECT. Both objects are deleted first and a
  # failure ends the build: until 2026-10-05 `|| true` left the old .o in place, and the
  # reduce family's PACK image was a stale 2026-10-04 object for every regen since.
  rm -f "$OUT/trisc$id.o" "$OUT/kernel_body_trisc$id.o"
  # The entry is Foundry's (fw/ft_trisc_entry.c): C, the kernel body's own flags, and only
  # the tt/ directory on its include path.
  run_compile "$OUT/trisc$id.compile.log" \
      $GCC $KERNEL_CFLAGS $DEF -DFT_TRISC_$T -DCOMPILE_FOR_TRISC=$id -DPROCESSOR_INDEX=$((id+2)) \
      -I"$HERE" -MMD -MF "$OUT/trisc$id.d" -c \
      "$HERE/fw/ft_trisc_entry.c" -o "$OUT/trisc$id.o" ||
      { echo "### ABORT: trisc$id ($T) did not compile"; exit 1; }
  ls -l "$OUT/trisc$id.o" 2>/dev/null || echo "FAILED trisc$id"
  # The C body, once per thread, with this thread's $DEF: FT_TRISC_* / COMPILE_FOR_TRISC
  # are what select which statements it keeps.
  if [ "$KERNEL_LANGUAGE" = c ]; then
    run_compile "$OUT/kernel_body_trisc$id.compile.log" \
        $GCC $KERNEL_CFLAGS $DEF -DFT_TRISC_$T -DCOMPILE_FOR_TRISC=$id -DPROCESSOR_INDEX=$((id+2)) \
        $INC -MMD -MF "$OUT/kernel_body_trisc$id.d" -c "$KERNEL" \
        -o "$OUT/kernel_body_trisc$id.o" ||
        { echo "### ABORT: kernel body trisc$id ($T) did not compile"; exit 1; }
    ls -l "$OUT/kernel_body_trisc$id.o" 2>/dev/null || echo "FAILED kernel body trisc$id"
  fi
done
echo __BUILD_DONE__


# ---- 3. Build the TRISC firmware ELF (kernel link needs its __fw_export_* symbols)
NM=$SFPI/compiler/bin/riscv-tt-elf-nm
SIZE=$SFPI/compiler/bin/riscv-tt-elf-size
LDFLAGS="-flto=auto -fno-exceptions -fno-rtti -nostdlib -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -Wno-stringop-overflow"

# Foundry's startup and C-library objects (fw/), shared by the firmware and kernel links.
rm -f "$OUT/ft_libc.o" "$OUT/ft_crt0.o"
$GCC -mcpu=$TT_MCPU_DM -std=c11 -O3 -Wall -Wextra -c "$HERE/fw/ft_libc.c" -o "$OUT/ft_libc.o" ||
    { echo "### ABORT: fw/ft_libc.c did not compile"; exit 1; }
$GCC -mcpu=$TT_MCPU_DM -c "$HERE/fw/ft_crt0.S" -o "$OUT/ft_crt0.o" ||
    { echo "### ABORT: fw/ft_crt0.S did not assemble"; exit 1; }
# The runtime ARCHIVE: our memcpy (fw/ft_libc.c -DFT_LIBC_RUNTIME) and binary32 helpers
# (fw/ft_softfloat.c), the only runtime calls GCC can emit here. Every link is -nostdlib, so no
# toolchain library reaches an image; built WITHOUT LTO and linked as an archive so a member
# joins only an image that calls it and every other image is unchanged.
rm -f "$OUT/ft_rt_libc.o" "$OUT/ft_softfloat.o" "$OUT/libft_rt.a"
AR=$SFPI/compiler/bin/riscv-tt-elf-ar
RT_CFLAGS="-mcpu=$TT_MCPU_DM -std=c11 -Os -Wall -Wextra -ffreestanding \
-fno-tree-loop-distribute-patterns"
# RUNTIME_DEFINES: red builds only (e.g. -DFT_SOFTFLOAT_RED_COMPARE_INVERTED); never shipped.
RT_CFLAGS="$RT_CFLAGS ${RUNTIME_DEFINES:-}"
[ -n "${RUNTIME_DEFINES:-}" ] && echo "### RUNTIME_DEFINES (RED BUILD): $RUNTIME_DEFINES"
$GCC $RT_CFLAGS -DFT_LIBC_RUNTIME -c "$HERE/fw/ft_libc.c" -o "$OUT/ft_rt_libc.o" &&
    $GCC $RT_CFLAGS -c "$HERE/fw/ft_softfloat.c" -o "$OUT/ft_softfloat.o" &&
    $AR rcs "$OUT/libft_rt.a" "$OUT/ft_rt_libc.o" "$OUT/ft_softfloat.o" ||
    { echo "### ABORT: runtime archive did not build"; exit 1; }

for id in 0 1 2; do
  case $id in 0) T=UNPACK;; 1) T=MATH;; 2) T=PACK;; esac
  echo "### firmware trisc$id"
  # Foundry's firmware (fw/ft_trisc_fw.c, clean-room Phase 5): C, the kernel flags.
  rm -f "$OUT/fw_trisc$id.o"
  run_compile "$OUT/fw_trisc$id.compile.log" \
      $GCC $KERNEL_CFLAGS $DEF -DFW_BUILD -DFT_TRISC_$T -DCOMPILE_FOR_TRISC=$id \
      -DPROCESSOR_INDEX=$((id+2)) -I"$HERE" -I"$HERE/kernels" -c "$HERE/fw/ft_trisc_fw.c" \
      -o "$OUT/fw_trisc$id.o" || { echo "### ABORT: firmware trisc$id did not compile"; exit 1; }
  python3 "$HERE/tools/emit_tensix_linker_script.py" --cc "$GCC" --memmap "$HERE/ft_tt_memmap.h" \
      --processor trisc$id --type firmware --template "$HERE/fw/ft_tensix_firmware.ld" \
      -o "$OUT/firmware_trisc$id.ld" || { echo "### ABORT: firmware_trisc$id.ld"; exit 1; }
  $GCC -mcpu=$TT_MCPU_COMPUTE $LDFLAGS -T"$OUT/firmware_trisc$id.ld" \
      "$OUT/ft_crt0.o" "$OUT/fw_trisc$id.o" "$OUT/ft_libc.o" "$OUT/libft_rt.a" -o "$OUT/fw_trisc$id.elf" 2>&1 | head -20
  [ -f "$OUT/fw_trisc$id.elf" ] && $SIZE "$OUT/fw_trisc$id.elf" || echo "FW LINK FAILED trisc$id"
done

# ---- KERNEL TEXT RELOCATION -------------------------------------------------
# main.ld places a kernel's .text at __fw_export_text_end, i.e. in the TAIL of its own
# processor's firmware hole -- and above that hole is the NEXT processor's firmware, not
# free L1 (docs/tt_launch_notes.md 2, corrected). The tails are tiny: TRISC0 gets 1360 B,
# TRISC1 1952 B, TRISC2 1392 B. vector_add fitted with 68 B to spare. Measured 2026-08-12,
# nothing else does:
#
#     matmul_compute    trisc0 1404 (44 B OVER)   trisc1  900 ok    trisc2 1316 ok
#     rms_norm_compute  trisc0 2448 (1088 OVER)   trisc1 2504 OVER  trisc2 2548 OVER
#
# So relocation is not an optimisation, it is the precondition for having an op library at
# all. __fw_export_text_end arrives from the WEAKENED firmware ELF via --just-symbols, so a
# strong --defsym overrides it and the linker script follows: .text lands at the given
# address and __kernel_data_lma right behind it, both of which the loader and the kernel's
# own _start already read from the ELF rather than assume. Verified: the relocated image is
# byte-for-byte the same text, entry == first segment vaddr, _start still 'T' (the
# --just-symbols trap of docs/tt_kernel_toolchain.md 4 does not fire).
#
# The host side must load a relocated image with ft_tt_op_image_load(), NOT
# ft_tt_launch_load_image() -- the latter enforces the in-hole rule this deliberately
# breaks. The addresses come from foundry/tt/ft_tt_op_l1_layout.h, which
# build_tensix_program.sh parses -- one source of truth, nothing to keep in step.
kernel_text_defsym() {
  local base
  eval "base=\${KERNEL_TEXT_BASE_TRISC$1:-}"
  [ -n "$base" ] && printf -- "-Wl,--defsym=__fw_export_text_end=%s" "$base"
}

# ---- 4. Link the kernel ELFs against the firmware export symbols -----------
for id in 0 1 2; do
  echo "### kernel trisc$id"
  python3 "$HERE/tools/emit_tensix_linker_script.py" --cc "$GCC" --memmap "$HERE/ft_tt_memmap.h" \
      --processor trisc$id --type kernel --template "$HERE/fw/ft_tensix_kernel.ld" \
      -o "$OUT/kernel_trisc$id.ld" || { echo "### ABORT: kernel_trisc$id.ld"; exit 1; }
  TE=$($NM "$OUT/fw_trisc$id.elf" | awk "/__fw_export_text_end/{print \$1}")
  LE=$($NM "$OUT/fw_trisc$id.elf" | awk "/__fw_export_ldm_end/{print \$1}")
  echo "    __fw_export_text_end=0x$TE  __fw_export_ldm_end=0x$LE"
  $SFPI/compiler/bin/riscv-tt-elf-objcopy --weaken "$OUT/fw_trisc$id.elf" "$OUT/fw_trisc${id}_weak.elf"
  KERNEL_BODY_OBJ=""
  [ "$KERNEL_LANGUAGE" = c ] && KERNEL_BODY_OBJ="$OUT/kernel_body_trisc$id.o"
  $GCC -mcpu=$TT_MCPU_COMPUTE $LDFLAGS -T"$OUT/kernel_trisc$id.ld" \
      $(kernel_text_defsym $id) \
      -Wl,--just-symbols="$OUT/fw_trisc${id}_weak.elf" \
      "$OUT/trisc$id.o" $KERNEL_BODY_OBJ "$OUT/ft_libc.o" "$OUT/libft_rt.a" -o "$OUT/trisc$id.elf" 2>&1 | head -20
  [ -f "$OUT/trisc$id.elf" ] && $SIZE "$OUT/trisc$id.elf" || echo "KERNEL LINK FAILED trisc$id"
  # Record which kernel this ELF came from; tools/emit_tensix_kernel_header.py reads the
  # sidecar for the generated header's source_name. Nothing above depends on it.
  basename "$KERNEL" > "$OUT/trisc$id.elf.src"
  # ...and record WHAT IT WAS BUILT FROM, by content -- the check the `.src` sidecar's
  # mtime comparison cannot make (tools/tensix_image_provenance.py).
  #
  # THE CB TABLE IS AN --extra-input BECAUSE THE COMPILER NEVER SEES IT. It is read by
  # tools/emit_tensix_cb_formats.py, which writes the ft_tensix_cb_formats.h these three
  # images are compiled against, so its content is part of the image and appears in no
  # depfile. The same goes for that emitter itself.
  CB_TABLE_PATH=${CB_TABLE:-}
  PROVENANCE_EXTRA=(--extra-input "$HERE/tools/emit_tensix_cb_formats.py")
  [ -n "$CB_TABLE_PATH" ] && PROVENANCE_EXTRA+=(--extra-input "$CB_TABLE_PATH")
  PROVENANCE_DEPS=(--depfile "$OUT/trisc$id.d")
  # AN ARRAY, NOT A `${VAR:+...}` EXPANSION. Measured 2026-08-26: written that way the
  # word-split leaves literal quotes in the key and the pair never reaches the recorder --
  # the variant image was built correctly and recorded a recipe IDENTICAL to the shipped
  # one, which is a program directory that rebuilds itself into the wrong kernel.
  RECIPE_DEFINES=()
  # A recipe is space-separated `key=value` pairs, so several defines are RECORDED comma-joined
# (rebuild_tensix_program.sh splits them back). Recorded space-joined, a reader stopped at the
# first define and a regen rebuilt `-DA -DB` as `-DA` alone (qwen36 S4, 2026-11-02).
[ -n "$KERNEL_DEFINES" ] && RECIPE_DEFINES=(--recipe "defines=$(echo $KERNEL_DEFINES | tr ' ' ',')")
  [ "$KERNEL_LANGUAGE" = c ] && PROVENANCE_DEPS+=(--depfile "$OUT/kernel_body_trisc$id.d")
  python3 "$HERE/tools/record_tensix_image_provenance.py" \
      --tt-root "$HERE" --image "$OUT/trisc$id.elf" --source "$KERNEL" \
      "${PROVENANCE_DEPS[@]}" "${PROVENANCE_EXTRA[@]}" \
      --recipe "toolchain=$($GCC -dumpversion)" \
      --recipe "ctargs=${CTARGS:-1,1,1}" \
      --recipe "watcher=${WATCHER:-0}" \
      --recipe "lightweight_asserts=${LIGHTWEIGHT_ASSERTS:-1}" \
      --recipe "relocate_kernel_text=${RELOCATE_KERNEL_TEXT:-0}" \
      --recipe "kernel_text_base=$(eval echo \${KERNEL_TEXT_BASE_TRISC$id:-})" \
      --recipe "cb_table=${CB_TABLE_PATH##*/}" "${RECIPE_DEFINES[@]}"
done
echo __LINK_DONE__
