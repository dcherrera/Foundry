/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ft_tensix_firmware.ld — layout of a Tensix FIRMWARE image (one per processor).
 *
 * tools/emit_tensix_linker_script.py prepends this processor's constants (FT_FIRMWARE_BASE,
 * FT_LOCAL_BASE, ...) from ft_tt_memmap.h; this file only arranges them.
 *
 *   .text   in L1 at the processor's firmware slot, `.start` (the reset entry) first.
 *   .data   in the processor's local data memory (Baby RISC-V local RAM,
 *   .bss    tt-isa-documentation BlackholeA0/TensixTile/BabyRISCV/README.md), read-only data
 *           included so constant tables are fetched from local RAM. The host stages this
 *           segment in L1 and the firmware copies it in (ft_tt_launch.c).
 *
 * Exported to the kernel link (which reads them through --just-symbols):
 *   __fw_export_text_end  first 16-byte-aligned L1 address after the firmware text, where a
 *                         kernel's text goes unless it is relocated;
 *   __fw_export_ldm_end   first 16-byte-aligned local address after the firmware's data;
 *   __stack_top           top of local memory; __global_pointer$ 0x7F0 above its base, so
 *                         gp-relative addressing reaches the whole of a TRISC's 4 KiB.
 *
 * `.segments` (non-loaded) records, for the host loader's size check (ft_tt_elf.c): text
 * address twice and its limit (the firmware slot), data address twice and its limit (local
 * memory less the processor's minimum stack, less where the data starts).
 *
 * Status: clean-room replacement of the staged tree's linker script, plan Phase 3.
 * Issues: None
 * Todo:   None
 */
OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS
{
    .text FT_FIRMWARE_BASE :
    {
        KEEP(*(.start))
        *(.text.unlikely .text.*_unlikely .text.unlikely.*)
        *(.text.exit .text.exit.*)
        *(.text.startup .text.startup.*)
        *(.text.hot .text.hot.*)
        *(.text .stub .text.* .gnu.linkonce.t.*)
        . = ALIGN(4);
    }
    __fw_export_text_end = ALIGN(16);

    .data FT_LOCAL_BASE :
    {
        __ldm_data_start = .;
        *(.rodata .rodata.* .gnu.linkonce.r.*)
        *(.srodata .srodata.*)
        *(.data .data.* .gnu.linkonce.d.*)
        *(.sdata .sdata.* .gnu.linkonce.s.*)
        . = ALIGN(4);
        __ldm_data_end = .;
    }
    .bss :
    {
        __ldm_bss_start = .;
        *(.sbss .sbss.* .gnu.linkonce.sb.*)
        *(.bss .bss.* .gnu.linkonce.b.*)
        *(COMMON)
        . = ALIGN(4);
        __ldm_bss_end = .;
    }
    __fw_export_ldm_end = ALIGN(16);

    __stack_top = FT_LOCAL_BASE + FT_LOCAL_SIZE;
    __global_pointer$ = FT_LOCAL_BASE + 0x7F0;
    __instrn_buffer = 0xFFE40000; /* INSTRN_BUF_BASE, BabyRISCV/PushTensixInstruction.md */

    .segments 0 (INFO) :
    {
        LONG(ADDR(.text)) LONG(ADDR(.text)) LONG(FT_FIRMWARE_SIZE)
        LONG(ADDR(.data)) LONG(ADDR(.data))
        LONG(FT_LOCAL_SIZE - FT_STACK_MIN_SIZE - (ADDR(.data) - FT_LOCAL_BASE))
    }

    ASSERT(__fw_export_text_end <= FT_FIRMWARE_BASE + FT_FIRMWARE_SIZE,
           "firmware text overflows its L1 slot")
    ASSERT(__fw_export_ldm_end <= __stack_top - FT_STACK_MIN_SIZE,
           "firmware data leaves less than the minimum stack")
}
