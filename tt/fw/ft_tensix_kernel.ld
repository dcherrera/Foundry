/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ft_tensix_kernel.ld — layout of a Tensix KERNEL image (one per processor).
 *
 * tools/emit_tensix_linker_script.py prepends this processor's constants from ft_tt_memmap.h.
 * A kernel is linked against its firmware with --just-symbols, which supplies
 * __fw_export_text_end, __fw_export_ldm_end, __global_pointer$ and __stack_top (see
 * ft_tensix_firmware.ld). A relocated kernel overrides __fw_export_text_end with --defsym.
 *
 *   .text   at __fw_export_text_end, the entry (`.start`, fw/ft_*_entry.c) first;
 *   .data   at __fw_export_ldm_end in local memory, loaded right behind the text
 *           (__kernel_data_lma) and copied in by the entry;
 *   .bss    after it; __stack_base is the first 16-byte-aligned address past it.
 *
 * `.segments` as in the firmware script; the text limit is the processor's kernel region,
 * measured from its firmware slot.
 *
 * Status: clean-room replacement of the staged tree's linker script, plan Phase 3.
 * Issues: None
 * Todo:   None
 */
OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS
{
    .text __fw_export_text_end :
    {
        KEEP(*(.start))
        *(.text.unlikely .text.*_unlikely .text.unlikely.*)
        *(.text.exit .text.exit.*)
        *(.text.startup .text.startup.*)
        *(.text.hot .text.hot.*)
        *(.text .stub .text.* .gnu.linkonce.t.*)
        . = ALIGN(4);
        __kernel_data_lma = .;
    }

    .data __fw_export_ldm_end : AT(__kernel_data_lma)
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
    __stack_base = ALIGN(16);
    __instrn_buffer = 0xFFE40000; /* INSTRN_BUF_BASE, BabyRISCV/PushTensixInstruction.md */

    .segments 0 (INFO) :
    {
        LONG(ADDR(.text)) LONG(ADDR(.text))
        LONG(FT_KERNEL_SIZE - (ADDR(.text) - FT_FIRMWARE_BASE))
        LONG(ADDR(.data)) LONG(ADDR(.data))
        LONG(FT_LOCAL_SIZE - FT_STACK_MIN_SIZE - (ADDR(.data) - FT_LOCAL_BASE))
    }

    ASSERT(__stack_base <= FT_LOCAL_BASE + FT_LOCAL_SIZE - FT_STACK_MIN_SIZE,
           "kernel data leaves less than the minimum stack")
}
