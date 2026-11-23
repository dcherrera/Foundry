/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ft_crt0.S — reset entry for the five Tensix firmware images.
 *
 * Every Baby RISC-V core leaves soft reset at a fixed or overridden pc
 * (tt-isa-documentation, BlackholeA0/TensixTile/SoftReset.md, "Upon leaving soft reset"):
 * GPRs are zero, so nothing above the ISA exists yet. This sets the two registers the RISC-V
 * psABI needs before any C runs — gp (so the linker's gp-relative relaxation is valid) and sp
 * (the top of this core's local data memory, a linker-script symbol) — then calls `main`.
 * Firmware `main` never returns; if it ever did, `exit` parks the core.
 *
 * Placed in `.start`, the section the linker script puts at the image's entry address.
 *
 * Status: clean-room replacement of the staged tree's start file, plan Phase 2.
 * Issues: None
 * Todo:   None
 */

    .section .start, "ax", @progbits
    .global _start
    .type   _start, @function
_start:
    /* gp must be materialised without relaxation: a relaxed form would read gp itself. */
    .option push
    .option norelax
    lui     gp, %hi(__global_pointer$)
    addi    gp, gp, %lo(__global_pointer$)
    .option pop

    lui     sp, %hi(__stack_top)
    addi    sp, sp, %lo(__stack_top)

    call    main
    tail    exit
    .size   _start, . - _start
