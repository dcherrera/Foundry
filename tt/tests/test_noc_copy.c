/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_noc_copy.c
 * @brief Off-hardware gate for `ft_tt_noc_copy.h` — the PIO copy shape and the
 *   window-walk arithmetic that BOTH transports now share.
 *
 *   This is where the shared code earns the right to be shared. The tt-kmd
 *   transport and the userspace-driver transport differ only in how a window is
 *   aimed; everything that moves a byte is this header, and this file proves it
 *   over host memory with no device anywhere in the source list.
 *
 *   THE FIXTURE PATTERN IS DELIBERATELY NON-PERIODIC, and the first thing this
 *   file checks is that it is. A 256-periodic fixture made every page identical
 *   in an earlier round here and turned a whole battery of byte comparisons
 *   vacuous — a copy that dropped a block, doubled one, or was off by 256 bytes
 *   would still have compared equal. So the generator is an LCG over the
 *   absolute offset, and its periodicity is a CHECK, not a comment.
 *
 *   WHAT THIS CANNOT TEST, stated so nobody assumes otherwise: that the store
 *   shape reaching a device is 16 B units in 256 B blocks (that is an
 *   instruction-selection property, checked in the disassembly on the card host),
 *   and that `ft_tt_noc_copy_store_fence()` retires a write-combining buffer —
 *   there are no WC buffers over ordinary host memory. Both are on-card
 *   questions and `test_udrv_card_noc.c` answers the second.
 *
 * @status Runs anywhere. No device, no root, no card.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_noc_copy.h"
#include "ft_tt_test_report.h"

/** A 2 MiB aperture, the only window size this board has. */
#define WINDOW_BYTES ((size_t)2u * 1024u * 1024u)

/** Bytes of guard either side of every destination buffer. A copy that writes
 *  one byte too many is a silent corruption of a neighbour, which is exactly
 *  the failure the window walk can produce, so every buffer is fenced. */
#define GUARD_BYTES 256u

/**
 * Guard filler, and THE TWO MUST DIFFER.
 *
 * This was a vacuous gate for one build of this file and the mutation battery
 * is what found it: with both buffers guarded by the same byte, a copy that
 * overran its destination by one unit read that unit out of the SOURCE's guard
 * and wrote the identical value into the DESTINATION's, so an over-run of
 * exactly the shape the walk can produce left every guard intact and every
 * check green. Giving the source a different filler makes the over-run write a
 * byte the destination guard cannot already hold.
 */
#define GUARD_BYTE 0xC7u
#define SOURCE_GUARD_BYTE 0x3Eu

/**
 * The fixture pattern: an LCG folded over the ABSOLUTE offset, so no two
 * 16-byte units, 256-byte blocks or 4 KiB pages are alike.
 */
static unsigned char pattern_byte(size_t offset, uint32_t salt)
{
    uint32_t state = (uint32_t)(offset + 1u) * 2654435761u + salt;

    state ^= state >> 15;
    state *= 2246822519u;
    state ^= state >> 13;
    return (unsigned char)(state & 0xFFu);
}

static void fill_pattern(unsigned char *buffer, size_t base_offset, size_t bytes, uint32_t salt)
{
    size_t index;

    for (index = 0u; index < bytes; index++)
        buffer[index] = pattern_byte(base_offset + index, salt);
}

/** Whether `bytes` of the pattern repeat with period `period`. */
static bool pattern_repeats_at(size_t bytes, size_t period, uint32_t salt)
{
    size_t index;

    if (period == 0u || bytes <= period)
        return false;

    for (index = 0u; index + period < bytes; index++) {
        if (pattern_byte(index, salt) != pattern_byte(index + period, salt))
            return false;
    }
    return true;
}

/* ------------------------------------------------------------------------ *
 * The fixture is not vacuous
 * ------------------------------------------------------------------------ */

static void stage_the_fixture_discriminates(void)
{
    const size_t span = 64u * 1024u;
    size_t distinct_units = 0u;
    size_t index;
    size_t probe;

    ft_tt_test_step("The fixture pattern — a comparison against a periodic pattern proves nothing");

    ft_tt_test_check("not 16-periodic (the copy UNIT)", !pattern_repeats_at(span, 16u, 0u), "");
    ft_tt_test_check("not 256-periodic (the copy BLOCK — the exact shape that made a battery "
                     "vacuous here once)",
                     !pattern_repeats_at(span, 256u, 0u), "");
    ft_tt_test_check("not 4096-periodic (a host page)", !pattern_repeats_at(span, 4096u, 0u), "");

    /* And that two different 16 B units really do differ, sampled across the
     * whole span rather than at one lucky offset. */
    for (probe = 0u; probe + 32u <= span; probe += 4096u) {
        bool differs = false;

        for (index = 0u; index < 16u; index++) {
            if (pattern_byte(probe + index, 0u) != pattern_byte(probe + 16u + index, 0u))
                differs = true;
        }
        if (differs)
            distinct_units++;
    }
    ft_tt_test_check_fmt("every sampled pair of adjacent 16 B units differs",
                         distinct_units == span / 4096u, "%zu of %zu", distinct_units,
                         span / 4096u);

    ft_tt_test_check("two salts give different bytes — a second write is distinguishable from "
                     "the first",
                     pattern_byte(0u, 0u) != pattern_byte(0u, 1u) ||
                         pattern_byte(1u, 0u) != pattern_byte(1u, 1u),
                     "");
}

/* ------------------------------------------------------------------------ *
 * Width
 * ------------------------------------------------------------------------ */

static void stage_widths(void)
{
    ft_tt_test_step("Transfer widths");

    ft_tt_test_check_fmt("BLOCK16 is 16 bytes",
                         ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_BLOCK16) == 16u, "%u",
                         ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_BLOCK16));
    ft_tt_test_check_fmt("WORD32 is 4 bytes",
                         ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_WORD32) == 4u, "%u",
                         ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_WORD32));
    ft_tt_test_check_fmt("an unknown width is 0, not a plausible default",
                         ft_tt_noc_transfer_width_bytes((ft_tt_noc_transfer_width)7) == 0u, "%u",
                         ft_tt_noc_transfer_width_bytes((ft_tt_noc_transfer_width)7));
    ft_tt_test_check_fmt("the block is 16 units of the unit size",
                         FT_TT_NOC_COPY_BLOCK_BYTES == 16u * FT_TT_NOC_COPY_UNIT_BYTES, "%u / %u",
                         FT_TT_NOC_COPY_BLOCK_BYTES, FT_TT_NOC_COPY_UNIT_BYTES);
}

/* ------------------------------------------------------------------------ *
 * The window walk
 * ------------------------------------------------------------------------ */

static void stage_window_slice(void)
{
    ft_tt_noc_window_slice slice;

    ft_tt_test_step("The window walk — pure arithmetic, and the half that corrupts a neighbour "
                    "rather than failing");

    memset(&slice, 0, sizeof(slice));
    ft_tt_test_check("a NULL out is refused",
                     !ft_tt_noc_window_slice_compute(0u, 16u, WINDOW_BYTES, NULL), "");
    ft_tt_test_check("a window size that is not a power of two is refused",
                     !ft_tt_noc_window_slice_compute(0u, 16u, 3u, &slice), "");
    ft_tt_test_check("a zero window size is refused",
                     !ft_tt_noc_window_slice_compute(0u, 16u, 0u, &slice), "");

    (void)ft_tt_noc_window_slice_compute(0u, 4096u, WINDOW_BYTES, &slice);
    ft_tt_test_check_fmt("a transfer inside one window takes one slice at offset 0",
                         slice.window_base == 0u && slice.offset_in_window == 0u &&
                             slice.bytes == 4096u,
                         "base 0x%llx off %zu bytes %zu", (unsigned long long)slice.window_base,
                         slice.offset_in_window, slice.bytes);

    (void)ft_tt_noc_window_slice_compute(WINDOW_BYTES + 4096u, 256u, WINDOW_BYTES, &slice);
    ft_tt_test_check_fmt("an address in the second window bases there, not at zero",
                         slice.window_base == WINDOW_BYTES && slice.offset_in_window == 4096u &&
                             slice.bytes == 256u,
                         "base 0x%llx off %zu", (unsigned long long)slice.window_base,
                         slice.offset_in_window);

    (void)ft_tt_noc_window_slice_compute(WINDOW_BYTES - 16u, 64u, WINDOW_BYTES, &slice);
    ft_tt_test_check_fmt("a transfer straddling a boundary is CUT at the boundary",
                         slice.window_base == 0u && slice.offset_in_window == WINDOW_BYTES - 16u &&
                             slice.bytes == 16u,
                         "%zu bytes of 64 in this slice", slice.bytes);

    (void)ft_tt_noc_window_slice_compute(WINDOW_BYTES, 16u, WINDOW_BYTES, &slice);
    ft_tt_test_check_fmt("an address exactly ON a boundary starts the next window at offset 0",
                         slice.window_base == WINDOW_BYTES && slice.offset_in_window == 0u,
                         "base 0x%llx off %zu", (unsigned long long)slice.window_base,
                         slice.offset_in_window);

    /* Walk a 5 MiB transfer the way a transport does and require the slices to
     * TILE the span: contiguous, no gap, no overlap, no zero-length step. */
    {
        const size_t total = 5u * 1024u * 1024u;
        uint64_t address = WINDOW_BYTES - 4096u;
        uint64_t expect = address;
        size_t remaining = total;
        size_t steps = 0u;
        size_t covered = 0u;
        bool contiguous = true;

        while (remaining > 0u && steps < 64u) {
            if (!ft_tt_noc_window_slice_compute(address, remaining, WINDOW_BYTES, &slice))
                break;
            if (slice.bytes == 0u)
                break;
            if (slice.window_base + slice.offset_in_window != expect)
                contiguous = false;
            expect += slice.bytes;
            address += slice.bytes;
            covered += slice.bytes;
            remaining -= slice.bytes;
            steps++;
        }

        ft_tt_test_check_fmt("a 5 MiB transfer is covered exactly", covered == total, "%zu of %zu",
                             covered, total);
        ft_tt_test_check("...by contiguous slices with no gap and no overlap", contiguous, "");
        ft_tt_test_check_fmt("...and it really did move the window, more than once", steps == 4u,
                             "%zu slices", steps);
    }
}

/* ------------------------------------------------------------------------ *
 * The copy
 * ------------------------------------------------------------------------ */

/** Allocate a guarded buffer: GUARD_BYTES of `filler`, `bytes` of payload, GUARD_BYTES. */
static unsigned char *guarded_alloc(size_t bytes, unsigned char filler)
{
    unsigned char *block = malloc(bytes + 2u * GUARD_BYTES);

    if (!block)
        return NULL;
    memset(block, filler, bytes + 2u * GUARD_BYTES);
    return block;
}

static bool guards_intact(const unsigned char *block, size_t bytes)
{
    size_t index;

    for (index = 0u; index < GUARD_BYTES; index++) {
        if (block[index] != GUARD_BYTE)
            return false;
        if (block[GUARD_BYTES + bytes + index] != GUARD_BYTE)
            return false;
    }
    return true;
}

/**
 * One copy in one direction at one length, byte-compared, with the guards
 * either side of the destination checked for over-run.
 */
static void check_one_copy(size_t bytes, bool is_read, ft_tt_noc_transfer_width width,
                           const char *label)
{
    unsigned char *source_block = guarded_alloc(bytes, SOURCE_GUARD_BYTE);
    unsigned char *destination_block = guarded_alloc(bytes, GUARD_BYTE);
    unsigned char *source;
    unsigned char *destination;
    bool equal;

    if (!source_block || !destination_block) {
        ft_tt_test_check_fmt("allocation", false, "%s", label);
        free(source_block);
        free(destination_block);
        return;
    }

    source = source_block + GUARD_BYTES;
    destination = destination_block + GUARD_BYTES;
    fill_pattern(source, 0u, bytes, 0u);

    /* `aperture` is host memory here. Which side of the copy it is decides
     * nothing about the shape — the same units and blocks are issued either
     * way — so both directions are exercised through the same entry point. */
    if (is_read)
        ft_tt_noc_copy_slice((volatile unsigned char *)source, destination, bytes, true, width);
    else
        ft_tt_noc_copy_slice((volatile unsigned char *)destination, source, bytes, false, width);

    equal = memcmp(source, destination, bytes) == 0;
    ft_tt_test_check_fmt(label, equal, "%zu bytes, %s", bytes, equal ? "byte-exact" : "MISMATCH");
    ft_tt_test_check_fmt("...and nothing outside the destination was written",
                         guards_intact(destination_block, bytes), "%s, %u B guard either side",
                         label, GUARD_BYTES);

    free(source_block);
    free(destination_block);
}

static void stage_the_copy(void)
{
    ft_tt_test_step("The PIO copy — every length class, both directions, guarded");

    /* Lengths chosen against the shape: under one unit's block, exactly one
     * block, a block plus a tail, several blocks, and a page. */
    check_one_copy(16u, false, FT_TT_NOC_WIDTH_BLOCK16, "block16 write, one unit");
    check_one_copy(240u, false, FT_TT_NOC_WIDTH_BLOCK16, "block16 write, one unit short of a block");
    check_one_copy(256u, false, FT_TT_NOC_WIDTH_BLOCK16, "block16 write, exactly one block");
    check_one_copy(272u, false, FT_TT_NOC_WIDTH_BLOCK16, "block16 write, a block and a tail");
    check_one_copy(4096u, false, FT_TT_NOC_WIDTH_BLOCK16, "block16 write, a page");
    check_one_copy(65536u, false, FT_TT_NOC_WIDTH_BLOCK16, "block16 write, 64 KiB");
    check_one_copy(4096u, true, FT_TT_NOC_WIDTH_BLOCK16, "block16 read, a page");
    check_one_copy(272u, true, FT_TT_NOC_WIDTH_BLOCK16, "block16 read, a block and a tail");

    check_one_copy(4u, false, FT_TT_NOC_WIDTH_WORD32, "word32 write, one word");
    check_one_copy(64u, false, FT_TT_NOC_WIDTH_WORD32, "word32 write, sixteen words");
    check_one_copy(4u, true, FT_TT_NOC_WIDTH_WORD32, "word32 read, one word");
    check_one_copy(4096u, true, FT_TT_NOC_WIDTH_WORD32, "word32 read, a page");

    /* Zero is a legal no-op and must not touch the destination at all. */
    {
        unsigned char destination[64];
        unsigned char source[64];

        memset(destination, GUARD_BYTE, sizeof(destination));
        memset(source, 0x11u, sizeof(source));
        ft_tt_noc_copy_slice((volatile unsigned char *)destination, source, 0u, false,
                             FT_TT_NOC_WIDTH_BLOCK16);
        ft_tt_test_check("a zero-byte copy writes nothing", destination[0] == GUARD_BYTE, "");
    }

    /* And the comparison can say no. Without this every equality above could be
     * a statement about two buffers that were already identical. */
    {
        unsigned char source[512];
        unsigned char destination[512];

        fill_pattern(source, 0u, sizeof(source), 0u);
        fill_pattern(destination, 0u, sizeof(destination), 1u);
        ft_tt_test_check("two DIFFERENT patterns compare unequal — the byte comparison "
                         "discriminates",
                         memcmp(source, destination, sizeof(source)) != 0, "");

        ft_tt_noc_copy_slice((volatile unsigned char *)destination, source, sizeof(source), false,
                             FT_TT_NOC_WIDTH_BLOCK16);
        ft_tt_test_check("...and equal once the copy has run",
                         memcmp(source, destination, sizeof(source)) == 0, "");
    }
}

/* ------------------------------------------------------------------------ */

int main(void)
{
    stage_the_fixture_discriminates();
    stage_widths();
    stage_window_slice();
    stage_the_copy();

    /* Callable everywhere; there are no write-combining buffers over host
     * memory, so this is a compile-and-link check, not a behavioural one. Said
     * out loud rather than dressed up as a gate. */
    ft_tt_noc_copy_store_fence();
    ft_tt_test_step("(the store fence is exercised on the card, not here — host memory has no "
                    "write-combining buffers to retire)");

    return ft_tt_test_finish();
}
