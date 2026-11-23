/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_noc.c — the named NoC read/write helpers (ft_tt_noc.{h,c}) and the
 * device-memory bump allocator (ft_tt_buffer.{h,c}).
 *
 * Two halves, deliberately separated:
 *
 *   unit    Pure arithmetic: the alignment rule table, the transfer-bounds
 *           check, and every allocator behaviour. NO DEVICE IS OPENED, so this
 *           runs on the Mac and is where the rules get proven wrong — a wrong
 *           NoC coordinate hangs the chip and costs a physical power cycle
 *           (spec §7.3), so as little as possible is discovered on silicon.
 *
 *   l1      Write and read back one Tensix's L1 through ft_tt_write_l1 /
 *           ft_tt_read_l1, on a core DERIVED from live telemetry.
 *   dram    Round-trip every enabled DRAM bank, and prove the banks are
 *           distinct endpoints rather than aliases of one.
 *   cross   A transfer that genuinely spans several 2 MiB windows. This is the
 *           re-windowing path, and ft_tt_noc_path_aim_count() is what tells a
 *           real multi-window run from a lucky in-window one.
 *   align   Alignment refusals against a live path: the helper must reject
 *           without issuing anything.
 *   bw      MB/s through the helpers, against the 287 MB/s WC baseline the raw
 *           window layer measured (spec §9.8).
 *
 * SAFETY: every coordinate comes from ft_tt_coord against the live grid, and
 * the Tensix used is index 1 or higher — index 0 is reserved for the launch
 * work happening in parallel.
 *
 * Build (unit half needs no fence; the device halves do):
 *   make -f Makefile.noc check          unit only
 *   make -f Makefile.noc PHASE1B=1 all  then ./build-noc/test_noc dram
 */

/* clock_gettime/CLOCK_MONOTONIC are POSIX and -std=c99 turns off glibc's implicit
 * feature-test macros. 200809L, not 199309L: the older level also hides snprintf. */
#define _POSIX_C_SOURCE 200809L

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_buffer.h"
#include "ft_tt_arch.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_noc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The Tensix this test is allowed to touch. Index 0 belongs to the launch
 * bring-up running concurrently; sharing a core across two agents is how a
 * half-loaded firmware image gets blamed on the wrong change. */
#define TEST_TENSIX_INDEX 1u

/* The DRAM bank the multi-MiB steps (cross, align, bw) write to. Bank 0 belongs to
 * the kernel-toolchain work running in parallel; the all-bank sweep in step_dram
 * still has to touch it, because "the banks are distinct endpoints" is exactly what
 * it proves, but nothing else here needs to. */
#define TEST_DRAM_BANK 1u

#define KIB 1024u
#define MIB (1024u * 1024u)

static int failures;
static int checks;

static void check(const char *what, bool ok, const char *detail)
{
    checks++;
    if (!ok)
        failures++;
    printf("  %-4s  %-58s %s\n", ok ? "ok" : "FAIL", what, detail ? detail : "");
}

static void check_rc(const char *what, int actual, int expected)
{
    char detail[96];

    snprintf(detail, sizeof(detail), "rc=%d (%s), expected %d (%s)", actual,
             ft_tt_error_string(actual), expected, ft_tt_error_string(expected));
    check(what, actual == expected, detail);
}

static double now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/** Position-dependent so a shifted, truncated or partially-landed transfer is
 *  visible instead of looking like a clean success. */
static void fill_pattern(unsigned char *buffer, size_t bytes, uint32_t seed)
{
    size_t index;

    for (index = 0; index + 4u <= bytes; index += 4u) {
        uint32_t word = seed ^ (uint32_t)(index / 4u) * 0x9E3779B9u;

        memcpy(buffer + index, &word, sizeof(word));
    }
}

/** memcmp with a first-difference report, because "mismatch" alone does not
 *  distinguish a dropped tail from a wrong window. */
static bool compare_pattern(const unsigned char *expected, const unsigned char *got, size_t bytes,
                            char *detail, size_t detail_bytes)
{
    size_t index;

    for (index = 0; index < bytes; index++) {
        if (expected[index] != got[index]) {
            snprintf(detail, detail_bytes,
                     "first difference at byte %zu of %zu (wrote 0x%02x, read 0x%02x)", index,
                     bytes, expected[index], got[index]);
            return false;
        }
    }
    snprintf(detail, detail_bytes, "%zu bytes byte-exact", bytes);
    return true;
}

/* ================================================================
 * unit: alignment and bounds
 * ================================================================ */

static void unit_alignment_rules(void)
{
    char detail[96];

    printf("\n-- alignment rules (ft_tt_noc_alignment) --\n");

    snprintf(detail, sizeof(detail), "L1 write %u, L1 read %u",
             ft_tt_noc_alignment(FT_TT_MEM_SPACE_L1, false),
             ft_tt_noc_alignment(FT_TT_MEM_SPACE_L1, true));
    check("L1 is 16 B in both directions",
          ft_tt_noc_alignment(FT_TT_MEM_SPACE_L1, false) == 16u &&
              ft_tt_noc_alignment(FT_TT_MEM_SPACE_L1, true) == 16u,
          detail);

    snprintf(detail, sizeof(detail), "DRAM write %u, DRAM read %u",
             ft_tt_noc_alignment(FT_TT_MEM_SPACE_DRAM, false),
             ft_tt_noc_alignment(FT_TT_MEM_SPACE_DRAM, true));
    check("DRAM writes take 16 B, DRAM reads take 64 B",
          ft_tt_noc_alignment(FT_TT_MEM_SPACE_DRAM, false) == 16u &&
              ft_tt_noc_alignment(FT_TT_MEM_SPACE_DRAM, true) == 64u,
          detail);

    check("an unknown space has no alignment", ft_tt_noc_alignment((ft_tt_memory_space)7, false) == 0u,
          NULL);

    snprintf(detail, sizeof(detail), "L1 %llu, DRAM %llu",
             (unsigned long long)ft_tt_noc_space_size(FT_TT_MEM_SPACE_L1),
             (unsigned long long)ft_tt_noc_space_size(FT_TT_MEM_SPACE_DRAM));
    check("space sizes are 1536 KiB and 0xff000000",
          ft_tt_noc_space_size(FT_TT_MEM_SPACE_L1) == (uint64_t)FT_TT_MEM_L1_SIZE &&
              ft_tt_noc_space_size(FT_TT_MEM_SPACE_DRAM) == (uint64_t)FT_TT_MEM_DRAM_SIZE,
          detail);
}

static void unit_transfer_checks(void)
{
    uint64_t l1_top = (uint64_t)FT_TT_MEM_L1_SIZE;
    uint64_t dram_top = (uint64_t)FT_TT_MEM_DRAM_SIZE;

    printf("\n-- transfer legality (ft_tt_noc_check_transfer) --\n");

    check_rc("L1 write, 16 B aligned both ways",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, 0x9F80u, 4096u), FT_TT_OK);
    check_rc("L1 write with a misaligned address is refused",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, 0x9F88u, 4096u), FT_TT_EINVAL);
    check_rc("L1 write with a misaligned length is refused",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, 0x9F80u, 4100u), FT_TT_EINVAL);
    check_rc("a zero-length transfer is legal",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, 0x9F80u, 0u), FT_TT_OK);

    check_rc("a 16 B-aligned DRAM WRITE is legal",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, false, 16u, 16u), FT_TT_OK);
    check_rc("the same address is refused for a DRAM READ (64 B rule)",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, 16u, 64u), FT_TT_EINVAL);
    check_rc("a 64 B-aligned DRAM read is legal",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, 64u, 64u), FT_TT_OK);
    check_rc("a DRAM read of 16 B is refused — the LENGTH needs 64 too",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, 64u, 16u), FT_TT_EINVAL);

    check_rc("a transfer ending exactly at the top of L1 is legal",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, l1_top - 16u, 16u), FT_TT_OK);
    check_rc("one 16 B unit past the top of L1 is refused",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, l1_top - 16u, 32u), FT_TT_EINVAL);
    check_rc("a transfer starting at the top of L1 is refused",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, l1_top, 16u), FT_TT_EINVAL);
    check_rc("a transfer ending exactly at the top of a DRAM bank is legal",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, dram_top - 64u, 64u), FT_TT_OK);
    check_rc("past the top of a DRAM bank is refused",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, dram_top, 64u), FT_TT_EINVAL);
    check_rc("an unknown space is refused",
             ft_tt_noc_check_transfer((ft_tt_memory_space)7, false, 0u, 16u), FT_TT_EINVAL);

    /* ---------------------------------------------------------------------
     * THE PAIRING. Everything above scores ONE end against ONE space, which is what the
     * checker could do until now — and it is why a transfer whose two ends are each legal
     * on their own could still be illegal. Upstream tests the CONGRUENCE and nothing
     * weaker.
     *
     * The first case below is the whole point: an L1 address at 0x9F80 and a DRAM address
     * at 0x40 BOTH pass their own rule — 0x9F80 is 16-aligned, 0x40 is 64-aligned — and
     * they sit at different offsets inside the 64-byte window the stricter end imposes, so
     * the transfer is not one the NoC will do. `ft_tt_noc_check_transfer()` calls it legal
     * from either side. That is the hole.
     * ------------------------------------------------------------------- */
    printf("\n-- transfer legality, BOTH ends (ft_tt_noc_check_transfer_pair) --\n");

    check_rc("each end of a 16-vs-64 mismatch passes its OWN rule: the L1 end",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, 0x9F90u, 64u), FT_TT_OK);
    check_rc("...and the DRAM end",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, 0x40u, 64u), FT_TT_OK);
    check_rc("but the PAIR is refused — 0x9F90 is 16 into the 64 B window and 0x40 is 0",
             ft_tt_noc_check_transfer_pair(FT_TT_MEM_SPACE_L1, 0x9F90u, FT_TT_MEM_SPACE_DRAM,
                                           0x40u, true, 64u),
             FT_TT_EINVAL);
    check_rc("move the L1 end onto the same offset and the pair is legal",
             ft_tt_noc_check_transfer_pair(FT_TT_MEM_SPACE_L1, 0x9FC0u, FT_TT_MEM_SPACE_DRAM,
                                           0x40u, true, 64u),
             FT_TT_OK);
    /* THE HISTORICAL CASE, replayed. 0x137A0 was the Phase 2 test map's first derived L1
     * base: 16- and 32-byte aligned, and 32 bytes into the 64-byte window. As the LOCAL end
     * of a DRAM read it produced tiles that were byte-exact up to element 992 of 1024 and
     * wrong in their last 64 bytes, with no refusal, no error, and three of four stages
     * still passing. Nothing in the checker could see it, because each
     * end passed its own rule. */
    check_rc("0x137A0 is a legal L1 address on its own",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, 0x137A0u, 2048u), FT_TT_OK);
    check_rc("and DRAM tile 0 is a legal DRAM read on its own",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, 0u, 2048u), FT_TT_OK);
    check_rc("the PAIR is refused — this is the 992-of-1024 bug, caught before it is issued",
             ft_tt_noc_check_transfer_pair(FT_TT_MEM_SPACE_L1, 0x137A0u, FT_TT_MEM_SPACE_DRAM,
                                           0u, true, 2048u),
             FT_TT_EINVAL);
    /* An L1-to-L1 pair is judged in a 16 B window, and both ends already have to be
     * 16-aligned to pass their own test — so the congruence is automatic there and adds
     * nothing. It is the DRAM READ direction, and only that direction, where the window is
     * wider than either end's own rule. Stated as a check so a future alignment change has
     * to face it. */
    check_rc("an L1-to-L1 pair is judged in a 16 B window, so congruence is automatic",
             ft_tt_noc_check_transfer_pair(FT_TT_MEM_SPACE_L1, 0x10000u, FT_TT_MEM_SPACE_L1,
                                           0x20010u, false, 64u),
             FT_TT_OK);
    check_rc("the pair still refuses what a single end would: a misaligned local address",
             ft_tt_noc_check_transfer_pair(FT_TT_MEM_SPACE_L1, 0x9F88u, FT_TT_MEM_SPACE_L1,
                                           0x9F88u, false, 64u),
             FT_TT_EINVAL);
    check_rc("and a span past the top of DRAM",
             ft_tt_noc_check_transfer_pair(FT_TT_MEM_SPACE_L1, 0x10000u, FT_TT_MEM_SPACE_DRAM,
                                           dram_top, true, 64u),
             FT_TT_EINVAL);
    /* The DRAM WRITE direction takes the 16-byte rule on the DRAM end, so the window the
     * pair is judged in is 16 and not 64 — a distinction that only exists because the
     * alignment is direction-dependent, and the reason the window is computed rather than
     * fixed. */
    check_rc("a DRAM WRITE judges the pair in a 16 B window, not a 64 B one",
             ft_tt_noc_check_transfer_pair(FT_TT_MEM_SPACE_L1, 0x9F90u, FT_TT_MEM_SPACE_DRAM,
                                           0x10u, false, 16u),
             FT_TT_OK);
}

/* ================================================================
 * unit: the shared window walk
 * ================================================================ */

/**
 * ft_tt_noc_window_slice_at() is the arithmetic BOTH device paths run on — the
 * named L1/DRAM helpers and ft_tt_launch's run control, which since the Phase 2
 * reconciliation goes through ft_tt_noc_transfer_raw() instead of its own copy
 * of this loop. It is also the piece that fails QUIETLY: a wrong window base is
 * still a valid transaction, just to the wrong address on the right core. So it
 * is a pure function, and it gets proven here rather than on silicon.
 */
static void unit_window_slices(void)
{
    const size_t window = 2u * MIB;
    ft_tt_noc_window_slice slice;
    char detail[160];

    printf("\n-- window slicing (ft_tt_noc_window_slice_at) --\n");

    check_rc("a transfer wholly inside one window slices whole",
             ft_tt_noc_window_slice_at(0x9F80u, 4096u, window, &slice), FT_TT_OK);
    snprintf(detail, sizeof(detail), "base 0x%llx, offset %zu, %zu bytes",
             (unsigned long long)slice.window_base, slice.offset_in_window, slice.bytes);
    check("it aims at 0 and takes all 4096 bytes",
          slice.window_base == 0u && slice.offset_in_window == 0x9F80u && slice.bytes == 4096u,
          detail);

    check_rc("a transfer that straddles a boundary is cut at it",
             ft_tt_noc_window_slice_at(window - 64u * KIB, 5u * MIB, window, &slice), FT_TT_OK);
    snprintf(detail, sizeof(detail), "base 0x%llx, offset %zu, %zu bytes",
             (unsigned long long)slice.window_base, slice.offset_in_window, slice.bytes);
    check("the first slice stops exactly at the 2 MiB line",
          slice.window_base == 0u && slice.offset_in_window == window - 64u * KIB &&
              slice.bytes == 64u * KIB,
          detail);

    check_rc("an address exactly on a boundary starts a fresh window",
             ft_tt_noc_window_slice_at(window, 32u, window, &slice), FT_TT_OK);
    snprintf(detail, sizeof(detail), "base 0x%llx, offset %zu",
             (unsigned long long)slice.window_base, slice.offset_in_window);
    check("base is the boundary and the offset is 0",
          slice.window_base == (uint64_t)window && slice.offset_in_window == 0u &&
              slice.bytes == 32u,
          detail);

    /* The soft-reset register is why this function cannot assume a memory space:
     * 0xFFB121B0 is in neither L1 nor a DRAM bank. */
    check_rc("the soft-reset register slices like anything else",
             ft_tt_noc_window_slice_at(0xFFB121B0u, 4u, window, &slice), FT_TT_OK);
    snprintf(detail, sizeof(detail), "base 0x%llx, offset 0x%zx",
             (unsigned long long)slice.window_base, slice.offset_in_window);
    /* 0xFFB00000 is NOT 2 MiB aligned, so the window lands 1 MiB lower and the
     * register sits 0x1121B0 into the aperture — still inside it. */
    check("base 0xFFA00000, offset 0x1121B0, 4 bytes",
          slice.window_base == 0xFFA00000u && slice.offset_in_window == 0x1121B0u &&
              slice.bytes == 4u && slice.offset_in_window < window,
          detail);

    check_rc("a zero-length transfer slices to nothing",
             ft_tt_noc_window_slice_at(0x9F80u, 0u, window, &slice), FT_TT_OK);
    check("and it asks for no bytes", slice.bytes == 0u, NULL);

    check_rc("a window size that is not a power of two is refused",
             ft_tt_noc_window_slice_at(0u, 16u, 3u * MIB, &slice), FT_TT_EINVAL);
    check_rc("a zero window size is refused", ft_tt_noc_window_slice_at(0u, 16u, 0u, &slice),
             FT_TT_EINVAL);
    check_rc("a NULL slice is refused", ft_tt_noc_window_slice_at(0u, 16u, window, NULL),
             FT_TT_EINVAL);

    /* Walk a whole 5 MiB transfer the way ft_tt_noc_transfer_raw() does. This is
     * the off-hardware twin of the `cross` step: same span, same expected four
     * window moves, no card. */
    {
        uint64_t address = (uint64_t)window - 64u * KIB;
        size_t remaining = 5u * MIB;
        size_t moved = 0;
        unsigned windows = 0;
        uint64_t previous_base = UINT64_MAX;
        bool bases_ascend = true;

        while (remaining > 0u && windows < 64u) {
            if (ft_tt_noc_window_slice_at(address, remaining, window, &slice) != FT_TT_OK)
                break;
            if (previous_base != UINT64_MAX && slice.window_base != previous_base + window)
                bases_ascend = false;
            previous_base = slice.window_base;
            address += slice.bytes;
            moved += slice.bytes;
            remaining -= slice.bytes;
            windows++;
        }
        snprintf(detail, sizeof(detail), "%u windows, %zu of %u bytes moved", windows, moved,
                 5u * MIB);
        check("a 5 MiB span 64 KiB below a boundary walks four windows and loses nothing",
              windows == 4u && moved == (size_t)5u * MIB && remaining == 0u && bases_ascend,
              detail);
    }
}

/**
 * The two store widths. WORD32 exists because a 4-byte device register cannot be
 * written 16 bytes at a time; if this table ever said otherwise, the launch path
 * would write the soft-reset register's three neighbours with it.
 */
static void unit_transfer_widths(void)
{
    char detail[128];

    printf("\n-- store widths (ft_tt_noc_transfer_width_bytes) --\n");

    snprintf(detail, sizeof(detail), "BLOCK16 %u, WORD32 %u",
             ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_BLOCK16),
             ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_WORD32));
    check("BLOCK16 is 16 B and WORD32 is 4 B",
          ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_BLOCK16) == 16u &&
              ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_WORD32) == 4u,
          detail);
    check("an unknown width has no size",
          ft_tt_noc_transfer_width_bytes((ft_tt_noc_transfer_width)9) == 0u, NULL);

    check_rc("a 4 B register access is legal at WORD32",
             ft_tt_noc_check_raw_transfer(FT_TT_NOC_WIDTH_WORD32, 0xFFB121B0u, 4u), FT_TT_OK);
    check_rc("the same 4 B access is REFUSED at BLOCK16 — a 16 B store over a register",
             ft_tt_noc_check_raw_transfer(FT_TT_NOC_WIDTH_BLOCK16, 0xFFB121B0u, 4u), FT_TT_EINVAL);
    check_rc("a 2-byte-misaligned address is refused at WORD32",
             ft_tt_noc_check_raw_transfer(FT_TT_NOC_WIDTH_WORD32, 0xFFB121B2u, 4u), FT_TT_EINVAL);
    check_rc("a length that is not a whole number of words is refused",
             ft_tt_noc_check_raw_transfer(FT_TT_NOC_WIDTH_WORD32, 0x1000u, 6u), FT_TT_EINVAL);
    check_rc("a 16 B-aligned bulk transfer is legal at BLOCK16",
             ft_tt_noc_check_raw_transfer(FT_TT_NOC_WIDTH_BLOCK16, 0x1000u, 4096u), FT_TT_OK);
    check_rc("an unknown width is refused",
             ft_tt_noc_check_raw_transfer((ft_tt_noc_transfer_width)9, 0u, 16u), FT_TT_EINVAL);
    check_rc("a zero-length raw transfer is legal",
             ft_tt_noc_check_raw_transfer(FT_TT_NOC_WIDTH_WORD32, 0x1000u, 0u), FT_TT_OK);
}

/* ================================================================
 * unit: the allocator
 * ================================================================ */

/** The topology this card reports, built by hand so the allocator can be tested
 *  with no device present. Matches what ARC telemetry returns here. */
static ft_tt_grid_info fake_grid(uint32_t dram_bank_count)
{
    ft_tt_grid_info grid;

    memset(&grid, 0, sizeof(grid));
    grid.valid = true;
    /* Off-hardware nothing selects an architecture, so the fixture names one. */
    grid.arch = ft_tt_arch_blackhole();
    grid.tensix_enabled_col = 16383u;
    grid.tensix_column_count = 14u;
    grid.tensix_core_count = 140u;
    grid.enabled_gddr = 255u;
    grid.dram_bank_count = dram_bank_count;
    return grid;
}

static void unit_allocator_init(void)
{
    ft_tt_allocator allocator;
    ft_tt_grid_info grid = fake_grid(8u);
    ft_tt_grid_info unread;
    char detail[128];

    printf("\n-- allocator setup --\n");

    memset(&unread, 0, sizeof(unread));
    check_rc("a grid that was never read from telemetry is refused",
             ft_tt_allocator_init(&allocator, NULL, &unread, 0u), FT_TT_EUNAVAIL);

    {
        ft_tt_grid_info zero_banks = fake_grid(0u);
        ft_tt_grid_info too_many = fake_grid(FT_TT_MAX_DRAM_BANKS + 1u);
        ft_tt_grid_info no_arch = fake_grid(8u);
        ft_tt_grid_info more_than_the_part;
        ft_tt_arch six_bank_part;

        check_rc("zero DRAM banks is refused",
                 ft_tt_allocator_init(&allocator, NULL, &zero_banks, 0u), FT_TT_EINVAL);
        check_rc("more banks than the arena array can hold is refused",
                 ft_tt_allocator_init(&allocator, NULL, &too_many, 0u), FT_TT_EINVAL);

        /* The bank count is meaningless without knowing the part it came from,
         * and guessing one is how a wrong bank becomes a NoC write to nowhere. */
        no_arch.arch = NULL;
        check_rc("a grid with no architecture is refused",
                 ft_tt_allocator_init(&allocator, NULL, &no_arch, 0u), FT_TT_EINVAL);

        /* The bound that is NOT the compiled-in constant: telemetry claiming more
         * banks than the SELECTED part has means the mask was misread. A part with
         * fewer banks than the capacity is the case a 6-bank arch would hit, and
         * the only way to reach it from here is to name one. */
        six_bank_part = *ft_tt_arch_blackhole();
        six_bank_part.dram.bank_count = 6u;
        more_than_the_part = fake_grid(8u);
        more_than_the_part.arch = &six_bank_part;
        check_rc("8 live banks on a 6-bank part is refused",
                 ft_tt_allocator_init(&allocator, NULL, &more_than_the_part, 0u), FT_TT_EINVAL);

        more_than_the_part.dram_bank_count = 6u;
        check_rc("...and 6 on the same part initialises",
                 ft_tt_allocator_init(&allocator, NULL, &more_than_the_part, 0u), FT_TT_OK);

        /* The bank bound at USE time must be the live count, not the array
         * capacity. Asked as an ALLOCATION and against EINVAL specifically:
         * querying free bytes on an out-of-range arena answers 0 either way,
         * because the untouched arenas are zeroed — so the obvious version of
         * this check passes on a bound that is wrong. */
        {
            ft_tt_buffer past_the_end;

            check_rc("bank 6 on a 6-bank part names no arena, whatever the capacity is",
                     ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_DRAM, 6u, 64u, &past_the_end),
                     FT_TT_EINVAL);
            check_rc("...and bank 5 does",
                     ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_DRAM, 5u, 64u, &past_the_end),
                     FT_TT_OK);
        }
    }

    check_rc("an L1 floor below the usable base is refused",
             ft_tt_allocator_init(&allocator, NULL, &grid, FT_TT_L1_USABLE_BASE - 16u),
             FT_TT_EINVAL);
    check_rc("an L1 floor at the top of L1 is refused",
             ft_tt_allocator_init(&allocator, NULL, &grid, FT_TT_MEM_L1_SIZE), FT_TT_EINVAL);

    check_rc("a default allocator initialises", ft_tt_allocator_init(&allocator, NULL, &grid, 0u),
             FT_TT_OK);

    snprintf(detail, sizeof(detail), "%zu free, FT_TT_L1_USABLE_SIZE is %u",
             ft_tt_allocator_free_bytes(&allocator, FT_TT_MEM_SPACE_L1, 0u), FT_TT_L1_USABLE_SIZE);
    check("the L1 arena is exactly the L1 above the reserved map",
          ft_tt_allocator_free_bytes(&allocator, FT_TT_MEM_SPACE_L1, 0u) ==
              (size_t)FT_TT_L1_USABLE_SIZE,
          detail);

    snprintf(detail, sizeof(detail), "MEM_MAX_KERNEL_SIZE %u overshoots by %u bytes",
             FT_TT_MEM_MAX_KERNEL_SIZE, FT_TT_MEM_MAX_KERNEL_SIZE - FT_TT_L1_USABLE_SIZE);
    check("and it is SMALLER than MEM_MAX_KERNEL_SIZE, which is not a bound",
          (size_t)FT_TT_L1_USABLE_SIZE < (size_t)FT_TT_MEM_MAX_KERNEL_SIZE, detail);

    snprintf(detail, sizeof(detail), "%zu free per bank",
             ft_tt_allocator_free_bytes(&allocator, FT_TT_MEM_SPACE_DRAM, 0u));
    check("each DRAM arena is one whole bank",
          ft_tt_allocator_free_bytes(&allocator, FT_TT_MEM_SPACE_DRAM, 0u) ==
              (size_t)FT_TT_MEM_DRAM_SIZE,
          detail);
    check("a bank index past the enabled count names no arena",
          ft_tt_allocator_free_bytes(&allocator, FT_TT_MEM_SPACE_DRAM, 8u) == 0u, NULL);
}

static void unit_allocator_alloc(void)
{
    ft_tt_allocator allocator;
    ft_tt_grid_info grid = fake_grid(8u);
    ft_tt_buffer first;
    ft_tt_buffer second;
    ft_tt_buffer dram;
    char detail[160];

    printf("\n-- allocation --\n");
    (void)ft_tt_allocator_init(&allocator, NULL, &grid, 0u);

    check_rc("a zero-byte allocation is refused",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, 0u, &first), FT_TT_EINVAL);
    check_rc("an L1 allocation with a non-zero bank is refused",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 1u, 64u, &first), FT_TT_EINVAL);

    check_rc("a 1-byte L1 allocation succeeds",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, 1u, &first), FT_TT_OK);
    snprintf(detail, sizeof(detail), "address 0x%llx, %zu asked, %zu reserved",
             (unsigned long long)first.address, first.bytes, first.reserved_bytes);
    check("it starts at the usable L1 floor and reserves a whole 16 B unit",
          first.address == (uint64_t)FT_TT_L1_USABLE_BASE && first.reserved_bytes == 16u, detail);

    check_rc("a second allocation follows it",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, 4096u, &second), FT_TT_OK);
    snprintf(detail, sizeof(detail), "0x%llx then 0x%llx", (unsigned long long)first.address,
             (unsigned long long)second.address);
    check("the second address is the first plus its reservation",
          second.address == first.address + (uint64_t)first.reserved_bytes, detail);

    check_rc("a DRAM allocation on bank 3 succeeds",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_DRAM, 3u, 100u, &dram), FT_TT_OK);
    snprintf(detail, sizeof(detail), "address 0x%llx, %zu reserved",
             (unsigned long long)dram.address, dram.reserved_bytes);
    check("DRAM rounds up to the 64 B read alignment", dram.reserved_bytes == 128u, detail);
    check_rc("and the result is legal for a DRAM READ, unaided",
             ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, dram.address,
                                      dram.reserved_bytes),
             FT_TT_OK);
    check("allocating on bank 3 did not move bank 4",
          ft_tt_allocator_used_bytes(&allocator, FT_TT_MEM_SPACE_DRAM, 4u) == 0u, NULL);
    check_rc("a bank past the enabled count is refused",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_DRAM, 8u, 64u, &dram), FT_TT_EINVAL);

    printf("\n-- exhaustion --\n");
    (void)ft_tt_allocator_reset(&allocator);
    check_rc("the whole usable L1 can be taken in one go",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, FT_TT_L1_USABLE_SIZE, &first),
             FT_TT_OK);
    check_rc("one more 16 B unit does not fit",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, 16u, &second), FT_TT_ENOMEM);

    (void)ft_tt_allocator_reset(&allocator);
    check_rc("MEM_MAX_KERNEL_SIZE bytes of L1 CANNOT be allocated — it overshoots real L1",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, FT_TT_MEM_MAX_KERNEL_SIZE,
                                &first),
             FT_TT_ENOMEM);

    (void)ft_tt_allocator_reset(&allocator);
    check_rc("a whole DRAM bank can be taken in one go",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_DRAM, 0u, FT_TT_MEM_DRAM_SIZE, &dram),
             FT_TT_OK);
    check_rc("and then a single 64 B unit does not",
             ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_DRAM, 0u, 64u, &second), FT_TT_ENOMEM);
}

static void unit_allocator_free(void)
{
    ft_tt_allocator allocator;
    ft_tt_grid_info grid = fake_grid(8u);
    ft_tt_buffer first;
    ft_tt_buffer second;
    ft_tt_buffer other_bank;
    char detail[128];

    printf("\n-- LIFO free --\n");
    (void)ft_tt_allocator_init(&allocator, NULL, &grid, 0u);
    (void)ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, 256u, &first);
    (void)ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, 256u, &second);

    check_rc("freeing the OLDER buffer first is refused, not silently ignored",
             ft_tt_buffer_free(&allocator, &first), FT_TT_EINVAL);
    snprintf(detail, sizeof(detail), "%u live", ft_tt_allocator_live_count(&allocator, FT_TT_MEM_SPACE_L1, 0u));
    check("and the refusal left the arena untouched",
          ft_tt_allocator_live_count(&allocator, FT_TT_MEM_SPACE_L1, 0u) == 2u &&
              ft_tt_allocator_used_bytes(&allocator, FT_TT_MEM_SPACE_L1, 0u) == 512u,
          detail);

    check_rc("freeing the newest buffer works", ft_tt_buffer_free(&allocator, &second), FT_TT_OK);
    check_rc("then the older one works too", ft_tt_buffer_free(&allocator, &first), FT_TT_OK);
    check("the arena is back to empty",
          ft_tt_allocator_used_bytes(&allocator, FT_TT_MEM_SPACE_L1, 0u) == 0u &&
              ft_tt_allocator_live_count(&allocator, FT_TT_MEM_SPACE_L1, 0u) == 0u,
          NULL);
    check_rc("a third free of an empty arena is refused",
             ft_tt_buffer_free(&allocator, &first), FT_TT_EINVAL);

    (void)ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_DRAM, 1u, 4096u, &other_bank);
    other_bank.logical_bank = 2u;
    check_rc("a buffer freed against the wrong bank is refused",
             ft_tt_buffer_free(&allocator, &other_bank), FT_TT_EINVAL);

    check_rc("reset drops everything", ft_tt_allocator_reset(&allocator), FT_TT_OK);
    check("every arena is back at its floor",
          ft_tt_allocator_used_bytes(&allocator, FT_TT_MEM_SPACE_DRAM, 1u) == 0u &&
              ft_tt_allocator_live_count(&allocator, FT_TT_MEM_SPACE_DRAM, 1u) == 0u,
          NULL);
}

static void run_unit(void)
{
    unit_alignment_rules();
    unit_transfer_checks();
    unit_window_slices();
    unit_transfer_widths();
    unit_allocator_init();
    unit_allocator_alloc();
    unit_allocator_free();
}

/* ================================================================
 * Hardware steps
 * ================================================================ */

typedef struct device_session {
    ft_tt_ctx *ctx;
    ft_tt_grid_info grid;
    ft_tt_allocator allocator;
    ft_tt_noc_path path;
} device_session;

/** Open the card, re-read the grid, and stand up an allocator and a WC path.
 *  Refuses to go further if the grid is not one the coordinate map can express
 *  — every device step below derives its endpoint from it. */
static int session_open(device_session *session)
{
    int rc;

    memset(session, 0, sizeof(*session));

    session->ctx = ft_tt_init_device(0);
    if (!session->ctx) {
        fprintf(stderr, "could not open /dev/tenstorrent/0\n");
        return FT_TT_ENODEV;
    }

    rc = ft_tt_arc_telemetry_read_grid(session->ctx, &session->grid);
    check_rc("ARC telemetry read", rc, FT_TT_OK);
    if (rc != FT_TT_OK)
        return rc;

    rc = ft_tt_coord_check_grid(&session->grid);
    check_rc("the grid is expressible by the coordinate map", rc, FT_TT_OK);
    if (rc != FT_TT_OK)
        return rc;

    rc = ft_tt_allocator_init(&session->allocator, session->ctx, &session->grid, 0u);
    check_rc("allocator initialised from the live grid", rc, FT_TT_OK);
    if (rc != FT_TT_OK)
        return rc;

    rc = ft_tt_noc_path_open(session->ctx, FT_TT_KMD_CACHE_WC, &session->path);
    check_rc("WC data path opened", rc, FT_TT_OK);
    return rc;
}

static void session_close(device_session *session)
{
    if (session->path.window)
        (void)ft_tt_noc_path_close(&session->path);
    if (session->ctx)
        ft_tt_ctx_free(session->ctx);
    memset(session, 0, sizeof(*session));
}

/** L1 round-trip on a Tensix that is NOT index 0. */
static void step_l1(device_session *session)
{
    const size_t bytes = 4u * KIB;
    ft_tt_noc_coord core;
    ft_tt_buffer buffer;
    unsigned char *expected = NULL;
    unsigned char *got = NULL;
    char detail[192];
    uint64_t aims_after_write;
    int rc;

    printf("\n-- L1 round-trip on Tensix %u --\n", TEST_TENSIX_INDEX);

    rc = ft_tt_coord_tensix_by_index(&session->grid, TEST_TENSIX_INDEX, FT_TT_COORD_TRANSLATED,
                                     &core);
    snprintf(detail, sizeof(detail), "translated (%u, %u)", core.x, core.y);
    check("Tensix coordinate derived from telemetry", rc == FT_TT_OK, detail);
    if (rc != FT_TT_OK)
        return;

    rc = ft_tt_buffer_alloc(&session->allocator, FT_TT_MEM_SPACE_L1, 0u, bytes, &buffer);
    check_rc("L1 buffer allocated", rc, FT_TT_OK);
    if (rc != FT_TT_OK)
        return;

    expected = malloc(bytes);
    got = malloc(bytes);
    if (!expected || !got) {
        check("host buffers", false, "out of memory");
        free(expected);
        free(got);
        return;
    }
    fill_pattern(expected, bytes, 0xA5A5A5A5u);
    memset(got, 0, bytes);

    rc = ft_tt_write_l1(&session->path, core, buffer.address, expected, bytes);
    check_rc("ft_tt_write_l1", rc, FT_TT_OK);
    aims_after_write = ft_tt_noc_path_aim_count(&session->path);

    rc = ft_tt_read_l1(&session->path, core, buffer.address, got, bytes);
    check_rc("ft_tt_read_l1", rc, FT_TT_OK);

    check("L1 round-trip is byte-exact", compare_pattern(expected, got, bytes, detail, sizeof(detail)),
          detail);

    snprintf(detail, sizeof(detail), "%llu aims for the write, %llu total",
             (unsigned long long)aims_after_write,
             (unsigned long long)ft_tt_noc_path_aim_count(&session->path));
    check("4 KiB inside one window costs ONE aim, and the read costs none",
          aims_after_write == 1u && ft_tt_noc_path_aim_count(&session->path) == 1u, detail);

    (void)ft_tt_buffer_free(&session->allocator, &buffer);
    free(expected);
    free(got);
}

/** Round-trip every enabled DRAM bank, then prove they are not aliases. */
static void step_dram(device_session *session)
{
    const size_t bytes = 64u * KIB;
    unsigned char *expected = malloc(bytes);
    unsigned char *got = malloc(bytes);
    ft_tt_buffer buffers[FT_TT_NUM_DRAM_BANKS];
    ft_tt_noc_coord coords[FT_TT_NUM_DRAM_BANKS];
    uint32_t banks = session->grid.dram_bank_count;
    uint32_t bank;
    char detail[192];

    memset(buffers, 0, sizeof(buffers));
    memset(coords, 0, sizeof(coords));

    printf("\n-- DRAM round-trip, all %u banks --\n", banks);
    if (!expected || !got) {
        check("host buffers", false, "out of memory");
        free(expected);
        free(got);
        return;
    }

    for (bank = 0; bank < banks; bank++) {
        char label[96];
        int rc = ft_tt_coord_dram_bank(&session->grid, bank, FT_TT_COORD_TRANSLATED, &coords[bank]);

        snprintf(label, sizeof(label), "bank %u coordinate derived", bank);
        snprintf(detail, sizeof(detail), "translated (%u, %u)", coords[bank].x, coords[bank].y);
        check(label, rc == FT_TT_OK, detail);
        if (rc != FT_TT_OK) {
            free(expected);
            free(got);
            return;
        }

        rc = ft_tt_buffer_alloc(&session->allocator, FT_TT_MEM_SPACE_DRAM, bank, bytes,
                                &buffers[bank]);
        if (rc != FT_TT_OK) {
            snprintf(label, sizeof(label), "bank %u buffer allocated", bank);
            check_rc(label, rc, FT_TT_OK);
            free(expected);
            free(got);
            return;
        }

        /* Seed by bank so bank n's data can never be mistaken for bank m's. */
        fill_pattern(expected, bytes, 0x5A5A0000u + bank);
        memset(got, 0, bytes);

        rc = ft_tt_write_dram(&session->path, coords[bank], buffers[bank].address, expected, bytes);
        snprintf(label, sizeof(label), "bank %u ft_tt_write_dram", bank);
        check_rc(label, rc, FT_TT_OK);

        rc = ft_tt_read_dram(&session->path, coords[bank], buffers[bank].address, got, bytes);
        snprintf(label, sizeof(label), "bank %u ft_tt_read_dram", bank);
        check_rc(label, rc, FT_TT_OK);

        snprintf(label, sizeof(label), "bank %u round-trip is byte-exact", bank);
        check(label, compare_pattern(expected, got, bytes, detail, sizeof(detail)), detail);
    }

    /* Every bank was written a different pattern and every bank was read back
     * correctly above; re-reading bank 0 now proves the later writes did not
     * land on it, i.e. these are eight endpoints and not one aliased eight
     * times. */
    if (banks > 1u) {
        int rc;

        fill_pattern(expected, bytes, 0x5A5A0000u);
        memset(got, 0, bytes);
        rc = ft_tt_read_dram(&session->path, coords[0], buffers[0].address, got, bytes);
        check_rc("re-read bank 0 after writing all the others", rc, FT_TT_OK);
        check("bank 0 still holds ITS pattern — the banks are distinct endpoints",
              compare_pattern(expected, got, bytes, detail, sizeof(detail)), detail);
    }

    for (bank = banks; bank > 0u; bank--)
        (void)ft_tt_buffer_free(&session->allocator, &buffers[bank - 1u]);
    free(expected);
    free(got);
}

/**
 * A transfer that starts 64 KiB below a 2 MiB boundary and runs for 5 MiB, so
 * it covers four windows. This is the re-windowing path; nothing else in the
 * tree exercises it.
 */
static void step_cross(device_session *session)
{
    const size_t bytes = 5u * MIB;
    const uint64_t offset = (uint64_t)2u * MIB - 64u * KIB;
    unsigned char *expected = malloc(bytes);
    unsigned char *got = malloc(bytes);
    ft_tt_buffer region;
    ft_tt_noc_coord bank_port;
    uint64_t start;
    char detail[192];
    uint64_t aims_before;
    uint64_t aims_after_write;
    int rc;

    printf("\n-- window-crossing transfer (%u MiB starting 64 KiB below a 2 MiB boundary) --\n",
           (unsigned)(bytes / MIB));
    if (!expected || !got) {
        check("host buffers", false, "out of memory");
        free(expected);
        free(got);
        return;
    }

    rc = ft_tt_coord_dram_bank(&session->grid, TEST_DRAM_BANK, FT_TT_COORD_TRANSLATED,
                               &bank_port);
    check_rc("test DRAM bank coordinate derived", rc, FT_TT_OK);
    if (rc != FT_TT_OK) {
        free(expected);
        free(got);
        return;
    }

    /* Reserve the whole span through the allocator, so the two Phase 1 pieces
     * are exercised together rather than each against a hand-picked address. */
    rc = ft_tt_buffer_alloc(&session->allocator, FT_TT_MEM_SPACE_DRAM, TEST_DRAM_BANK,
                            (size_t)offset + bytes,
                            &region);
    check_rc("DRAM region allocated for the crossing", rc, FT_TT_OK);
    if (rc != FT_TT_OK) {
        free(expected);
        free(got);
        return;
    }
    start = region.address + offset;

    fill_pattern(expected, bytes, 0xC0FFEE00u);
    memset(got, 0, bytes);

    /* Park the window on the LAST span this transfer will touch, so the walk
     * cannot start out already in position and quietly need one aim fewer. */
    (void)ft_tt_write_dram(&session->path, bank_port, start + 4u * MIB, expected, 64u);
    aims_before = ft_tt_noc_path_aim_count(&session->path);

    rc = ft_tt_write_dram(&session->path, bank_port, start, expected, bytes);
    check_rc("ft_tt_write_dram across four windows", rc, FT_TT_OK);
    aims_after_write = ft_tt_noc_path_aim_count(&session->path);

    snprintf(detail, sizeof(detail), "%llu window moves",
             (unsigned long long)(aims_after_write - aims_before));
    check("the write really did move the window four times",
          aims_after_write - aims_before == 4u, detail);

    rc = ft_tt_read_dram(&session->path, bank_port, start, got, bytes);
    check_rc("ft_tt_read_dram across four windows", rc, FT_TT_OK);
    snprintf(detail, sizeof(detail), "%llu window moves",
             (unsigned long long)(ft_tt_noc_path_aim_count(&session->path) - aims_after_write));
    check("and so did the read", ft_tt_noc_path_aim_count(&session->path) - aims_after_write == 4u,
          detail);

    check("the whole span survived re-windowing",
          compare_pattern(expected, got, bytes, detail, sizeof(detail)), detail);

    (void)ft_tt_buffer_free(&session->allocator, &region);
    free(expected);
    free(got);
}

/** Alignment refusals against a live path: nothing may reach the device. */
static void step_align(device_session *session)
{
    unsigned char scratch[128];
    ft_tt_noc_coord bank_port;
    ft_tt_noc_coord core;
    ft_tt_noc_coord bad_coord;
    uint64_t aims_before;
    char detail[128];
    int rc;

    printf("\n-- alignment refusals on a live path --\n");
    memset(scratch, 0x11, sizeof(scratch));

    rc = ft_tt_coord_dram_bank(&session->grid, TEST_DRAM_BANK, FT_TT_COORD_TRANSLATED,
                               &bank_port);
    check_rc("test DRAM bank coordinate derived", rc, FT_TT_OK);
    if (rc != FT_TT_OK)
        return;
    rc = ft_tt_coord_tensix_by_index(&session->grid, TEST_TENSIX_INDEX, FT_TT_COORD_TRANSLATED,
                                     &core);
    check_rc("Tensix coordinate derived", rc, FT_TT_OK);
    if (rc != FT_TT_OK)
        return;

    aims_before = ft_tt_noc_path_aim_count(&session->path);

    check_rc("a misaligned DRAM write is refused",
             ft_tt_write_dram(&session->path, bank_port, 8u, scratch, 64u), FT_TT_EINVAL);
    check_rc("a 16 B-aligned DRAM read is refused (64 B rule)",
             ft_tt_read_dram(&session->path, bank_port, 16u, scratch, 64u), FT_TT_EINVAL);
    check_rc("an L1 write running past the top of L1 is refused",
             ft_tt_write_l1(&session->path, core, FT_TT_MEM_L1_SIZE - 16u, scratch, 32u),
             FT_TT_EINVAL);
    check_rc("a NULL host buffer is refused",
             ft_tt_write_dram(&session->path, bank_port, 0u, NULL, 64u), FT_TT_EINVAL);

    /* A coordinate too large for the 6-bit x_end/y_end fields would be
     * truncated into a DIFFERENT, live endpoint. That is the failure mode that
     * costs a power cycle, so it must be refused rather than masked. */
    bad_coord.x = 64u;
    bad_coord.y = 0u;
    check_rc("a coordinate that does not fit the 6-bit TLB fields is refused",
             ft_tt_write_dram(&session->path, bad_coord, 0u, scratch, 64u), FT_TT_EINVAL);

    snprintf(detail, sizeof(detail), "%llu aims before, %llu after",
             (unsigned long long)aims_before,
             (unsigned long long)ft_tt_noc_path_aim_count(&session->path));
    check("not one of those refusals aimed the window",
          ft_tt_noc_path_aim_count(&session->path) == aims_before, detail);
}

/** MB/s through the helpers, against the raw-window 287 MB/s WC baseline. */
static void step_bandwidth(device_session *session)
{
    const size_t bytes = 8u * MIB;
    unsigned char *host = malloc(bytes);
    ft_tt_buffer region;
    ft_tt_noc_coord bank_port;
    double seconds;
    double megabytes = (double)bytes / (1024.0 * 1024.0);
    double t0;
    int rc;

    printf("\n-- bandwidth through the helpers (WC, %u MiB, four windows) --\n",
           (unsigned)(bytes / MIB));
    if (!host) {
        check("host buffer", false, "out of memory");
        return;
    }
    memset(host, 0x5A, bytes);

    rc = ft_tt_coord_dram_bank(&session->grid, TEST_DRAM_BANK, FT_TT_COORD_TRANSLATED,
                               &bank_port);
    check_rc("test DRAM bank coordinate derived", rc, FT_TT_OK);
    if (rc == FT_TT_OK)
        rc = ft_tt_buffer_alloc(&session->allocator, FT_TT_MEM_SPACE_DRAM, TEST_DRAM_BANK, bytes,
                                   &region);
    check_rc("DRAM region allocated", rc, FT_TT_OK);
    if (rc != FT_TT_OK) {
        free(host);
        return;
    }

    t0 = now_seconds();
    rc = ft_tt_write_dram(&session->path, bank_port, region.address, host, bytes);
    seconds = now_seconds() - t0;
    check_rc("bulk write", rc, FT_TT_OK);
    printf("       host -> card  %8.1f MB/s   (raw window layer measured 287 MB/s WC)\n",
           megabytes / seconds);

    t0 = now_seconds();
    rc = ft_tt_read_dram(&session->path, bank_port, region.address, host, bytes);
    seconds = now_seconds() - t0;
    check_rc("bulk read", rc, FT_TT_OK);
    printf("       card -> host  %8.1f MB/s   (raw window layer measured 2.3 MB/s)\n",
           megabytes / seconds);

    (void)ft_tt_buffer_free(&session->allocator, &region);
    free(host);
}

/* ================================================================
 * main
 * ================================================================ */

static void usage(void)
{
    fprintf(stderr, "usage: test_noc [unit|l1|dram|cross|align|bw|hw]\n"
                    "  unit  no device is opened (default)\n"
                    "  hw    l1 + dram + cross + align, in that order\n");
}

int main(int argc, char **argv)
{
    const char *step = (argc > 1) ? argv[1] : "unit";
    device_session session;
    bool want_l1;
    bool want_dram;
    bool want_cross;
    bool want_align;
    bool want_bw;

    printf("=== test_noc: %s ===\n", step);

    if (strcmp(step, "unit") == 0) {
        run_unit();
        printf("\n%d checks, %d failed\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }

    want_l1 = strcmp(step, "l1") == 0 || strcmp(step, "hw") == 0;
    want_dram = strcmp(step, "dram") == 0 || strcmp(step, "hw") == 0;
    want_cross = strcmp(step, "cross") == 0 || strcmp(step, "hw") == 0;
    want_align = strcmp(step, "align") == 0 || strcmp(step, "hw") == 0;
    want_bw = strcmp(step, "bw") == 0;

    if (!want_l1 && !want_dram && !want_cross && !want_align && !want_bw) {
        usage();
        return 2;
    }

    if (session_open(&session) != FT_TT_OK) {
        session_close(&session);
        printf("\n%d checks, %d failed\n", checks, failures);
        return 1;
    }

    if (failures == 0 && want_l1)
        step_l1(&session);
    if (failures == 0 && want_dram)
        step_dram(&session);
    if (failures == 0 && want_cross)
        step_cross(&session);
    if (failures == 0 && want_align)
        step_align(&session);
    if (failures == 0 && want_bw)
        step_bandwidth(&session);

    session_close(&session);
    printf("\n%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
