/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_first_write.c — the first bytes this backend ever writes to the card.
 *
 * Run one step at a time and inspect between them. The steps are ordered by how
 * much they can teach and how much they can cost:
 *
 *   arc     re-read ARC telemetry. Already proven, so this is a free check that
 *           nothing drifted since the last run. Reads only.
 *   tensix  write a pattern into one derived Tensix's L1 and read it back. This
 *           is the first write. On THIS card all 14 columns are enabled, so
 *           TRANSLATED == NOC0 for every Tensix — which is exactly why this step
 *           cannot prove the translation layer. It proves the write path.
 *   dram    write a pattern into DRAM bank 0 and read it back. This IS the
 *           translation proof: translated DRAM lives at (17-18, 12-23), which
 *           exists nowhere in NoC0 space, so if
 *           "TLBs are aimed with TRANSLATED coordinates"
 *           is wrong,
 *           this is where it shows up.
 *
 * Every coordinate is DERIVED from ft_tt_coord against the live grid. Nothing
 * here is hardcoded or probed — a wrong endpoint hangs the chip and costs a
 * physical power cycle (spec §7.3), and that has already happened once.
 *
 * Build (the fence must be opened explicitly):
 *   cc -std=c99 -Wall -Wextra -Werror -DFT_TT_PHASE1B_NOC_ENABLED -I.. \
 *      ../ft_tt_kmd.c ../ft_tt_device.c ../ft_tt_tlb.c \
 *      ../ft_tt_arc_telemetry.c ../ft_tt_coord.c test_first_write.c \
 *      -o build/test_first_write
 */

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_tlb.h"

#include <stdio.h>
#include <string.h>

/* A 2 MiB window must be aimed at a 2 MiB-aligned address.
 * L1 is 1536 KiB, so aiming at 0 covers all of it and we index inside. */
#define WINDOW_BYTES   (2u * 1024u * 1024u)
#define L1_AIM_BASE    0u
#define L1_SCRATCH_OFF 0x100000u /* 1 MiB into L1: clear of mailbox and firmware */
#define DRAM_AIM_BASE  0x200000u /* 2 MiB into the bank, 2 MiB aligned */

#define PATTERN_WORDS 64u /* 256 bytes — a multiple of the 16 B access rule */

static int failures;

static void check(const char *what, bool ok, const char *detail)
{
    printf("  %-4s  %-46s %s\n", ok ? "ok" : "FAIL", what, detail ? detail : "");
    if (!ok) {
        failures++;
    }
}

/** Fill a distinctive, position-dependent pattern so a partial or shifted
 *  write is visible rather than looking like a clean success. */
static void fill_pattern(uint32_t *buf, uint32_t seed)
{
    for (uint32_t i = 0; i < PATTERN_WORDS; i++) {
        buf[i] = seed ^ (i * 0x9E3779B9u);
    }
}

/**
 * Aim `window` at (coord, aim_base), write `PATTERN_WORDS` words at
 * `offset` inside the aperture, read them back and compare.
 */
static int write_read_back(ft_tt_ctx *ctx, ft_tt_tlb_window *window,
                           ft_tt_noc_coord coord, uint64_t aim_base, uint32_t offset,
                           uint32_t seed, const char *label)
{
    char detail[160];

    int rc = ft_tt_tlb_window_aim(ctx, window, (uint16_t)coord.x, (uint16_t)coord.y, aim_base);
    snprintf(detail, sizeof(detail), "(%u, %u) base 0x%llx rc=%d", coord.x, coord.y,
             (unsigned long long)aim_base, rc);
    check("window aimed", rc == FT_TT_OK, detail);
    if (rc != FT_TT_OK) {
        return rc;
    }

    volatile unsigned char *aperture = ft_tt_tlb_window_aperture(window);
    if (aperture == NULL) {
        check("aperture is mapped", false, "NULL");
        return FT_TT_EINVAL;
    }

    uint32_t expected[PATTERN_WORDS];
    uint32_t got[PATTERN_WORDS];
    fill_pattern(expected, seed);
    memset(got, 0, sizeof(got));

    volatile uint32_t *target = (volatile uint32_t *)(void *)(aperture + offset);

    /* Read the region first, purely for the log: if it already contains our
     * pattern from a previous run, a no-op write would look like success. */
    uint32_t before = target[0];

    for (uint32_t i = 0; i < PATTERN_WORDS; i++) {
        target[i] = expected[i];
    }
    for (uint32_t i = 0; i < PATTERN_WORDS; i++) {
        got[i] = target[i];
    }

    bool match = memcmp(expected, got, sizeof(expected)) == 0;
    if (match) {
        snprintf(detail, sizeof(detail), "%u words at +0x%x (was 0x%08x, now 0x%08x)",
                 PATTERN_WORDS, offset, before, got[0]);
    } else {
        uint32_t first = 0;
        while (first < PATTERN_WORDS && expected[first] == got[first]) {
            first++;
        }
        snprintf(detail, sizeof(detail), "mismatch at word %u: wrote 0x%08x read 0x%08x", first,
                 expected[first], got[first]);
    }
    check(label, match, detail);

    return match ? FT_TT_OK : FT_TT_EIO;
}

static int step_arc(ft_tt_ctx *ctx, ft_tt_grid_info *grid)
{
    int rc = ft_tt_arc_telemetry_read_grid(ctx, grid);
    char detail[128];
    snprintf(detail, sizeof(detail), "rc=%d", rc);
    check("ARC telemetry re-read", rc == FT_TT_OK, detail);
    if (rc != FT_TT_OK) {
        return rc;
    }

    snprintf(detail, sizeof(detail), "cols=0x%x (%u), gddr=%u banks", grid->tensix_enabled_col,
             grid->tensix_column_count, grid->dram_bank_count);
    check("topology unchanged", grid->tensix_enabled_col == 16383u && grid->dram_bank_count == 8u,
          detail);

    rc = ft_tt_coord_check_grid(grid);
    snprintf(detail, sizeof(detail), "rc=%d", rc);
    check("grid is expressible by the coordinate map", rc == FT_TT_OK, detail);
    return rc;
}

int main(int argc, char **argv)
{
    const char *step = (argc > 1) ? argv[1] : "arc";

    ft_tt_ctx *ctx = ft_tt_init_device(0);
    if (ctx == NULL) {
        fprintf(stderr, "could not open device 0\n");
        return 1;
    }

    printf("=== step: %s ===\n", step);

    ft_tt_grid_info grid;
    int rc = step_arc(ctx, &grid);
    if (rc != FT_TT_OK || failures != 0) {
        printf("\nstopping: the grid must be sound before anything is written\n");
        ft_tt_ctx_free(ctx);
        return 1;
    }

    if (strcmp(step, "arc") == 0) {
        printf("\n%d checks failed\n", failures);
        ft_tt_ctx_free(ctx);
        return failures == 0 ? 0 : 1;
    }

    ft_tt_tlb_window *window = NULL;
    rc = ft_tt_tlb_window_alloc(ctx, WINDOW_BYTES, FT_TT_KMD_CACHE_UC, &window);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "window alloc failed: %d\n", rc);
        ft_tt_ctx_free(ctx);
        return 1;
    }

    if (strcmp(step, "tensix") == 0) {
        ft_tt_noc_coord noc0;
        ft_tt_noc_coord translated;
        char detail[160];

        rc = ft_tt_coord_tensix_by_index(&grid, 0, FT_TT_COORD_NOC0, &noc0);
        int rc2 = ft_tt_coord_tensix_by_index(&grid, 0, FT_TT_COORD_TRANSLATED, &translated);
        snprintf(detail, sizeof(detail), "noc0 (%u, %u), translated (%u, %u)", noc0.x, noc0.y,
                 translated.x, translated.y);
        check("Tensix 0 derived in both systems", rc == FT_TT_OK && rc2 == FT_TT_OK, detail);

        /* On a 14/14 card these must agree. If they do not, the mapping is
         * wrong and nothing should be written. */
        check("translated == noc0 on this un-harvested card",
              noc0.x == translated.x && noc0.y == translated.y, detail);

        if (failures == 0) {
            (void)write_read_back(ctx, window, translated, L1_AIM_BASE, L1_SCRATCH_OFF,
                                  0xA5A5A5A5u, "L1 pattern survives write -> read");
        }
    } else if (strcmp(step, "dram") == 0) {
        ft_tt_noc_coord noc0;
        ft_tt_noc_coord translated;
        char detail[160];

        rc = ft_tt_coord_dram_bank(&grid, 0, FT_TT_COORD_NOC0, &noc0);
        int rc2 = ft_tt_coord_dram_bank(&grid, 0, FT_TT_COORD_TRANSLATED, &translated);
        snprintf(detail, sizeof(detail), "noc0 (%u, %u), translated (%u, %u)", noc0.x, noc0.y,
                 translated.x, translated.y);
        check("DRAM bank 0 derived in both systems", rc == FT_TT_OK && rc2 == FT_TT_OK, detail);

        /* The translation proof: these MUST differ, and translated must be in
         * the alias grid (x 17-18), which is not a NoC0 address at all. */
        check("translated differs from noc0 — this is the alias grid",
              !(noc0.x == translated.x && noc0.y == translated.y), detail);
        check("translated DRAM x is in the alias range 17-18",
              translated.x == 17u || translated.x == 18u, detail);

        if (failures == 0) {
            (void)write_read_back(ctx, window, translated, DRAM_AIM_BASE, 0u, 0x5A5A5A5Au,
                                  "DRAM pattern survives write -> read");
        }
    } else {
        fprintf(stderr, "unknown step '%s' (use: arc | tensix | dram)\n", step);
        failures++;
    }

    (void)ft_tt_tlb_window_free(ctx, window);
    printf("\n%d checks failed\n", failures);
    ft_tt_ctx_free(ctx);
    return failures == 0 ? 0 : 1;
}
