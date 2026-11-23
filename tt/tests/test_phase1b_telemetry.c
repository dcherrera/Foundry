/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase1b_telemetry.c — the first NoC transaction this backend ever issues.
 *
 * Phase 1b's opening move is deliberately the safest one available: aim a TLB
 * window at the endpoint tt-kmd itself is already aimed at, and only read.
 *
 *   - The endpoint is ARC at NoC0 (8, 0), CSM base 0x10000000. That is not a
 *     guess and not a probe: it was decoded from the driver's own live window
 *     configuration through a read-only mapping of the TLB register file
 *     (docs/tt_bringup_log.md finding 3), and the driver's reads through it
 *     demonstrably work — sysfs is populated and tt_heartbeat ticks.
 *   - Nothing is written to the device. The sequence claims a window, aims it,
 *     reads, and releases it.
 *
 * This matters because a wrong endpoint does not fail a read, it hangs the NoC,
 * and recovery is a physical power cycle (spec §7.3). One has already been spent.
 *
 * The result is checked against an independent oracle rather than merely
 * printed: pyluwen read the same five values off this card before any of this C
 * existed, so the expected numbers below are ground truth obtained by a
 * completely separate path. Agreement is evidence; a print is not.
 *
 * Build (the fence must be opened explicitly):
 *   cc -std=c99 -Wall -Wextra -Werror -DFT_TT_PHASE1B_NOC_ENABLED -I.. \
 *      ../ft_tt_kmd.c ../ft_tt_device.c ../ft_tt_tlb.c \
 *      ../ft_tt_arc_telemetry.c test_phase1b_telemetry.c -o build/test_phase1b
 */

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_device.h"

#include <stdio.h>

/* Ground truth read from this card via pyluwen, independently of this code. */
#define ORACLE_TENSIX_ENABLED_COL 16383u /* 0b11111111111111 — 14 columns */
#define ORACLE_ENABLED_ETH        16095u
#define ORACLE_ENABLED_GDDR       255u   /* 8 banks */
#define ORACLE_ENABLED_L2CPU      15u    /* 4 clusters */
#define ORACLE_TENSIX_COLUMNS     14u
#define ORACLE_DRAM_BANKS         8u

static int failures;

static void check_u32(const char *what, uint32_t got, uint32_t expected)
{
    if (got == expected) {
        printf("  ok    %-34s = %u\n", what, got);
    } else {
        printf("  FAIL  %-34s = %u (expected %u)\n", what, got, expected);
        failures++;
    }
}

int main(void)
{
    ft_tt_ctx *ctx = ft_tt_init_device(0);
    if (ctx == NULL) {
        fprintf(stderr, "could not open device 0\n");
        return 1;
    }

    printf("Phase 1b — first NoC read (ARC telemetry via NoC0 (8,0) CSM 0x10000000)\n");

    ft_tt_grid_info grid;
    int rc = ft_tt_arc_telemetry_read_grid(ctx, &grid);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "  ft_tt_arc_telemetry_read_grid failed: %d\n", rc);
        fprintf(stderr, "  (FT_TT_EPERM here means the Phase 1a fence is still closed —\n"
                        "   rebuild with -DFT_TT_PHASE1B_NOC_ENABLED)\n");
        ft_tt_ctx_free(ctx);
        return 1;
    }

    check_u32("grid is valid", grid.valid ? 1u : 0u, 1u);
    check_u32("tensix_enabled_col", grid.tensix_enabled_col, ORACLE_TENSIX_ENABLED_COL);
    check_u32("enabled_eth", grid.enabled_eth, ORACLE_ENABLED_ETH);
    check_u32("enabled_gddr", grid.enabled_gddr, ORACLE_ENABLED_GDDR);
    check_u32("enabled_l2cpu", grid.enabled_l2cpu, ORACLE_ENABLED_L2CPU);
    check_u32("noc_translation_enabled", grid.noc_translation_enabled, 1u);
    check_u32("tensix_column_count", grid.tensix_column_count, ORACLE_TENSIX_COLUMNS);
    check_u32("dram_bank_count", grid.dram_bank_count, ORACLE_DRAM_BANKS);

    /* mem_info refused to guess in Phase 1a; with a grid it must now answer. */
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    rc = ft_tt_mem_info(ctx, &free_bytes, &total_bytes);
    if (rc == FT_TT_OK) {
        printf("  ok    ft_tt_mem_info now answers        = %.2f GiB total\n",
               (double)total_bytes / (1024.0 * 1024.0 * 1024.0));
    } else {
        printf("  FAIL  ft_tt_mem_info still refuses (rc=%d)\n", rc);
        failures++;
    }

    printf("\n%d checks failed\n", failures);
    ft_tt_ctx_free(ctx);
    return failures == 0 ? 0 : 1;
}
