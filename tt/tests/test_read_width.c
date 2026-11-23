/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_read_width.c — is the card→host read rate a latency ceiling or a
 * transaction-count effect?
 *
 * spec §9.8 currently says 2.3 MB/s is "a ceiling, not a tuning target", on the
 * reasoning that every uncached read is a serialised non-posted PCIe round-trip.
 * If that is the whole story, widening the load should change nothing. If the
 * rate instead tracks the number of load instructions, it should scale with load
 * width.
 *
 * Same buffer, same window, same aim, back to back — only the load width differs.
 */

#define _POSIX_C_SOURCE 199309L

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_tlb.h"

#include <emmintrin.h> /* SSE2 only — this host has no AVX */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WINDOW_BYTES   (2u * 1024u * 1024u)
#define DRAM_AIM_BASE  0x200000u
#define TRANSFER_BYTES WINDOW_BYTES
#define REPEATS        4u

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void read_8b(unsigned char *dst, volatile const unsigned char *src, size_t bytes)
{
    uint64_t *d = (uint64_t *)(void *)dst;
    volatile const uint64_t *s = (volatile const uint64_t *)(const void *)src;
    for (size_t i = 0; i < bytes / 8u; i++) {
        d[i] = s[i];
    }
}

static void read_16b(unsigned char *dst, volatile const unsigned char *src, size_t bytes)
{
    for (size_t off = 0; off < bytes; off += 16u) {
        __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(src + off));
        _mm_storeu_si128((__m128i *)(void *)(dst + off), v);
    }
}

int main(void)
{
    ft_tt_ctx *ctx = ft_tt_init_device(0);
    if (ctx == NULL) {
        return 1;
    }

    ft_tt_grid_info grid;
    ft_tt_noc_coord dram;
    if (ft_tt_arc_telemetry_read_grid(ctx, &grid) != FT_TT_OK ||
        ft_tt_coord_check_grid(&grid) != FT_TT_OK ||
        ft_tt_coord_dram_bank(&grid, 0, FT_TT_COORD_TRANSLATED, &dram) != FT_TT_OK) {
        ft_tt_ctx_free(ctx);
        return 1;
    }

    ft_tt_tlb_window *w = NULL;
    if (ft_tt_tlb_window_alloc(ctx, WINDOW_BYTES, FT_TT_KMD_CACHE_WC, &w) != FT_TT_OK) {
        ft_tt_ctx_free(ctx);
        return 1;
    }
    if (ft_tt_tlb_window_aim(ctx, w, (uint16_t)dram.x, (uint16_t)dram.y, DRAM_AIM_BASE) !=
        FT_TT_OK) {
        ft_tt_ctx_free(ctx);
        return 1;
    }

    volatile unsigned char *ap = ft_tt_tlb_window_aperture(w);
    unsigned char *buf = malloc(TRANSFER_BYTES);
    double mb = (double)TRANSFER_BYTES * REPEATS / (1024.0 * 1024.0);

    printf("DRAM bank 0 translated (%u, %u), WC, %u MiB x %u, reads only\n\n", dram.x, dram.y,
           TRANSFER_BYTES / (1024u * 1024u), REPEATS);

    double t0 = now_seconds();
    for (uint32_t r = 0; r < REPEATS; r++) {
        read_8b(buf, ap, TRANSFER_BYTES);
    }
    double s8 = now_seconds() - t0;

    t0 = now_seconds();
    for (uint32_t r = 0; r < REPEATS; r++) {
        read_16b(buf, ap, TRANSFER_BYTES);
    }
    double s16 = now_seconds() - t0;

    printf("   8 B loads   %6.2f MB/s\n", mb / s8);
    printf("  16 B loads   %6.2f MB/s\n", mb / s16);
    printf("\n  ratio %.2fx — %s\n", s8 / s16,
           (s8 / s16) > 1.5 ? "scales with load width: transaction-bound, NOT a latency ceiling"
                            : "flat: consistent with a latency ceiling");

    free(buf);
    (void)ft_tt_tlb_window_free(ctx, w);
    ft_tt_ctx_free(ctx);
    return 0;
}
