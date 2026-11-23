/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_bandwidth.c — the two numbers Phase 1 still owes.
 *
 *   1. What a TLB window reprogram costs. With no BAR4 there are no 4 GiB
 *      windows, so every DRAM touch beyond 2 MiB moves a window — ~16,000 of
 *      them to cross 32 GB. Phase 0 established this is three posted MMIO
 *      stores rather than an ioctl; this measures it.
 *
 *   2. Achieved host<->card MB/s, against the ~500 MB/s an x1 Gen2 link allows.
 *      The gap between the two is the window and access overhead.
 *
 * Both cache modes are measured because the choice is not obvious: UC is
 * correct for registers, WC combines writes into larger PCIe bursts but makes
 * reads slow. Which to use for bulk weight loading is a real decision and it
 * should rest on a number, not a preference.
 *
 * Reads and writes go to DRAM bank 0 at a 2 MiB-aligned offset. Nothing is
 * loaded on the card, so this region is free. All coordinates are derived from
 * ft_tt_coord against live telemetry — never hardcoded, never probed.
 *
 * Build:
 *   cc -std=c99 -O2 -Wall -Wextra -Werror -DFT_TT_PHASE1B_NOC_ENABLED -I.. \
 *      ../ft_tt_kmd.c ../ft_tt_device.c ../ft_tt_tlb.c \
 *      ../ft_tt_arc_telemetry.c ../ft_tt_coord.c test_bandwidth.c \
 *      -o build/test_bandwidth
 */

/* clock_gettime/CLOCK_MONOTONIC are POSIX, and -std=c99 -pedantic hides them. */
#define _POSIX_C_SOURCE 199309L

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_tlb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>

/** Longest "/sys/bus/pci/devices/0000:03:00.0"-shaped path, with room to spare. */
#define BLACKHOLE_SYSFS_PATH_MAX 96

/**
 * Find this host's Blackhole in sysfs and write its device directory to `out`.
 *
 * Scanning rather than hardcoding because the BDF is a property of the
 * MOTHERBOARD, not of the card: 0000:12:00.0 on one p150a host, 0000:03:00.0 on
 * another. A literal here does not fail loudly — it reports "unknown" forever.
 *
 * @return 0 when a card was found, -1 otherwise
 */
static int blackhole_sysfs_directory(char *out, size_t out_size)
{
    const char *devices_root = "/sys/bus/pci/devices";
    struct dirent *entry;
    DIR *dir = opendir(devices_root);
    int found = -1;

    if (dir == NULL)
        return -1;
    while (found != 0 && (entry = readdir(dir)) != NULL) {
        char identity_path[BLACKHOLE_SYSFS_PATH_MAX + 16];
        char identity[16];
        FILE *f;

        if (entry->d_name[0] == '.')
            continue;
        (void)snprintf(identity_path, sizeof(identity_path), "%s/%s/device", devices_root,
                       entry->d_name);
        f = fopen(identity_path, "r");
        if (f == NULL)
            continue;
        if (fgets(identity, sizeof(identity), f) != NULL && strstr(identity, "0xb140") != NULL) {
            (void)snprintf(out, out_size, "%s/%s", devices_root, entry->d_name);
            found = 0;
        }
        fclose(f);
    }
    closedir(dir);
    return found;
}

#define WINDOW_BYTES  (2u * 1024u * 1024u)
#define DRAM_AIM_BASE 0x200000u

#define AIM_ITERATIONS   2000u
#define TRANSFER_BYTES   WINDOW_BYTES
#define TRANSFER_REPEATS 4u

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/** Bulk copy shaped the way tt-umd shapes it: 16 x 16 B per 256 B block, which
 *  is what keeps the same number of PCIe transactions in flight. Written with
 *  plain 64-bit stores because this host has no AVX (spec §8.4) — tt-umd's own
 *  device_memcpy.cpp is AVX2 and will not run here. */
static void block_write(volatile unsigned char *dst, const unsigned char *src, size_t bytes)
{
    volatile uint64_t *d = (volatile uint64_t *)(void *)dst;
    const uint64_t *s = (const uint64_t *)(const void *)src;
    size_t words = bytes / 8u;

    for (size_t i = 0; i < words; i++) {
        d[i] = s[i];
    }
}

static void block_read(unsigned char *dst, volatile const unsigned char *src, size_t bytes)
{
    uint64_t *d = (uint64_t *)(void *)dst;
    volatile const uint64_t *s = (volatile const uint64_t *)(const void *)src;
    size_t words = bytes / 8u;

    for (size_t i = 0; i < words; i++) {
        d[i] = s[i];
    }
}

static void measure_mode(ft_tt_ctx *ctx, ft_tt_noc_coord dram,
                         enum ft_tt_kmd_cache_mode mode, const char *mode_name,
                         unsigned char *host_buf)
{
    ft_tt_tlb_window *window = NULL;
    int rc = ft_tt_tlb_window_alloc(ctx, WINDOW_BYTES, mode, &window);
    if (rc != FT_TT_OK) {
        printf("  %s: window alloc failed (%d)\n", mode_name, rc);
        return;
    }

    rc = ft_tt_tlb_window_aim(ctx, window, (uint16_t)dram.x, (uint16_t)dram.y, DRAM_AIM_BASE);
    if (rc != FT_TT_OK) {
        printf("  %s: aim failed (%d)\n", mode_name, rc);
        (void)ft_tt_tlb_window_free(ctx, window);
        return;
    }

    volatile unsigned char *aperture = ft_tt_tlb_window_aperture(window);

    /* --- window reprogram cost ------------------------------------------- */
    double t0 = now_seconds();
    for (uint32_t i = 0; i < AIM_ITERATIONS; i++) {
        /* alternate the target so nothing can be elided as a no-op */
        uint64_t base = DRAM_AIM_BASE + ((i & 1u) ? WINDOW_BYTES : 0u);
        (void)ft_tt_tlb_window_aim(ctx, window, (uint16_t)dram.x, (uint16_t)dram.y, base);
    }
    double aim_seconds = now_seconds() - t0;
    printf("  %-2s  reprogram      %8.3f us each  (%u iterations)\n", mode_name,
           aim_seconds * 1e6 / (double)AIM_ITERATIONS, AIM_ITERATIONS);

    (void)ft_tt_tlb_window_aim(ctx, window, (uint16_t)dram.x, (uint16_t)dram.y, DRAM_AIM_BASE);

    /* --- write --- */
    t0 = now_seconds();
    for (uint32_t r = 0; r < TRANSFER_REPEATS; r++) {
        block_write(aperture, host_buf, TRANSFER_BYTES);
    }
    double write_seconds = now_seconds() - t0;
    double write_mb = (double)TRANSFER_BYTES * TRANSFER_REPEATS / (1024.0 * 1024.0);
    printf("  %-2s  host -> card   %8.1f MB/s\n", mode_name, write_mb / write_seconds);

    /* --- read --- */
    t0 = now_seconds();
    for (uint32_t r = 0; r < TRANSFER_REPEATS; r++) {
        block_read(host_buf, aperture, TRANSFER_BYTES);
    }
    double read_seconds = now_seconds() - t0;
    printf("  %-2s  card -> host   %8.1f MB/s\n", mode_name, write_mb / read_seconds);

    (void)ft_tt_tlb_window_free(ctx, window);
}

int main(void)
{
    ft_tt_ctx *ctx = ft_tt_init_device(0);
    if (ctx == NULL) {
        fprintf(stderr, "could not open device 0\n");
        return 1;
    }

    ft_tt_grid_info grid;
    if (ft_tt_arc_telemetry_read_grid(ctx, &grid) != FT_TT_OK ||
        ft_tt_coord_check_grid(&grid) != FT_TT_OK) {
        fprintf(stderr, "grid not sound; refusing to measure\n");
        ft_tt_ctx_free(ctx);
        return 1;
    }

    ft_tt_noc_coord dram;
    if (ft_tt_coord_dram_bank(&grid, 0, FT_TT_COORD_TRANSLATED, &dram) != FT_TT_OK) {
        fprintf(stderr, "could not derive DRAM bank 0\n");
        ft_tt_ctx_free(ctx);
        return 1;
    }

    unsigned char *host_buf = malloc(TRANSFER_BYTES);
    if (host_buf == NULL) {
        ft_tt_ctx_free(ctx);
        return 1;
    }
    memset(host_buf, 0x5A, TRANSFER_BYTES);

    /* READ the link state, never assert it. This line used to print "link is x1 Gen2"
     * as a constant, and on 2026-08-13 it printed exactly that while the link was
     * actually trained at Gen1 -- host bandwidth had halved to 152 MB/s and no suite
     * could see it, because the only thing that mentioned the link speed was a string
     * literal. That is the dev-notes §3 family: a banner asserting the thing it should
     * measure. A single Retrain Link write restored Gen2 and 285.6 MB/s.
     *
     * Reading it from sysfs rather than a header: the ceiling depends on what the link
     * NEGOTIATED, which is a runtime property and can and did change under us. */
    {
        char link_speed[32] = "unknown";
        char link_width[32] = "unknown";
        char card_directory[BLACKHOLE_SYSFS_PATH_MAX];
        char link_attribute[BLACKHOLE_SYSFS_PATH_MAX + 32];
        FILE *f;

        /* The BDF is FOUND, not written down: it was 0000:12:00.0 on one p150a host
         * and is 0000:03:00.0 on another, and a stale literal here reports "unknown"
         * forever rather than failing — the exact banner-instead-of-measurement
         * defect the comment above is about. */
        if (blackhole_sysfs_directory(card_directory, sizeof(card_directory)) != 0)
            (void)snprintf(card_directory, sizeof(card_directory), "/nonexistent");

        (void)snprintf(link_attribute, sizeof(link_attribute), "%s/current_link_speed",
                       card_directory);
        f = fopen(link_attribute, "r");

        if (f != NULL) {
            if (fgets(link_speed, sizeof(link_speed), f) == NULL)
                (void)snprintf(link_speed, sizeof(link_speed), "unknown");
            fclose(f);
        }
        (void)snprintf(link_attribute, sizeof(link_attribute), "%s/current_link_width",
                       card_directory);
        f = fopen(link_attribute, "r");
        if (f != NULL) {
            if (fgets(link_width, sizeof(link_width), f) == NULL)
                (void)snprintf(link_width, sizeof(link_width), "unknown");
            fclose(f);
        }
        link_speed[strcspn(link_speed, "\n")] = '\0';
        link_width[strcspn(link_width, "\n")] = '\0';
        printf("DRAM bank 0 translated (%u, %u); link NEGOTIATED x%s at %s\n", dram.x, dram.y,
               link_width, link_speed);
        if (strstr(link_speed, "2.5") != NULL)
            printf("  WARNING: 2.5GT/s is Gen1. This link trains at Gen2 (5GT/s) and has\n"
                   "  been found sitting at Gen1 with host bandwidth halved. Retrain the\n"
                   "  UPSTREAM BRIDGE — derive it, never hardcode it:\n"
                   "    B=$(basename $(dirname $(readlink -f %s)))\n"
                   "    sudo setpci -s $B CAP_EXP+10.w=0020:0020\n",
                   card_directory);
    }
    printf("transfer %u MiB per direction, %u repeats\n\n", TRANSFER_BYTES / (1024u * 1024u),
           TRANSFER_REPEATS);

    measure_mode(ctx, dram, FT_TT_KMD_CACHE_UC, "UC", host_buf);
    printf("\n");
    measure_mode(ctx, dram, FT_TT_KMD_CACHE_WC, "WC", host_buf);

    free(host_buf);
    ft_tt_ctx_free(ctx);
    return 0;
}
