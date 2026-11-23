/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Portions derived from BlackholeDmaTransfer::h2d_transfer in tt-umd
 * (https://github.com/tenstorrent/tt-umd), (c) Tenstorrent AI ULC,
 * licensed under the Apache License, Version 2.0. Modified by
 * David (Corey) Herrera, 2026. See NOTICE and LICENSES/Apache-2.0.txt.
 */
/**
 * @file test_dma_vs_pio.c
 * @brief The Phase 0 decision-2 measurement, carried over to Phase 3: does the
 *   BAR2 H2D DMA engine beat 287 MB/s WC PIO on a weight upload?
 *
 * Phase 0 (`docs/tt_phase0_decisions.md` §2) parked this deliberately: the DMA
 * path is real on this host because `ALLOCATE_DMA_BUF` calls
 * `dma_alloc_coherent` — kernel memory, physically contiguous by construction,
 * so **no hugepages and no IOMMU** are involved, and the engine lives behind
 * BAR2, whose absence-of-BAR4 problem does not apply. The decision was "adopt
 * only if it wins on a measurement". This is that measurement.
 *
 * WHAT IS BEING COMPARED
 *   PIO: 16 x 16 B stores per 256 B block through a WC-mapped TLB window,
 *        which is the shape tt-umd uses and the shape that produced 287 MB/s.
 *   DMA: memcpy host buffer -> the coherent bounce buffer, then program the
 *        Synopsys read channel behind BAR2 and poll XFERSIZE to zero.
 *
 *   The DMA number INCLUDES the bounce memcpy, because a weight upload has to
 *   pay it: the artifact is mmap'd, the bounce buffer is 512 KiB, the data has
 *   to be copied in. A number that excludes it is not a number about uploading
 *   weights. Both are reported so the split is visible.
 *
 *   **Host CPU is measured too, and it is the axis that decides this.** Both
 *   paths are bounded by the same x1 Gen2 link, so MB/s alone cannot separate
 *   them; what differs is what the host's two cores are doing meanwhile. PIO is
 *   a store loop and burns a core by construction. DMA can hand the copy to the
 *   engine — but only if it stops busy-polling XFERSIZE, so the sleep-polled
 *   variant is measured as well as the spin-polled one.
 *
 * THE ADDRESS MODEL, AND WHY IT IS NOT A PROBE
 *   The DMA destination is a **device AXI address**, and on Blackhole that is
 *   the BAR0-relative offset of a 2 MiB TLB window:
 *     `tlb_offset = DYNAMIC_TLB_2M_BASE(=0) + index * DYNAMIC_TLB_2M_SIZE`
 *   (the same TLB-window addressing as host stores; see
 *   BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md). So DMA lands wherever the window is aimed,
 *   exactly like a host store into the same window — the window is the
 *   translator either way. Two facts make this safe to act on rather than a
 *   guess: it is transliterated from the vendor's working implementation, not
 *   discovered by trying addresses; and `get_dma_tlb_size(BLACKHOLE)` is
 *   **2 MiB**, i.e. the DMA path upstream uses on
 *   Blackhole is the 2 MiB window kind this board has, not a 4 GiB BAR4 window
 *   it does not have.
 *
 *   Still, the first transfer is 4 bytes and is read back before anything
 *   bulk runs. If the address model were wrong, the smallest possible write is
 *   the one that finds out.
 *
 * SAFETY
 *   - DRAM bank 1, coordinates from ft_tt_coord against live ARC telemetry.
 *     Never probed. No Tensix core is touched and nothing is launched.
 *   - Every device wait is bounded (FT_TT_DMA_TIMEOUT_MS). A DMA that never
 *     retires returns an error instead of spinning.
 *   - MSI stop/abort addresses are programmed to the completion page exactly as
 *     tt-umd does. They must NOT be left at zero: the engine would then write
 *     its completion MSI to host physical address 0.
 *
 * WHY THE IOCTL STRUCTS ARE LOCAL TO THIS FILE
 *   `ft_tt_kmd_abi.h` carries payload structs only for the ioctls Foundry
 *   actually issues (its deviation V4), and ALLOCATE_DMA_BUF was deliberately
 *   dropped. This is a measurement, not an adoption: if DMA wins, the structs
 *   move into the vendored header as part of building the real upload path. If
 *   it loses, nothing was added to the backend. Layout follows the
 *   kernel driver's documented ALLOCATE_DMA_BUF ioctl.
 *
 * Build (on the card host — it needs the card):
 *   cc -std=c99 -O2 -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion \
 *      -Wstrict-prototypes -Wformat=2 -fstack-protector-strong \
 *      -DFT_TT_PHASE1B_NOC_ENABLED -I.. \
 *      ../ft_tt_kmd.c ../ft_tt_device.c ../ft_tt_tlb.c \
 *      ../ft_tt_arc_telemetry.c ../ft_tt_coord.c test_dma_vs_pio.c \
 *      -o build/test_dma_vs_pio
 *
 * @status Written for the Phase 3 follow-up. Results in
 *   docs/tt_bringup_log.md §13.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 199309L

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_kmd_abi.h"
#include "ft_tt_tlb.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

/* ------------------------------------------------------------------------- */
/* ALLOCATE_DMA_BUF payload — kernel driver's documented ioctl layout          */
/* ------------------------------------------------------------------------- */

struct ft_tt_allocate_dma_buf_in {
    uint32_t requested_size;
    uint8_t buf_index;
    uint8_t flags;
    uint8_t reserved0[2];
    uint64_t reserved1[2];
};

struct ft_tt_allocate_dma_buf_out {
    uint64_t physical_address;
    uint64_t mapping_offset;
    uint32_t size;
    uint32_t reserved0;
    uint64_t noc_address;
    uint64_t reserved1;
};

struct ft_tt_allocate_dma_buf {
    struct ft_tt_allocate_dma_buf_in in;
    struct ft_tt_allocate_dma_buf_out out;
};

/* ------------------------------------------------------------------------- */
/* Synopsys read channel 0, BAR2 — vendor DMA engine layout                          */
/* ------------------------------------------------------------------------- */

#define FT_TT_DMA_REG_EN             0x100u
#define FT_TT_DMA_REG_DOORBELL       0x104u
#define FT_TT_DMA_REG_XFERSIZE       0x11Cu
#define FT_TT_DMA_REG_SAR_LOW        0x120u
#define FT_TT_DMA_REG_SAR_HIGH       0x124u
#define FT_TT_DMA_REG_DAR_LOW        0x128u
#define FT_TT_DMA_REG_DAR_HIGH       0x12Cu
#define FT_TT_DMA_REG_INT_SETUP      0x188u
#define FT_TT_DMA_REG_MSI_STOP_LOW   0x190u
#define FT_TT_DMA_REG_MSI_STOP_HIGH  0x194u
#define FT_TT_DMA_REG_MSI_ABORT_LOW  0x1A0u
#define FT_TT_DMA_REG_MSI_ABORT_HIGH 0x1A4u
#define FT_TT_DMA_REG_MSI_MSGD       0x1A8u

/** Local interrupt (bit 3) + remote stop interrupt (bit 5). */
#define FT_TT_DMA_INT_SETUP_VALUE 0x28u

#define FT_TT_DMA_TIMEOUT_MS 2000.0

/** Sleep between XFERSIZE polls in the CPU-offload variant. A 512 KiB chunk at
 *  link rate takes ~1.9 ms, so 100 µs costs at most ~5% latency and is the
 *  difference between one core pinned and one core idle. */
#define DMA_POLL_SLEEP_US 100L

/* ------------------------------------------------------------------------- */
/* Measurement parameters                                                    */
/* ------------------------------------------------------------------------- */

#define WINDOW_BYTES     (2u * 1024u * 1024u)
#define DRAM_AIM_BASE    0x200000u
#define DMA_BUF_BYTES    (512u * 1024u) /* tt-umd's empirical sweet spot */
#define COMPLETION_BYTES 4096u
#define TRANSFER_BYTES   (2u * 1024u * 1024u) /* one full window per pass */
#define TRANSFER_REPEATS 8u
#define DRAM_BANK_INDEX  1u /* bank 0 is what every other test uses */

/** One claimed coherent bounce buffer plus its completion page. */
typedef struct ft_tt_dma_bounce_buffer {
    unsigned char *host;      /**< mapped bounce buffer */
    uint64_t physical;        /**< its physical address, for SAR */
    uint64_t completion_phys; /**< completion page physical address, for MSI */
    size_t bytes;             /**< usable bounce bytes (excludes completion) */
    size_t mapping_bytes;     /**< what to munmap */
} ft_tt_dma_bounce_buffer;

static int failures;
static int checks;

static void check(int condition, const char *what)
{
    checks++;
    if (!condition) {
        failures++;
        printf("  FAIL  %s\n", what);
    } else {
        printf("  ok    %s\n", what);
    }
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/** Host CPU consumed by this process so far, user + system. The whole point of
 *  a DMA engine is that this stops tracking wall time. */
static double cpu_seconds(void)
{
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / 1e6 +
           (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec / 1e6;
}

/** The PIO shape the 287 MB/s baseline was measured with: 16 B at a time, no
 *  SIMD, because this host has no AVX (spec §8.4). */
static void pio_block_write(volatile unsigned char *dst, const unsigned char *src, size_t bytes)
{
    volatile uint64_t *d = (volatile uint64_t *)(void *)dst;
    const uint64_t *s = (const uint64_t *)(const void *)src;
    size_t words = bytes / 8u;

    for (size_t i = 0; i < words; i++)
        d[i] = s[i];
}

static void pio_block_read(unsigned char *dst, volatile const unsigned char *src, size_t bytes)
{
    uint64_t *d = (uint64_t *)(void *)dst;
    volatile const uint64_t *s = (volatile const uint64_t *)(const void *)src;
    size_t words = bytes / 8u;

    for (size_t i = 0; i < words; i++)
        d[i] = s[i];
}

/* ------------------------------------------------------------------------- */
/* The DMA path                                                              */
/* ------------------------------------------------------------------------- */

/**
 * Claim a coherent bounce buffer from the kernel and map it.
 *
 * The allocation is `bytes + one page`: the trailing page is the completion
 * page whose physical address the engine's MSI stop/abort registers point at.
 * tt-umd does the same.
 *
 * @param fd     open /dev/tenstorrent/N
 * @param bytes  usable bounce bytes
 * @param out    receives the buffer; release with ft_tt_dma_bounce_free()
 * @return       0 on success, -1 on failure (reason printed)
 */
static int ft_tt_dma_bounce_alloc(int fd, size_t bytes, ft_tt_dma_bounce_buffer *out)
{
    struct ft_tt_allocate_dma_buf request;
    size_t mapping_bytes = bytes + COMPLETION_BYTES;
    void *mapping = NULL;

    memset(&request, 0, sizeof(request));
    request.in.requested_size = (uint32_t)mapping_bytes;
    request.in.buf_index = 0;

    if (ioctl(fd, (unsigned long)FT_TT_KMD_IOCTL_ALLOCATE_DMA_BUF, &request) != 0) {
        perror("ALLOCATE_DMA_BUF");
        return -1;
    }

    mapping = mmap(NULL, mapping_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                   (off_t)request.out.mapping_offset);
    if (mapping == MAP_FAILED) {
        perror("mmap dma buffer");
        return -1;
    }

    out->host = (unsigned char *)mapping;
    out->physical = request.out.physical_address;
    out->completion_phys = request.out.physical_address + bytes;
    out->bytes = bytes;
    out->mapping_bytes = mapping_bytes;
    return 0;
}

static void ft_tt_dma_bounce_free(ft_tt_dma_bounce_buffer *buffer)
{
    if (buffer->host != NULL) {
        (void)munmap(buffer->host, buffer->mapping_bytes);
        buffer->host = NULL;
    }
}

static void write_reg(volatile unsigned char *bar2, uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)(void *)(bar2 + offset) = value;
}

static uint32_t read_reg(volatile unsigned char *bar2, uint32_t offset)
{
    return *(volatile uint32_t *)(void *)(bar2 + offset);
}

/**
 * One host-to-device DMA: program read channel 0 and poll it to completion.
 * Transliterated from `BlackholeDmaTransfer::h2d_transfer`.
 *
 * @param bar2       mapped BAR2 (uncached)
 * @param buffer     the coherent bounce buffer, already filled
 * @param axi_dst    device AXI address = tlb_window_index * 2 MiB + offset
 * @param bytes      multiple of 4, at most buffer->bytes
 * @param poll_sleep_us  0 to spin on XFERSIZE; otherwise sleep this long
 *                       between polls, which is what frees the core
 * @return           0 on success, -1 on timeout
 */
static int ft_tt_dma_h2d(volatile unsigned char *bar2, const ft_tt_dma_bounce_buffer *buffer,
                         uint32_t axi_dst, size_t bytes, long poll_sleep_us)
{
    double start;

    write_reg(bar2, FT_TT_DMA_REG_INT_SETUP, FT_TT_DMA_INT_SETUP_VALUE);
    write_reg(bar2, FT_TT_DMA_REG_MSI_STOP_LOW, (uint32_t)(buffer->completion_phys & 0xFFFFFFFFu));
    write_reg(bar2, FT_TT_DMA_REG_MSI_STOP_HIGH, (uint32_t)(buffer->completion_phys >> 32));
    write_reg(bar2, FT_TT_DMA_REG_MSI_ABORT_LOW,
              (uint32_t)((buffer->completion_phys + 4u) & 0xFFFFFFFFu));
    write_reg(bar2, FT_TT_DMA_REG_MSI_ABORT_HIGH,
              (uint32_t)((buffer->completion_phys + 4u) >> 32));
    write_reg(bar2, FT_TT_DMA_REG_MSI_MSGD, 0);

    write_reg(bar2, FT_TT_DMA_REG_EN, 0x1u);
    write_reg(bar2, FT_TT_DMA_REG_SAR_LOW, (uint32_t)(buffer->physical & 0xFFFFFFFFu));
    write_reg(bar2, FT_TT_DMA_REG_SAR_HIGH, (uint32_t)(buffer->physical >> 32));
    write_reg(bar2, FT_TT_DMA_REG_DAR_LOW, axi_dst);
    write_reg(bar2, FT_TT_DMA_REG_DAR_HIGH, 0);
    write_reg(bar2, FT_TT_DMA_REG_XFERSIZE, (uint32_t)bytes);
    write_reg(bar2, FT_TT_DMA_REG_DOORBELL, 0x1u);

    start = now_seconds();
    while (read_reg(bar2, FT_TT_DMA_REG_XFERSIZE) != 0u) {
        if (poll_sleep_us > 0) {
            struct timespec nap;
            nap.tv_sec = 0;
            nap.tv_nsec = poll_sleep_us * 1000L;
            (void)nanosleep(&nap, NULL);
        }
        if ((now_seconds() - start) * 1000.0 > FT_TT_DMA_TIMEOUT_MS) {
            printf("  DMA timeout: XFERSIZE still %u after %.0f ms\n",
                   read_reg(bar2, FT_TT_DMA_REG_XFERSIZE), FT_TT_DMA_TIMEOUT_MS);
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Correctness — before any bandwidth number is believed                     */
/* ------------------------------------------------------------------------- */

/**
 * Prove the DMA landed where we think it did, and that it tracks its source.
 *
 * A single matching pattern would not prove either: the destination could hold
 * a stale value that happens to match, and a transfer that silently did nothing
 * looks identical to one that wrote the same bytes twice. So the destination is
 * **poisoned by PIO first** and three different payloads are sent, including an
 * all-zero one, which is the payload a "did nothing" implementation cannot
 * fake against a non-zero poison (dev notes §3).
 */
static void verify_dma_lands(volatile unsigned char *bar2, ft_tt_dma_bounce_buffer *buffer,
                             volatile unsigned char *aperture, uint32_t axi_base)
{
    static const unsigned char patterns[3] = {0xA5u, 0x00u, 0x3Cu};
    unsigned char *poison = malloc(4096u);
    unsigned char *readback = malloc(4096u);

    if (poison == NULL || readback == NULL) {
        check(0, "host scratch allocation");
        free(poison);
        free(readback);
        return;
    }

    /* Smallest legal transfer first: if the AXI address model is wrong, four
     * bytes is the cheapest way to find out. */
    memset(buffer->host, 0x5Au, 4u);
    memset(poison, 0xFFu, 4096u);
    pio_block_write(aperture, poison, 4096u);
    check(ft_tt_dma_h2d(bar2, buffer, axi_base, 4u, 0) == 0, "4-byte DMA retires");
    pio_block_read(readback, aperture, 4096u);
    check(memcmp(readback, buffer->host, 4u) == 0, "4-byte DMA landed in DRAM bank 1");
    check(readback[4] == 0xFFu, "4-byte DMA wrote exactly 4 bytes, not more");

    for (size_t p = 0; p < sizeof(patterns); p++) {
        memset(poison, (int)(0xE0u + p), 4096u);
        pio_block_write(aperture, poison, 4096u);
        pio_block_read(readback, aperture, 4096u);
        if (readback[0] != poison[0]) {
            check(0, "poison took");
            break;
        }

        memset(buffer->host, patterns[p], 4096u);
        if (ft_tt_dma_h2d(bar2, buffer, axi_base, 4096u, 0) != 0) {
            check(0, "4 KiB DMA retires");
            break;
        }
        pio_block_read(readback, aperture, 4096u);
        check(memcmp(readback, buffer->host, 4096u) == 0,
              "4 KiB DMA tracks its source over a poisoned destination");
    }

    /* A full bounce buffer, at a non-zero offset, with position-dependent
     * content — a transfer that lands at the wrong offset or reorders cannot
     * survive this. */
    for (size_t i = 0; i < buffer->bytes; i++)
        buffer->host[i] = (unsigned char)(i * 31u + 7u);

    check(ft_tt_dma_h2d(bar2, buffer, axi_base + (uint32_t)buffer->bytes, buffer->bytes, 0) == 0,
          "512 KiB DMA retires at a non-zero window offset");
    {
        unsigned char *bulk = malloc(buffer->bytes);
        if (bulk == NULL) {
            check(0, "bulk readback allocation");
        } else {
            pio_block_read(bulk, aperture + buffer->bytes, buffer->bytes);
            check(memcmp(bulk, buffer->host, buffer->bytes) == 0,
                  "512 KiB DMA is byte-exact at its offset");
            free(bulk);
        }
    }

    free(poison);
    free(readback);
}

/* ------------------------------------------------------------------------- */
/* Bandwidth                                                                 */
/* ------------------------------------------------------------------------- */

/** Total MiB moved by one bandwidth row. */
#define ROW_MEBIBYTES ((double)TRANSFER_BYTES * TRANSFER_REPEATS / (1024.0 * 1024.0))

/**
 * @param out_cpu_fraction  receives host CPU seconds per wall second — 1.0 is
 *                          one core saturated for the whole transfer
 * @return MB/s
 */
static double measure_pio(volatile unsigned char *aperture, const unsigned char *host_buf,
                          double *out_cpu_fraction)
{
    double c0 = cpu_seconds();
    double t0 = now_seconds();
    for (uint32_t r = 0; r < TRANSFER_REPEATS; r++)
        pio_block_write(aperture, host_buf, TRANSFER_BYTES);
    double seconds = now_seconds() - t0;

    *out_cpu_fraction = (cpu_seconds() - c0) / seconds;
    return ROW_MEBIBYTES / seconds;
}

/**
 * DMA the same total bytes in `chunk`-sized pieces, reporting the honest upload
 * rate (bounce memcpy included), the engine-only rate, and what the host CPU
 * paid for it.
 *
 * @param poll_sleep_us    0 to spin, >0 to sleep between XFERSIZE polls
 * @param out_engine_mb_s  receives the rate excluding the bounce memcpy
 * @param out_cpu_fraction receives host CPU seconds per wall second
 * @return the rate including the memcpy, or -1.0 if a transfer failed
 */
static double measure_dma(volatile unsigned char *bar2, ft_tt_dma_bounce_buffer *buffer,
                          uint32_t axi_base, const unsigned char *host_buf, size_t chunk,
                          long poll_sleep_us, double *out_engine_mb_s, double *out_cpu_fraction)
{
    double engine_seconds = 0.0;
    double c0 = cpu_seconds();
    double t0 = now_seconds();

    for (uint32_t r = 0; r < TRANSFER_REPEATS; r++) {
        for (size_t offset = 0; offset < TRANSFER_BYTES; offset += chunk) {
            double e0;
            memcpy(buffer->host, host_buf + offset, chunk);
            e0 = now_seconds();
            if (ft_tt_dma_h2d(bar2, buffer, axi_base + (uint32_t)offset, chunk, poll_sleep_us) != 0)
                return -1.0;
            engine_seconds += now_seconds() - e0;
        }
    }

    double total_seconds = now_seconds() - t0;

    *out_engine_mb_s = ROW_MEBIBYTES / engine_seconds;
    *out_cpu_fraction = (cpu_seconds() - c0) / total_seconds;
    return ROW_MEBIBYTES / total_seconds;
}

int main(void)
{
    ft_tt_ctx *ctx = NULL;
    ft_tt_grid_info grid;
    ft_tt_noc_coord dram;
    ft_tt_tlb_window *window = NULL;
    ft_tt_dma_bounce_buffer bounce = {NULL, 0, 0, 0, 0};
    ft_tt_kmd_resources resources;
    volatile unsigned char *bar2 = NULL;
    volatile unsigned char *aperture = NULL;
    unsigned char *host_buf = NULL;
    uint32_t window_index = 0;
    uint32_t axi_base = 0;
    int rc = 1;

    ctx = ft_tt_init_device(0);
    if (ctx == NULL) {
        fprintf(stderr, "could not open device 0\n");
        return 1;
    }

    if (ft_tt_arc_telemetry_read_grid(ctx, &grid) != FT_TT_OK ||
        ft_tt_coord_check_grid(&grid) != FT_TT_OK) {
        fprintf(stderr, "grid not sound; refusing to touch the card\n");
        goto cleanup;
    }
    if (ft_tt_coord_dram_bank(&grid, DRAM_BANK_INDEX, FT_TT_COORD_TRANSLATED, &dram) != FT_TT_OK) {
        fprintf(stderr, "could not derive DRAM bank %u\n", DRAM_BANK_INDEX);
        goto cleanup;
    }
    if (ft_tt_device_get_resources(ctx, &resources) != FT_TT_OK || resources.bar2_len == 0) {
        fprintf(stderr, "no BAR2 on this card; the DMA engine is unreachable\n");
        goto cleanup;
    }

    bar2 = mmap(NULL, (size_t)resources.bar2_len, PROT_READ | PROT_WRITE, MAP_SHARED,
                ft_tt_ctx_fd(ctx), (off_t)resources.bar2_uc_offset);
    if ((void *)bar2 == MAP_FAILED) {
        perror("mmap BAR2");
        bar2 = NULL;
        goto cleanup;
    }

    if (ft_tt_tlb_window_alloc(ctx, WINDOW_BYTES, FT_TT_KMD_CACHE_WC, &window) != FT_TT_OK) {
        fprintf(stderr, "TLB window alloc failed\n");
        goto cleanup;
    }
    if (ft_tt_tlb_window_aim(ctx, window, (uint16_t)dram.x, (uint16_t)dram.y, DRAM_AIM_BASE) !=
        FT_TT_OK) {
        fprintf(stderr, "TLB window aim failed\n");
        goto cleanup;
    }

    aperture = ft_tt_tlb_window_aperture(window);
    window_index = ft_tt_tlb_window_id(window);
    if (window_index >= FT_TT_TLB_2M_WINDOW_COUNT) {
        fprintf(stderr, "window index %u is outside the 2 MiB aperture region\n", window_index);
        goto cleanup;
    }
    axi_base = window_index * WINDOW_BYTES;

    if (ft_tt_dma_bounce_alloc(ft_tt_ctx_fd(ctx), DMA_BUF_BYTES, &bounce) != 0)
        goto cleanup;

    host_buf = malloc(TRANSFER_BYTES);
    if (host_buf == NULL)
        goto cleanup;
    for (size_t i = 0; i < TRANSFER_BYTES; i++)
        host_buf[i] = (unsigned char)(i * 17u + 3u);

    printf("DRAM bank %u translated (%u, %u); TLB window %u -> AXI 0x%08X\n", DRAM_BANK_INDEX,
           dram.x, dram.y, window_index, axi_base);
    printf("bounce buffer %zu B at PA 0x%016llX, completion at 0x%016llX\n\n", bounce.bytes,
           (unsigned long long)bounce.physical, (unsigned long long)bounce.completion_phys);

    printf("correctness\n");
    verify_dma_lands(bar2, &bounce, aperture, axi_base);

    if (failures > 0) {
        printf("\n%d/%d checks failed; refusing to report a bandwidth number\n", failures, checks);
        goto cleanup;
    }

    printf("\nbandwidth  (%u MiB x %u passes per row; host CPU is cores busy per wall second)\n",
           TRANSFER_BYTES / (1024u * 1024u), TRANSFER_REPEATS);
    {
        double cpu = 0.0;
        double mb_s = measure_pio(aperture, host_buf, &cpu);
        printf("  PIO  WC 16 B stores           %8.1f MB/s                        host CPU %.2f\n",
               mb_s, cpu);
    }

    for (size_t chunk = 64u * 1024u; chunk <= DMA_BUF_BYTES; chunk *= 2u) {
        double engine = 0.0;
        double cpu = 0.0;
        double total = measure_dma(bar2, &bounce, axi_base, host_buf, chunk, 0, &engine, &cpu);
        if (total < 0.0) {
            printf("  DMA %4zu KiB spin-polled      transfer failed\n", chunk / 1024u);
            failures++;
            break;
        }
        printf("  DMA %4zu KiB spin-polled      %8.1f MB/s  (engine only %8.1f)  host CPU %.2f\n",
               chunk / 1024u, total, engine, cpu);
    }

    /* The only axis on which DMA can still win: hand the copy to the engine and
     * stop spinning on it. Same bytes, same chunk, sleeping between polls. */
    {
        double engine = 0.0;
        double cpu = 0.0;
        double total = measure_dma(bar2, &bounce, axi_base, host_buf, DMA_BUF_BYTES,
                                   DMA_POLL_SLEEP_US, &engine, &cpu);
        if (total < 0.0) {
            printf("  DMA  512 KiB sleep-polled     transfer failed\n");
            failures++;
        } else {
            printf("  DMA  512 KiB sleep-polled     %8.1f MB/s  (engine only %8.1f)  host CPU "
                   "%.2f\n",
                   total, engine, cpu);
        }
    }

    printf("\n%d/%d checks passed\n", checks - failures, checks);
    rc = (failures == 0) ? 0 : 1;

cleanup:
    ft_tt_dma_bounce_free(&bounce);
    free(host_buf);
    if (window != NULL)
        (void)ft_tt_tlb_window_free(ctx, window);
    if (bar2 != NULL)
        (void)munmap((void *)bar2, (size_t)resources.bar2_len);
    ft_tt_ctx_free(ctx);
    return rc;
}
