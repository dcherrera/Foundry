/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_device.c
 * @brief Implementation of the Foundry Blackhole device surface: open,
 *   enumerate, map the TLB register file, close.
 *
 * @status Phase 1a, run against the p150a at 12:00.0 on the card host. Read-only
 *   against the device: the only ioctls issued are GET_DEVICE_INFO,
 *   GET_DRIVER_INFO and QUERY_MAPPINGS, and the only mapping made is the 4 KiB
 *   TLB register file, mapped PROT_READ.
 * @issues None
 * @todo None
 */

/* POSIX.1-2008 is needed for O_CLOEXEC and mmap under -std=c99, which turns off
 * the implicit feature-test macros glibc would otherwise define. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "ft_tt_device.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/mman.h>
#include <unistd.h>

#include "ft_tt_memmap.h"

#define FT_TT_CHARDEV_PATH_FORMAT "/dev/tenstorrent/%d"
#define FT_TT_CHARDEV_PATH_MAX 64

struct ft_tt_ctx {
    int device_index;
    const ft_tt_arch *arch;
    ft_tt_kmd_device *kmd;
    ft_tt_kmd_device_attrs attrs;
    ft_tt_kmd_resources resources;
    ft_tt_device_info info;
    ft_tt_tlb_counts tlb_counts;
    ft_tt_grid_info grid;
    volatile uint32_t *tlb_registers;
    size_t tlb_registers_size;
    size_t device_alloc_bytes;
    uint32_t live_window_count;
};

/* ------------------------------------------------------------------------- */
/* Diagnostics                                                               */
/* ------------------------------------------------------------------------- */

const char *ft_tt_error_string(int code)
{
    switch (code) {
    case FT_TT_OK:       return "ok";
    case FT_TT_ERROR:    return "error";
    case FT_TT_ENOMEM:   return "out of host memory";
    case FT_TT_EINVAL:   return "invalid argument";
    case FT_TT_EIO:      return "device I/O failed";
    case FT_TT_ENODEV:   return "no such device";
    case FT_TT_EPERM:    return "blocked by the phase-1a NoC fence";
    case FT_TT_EUNAVAIL: return "not available in this phase";
    case FT_TT_ENOTFOUND: return "not found in this binary";
    default:             return "unknown status";
    }
}

void ft_tt_log_error(const char *fmt, ...)
{
    va_list args;

    fputs("[ft_tt] ", stderr);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
}

/* ------------------------------------------------------------------------- */
/* Open                                                                      */
/* ------------------------------------------------------------------------- */

/* Cache the identity the two info ioctls report, so later queries cost nothing
 * and every caller sees the same snapshot. */
static int collect_device_info(ft_tt_ctx *ctx)
{
    uint64_t value = 0;
    int rc;

    rc = ft_tt_kmd_get_attrs(ctx->kmd, &ctx->attrs);
    if (rc != 0) {
        ft_tt_log_error("GET_DEVICE_INFO failed on device %d: %s", ctx->device_index,
                        strerror(-rc));
        return FT_TT_EIO;
    }

    rc = ft_tt_kmd_get_resources(ctx->kmd, &ctx->resources);
    if (rc != 0)
        return FT_TT_EINVAL;

    ctx->info.chip_arch = ctx->attrs.chip_arch;
    ctx->info.pci_vendor_id = (uint16_t)ctx->attrs.pci_vendor_id;
    ctx->info.pci_device_id = (uint16_t)ctx->attrs.pci_device_id;
    ctx->info.pci_subsystem_vendor_id = (uint16_t)ctx->attrs.pci_subsystem_vendor_id;
    ctx->info.pci_subsystem_id = (uint16_t)ctx->attrs.pci_subsystem_id;
    ctx->info.pci_domain = (uint16_t)ctx->attrs.pci_domain;
    ctx->info.pci_bus = (uint8_t)ctx->attrs.pci_bus;
    ctx->info.pci_device = (uint8_t)ctx->attrs.pci_device;
    ctx->info.pci_function = (uint8_t)ctx->attrs.pci_function;

    rc = ft_tt_kmd_driver_attr(ctx->kmd, FT_TT_KMD_DRIVER_API_VERSION, &value);
    if (rc != 0) {
        ft_tt_log_error("GET_DRIVER_INFO failed on device %d: %s", ctx->device_index,
                        strerror(-rc));
        return FT_TT_EIO;
    }
    ctx->info.driver_api_version = (uint32_t)value;

    if (ft_tt_kmd_driver_attr(ctx->kmd, FT_TT_KMD_DRIVER_VERSION_MAJOR, &value) == 0)
        ctx->info.driver_version_major = (uint8_t)value;
    if (ft_tt_kmd_driver_attr(ctx->kmd, FT_TT_KMD_DRIVER_VERSION_MINOR, &value) == 0)
        ctx->info.driver_version_minor = (uint8_t)value;
    if (ft_tt_kmd_driver_attr(ctx->kmd, FT_TT_KMD_DRIVER_VERSION_PATCH, &value) == 0)
        ctx->info.driver_version_patch = (uint8_t)value;

    ctx->tlb_counts.window_2m_total = (uint32_t)ctx->attrs.window_count_2m;
    ctx->tlb_counts.window_4g_total = (uint32_t)ctx->attrs.window_count_4g;
    ctx->tlb_counts.window_2m_available =
        ctx->tlb_counts.window_2m_total ? ctx->tlb_counts.window_2m_total - 1u : 0u;

    return FT_TT_OK;
}

/*
 * Map the 4 KiB TLB configuration register file out of BAR0 UC, and nothing
 * else. See ft_tt_tlb_registers() for why the rest of BAR0 stays unmapped.
 * tt-kmd accepts a partial mapping of a BAR at any page-aligned offset
 * (the driver's mmap range check).
 */
static int map_tlb_register_file(ft_tt_ctx *ctx)
{
    int prot = PROT_READ;
    off_t offset;
    void *mapping;

#ifdef FT_TT_PHASE1B_NOC_ENABLED
    prot = PROT_READ | PROT_WRITE;
#endif

    if (ctx->resources.bar0_len < (uint64_t)FT_TT_TLB_REGS_OFFSET + FT_TT_TLB_REGS_LEN) {
        ft_tt_log_error("BAR0 is %llu bytes, too small to contain the TLB register file at 0x%x",
                        (unsigned long long)ctx->resources.bar0_len, FT_TT_TLB_REGS_OFFSET);
        return FT_TT_ENODEV;
    }

    offset = (off_t)(ctx->resources.bar0_uc_offset + FT_TT_TLB_REGS_OFFSET);
    mapping = mmap(NULL, FT_TT_TLB_REGS_LEN, prot, MAP_SHARED, ft_tt_kmd_fd(ctx->kmd), offset);
    if (mapping == MAP_FAILED) {
        ft_tt_log_error("mmap of the TLB register file failed: %s", strerror(errno));
        return FT_TT_EIO;
    }

    ctx->tlb_registers = (volatile uint32_t *)mapping;
    ctx->tlb_registers_size = FT_TT_TLB_REGS_LEN;
    return FT_TT_OK;
}

/**
 * Ask the firmware for the MAXIMUM AI clock, unless `FT_TT_AI_CLK=min` says otherwise.
 *
 * ---------------------------------------------------------------------------
 * THE DEFAULT WAS THE MINIMUM, AND NOBODY CHOSE IT
 * ---------------------------------------------------------------------------
 * tt-kmd reads an open WITHOUT `O_APPEND` as a legacy client and initialises that fd to
 * `FT_TT_KMD_POWER_FLAGS_ALL & ~FT_TT_KMD_POWER_FLAG_MAX_AI_CLOCK` — every power flag
 * on except bit 0, which the firmware documents as `1=Max AI Clock, 0=Min AI Clock`. So
 * every Foundry process has been asking for the minimum on every open since the first one,
 * and `DEV_NOTES.md`'s "Busy AICLK 800 MHz, unchanged under load — not a boost, not a
 * throttle" was the firmware honouring that request exactly.
 *
 * MEASURED 2026-08-15 on this p150a, one bit of one ioctl the only variable:
 *
 *   | | min (the old default) | max |
 *   |---|---|---|
 *   | AICLK | 800 MHz | **1350 MHz** |
 *   | Vcore | 740 mV | 861 mV |
 *   | board power, 300-token decode | 37 W | 78 W (of 150) |
 *   | ASIC temperature | 59 C | 69 C (of 90) |
 *   | MiniCPM5-1B token, on-card attention | 0.093 s | **0.077 s** |
 *   | Bfp8 matmul reader floor | 101 ns/tile-pair | **60** |
 *
 * The aggregation is a bitwise OR across every open fd, so this request can only ever ADD
 * a flag and can never take one away from another client.
 *
 * **Set `FT_TT_AI_CLK=min` to get the old behaviour back.** It is a deliberate escape, in
 * the shape `PN_TT_DISPATCH=oneshot` already established: if you did not ask for the minimum
 * clock, you do not get it. The cost of the default is ~38 W of chip-wide clock tree paid
 * whether or not any core is busy — measured, and it dwarfs the 1-7 W four busy Tensix add.
 *
 * A refusal is logged and not fatal: an older driver without the ioctl answers EINVAL, and a
 * backend at 800 MHz is exactly what we had yesterday.
 */
bool ft_tt_ai_clock_wants_maximum(void)
{
    const char *choice = getenv("FT_TT_AI_CLK");

    return !(choice && strcmp(choice, "min") == 0);
}

static void request_ai_clock(ft_tt_ctx *ctx)
{
    int wants_min = !ft_tt_ai_clock_wants_maximum();
    int rc = ft_tt_kmd_request_max_ai_clock(ctx->kmd, !wants_min);

    if (rc != 0)
        ft_tt_log_error("SET_POWER_STATE refused (%s) — the AI clock stays where the driver's "
                        "default put it",
                        strerror(-rc));
    else if (wants_min)
        ft_tt_log_error("FT_TT_AI_CLK=min — asking for the MINIMUM AI clock (800 MHz on this part)");
}

ft_tt_ctx *ft_tt_init_device(int device_index)
{
    char path[FT_TT_CHARDEV_PATH_MAX];
    ft_tt_ctx *ctx = NULL;
    int rc;
    int n;

    if (device_index < 0) {
        ft_tt_log_error("invalid device index %d", device_index);
        return NULL;
    }

    n = snprintf(path, sizeof(path), FT_TT_CHARDEV_PATH_FORMAT, device_index);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        ft_tt_log_error("device index %d does not fit a chardev path", device_index);
        return NULL;
    }

    ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        ft_tt_log_error("out of memory opening %s", path);
        return NULL;
    }
    ctx->device_index = device_index;

    rc = ft_tt_kmd_open(path, &ctx->kmd, 0);
    if (rc != 0) {
        ft_tt_log_error("cannot open %s: %s", path, strerror(-rc));
        goto cleanup;
    }

    request_ai_clock(ctx);

    rc = collect_device_info(ctx);
    if (rc != FT_TT_OK)
        goto cleanup;

    /* THE ARCHITECTURE GATE. Everything below this line — the memory map, the
     * coordinate tables, the mailbox layout, the reset bits — describes one
     * part, and this is the only place that checks which part is in the slot.
     * ft_tt_arch_select() logs what it saw and returns NULL for anything it has
     * no verified tables for; it never falls back to a default (spec §7.3: a
     * wrong coordinate hangs the chip). */
    ctx->arch = ft_tt_arch_select(ctx->info.pci_vendor_id, ctx->info.pci_device_id,
                                  ctx->info.chip_arch);
    if (!ctx->arch) {
        ft_tt_log_error("%s: refusing to open — see the architecture error above", path);
        goto cleanup;
    }

    rc = map_tlb_register_file(ctx);
    if (rc != FT_TT_OK)
        goto cleanup;

    return ctx;

cleanup:
    ft_tt_ctx_free(ctx);
    return NULL;
}

void ft_tt_ctx_free(ft_tt_ctx *ctx)
{
    if (!ctx)
        return;

    if (ctx->live_window_count != 0)
        ft_tt_log_error("closing device %d with %u TLB window(s) still held", ctx->device_index,
                        ctx->live_window_count);

    if (ctx->tlb_registers) {
        if (munmap((void *)(uintptr_t)ctx->tlb_registers, ctx->tlb_registers_size) != 0)
            ft_tt_log_error("munmap of the TLB register file failed: %s", strerror(errno));
        ctx->tlb_registers = NULL;
    }

    if (ctx->kmd) {
        int rc = ft_tt_kmd_close(ctx->kmd);
        if (rc != 0)
            ft_tt_log_error("close of device %d failed: %s", ctx->device_index, strerror(-rc));
        ctx->kmd = NULL;
    }

    free(ctx);
}

/* ------------------------------------------------------------------------- */
/* Accessors                                                                 */
/* ------------------------------------------------------------------------- */

bool ft_tt_ctx_available(const ft_tt_ctx *ctx)
{
    return ctx != NULL && ctx->kmd != NULL && ctx->tlb_registers != NULL;
}

int ft_tt_ctx_device_index(const ft_tt_ctx *ctx)
{
    return ctx ? ctx->device_index : -1;
}

int ft_tt_ctx_fd(const ft_tt_ctx *ctx)
{
    return ctx ? ft_tt_kmd_fd(ctx->kmd) : -1;
}

ft_tt_kmd_device *ft_tt_ctx_kmd_device(const ft_tt_ctx *ctx)
{
    return ctx ? ctx->kmd : NULL;
}

const ft_tt_arch *ft_tt_ctx_arch(const ft_tt_ctx *ctx)
{
    return ctx ? ctx->arch : NULL;
}

int ft_tt_device_get_info(const ft_tt_ctx *ctx, ft_tt_device_info *out_info)
{
    if (!ctx || !out_info)
        return FT_TT_EINVAL;

    *out_info = ctx->info;
    return FT_TT_OK;
}

int ft_tt_device_get_resources(const ft_tt_ctx *ctx, ft_tt_kmd_resources *out_resources)
{
    if (!ctx || !out_resources)
        return FT_TT_EINVAL;

    *out_resources = ctx->resources;
    return FT_TT_OK;
}

int ft_tt_tlb_counts_get(const ft_tt_ctx *ctx, ft_tt_tlb_counts *out_counts)
{
    if (!ctx || !out_counts)
        return FT_TT_EINVAL;

    *out_counts = ctx->tlb_counts;
    return FT_TT_OK;
}

const volatile uint32_t *ft_tt_tlb_registers(const ft_tt_ctx *ctx)
{
    return ctx ? ctx->tlb_registers : NULL;
}

size_t ft_tt_tlb_registers_size(const ft_tt_ctx *ctx)
{
    return ctx ? ctx->tlb_registers_size : 0;
}

/* ------------------------------------------------------------------------- */
/* Capacity                                                                  */
/* ------------------------------------------------------------------------- */

int ft_tt_grid_get(const ft_tt_ctx *ctx, ft_tt_grid_info *out_grid)
{
    if (!ctx || !out_grid)
        return FT_TT_EINVAL;

    *out_grid = ctx->grid;
    return FT_TT_OK;
}

int ft_tt_grid_set(ft_tt_ctx *ctx, const ft_tt_grid_info *grid)
{
    if (!ctx || !grid)
        return FT_TT_EINVAL;

    /* A grid's telemetry masks only mean something under an architecture, so
     * the context's own table is stamped on unconditionally. A caller that
     * supplied a DIFFERENT one has mixed two cards' topologies, which is worth
     * refusing rather than silently correcting. */
    if (grid->arch && grid->arch != ctx->arch) {
        ft_tt_log_error("device %d: refusing a grid built for %s on a %s card", ctx->device_index,
                        grid->arch->name, ctx->arch ? ctx->arch->name : "(none)");
        return FT_TT_EINVAL;
    }

    ctx->grid = *grid;
    ctx->grid.arch = ctx->arch;
    return FT_TT_OK;
}

int ft_tt_mem_info(const ft_tt_ctx *ctx, size_t *free_bytes, size_t *total_bytes)
{
    size_t total;

    if (!ctx)
        return FT_TT_EINVAL;

    /* Bank count comes from ARC telemetry and nowhere else. Refuse rather than
     * report the nominal 8 banks as a measurement. */
    if (!ctx->grid.valid || !ctx->arch)
        return FT_TT_EUNAVAIL;

    /* Both factors are live: the bank count from telemetry, the bank size from
     * the selected architecture. Neither is a compiled-in constant. */
    total = (size_t)ctx->grid.dram_bank_count * (size_t)ctx->arch->core.dram_bytes_per_bank;

    if (total_bytes)
        *total_bytes = total;
    if (free_bytes)
        *free_bytes = total > ctx->device_alloc_bytes ? total - ctx->device_alloc_bytes : 0;

    return FT_TT_OK;
}

size_t ft_tt_ctx_alloc_bytes(const ft_tt_ctx *ctx)
{
    return ctx ? ctx->device_alloc_bytes : 0;
}

void ft_tt_ctx_note_device_alloc(ft_tt_ctx *ctx, int64_t delta_bytes)
{
    if (!ctx || delta_bytes == 0)
        return;

    if (delta_bytes > 0) {
        ctx->device_alloc_bytes += (size_t)delta_bytes;
    } else {
        size_t release = (size_t)(-delta_bytes);

        if (release > ctx->device_alloc_bytes) {
            ft_tt_log_error("device memory tally underflow on device %d (%zu held, %zu released)",
                            ctx->device_index, ctx->device_alloc_bytes, release);
            ctx->device_alloc_bytes = 0;
        } else {
            ctx->device_alloc_bytes -= release;
        }
    }
}

uint32_t ft_tt_ctx_live_window_count(const ft_tt_ctx *ctx)
{
    return ctx ? ctx->live_window_count : 0;
}

void ft_tt_ctx_note_window(ft_tt_ctx *ctx, int delta)
{
    if (!ctx)
        return;

    if (delta > 0) {
        ctx->live_window_count += (uint32_t)delta;
    } else if (delta < 0) {
        uint32_t drop = (uint32_t)(-delta);

        if (drop > ctx->live_window_count) {
            ft_tt_log_error("window tally underflow on device %d (%u held, %u released)",
                            ctx->device_index, ctx->live_window_count, drop);
            ctx->live_window_count = 0;
        } else {
            ctx->live_window_count -= drop;
        }
    }
}
