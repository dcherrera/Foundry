/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_runtime.c
 * @brief Implementation of the serving-process host runtime (see ft_tt_runtime.h).
 *
 * @status Written for Phase 7. Carries the TRANSPORT SEAM since 2026-08-17:
 *   `ft_tt_runtime_open()` selects tt-kmd or foundry's own userspace driver from
 *   `FT_TT_DRIVER`, defaulting to `kmd`, and a serve has run on the card over
 *   both. Nothing above this file changed.
 * @issues None
 * @todo None
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

/** Firmware ELF names build_tensix_program.sh writes, indexed by FT_TT_PROC_*. */
static const char *const k_firmware_file_names[FT_TT_PROC_COUNT] = {
    "fw_brisc.elf", "fw_ncrisc.elf", "fw_trisc0.elf", "fw_trisc1.elf", "fw_trisc2.elf"
};

/** Kernel ELF names from the same build, indexed the same way. */
static const char *const k_kernel_file_names[FT_TT_PROC_COUNT] = {
    "brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf"
};

const char *ft_tt_runtime_kernel_image_name(unsigned processor)
{
    if (processor >= FT_TT_PROC_COUNT)
        return NULL;
    return k_kernel_file_names[processor];
}

/* ================================================================
 * Files
 * ================================================================ */

/** Read a whole file into a fresh buffer. Caller frees. */
static int read_whole_file(const char *path, unsigned char **out_data, size_t *out_size)
{
    FILE *file = fopen(path, "rb");
    unsigned char *buffer = NULL;
    long size;

    if (!file) {
        ft_tt_log_error("tt runtime: cannot open %s", path);
        return FT_TT_EIO;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return FT_TT_EIO;
    }
    buffer = malloc((size_t)size);
    if (!buffer) {
        fclose(file);
        return FT_TT_ENOMEM;
    }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        ft_tt_log_error("tt runtime: short read on %s", path);
        free(buffer);
        fclose(file);
        return FT_TT_EIO;
    }
    fclose(file);
    *out_data = buffer;
    *out_size = (size_t)size;
    return FT_TT_OK;
}

/** Copy the `.src` sidecar's first line into `out`, or fall back to the ELF name. A
 *  missing sidecar is not an error: it only costs a log line its provenance. */
static void read_source_name(const char *directory, const char *elf_name, char *out,
                             size_t out_bytes)
{
    char path[512];
    FILE *file = NULL;
    size_t length;

    (void)snprintf(out, out_bytes, "%s", elf_name);
    if (snprintf(path, sizeof(path), "%s/%s.src", directory, elf_name) < 0)
        return;
    file = fopen(path, "r");
    if (!file)
        return;
    if (fgets(out, (int)out_bytes, file)) {
        length = strlen(out);
        while (length > 0 && (out[length - 1] == '\n' || out[length - 1] == '\r'))
            out[--length] = '\0';
        if (length == 0)
            (void)snprintf(out, out_bytes, "%s", elf_name);
    }
    fclose(file);
}

/** The shared body of both loaders: read `names[p]` for every p in `mask`. */
static int load_named_images(const char *directory, uint32_t processor_mask,
                             const char *const *names, ft_tt_runtime_images *out_images)
{
    unsigned processor;

    if (!directory || !out_images || processor_mask == 0u)
        return FT_TT_EINVAL;
    memset(out_images, 0, sizeof(*out_images));
    (void)snprintf(out_images->name, sizeof(out_images->name), "%s", directory);

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        char path[512];
        int rc;

        if ((processor_mask & (1u << processor)) == 0u)
            continue;
        if (snprintf(path, sizeof(path), "%s/%s", directory, names[processor]) < 0) {
            ft_tt_runtime_images_release(out_images);
            return FT_TT_EINVAL;
        }
        rc = read_whole_file(path, &out_images->owned_data[processor],
                             &out_images->elf_size[processor]);
        if (rc != FT_TT_OK) {
            ft_tt_runtime_images_release(out_images);
            return rc;
        }
        out_images->elf_data[processor] = out_images->owned_data[processor];
        read_source_name(directory, names[processor], out_images->source_names[processor],
                         sizeof(out_images->source_names[processor]));
    }
    out_images->source = FT_TT_PROGRAM_SOURCE_DIRECTORY;
    return FT_TT_OK;
}

int ft_tt_runtime_firmware_load(const char *directory, ft_tt_runtime_images *out_images)
{
    uint32_t all = 0u;
    unsigned processor;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++)
        all |= 1u << processor;
    return load_named_images(directory, all, k_firmware_file_names, out_images);
}

int ft_tt_runtime_program_load(const char *directory, uint32_t processor_mask,
                               ft_tt_runtime_images *out_images)
{
    unsigned processor;
    unsigned count = 0;
    int rc = load_named_images(directory, processor_mask, k_kernel_file_names, out_images);

    if (rc != FT_TT_OK)
        return rc;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if ((processor_mask & (1u << processor)) == 0u)
            continue;
        out_images->images[count].source_name = out_images->source_names[processor];
        out_images->images[count].processor = processor;
        out_images->images[count].elf_data = out_images->elf_data[processor];
        out_images->images[count].elf_size = out_images->elf_size[processor];
        count++;
    }
    out_images->program.name = out_images->name;
    out_images->program.toolchain = "on-disk";
    out_images->program.compile_time_args = "on-disk";
    out_images->program.images = out_images->images;
    out_images->program.image_count = count;
    return FT_TT_OK;
}

void ft_tt_runtime_images_release(ft_tt_runtime_images *images)
{
    unsigned processor;

    if (!images)
        return;
    /* `owned_data`, NOT `elf_data`: an embedded image points into a `static const` array,
     * and free()ing that is the one way this struct could turn the portability win into a
     * crash. Two arrays make it unrepresentable rather than merely avoided. */
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        free(images->owned_data[processor]);
        images->owned_data[processor] = NULL;
        images->elf_data[processor] = NULL;
        images->elf_size[processor] = 0;
    }
    images->program.images = NULL;
    images->program.image_count = 0;
    images->source = FT_TT_PROGRAM_SOURCE_NONE;
}

/* ================================================================
 * The bulk DRAM accessor
 * ================================================================ */

/**
 * The ft_tt_noc_access vtable over ft_tt_{read,write}_dram(), and nothing else.
 *
 * Every byte moves at FT_TT_NOC_WIDTH_BLOCK16 through a WC window, with the DRAM
 * space bounds and the 64 B read / 16 B write alignment those helpers enforce. A
 * register address does not fit the space and a 4-byte length does not fit the unit,
 * so this accessor CANNOT carry run control even by mistake — which is the point of
 * having two rather than one wider one.
 */
static int bulk_dram_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                           size_t bytes)
{
    ft_tt_noc_path *path = (ft_tt_noc_path *)user;
    ft_tt_noc_coord coord;
    int rc;

    if (!path || !src)
        return FT_TT_EINVAL;

    coord.x = x;
    coord.y = y;
    rc = ft_tt_write_dram(path, coord, addr, src, bytes);
    /* WC stores may sit in the write-combining buffer indefinitely. The UC run-control
     * accesses that follow (launch args, go message) would drain it as a side effect of
     * x86's ordering rules, but relying on that makes correctness depend on the ORDER of
     * two unrelated modules' calls. One store fence per transfer costs nothing measurable
     * against a PCIe round trip and removes the question. */
#if defined(__SSE2__)
    _mm_sfence();
#endif
    return rc;
}

static int bulk_dram_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst,
                          size_t bytes)
{
    ft_tt_noc_path *path = (ft_tt_noc_path *)user;
    ft_tt_noc_coord coord;

    if (!path || !dst)
        return FT_TT_EINVAL;

    coord.x = x;
    coord.y = y;
    return ft_tt_read_dram(path, coord, addr, dst, bytes);
}

/**
 * The same vtable over ft_tt_{read,write}_l1(), so a caller that wants more than a word out
 * of a core's L1 does not pay the register accessor's 4 bytes per PCIe transaction.
 *
 * The L1 space check keeps it honest in both directions: 16-byte alignment in address AND
 * length means a 4-byte control word, a go message or a soft-reset register cannot come
 * through here, and an address outside L1 is a refusal rather than a NoC transaction.
 */
static int bulk_l1_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                         size_t bytes)
{
    ft_tt_noc_path *path = (ft_tt_noc_path *)user;
    ft_tt_noc_coord coord;
    int rc;

    if (!path || !src)
        return FT_TT_EINVAL;

    coord.x = x;
    coord.y = y;
    rc = ft_tt_write_l1(path, coord, addr, src, bytes);
    /* Same store fence as the DRAM arm, for the same reason: a WC store may sit in the
     * write-combining buffer, and correctness must not depend on an unrelated module's
     * later UC access draining it. */
#if defined(__SSE2__)
    _mm_sfence();
#endif
    return rc;
}

static int bulk_l1_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst,
                        size_t bytes)
{
    ft_tt_noc_path *path = (ft_tt_noc_path *)user;
    ft_tt_noc_coord coord;

    if (!path || !dst)
        return FT_TT_EINVAL;

    coord.x = x;
    coord.y = y;
    return ft_tt_read_l1(path, coord, addr, dst, bytes);
}

/* ================================================================
 * The card
 * ================================================================ */

int ft_tt_runtime_driver_select(ft_tt_runtime_driver *out_driver, bool *out_explicit)
{
    const char *choice = getenv(FT_TT_RUNTIME_DRIVER_ENVIRONMENT);

    if (!out_driver)
        return FT_TT_EINVAL;
    if (out_explicit)
        *out_explicit = false;
    if (!choice || choice[0] == '\0') {
        *out_driver = FT_TT_RUNTIME_DRIVER_KMD;
        return FT_TT_OK;
    }
    if (strcmp(choice, "kmd") == 0) {
        *out_driver = FT_TT_RUNTIME_DRIVER_KMD;
    } else if (strcmp(choice, "udrv") == 0) {
        *out_driver = FT_TT_RUNTIME_DRIVER_UDRV;
    } else {
        /* Not defaulted. A typo here would run a whole round on the transport
         * the operator was trying to leave, and every number in it would be
         * attributed to the other one. */
        ft_tt_log_error("%s='%s' names no transport — it must be kmd or udrv. REFUSING",
                        FT_TT_RUNTIME_DRIVER_ENVIRONMENT, choice);
        return FT_TT_EINVAL;
    }
    if (out_explicit)
        *out_explicit = true;
    return FT_TT_OK;
}

const char *ft_tt_runtime_driver_string(ft_tt_runtime_driver driver)
{
    switch (driver) {
    case FT_TT_RUNTIME_DRIVER_KMD:
        return "kmd";
    case FT_TT_RUNTIME_DRIVER_UDRV:
        return "udrv";
    default:
        return "unknown";
    }
}

int ft_tt_runtime_pci_address(const ft_tt_runtime *runtime, char *out, size_t out_bytes)
{
    ft_tt_device_info info;

    if (!runtime || !out || out_bytes == 0)
        return FT_TT_EINVAL;
    out[0] = '\0';

    if (runtime->driver == FT_TT_RUNTIME_DRIVER_UDRV) {
        const char *address = ft_tt_runtime_udrv_pci_address(runtime->udrv);

        if (!address || address[0] == '\0')
            return FT_TT_EUNAVAIL;
        (void)snprintf(out, out_bytes, "%s", address);
        return FT_TT_OK;
    }

    if (!runtime->ctx || ft_tt_device_get_info(runtime->ctx, &info) != FT_TT_OK)
        return FT_TT_EUNAVAIL;
    (void)snprintf(out, out_bytes, "%04x:%02x:%02x.%x", info.pci_domain, info.pci_bus,
                   info.pci_device, info.pci_function);
    return FT_TT_OK;
}

int ft_tt_runtime_window_stats_read(const ft_tt_runtime *runtime, ft_tt_runtime_window_stats *out)
{
    if (!out)
        return FT_TT_EINVAL;
    memset(out, 0, sizeof(*out));
    if (!runtime)
        return FT_TT_EINVAL;
    if (runtime->driver == FT_TT_RUNTIME_DRIVER_UDRV)
        return ft_tt_runtime_udrv_window_stats_read(runtime->udrv, out);

    out->register_read = runtime->binding.path.bytes_read;
    out->register_written = runtime->binding.path.bytes_written;
    out->register_aims = ft_tt_noc_path_aim_count(&runtime->binding.path);
    out->bulk_read = runtime->bulk_path.bytes_read;
    out->bulk_written = runtime->bulk_path.bytes_written;
    out->bulk_aims = ft_tt_noc_path_aim_count(&runtime->bulk_path);
    /* tt-kmd's CONFIGURE_TLB is a syscall that reports its own failures, so
     * there is no readback to disagree and no counter for one. Zero here means
     * "not applicable", not "none happened" — the udrv arm is where it counts. */
    out->aim_failures = 0u;
    return FT_TT_OK;
}

uint64_t ft_tt_runtime_dram_bytes(const ft_tt_runtime *runtime)
{
    if (!runtime || !runtime->grid.valid || !runtime->grid.arch)
        return 0u;
    /* Both factors are live: the bank count from telemetry, the bank size from
     * the selected architecture. This is ft_tt_mem_info()'s own arithmetic with
     * the context — which one of the two transports does not have — removed. */
    return (uint64_t)runtime->grid.dram_bank_count *
           (uint64_t)runtime->grid.arch->core.dram_bytes_per_bank;
}

/** The tt-kmd half of the open: chardev, telemetry, two ioctl-allocated windows. */
static int runtime_open_kmd(int device_index, ft_tt_runtime *runtime)
{
    int rc;

    runtime->ctx = ft_tt_init_device(device_index);
    if (!runtime->ctx) {
        ft_tt_log_error("tt runtime: could not open device %d", device_index);
        return FT_TT_ENODEV;
    }

    rc = ft_tt_arc_telemetry_read_grid(runtime->ctx, &runtime->grid);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("tt runtime: telemetry read failed: %s", ft_tt_error_string(rc));
        return rc;
    }
    /* A topology we cannot map is a topology we must not probe: a wrong NoC endpoint hangs
     * the chip and costs a power cycle (spec §7.3). Refuse before claiming a window. */
    rc = ft_tt_coord_check_grid(&runtime->grid);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("tt runtime: grid is not a topology we can map; refusing the card");
        return rc;
    }

    rc = ft_tt_tlb_window_alloc(runtime->ctx, FT_TT_RUNTIME_WINDOW_BYTES, FT_TT_KMD_CACHE_UC,
                                &runtime->window);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_noc_over_window(runtime->ctx, runtime->window, &runtime->binding,
                                      &runtime->noc);
    if (rc != FT_TT_OK)
        return rc;

    /* A SECOND window, write-combined, for bulk DRAM. There are 201 and the backend uses
     * two, so this is not a scarce resource; what it buys is that data stops inheriting the
     * register accessor's 4-byte width. */
    rc = ft_tt_noc_path_open(runtime->ctx, FT_TT_KMD_CACHE_WC, &runtime->bulk_path);
    if (rc != FT_TT_OK)
        return rc;
    runtime->bulk_dram.write = bulk_dram_write;
    runtime->bulk_dram.read = bulk_dram_read;
    runtime->bulk_l1.write = bulk_l1_write;
    runtime->bulk_l1.read = bulk_l1_read;
    return FT_TT_OK;
}

int ft_tt_runtime_open(int device_index, ft_tt_runtime *out_runtime)
{
    ft_tt_runtime_driver driver = FT_TT_RUNTIME_DRIVER_KMD;
    int rc = ft_tt_runtime_driver_select(&driver, NULL);

    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_runtime_open_with_driver(device_index, driver, out_runtime);
}

int ft_tt_runtime_open_with_driver(int device_index, ft_tt_runtime_driver driver,
                                   ft_tt_runtime *out_runtime)
{
    int rc;

    if (!out_runtime)
        return FT_TT_EINVAL;
    if (driver != FT_TT_RUNTIME_DRIVER_KMD && driver != FT_TT_RUNTIME_DRIVER_UDRV)
        return FT_TT_EINVAL;

    /* Filled IN PLACE rather than in a stack copy: on the tt-kmd path the
     * accessors' `user` pointers are into the runtime itself, and a copy after
     * the fact is what makes them need re-binding. Building here removes that
     * whole class of mistake, and the udrv path's pointers are into its own
     * heap handle and never needed it. */
    memset(out_runtime, 0, sizeof(*out_runtime));
    out_runtime->driver = driver;

    if (driver == FT_TT_RUNTIME_DRIVER_UDRV)
        rc = ft_tt_runtime_udrv_open(device_index, &out_runtime->udrv, &out_runtime->grid,
                                     &out_runtime->noc, &out_runtime->bulk_dram,
                                     &out_runtime->bulk_l1);
    else
        rc = runtime_open_kmd(device_index, out_runtime);

    if (rc != FT_TT_OK) {
        /* NO FALLBACK. An explicit transport that cannot be honoured is a
         * refusal; the caller decides whether that ends the process, and
         * ft_tt_runtime_driver_select()'s `explicit` flag is how it tells an
         * absent card from a broken instruction. */
        ft_tt_runtime_close(out_runtime);
        return rc;
    }

    if (driver == FT_TT_RUNTIME_DRIVER_KMD) {
        out_runtime->noc.user = &out_runtime->binding;
        out_runtime->bulk_dram.user = &out_runtime->bulk_path;
        out_runtime->bulk_l1.user = &out_runtime->bulk_path;
    }
    return FT_TT_OK;
}

void ft_tt_runtime_close(ft_tt_runtime *runtime)
{
    if (!runtime)
        return;
    if (runtime->driver == FT_TT_RUNTIME_DRIVER_UDRV) {
        ft_tt_runtime_udrv_close(runtime->udrv);
    } else {
        (void)ft_tt_noc_path_close(&runtime->bulk_path);
        if (runtime->window && runtime->ctx)
            (void)ft_tt_tlb_window_free(runtime->ctx, runtime->window);
        ft_tt_ctx_free(runtime->ctx);
    }
    memset(runtime, 0, sizeof(*runtime));
}

/** Monotonic seconds, or 0 when no timing was asked for. */
static double boot_clock(const ft_tt_runtime_boot_timing *timing)
{
    struct timespec now;

    if (!timing || clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int ft_tt_runtime_boot_core(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            const ft_tt_runtime_images *firmware)
{
    return ft_tt_runtime_boot_core_timed(noc, x, y, firmware, NULL);
}

int ft_tt_runtime_boot_core_timed(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  const ft_tt_runtime_images *firmware,
                                  ft_tt_runtime_boot_timing *timing)
{
    unsigned processor;
    double mark = boot_clock(timing), now;
    int rc;

    if (!noc || !firmware)
        return FT_TT_EINVAL;

    /* The five RISC-Vs and NOTHING ELSE. The same register has bits for the matrix backend,
     * and asserting them here corrupts that core's compute permanently — see the warning
     * block in ft_tt_launch.h. The firmware brings the backend up itself. */
    rc = ft_tt_launch_assert_reset(noc, x, y, FT_TT_SOFT_RESET_ALL_RISCS);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_clear_mailbox(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;
    now = boot_clock(timing);
    if (timing)
        timing->reset_seconds += now - mark;
    mark = now;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        rc = ft_tt_launch_load_image(noc, x, y, processor, FT_TT_IMAGE_FIRMWARE,
                                     firmware->elf_data[processor],
                                     firmware->elf_size[processor], NULL);
        if (rc != FT_TT_OK)
            return rc;
        if (timing)
            timing->image_bytes += firmware->elf_size[processor];
    }
    now = boot_clock(timing);
    if (timing)
        timing->images_seconds += now - mark;
    mark = now;

    /* RUN_MSG_DONE is 0, so a cleared mailbox already reads DONE. Without this the boot
     * wait below is vacuous — it would pass on a dead core. */
    rc = ft_tt_launch_prime_firmware_handshake(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;

    /* BRISC only: its firmware releases the other four itself, and releasing them from the
     * host races it. */
    rc = ft_tt_launch_deassert_reset(noc, x, y, FT_TT_SOFT_RESET_BRISC, false);
    if (rc != FT_TT_OK)
        return rc;

    rc = ft_tt_launch_wait_firmware_ready(noc, x, y, FT_TT_RUNTIME_BOOT_TIMEOUT_MS);
    if (timing)
        timing->handshake_seconds += boot_clock(timing) - mark;
    return rc;
}
