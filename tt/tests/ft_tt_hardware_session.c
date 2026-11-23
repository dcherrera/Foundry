/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_hardware_session.c
 * @brief Implementation of the on-card Phase 2 test scaffolding.
 *
 * @status Runs on the p150a test card, over EITHER transport since
 *   2026-08-18: the open goes through `ft_tt_runtime_open_with_driver()` and
 *   `FT_TT_DRIVER=udrv` runs the on-card suites with no vendor module.
 * @issues None
 * @todo None
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_hardware_session.h"

#include "ft_tt_udrv.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

/** File names of the five firmware images, indexed by FT_TT_PROC_*. */
static const char *const k_firmware_file_names[FT_TT_PROC_COUNT] = {
    "fw_brisc.elf", "fw_ncrisc.elf", "fw_trisc0.elf", "fw_trisc1.elf", "fw_trisc2.elf"
};

int ft_tt_hardware_session_open(ft_tt_hardware_session *out_session)
{
    ft_tt_runtime_driver driver = FT_TT_RUNTIME_DRIVER_KMD;
    int rc;

    if (!out_session)
        return FT_TT_EINVAL;
    /* Filled IN PLACE. The accessors' `user` pointers are into `runtime`, which is
     * inside this struct, so a build-then-copy would need the same re-binding the
     * runtime removed — and getting that wrong is a wrong window, not a crash. */
    memset(out_session, 0, sizeof(*out_session));

    rc = ft_tt_runtime_driver_select(&driver, NULL);
    if (rc != FT_TT_OK)
        return rc; /* already explained: an unrecognised transport is a refusal */

    rc = ft_tt_runtime_open_with_driver(0, driver, &out_session->runtime);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("session: could not open device 0 over %s: %s",
                        ft_tt_runtime_driver_string(driver), ft_tt_error_string(rc));
        return rc;
    }

    out_session->noc = out_session->runtime.noc;
    out_session->grid = out_session->runtime.grid;
    /* On stderr, not stdout: an on-card suite's stdout is compared between the two
     * transports, and a line naming the transport would be the one guaranteed
     * difference in every such comparison. */
    ft_tt_log_error("session: card open over %s, %u Tensix in %u columns",
                    ft_tt_runtime_driver_string(driver), out_session->grid.tensix_core_count,
                    out_session->grid.tensix_column_count);
    return FT_TT_OK;
}

void ft_tt_hardware_session_close(ft_tt_hardware_session *session)
{
    unsigned slot;

    if (!session)
        return;
    /* Cores go back first: a claim outliving the window that drove it would name
     * a process that can no longer reach the card. */
    for (slot = 0; slot < FT_TT_HARDWARE_SESSION_MAX_CLAIMS; slot++) {
        ft_tt_tensix_claim_release(session->cores[slot]);
        session->cores[slot] = NULL;
    }
    ft_tt_runtime_close(&session->runtime);
    memset(session, 0, sizeof(*session));
}

ft_tt_ctx *ft_tt_hardware_session_kmd_context(const ft_tt_hardware_session *session)
{
    if (!session || session->runtime.driver != FT_TT_RUNTIME_DRIVER_KMD)
        return NULL;
    return session->runtime.ctx;
}

/**
 * The card identity claims coordinate under: the PCI BDF, so two cards in one box
 * never share a core pool.
 *
 * Through `ft_tt_runtime_pci_address()`, which answers on BOTH transports. It used
 * to go through `ft_tt_device_get_info()` on the tt-kmd context, which is the same
 * mistake `ft_tt_core` had: the claim is a flock under /tmp and has never wanted a
 * device handle, and routing its NAME through one is what tied it to a chardev.
 *
 * Falls back to NULL — which lets ft_tt_tensix_claim read the environment and then
 * its own single-card default — when neither transport can answer. A claim under a
 * slightly wrong name still excludes everyone using the same name, which is
 * strictly better than not claiming at all.
 */
static const char *session_card_identity(const ft_tt_hardware_session *session,
                                         char *buffer, size_t buffer_bytes)
{
    if (ft_tt_runtime_pci_address(&session->runtime, buffer, buffer_bytes) != FT_TT_OK ||
        buffer[0] == '\0')
        return NULL;
    return buffer;
}

int ft_tt_hardware_session_claim_cores(ft_tt_hardware_session *session, const uint32_t *indices,
                                       size_t count, const char *reason,
                                       ft_tt_tensix_claim_retire_fn retire, void *retire_context)
{
    char card[FT_TT_UDRV_CLAIM_NAMESPACE_MAX];
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    unsigned free_slot = 0;
    int rc;

    if (!session || !indices || count == 0u)
        return FT_TT_EINVAL;
    while (free_slot < FT_TT_HARDWARE_SESSION_MAX_CLAIMS && session->cores[free_slot])
        free_slot++;
    if (free_slot == FT_TT_HARDWARE_SESSION_MAX_CLAIMS) {
        ft_tt_log_error("session: no room for another core claim (max %u)",
                        FT_TT_HARDWARE_SESSION_MAX_CLAIMS);
        return FT_TT_EINVAL;
    }

    memset(&request, 0, sizeof(request));
    request.card = session_card_identity(session, card, sizeof(card));
    request.indices = indices;
    request.count = count;
    request.reason = reason;
    /* reserved stays NULL on purpose: that is what reads FT_TT_TENSIX_RESERVED,
     * which is how an operator keeps a sweep out of another family's cores and
     * out of anything a serve predating this module is using. */

    request.retire = retire;
    request.retire_context = retire_context;

    rc = ft_tt_tensix_claim_acquire(&request, &session->cores[free_slot], &refusal);
    if (rc == FT_TT_UDRV_OK)
        return FT_TT_OK;

    ft_tt_tensix_claim_refusal_print(&refusal);
    session->cores[free_slot] = NULL;
    /* FT_TT_UDRV_EAGAIN and FT_TT_EPERM are the same number by design (see
     * ft_tt_udrv.h), but the caller's table is ft_tt_device.h's, so name it. */
    return (rc == FT_TT_UDRV_EAGAIN) ? FT_TT_EPERM : rc;
}

int ft_tt_hardware_session_tensix(const ft_tt_hardware_session *session, uint32_t index,
                                  uint16_t *out_x, uint16_t *out_y)
{
    ft_tt_noc_coord core;
    int rc;

    if (!session || !out_x || !out_y)
        return FT_TT_EINVAL;

    rc = ft_tt_coord_tensix_by_index(&session->grid, index, FT_TT_COORD_TRANSLATED, &core);
    if (rc != FT_TT_OK)
        return rc;

    *out_x = (uint16_t)core.x;
    *out_y = (uint16_t)core.y;
    return FT_TT_OK;
}

int ft_tt_tensix_index_from_environment(uint32_t *out_index)
{
    const char *value = getenv(FT_TT_TENSIX_INDEX_ENVIRONMENT);
    char *end = NULL;
    unsigned long parsed;

    if (!out_index)
        return FT_TT_EINVAL;
    if (!value || value[0] == '\0') {
        *out_index = 0u;
        return FT_TT_OK;
    }

    errno = 0;
    parsed = strtoul(value, &end, 10);
    /* Only the SYNTAX is checked here. The bound used to be FT_TT_LAUNCH_MAX_CORES, a
     * literal 140 that is wrong on a 120-core p150 — and the refusal that matters was
     * always the LIVE one in ft_tt_coord_tensix_by_index(), which bounds against
     * ft_tt_coord_tensix_core_count(). Two bounds, one derived and one constant, is how
     * the constant survives. */
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX) {
        ft_tt_log_error("session: %s=\"%s\" is not a Tensix index",
                        FT_TT_TENSIX_INDEX_ENVIRONMENT, value);
        return FT_TT_EINVAL;
    }
    *out_index = (uint32_t)parsed;
    return FT_TT_OK;
}

int ft_tt_hardware_session_tensix_from_environment(ft_tt_hardware_session *session,
                                                   uint16_t *out_x, uint16_t *out_y,
                                                   uint32_t *out_index)
{
    uint32_t index = 0;
    int rc = ft_tt_tensix_index_from_environment(&index);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_hardware_session_tensix(session, index, out_x, out_y);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("session: no enabled Tensix at index %u (%s)", index,
                        ft_tt_error_string(rc));
        return rc;
    }

    /* Claim BEFORE announcing the coordinate: a log line saying "target Tensix 71"
     * for a core another process owns is the line that sends someone debugging
     * the wrong thing. `retire` is NULL here because this entry point serves
     * every single-core harness and does not know what any of them run — a
     * harness with a resident queue supplies its own via
     * ft_tt_hardware_session_claim_cores(). */
    rc = ft_tt_hardware_session_claim_cores(session, &index, 1u,
                                            "single-core harness (FT_TT_TENSIX_INDEX)", NULL,
                                            NULL);
    if (rc != FT_TT_OK)
        return rc;

    printf("target Tensix %u at translated (%u, %u)  [%s, claimed]\n", index, (unsigned)*out_x,
           (unsigned)*out_y, FT_TT_TENSIX_INDEX_ENVIRONMENT);
    if (out_index)
        *out_index = index;
    return FT_TT_OK;
}

const char *ft_tt_firmware_directory(void)
{
    static char fallback[4096];
    const char *value = getenv(FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT);
    const char *home = getenv("HOME");

    if (value && value[0] != '\0')
        return value;
    /* Default: $HOME/ttprog, the conventional install location for the boot firmware. */
    if (home && home[0] != '\0' &&
        snprintf(fallback, sizeof fallback, "%s/ttprog", home) < (int)sizeof fallback)
        return fallback;
    ft_tt_log_error("session: %s is not set and $HOME is unknown -- point it at the program "
                    "directory whose fw_*.elf to boot (built by build_tensix_program.sh, e.g. "
                    "tt/programs/<op>)", FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT);
    return NULL;
}

/** Read a whole file into a fresh buffer. */
static int read_whole_file(const char *path, unsigned char **out_data, size_t *out_size)
{
    FILE *file = fopen(path, "rb");
    unsigned char *buffer = NULL;
    long size;

    if (!file) {
        ft_tt_log_error("session: cannot open %s", path);
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
        ft_tt_log_error("session: short read on %s", path);
        free(buffer);
        fclose(file);
        return FT_TT_EIO;
    }
    fclose(file);
    *out_data = buffer;
    *out_size = (size_t)size;
    return FT_TT_OK;
}

int ft_tt_firmware_set_load(const char *directory, ft_tt_firmware_set *out_set)
{
    unsigned processor;

    if (!directory || !out_set)
        return FT_TT_EINVAL;
    memset(out_set, 0, sizeof(*out_set));

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        char path[512];
        int rc;
        int written = snprintf(path, sizeof(path), "%s/%s", directory,
                               k_firmware_file_names[processor]);

        if (written < 0 || (size_t)written >= sizeof(path)) {
            ft_tt_firmware_set_release(out_set);
            return FT_TT_EINVAL;
        }
        rc = read_whole_file(path, &out_set->elf_data[processor], &out_set->elf_size[processor]);
        if (rc != FT_TT_OK) {
            ft_tt_firmware_set_release(out_set);
            return rc;
        }
    }
    return FT_TT_OK;
}

void ft_tt_firmware_set_release(ft_tt_firmware_set *set)
{
    unsigned processor;

    if (!set)
        return;
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        free(set->elf_data[processor]);
        set->elf_data[processor] = NULL;
        set->elf_size[processor] = 0;
    }
}

/**
 * Read the Tensix instruction FIFO's status and say whether a boot can push into it.
 *
 * **The one thing a read CAN say about a wedged core.** BRISC firmware's `device_setup()`
 * pushes ZEROACC / SFPENCC / load-const / four SEMINITs into THREAD 0's instruction FIFO
 * (in its boot path) *before* it releases the
 * four subordinates. A write to `INSTRN_BUF_BASE` stalls the RISC-V core until the FIFO
 * has space, so a FIFO that cannot accept one stalls BRISC in a
 * single store — with the subordinates still held and the go signal still INIT. That is
 * the `0x00047000` fingerprint exactly, and it is our own intermediate register state
 * rather than anything the hardware invented.
 *
 * MEASURED 2026-08-17: 140 of 140 healthy cores read 0x77 with `invalid_instrn` 0; the
 * core the 1x5 multicast relaunch arm wedged read **0x66 with `invalid_instrn` 1**.
 */
static int read_instrn_buf_status(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  uint32_t *out_status, uint32_t *out_invalid)
{
    int rc = noc->read(noc->user, x, y, FT_TT_TENSIX_INSTRN_BUF_STATUS_ADDR, out_status,
                       sizeof(*out_status));

    if (rc != FT_TT_OK)
        return rc;
    return noc->read(noc->user, x, y, FT_TT_TENSIX_DBG_INVALID_INSTRN_ADDR, out_invalid,
                     sizeof(*out_invalid));
}

int ft_tt_core_boot_firmware(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                             const ft_tt_firmware_set *firmware,
                             ft_tt_image_placement placements[FT_TT_PROC_COUNT])
{
    unsigned processor;
    uint32_t instrn_buf_status = 0u;
    uint32_t invalid_instrn = 0u;
    int rc;

    if (!noc || !firmware)
        return FT_TT_EINVAL;
    if (placements)
        memset(placements, 0, sizeof(*placements) * FT_TT_PROC_COUNT);

    /* The five RISC-Vs and NOTHING ELSE. The same register has bits for the matrix
     * backend, and asserting them here corrupts that core's compute permanently — see
     * the warning block in ft_tt_launch.h. The firmware brings the backend up itself. */
    rc = ft_tt_launch_assert_reset(noc, x, y, FT_TT_SOFT_RESET_ALL_RISCS);
    if (rc != FT_TT_OK)
        return rc;

    /* AFTER the assert and BEFORE the deassert, which is the only place this reading is
     * attributable: it says what the PARK left behind, not what the boot did. Reported
     * always, so a clean boot records its own baseline. */
    if (read_instrn_buf_status(noc, x, y, &instrn_buf_status, &invalid_instrn) != FT_TT_OK)
        instrn_buf_status = FT_TT_TENSIX_INSTRN_BUF_UNREAD;
    if (instrn_buf_status == FT_TT_TENSIX_INSTRN_BUF_UNREAD)
        ; /* a diagnostic read that did not answer must not abort a boot */
    else if (instrn_buf_status != FT_TT_TENSIX_INSTRN_BUF_READY)
        ft_tt_log_error("boot: Tensix (%u, %u) instrn_buf_status=0x%08x (want 0x%02x) "
                        "invalid_instrn=0x%08x with all five RISC-Vs held — the Tensix "
                        "instruction FIFO cannot accept a write, so device_setup() will "
                        "stall BRISC before it releases the subordinates",
                        (unsigned)x, (unsigned)y, instrn_buf_status,
                        (unsigned)FT_TT_TENSIX_INSTRN_BUF_READY, invalid_instrn);
    else if (getenv("FT_TT_TRACE_INSTRN_BUF"))
        ft_tt_log_error("boot: Tensix (%u, %u) instrn_buf_status=0x%08x READY, "
                        "invalid_instrn=0x%08x (trace)", (unsigned)x, (unsigned)y,
                        instrn_buf_status, invalid_instrn);

    rc = ft_tt_launch_clear_mailbox(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        rc = ft_tt_launch_load_image(noc, x, y, processor, FT_TT_IMAGE_FIRMWARE,
                                     firmware->elf_data[processor], firmware->elf_size[processor],
                                     placements ? &placements[processor] : NULL);
        if (rc != FT_TT_OK)
            return rc;
    }

    /* RUN_MSG_DONE is 0, so a cleared mailbox already reads DONE. Without this the
     * boot wait below is vacuous — see ft_tt_launch.h. */
    rc = ft_tt_launch_prime_firmware_handshake(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;

    /* BRISC only: its firmware releases the other four itself. */
    rc = ft_tt_launch_deassert_reset(noc, x, y, FT_TT_SOFT_RESET_BRISC, false);
    if (rc != FT_TT_OK)
        return rc;

    return ft_tt_launch_wait_firmware_ready(noc, x, y, FT_TT_FIRMWARE_BOOT_TIMEOUT_MS);
}

int ft_tt_core_load_program(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            const ft_tt_kernel_program *program,
                            ft_tt_image_placement placements[FT_TT_PROC_COUNT])
{
    unsigned index;

    if (!noc || !program)
        return FT_TT_EINVAL;
    if (placements)
        memset(placements, 0, sizeof(*placements) * FT_TT_PROC_COUNT);

    for (index = 0; index < program->image_count; index++) {
        const ft_tt_kernel_image *image = &program->images[index];
        ft_tt_image_placement place;
        int rc = ft_tt_launch_load_image(noc, x, y, image->processor, FT_TT_IMAGE_KERNEL,
                                         image->elf_data, image->elf_size, &place);

        if (rc != FT_TT_OK)
            return rc;
        if (placements && image->processor < FT_TT_PROC_COUNT)
            placements[image->processor] = place;
    }
    return FT_TT_OK;
}

int ft_tt_core_park(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    uint32_t instrn_buf_status = 0u;
    uint32_t invalid_instrn = 0u;

    /* Read BEFORE the assert. Together with the reading ft_tt_core_boot_firmware() takes
     * after its own assert, this brackets the park: it separates "the previous run left
     * the Tensix instruction FIFO stuck" from "asserting reset is what stuck it". Reading
     * only; a park that refused here would change every suite's teardown on a baseline
     * that has not been measured for a RUNNING core. */
    if (read_instrn_buf_status(noc, x, y, &instrn_buf_status, &invalid_instrn) == FT_TT_OK &&
        (instrn_buf_status != FT_TT_TENSIX_INSTRN_BUF_READY ||
         getenv("FT_TT_TRACE_INSTRN_BUF")))
        ft_tt_log_error("park: Tensix (%u, %u) instrn_buf_status=0x%08x (ready is 0x%02x) "
                        "invalid_instrn=0x%08x BEFORE the assert%s",
                        (unsigned)x, (unsigned)y, instrn_buf_status,
                        (unsigned)FT_TT_TENSIX_INSTRN_BUF_READY, invalid_instrn,
                        instrn_buf_status == FT_TT_TENSIX_INSTRN_BUF_READY
                            ? " (trace)"
                            : " — the Tensix instruction FIFO is ALREADY stuck, and reset "
                              "removes the only processors that could drain it");
    return ft_tt_launch_assert_reset(noc, x, y, FT_TT_SOFT_RESET_ALL_RISCS);
}

static uint64_t monotonic_milliseconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)(now.tv_nsec / 1000000L);
}

int ft_tt_core_wait_sync_registers_ready(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                         uint32_t timeout_ms)
{
    uint64_t deadline;

    if (!noc || timeout_ms == 0u)
        return FT_TT_EINVAL;

    deadline = monotonic_milliseconds() + timeout_ms;
    for (;;) {
        uint8_t sync[FT_TT_SUBORDINATE_MAP_SIZE];
        int rc = noc->read(noc->user, x, y, FT_TT_MB_SUBORDINATE_SYNC_ADDR, sync, sizeof(sync));

        if (rc != FT_TT_OK)
            return rc;
        /* subordinate_sync is [NCRISC, TRISC0, TRISC1, TRISC2]. */
        if (sync[1] == FT_TT_RUN_SYNC_MSG_DONE)
            return FT_TT_OK;
        if (monotonic_milliseconds() >= deadline) {
            ft_tt_log_error("session: TRISC0 still at 0x%02x after %u ms; sync registers were "
                            "never re-initialised", sync[1], timeout_ms);
            return FT_TT_EIO;
        }
    }
}
