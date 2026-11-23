/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_boot.c — checklist steps 1-4 from docs/tt_launch_notes.md §6.
 *
 * This stops at the offset proof and does not run a kernel. Step 4 is the gate:
 * after loading firmware and deasserting BRISC, we read the mailbox back and look
 * for values *the firmware itself wrote*. Seeing them at the offsets we predicted
 * is the only real evidence that the transliterated mailbox layout matches the
 * binary — everything after it writes to addresses this step validates.
 *
 * Nothing here guesses a coordinate: the target Tensix comes from ft_tt_coord
 * against live ARC telemetry.
 *
 * Build:
 *   cc -std=c99 -O1 -Wall -Wextra -Werror -DFT_TT_PHASE1B_NOC_ENABLED -I.. \
 *      ../ft_tt_kmd.c ../ft_tt_device.c ../ft_tt_tlb.c ../ft_tt_arc_telemetry.c \
 *      ../ft_tt_coord.c ../ft_tt_elf.c ../ft_tt_kernel_blob.c ../ft_tt_launch.c \
 *      test_phase2_boot.c -o build/test_phase2_boot
 */

#define _POSIX_C_SOURCE 199309L

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_launch.h"
#include "ft_tt_memmap.h"
#include "ft_tt_tlb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WINDOW_BYTES (2u * 1024u * 1024u)

static int failures;

static void check(const char *what, bool ok, const char *detail)
{
    printf("  %-4s  %-44s %s\n", ok ? "ok" : "FAIL", what, detail ? detail : "");
    if (!ok) {
        failures++;
    }
}

/** Slurp an ELF. Returns NULL and reports on failure. */
static unsigned char *load_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "  cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)size);
    if (buf == NULL || fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *out_size = (size_t)size;
    return buf;
}

/** File names of the five firmware images, indexed by FT_TT_PROC_*. The DIRECTORY comes
 *  from ft_tt_firmware_directory() so two workers can point at their own build. */
static const char *const FW_FILE_NAMES[FT_TT_PROC_COUNT] = {
    "fw_brisc.elf", "fw_ncrisc.elf", "fw_trisc0.elf", "fw_trisc1.elf", "fw_trisc2.elf"
};

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
        fprintf(stderr, "grid not sound; refusing to touch a core\n");
        ft_tt_ctx_free(ctx);
        return 1;
    }

    uint32_t tensix_index = 0;
    ft_tt_noc_coord core;
    if (ft_tt_tensix_index_from_environment(&tensix_index) != FT_TT_OK ||
        ft_tt_coord_tensix_by_index(&grid, tensix_index, FT_TT_COORD_TRANSLATED, &core) !=
            FT_TT_OK) {
        fprintf(stderr, "could not derive Tensix %u\n", tensix_index);
        ft_tt_ctx_free(ctx);
        return 1;
    }
    printf("target Tensix %u at translated (%u, %u)  [%s]\n\n", tensix_index, core.x, core.y,
           FT_TT_TENSIX_INDEX_ENVIRONMENT);

    ft_tt_tlb_window *window = NULL;
    if (ft_tt_tlb_window_alloc(ctx, WINDOW_BYTES, FT_TT_KMD_CACHE_UC, &window) != FT_TT_OK) {
        ft_tt_ctx_free(ctx);
        return 1;
    }

    ft_tt_noc_window_binding binding;
    ft_tt_noc_access noc;
    int rc = ft_tt_launch_noc_over_window(ctx, window, &binding, &noc);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "noc_over_window failed: %d\n", rc);
        (void)ft_tt_tlb_window_free(ctx, window);
        ft_tt_ctx_free(ctx);
        return 1;
    }

    char detail[200];
    uint16_t x = (uint16_t)core.x;
    uint16_t y = (uint16_t)core.y;

    /* --- step 2: read the reset register, do not write it ------------------ */
    printf("step 2 — read the soft-reset register (first non-L1 address)\n");
    uint32_t reset = 0;
    rc = ft_tt_launch_read_reset(&noc, x, y, &reset);
    snprintf(detail, sizeof(detail), "0x%08x (rc=%d)", reset, rc);
    check("reset register readable", rc == FT_TT_OK, detail);
    check("all five RISC-Vs are held in reset",
          (reset & FT_TT_SOFT_RESET_ALL_RISCS) == FT_TT_SOFT_RESET_ALL_RISCS, detail);
    if (failures != 0) {
        goto done;
    }

    /* --- step 3: assert reset, clear the mailbox, read back zero ----------- */
    printf("\nstep 3 — assert reset and clear the mailbox\n");
    rc = ft_tt_launch_assert_reset(&noc, x, y, FT_TT_SOFT_RESET_ALL_RISCS);
    snprintf(detail, sizeof(detail), "rc=%d", rc);
    check("reset asserted", rc == FT_TT_OK, detail);

    rc = ft_tt_launch_clear_mailbox(&noc, x, y);
    snprintf(detail, sizeof(detail), "rc=%d", rc);
    check("mailbox cleared", rc == FT_TT_OK, detail);

    ft_tt_launch_state state;
    rc = ft_tt_launch_read_state(&noc, x, y, &state);
    snprintf(detail, sizeof(detail), "go[0].signal=0x%02x sync=%02x%02x%02x%02x rd_ptr=%u",
             state.go_messages[0].signal, state.subordinate_sync[0], state.subordinate_sync[1],
             state.subordinate_sync[2], state.subordinate_sync[3], state.launch_msg_rd_ptr);
    check("mailbox reads back clear", rc == FT_TT_OK && state.go_messages[0].signal == 0u,
          detail);
    if (failures != 0) {
        goto done;
    }

    /* --- step 4: load firmware, prime, deassert BRISC, THE OFFSET PROOF ---- */
    printf("\nstep 4 — load firmware, prime, deassert BRISC (the offset proof)\n");
    if (ft_tt_firmware_directory() == NULL) {
        check("firmware directory named (FT_TT_FIRMWARE_DIR)", false, "unset");
        goto done;
    }
    for (unsigned p = 0; p < FT_TT_PROC_COUNT; p++) {
        char path[512];
        size_t size = 0;
        unsigned char *elf;

        (void)snprintf(path, sizeof(path), "%s/%s", ft_tt_firmware_directory(), FW_FILE_NAMES[p]);
        elf = load_file(path, &size);
        if (elf == NULL) {
            check("firmware image loaded", false, path);
            goto done;
        }
        rc = ft_tt_launch_load_image(&noc, x, y, p, FT_TT_IMAGE_FIRMWARE, elf, size, NULL);
        snprintf(detail, sizeof(detail), "proc %u, %zu bytes, rc=%d", p, size, rc);
        check("firmware image loaded", rc == FT_TT_OK, detail);
        free(elf);
        if (failures != 0) {
            goto done;
        }
    }

    rc = ft_tt_launch_prime_firmware_handshake(&noc, x, y);
    snprintf(detail, sizeof(detail), "rc=%d — RUN_MSG_INIT, not DONE(=0)", rc);
    check("handshake primed", rc == FT_TT_OK, detail);

    rc = ft_tt_launch_deassert_reset(&noc, x, y, FT_TT_SOFT_RESET_BRISC, false);
    snprintf(detail, sizeof(detail), "rc=%d — BRISC only; it releases the others", rc);
    check("BRISC deasserted", rc == FT_TT_OK, detail);

    rc = ft_tt_launch_wait_firmware_ready(&noc, x, y, 2000);
    snprintf(detail, sizeof(detail), "rc=%d", rc);
    check("firmware reported ready within 2 s", rc == FT_TT_OK, detail);

    rc = ft_tt_launch_read_state(&noc, x, y, &state);
    if (rc != FT_TT_OK) {
        check("state readable after boot", false, "read failed");
        goto done;
    }

    snprintf(detail, sizeof(detail), "0x%02x (BRISC firmware writes DONE at the end of boot)",
             state.go_messages[0].signal);
    check("go signal went 0x40 -> 0x00", state.go_messages[0].signal == 0u, detail);

    bool sync_clear = true;
    for (unsigned i = 0; i < 4u; i++) {
        if (state.subordinate_sync[i] != 0u) {
            sync_clear = false;
        }
    }
    snprintf(detail, sizeof(detail), "%02x %02x %02x %02x — all four subordinates done",
             state.subordinate_sync[0], state.subordinate_sync[1], state.subordinate_sync[2],
             state.subordinate_sync[3]);
    check("subordinate_sync reads 00 00 00 00", sync_clear, detail);

    snprintf(detail, sizeof(detail), "%u (BRISC firmware zeroes it at boot)", state.launch_msg_rd_ptr);
    check("launch_msg_rd_ptr reads 0", state.launch_msg_rd_ptr == 0u, detail);

    if (failures != 0) {
        printf("\n--- state dump ---\n");
        ft_tt_launch_dump_state(&state, x, y);
    }

done:
    /* Leave the core parked in reset rather than running. */
    (void)ft_tt_launch_assert_reset(&noc, x, y, FT_TT_SOFT_RESET_ALL_RISCS);
    (void)ft_tt_tlb_window_free(ctx, window);
    printf("\n%d checks failed\n", failures);
    ft_tt_ctx_free(ctx);
    return failures == 0 ? 0 : 1;
}
