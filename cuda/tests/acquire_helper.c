/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * acquire_helper.c — Simple CLI to acquire/release K80 dies via bar_swap
 *
 * Usage:
 *   ./acquire_helper acquire <die_id>   # Acquire die into MMIO slot
 *   ./acquire_helper release <die_id>   # Release die from MMIO slot
 *   ./acquire_helper status             # Show current state
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "barswap/barswap.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s {acquire|release|status} [die_id]\n", argv[0]);
        return 1;
    }

    bs_context *ctx = bs_init();
    if (!ctx) {
        fprintf(stderr, "Failed to open /dev/bar_swap\n");
        return 1;
    }

    if (strcmp(argv[1], "status") == 0) {
        printf("Dies: %u, Slots: %u, Active: %d\n",
               bs_num_dies(ctx), bs_num_slots(ctx), bs_active_die(ctx));
        bs_free(ctx);
        return 0;
    }

    if (argc < 3) {
        fprintf(stderr, "Need die_id argument\n");
        bs_free(ctx);
        return 1;
    }

    int die = atoi(argv[2]);

    if (strcmp(argv[1], "acquire") == 0) {
        bs_gpu *gpu = bs_acquire(ctx, die);
        if (!gpu) {
            fprintf(stderr, "Failed to acquire die %d\n", die);
            bs_free(ctx);
            return 1;
        }
        printf("Acquired die %d (BAR1=0x%llx)\n", die,
               (unsigned long long)gpu->bar1_addr);
        /* Don't release — leave it acquired for nvidia */
        /* gpu handle leaks intentionally */
        bs_free(ctx);
        return 0;
    }

    if (strcmp(argv[1], "release") == 0) {
        /* Release via fresh acquire context — the ioctl still works */
        bs_gpu *gpu = bs_acquire(ctx, die);
        if (gpu) bs_release(ctx, gpu);
        bs_free(ctx);
        printf("Released die %d\n", die);
        return 0;
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    bs_free(ctx);
    return 1;
}
