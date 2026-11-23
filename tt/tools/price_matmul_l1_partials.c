/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_launch.h"
#include "ft_tt_kernel_blob.h"

/* ---------------------------------------------------------------------------
 * The link stubs. See the file header: none of these is on the path being priced.
 * --------------------------------------------------------------------------- */
void ft_tt_log_error(const char *format, ...);
void ft_tt_log_error(const char *format, ...)
{
    (void)format;
}

unsigned ft_tt_kernel_program_enables(const ft_tt_kernel_program *program)
{
    (void)program;
    return 0u;
}

int ft_tt_launch_load_image(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            unsigned processor, ft_tt_image_kind kind, const void *elf_data,
                            size_t elf_size, ft_tt_image_placement *out_place)
{
    (void)noc; (void)x; (void)y; (void)processor; (void)kind; (void)elf_data; (void)elf_size;
    (void)out_place;
    return FT_TT_EINVAL;
}

int ft_tt_launch_plant_runtime_args(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                    uint32_t l1_address, const uint32_t *args, unsigned count)
{
    (void)noc; (void)x; (void)y; (void)l1_address; (void)args; (void)count;
    return FT_TT_EINVAL;
}

int ft_tt_launch_write_launch_message(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                      const ft_tt_launch_config *config)
{
    (void)noc; (void)x; (void)y; (void)config;
    return FT_TT_EINVAL;
}

int ft_tt_launch_send_go(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    (void)noc; (void)x; (void)y;
    return FT_TT_EINVAL;
}

int ft_tt_launch_wait_done(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           uint32_t timeout_ms, ft_tt_launch_state *out_state)
{
    (void)noc; (void)x; (void)y; (void)timeout_ms; (void)out_state;
    return FT_TT_EINVAL;
}

int ft_tt_launch_encode_cb_config(const ft_tt_launch_cb_entry *entries, unsigned entry_count,
                                  void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    unsigned index;

    /* The real encoder's own two refusals, spelled from its contract: a short buffer and an
     * index outside the CB file. Everything else it does is writing four words per entry. */
    if (!entries || !buffer || !out_mask)
        return FT_TT_EINVAL;
    if (buffer_bytes < (size_t)FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4u)
        return FT_TT_EINVAL;
    *out_mask = 0;
    for (index = 0; index < entry_count; index++) {
        if (entries[index].index >= FT_TT_NUM_CIRCULAR_BUFFERS)
            return FT_TT_EINVAL;
        *out_mask |= (uint64_t)1u << entries[index].index;
    }
    return FT_TT_OK;
}
static int fits(uint32_t mb, uint32_t nb, uint32_t depth, uint32_t wtile, int partials,
                uint32_t *out_bytes)
{
    ft_tt_op_cb_request r[4];
    ft_tt_launch_cb_entry placed[4];
    unsigned char buffer[4096];
    uint64_t mask = 0;
    unsigned n = 3;
    int rc;

    r[0].index = 0u;  r[0].page_bytes = FT_TT_OP_TILE_BYTES_BF16; r[0].page_count = depth * mb * 2u;
    r[1].index = 1u;  r[1].page_bytes = wtile;                    r[1].page_count = depth * nb * 2u;
    r[2].index = 16u; r[2].page_bytes = FT_TT_OP_TILE_BYTES_BF16; r[2].page_count = 2u;
    if (partials) {
        r[3].index = 17u; r[3].page_bytes = FT_TT_OP_TILE_BYTES_FP32; r[3].page_count = mb * nb;
        n = 4;
    }
    *out_bytes = 0;
    for (unsigned i = 0; i < n; i++)
        *out_bytes += r[i].page_bytes * r[i].page_count;
    rc = ft_tt_op_cb_place(r, n, placed, buffer, sizeof(buffer), &mask);
    return rc == FT_TT_OK;
}

int main(void)
{
    const uint32_t wtile = FT_TT_OP_TILE_BYTES_BFP8_B; /* the Ornith artifact's weight tile */
    const uint32_t a = FT_TT_OP_TILE_BYTES_BF16, b = wtile;
    const uint32_t arena = FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE;
    /* The shipped point, in the same units as everything below: bytes a pass-unit, and
     * transactions a position in units of Nt*Kt/depth/32. */
    const double base_bytes = (double)a / 8.0 + (double)b / 1.0;
    const double base_tx = (1.0 + 8.0) / 8.0;

    printf("arena %u B (0x%x..0x%x), weight tile %u B, activation tile %u B\n\n",
           arena, FT_TT_OP_CB_ARENA_BASE, FT_TT_OP_CB_ARENA_END, wtile, a);
    printf("shipped MB=1 NB=8: bytes %.0f, tx %.4f\n\n", base_bytes, base_tx);

    for (uint32_t depth = 8; depth >= 2; depth /= 2) {
        printf("=== depth %u\n", depth);
        printf("  MB  NB   bytes   x    tx      x     Mt>=  L1(DST)   L1(+partials)  fits?\n");
        for (uint32_t mb = 1; mb <= 16; mb *= 2) {
            for (uint32_t nb = 1; nb <= 16; nb *= 2) {
                uint32_t nodst, withp;
                int ok_dst = fits(mb, nb, depth, wtile, 0, &nodst);
                int ok_p = fits(mb, nb, depth, wtile, 1, &withp);
                double bytes = (double)a / nb + (double)b / mb;
                double tx = (double)(mb + nb) / (double)(mb * nb) * (double)mb / (double)mb;
                /* transactions a position, same units: (MB+NB) reads per pass-block, a pass
                 * covering MB bands of 32 positions and Nt/NB column blocks. */
                tx = (double)(mb + nb) / ((double)nb * (double)mb);
                if (mb * nb > 8u && !ok_p)
                    continue;
                printf("  %2u  %2u  %6.0f  %4.2f  %6.4f  %4.2f  %4u  %8u  %13u  %s%s\n",
                       mb, nb, bytes, base_bytes / bytes, tx, base_tx / tx, mb, nodst, withp,
                       ok_p ? "yes" : "NO ",
                       (mb * nb <= 8u) ? "  (reachable today, DST)" : "  (needs PX4)");
            }
        }
        printf("\n");
    }
    return 0;
}
