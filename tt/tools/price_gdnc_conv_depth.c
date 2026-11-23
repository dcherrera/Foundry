/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file price_gdnc_conv_depth.c
 * @brief PRICE THE gdnc CONV BLOCK DEPTH LADDER OFF HARDWARE, by calling the SHIPPING
 *        `ft_tt_op_gdnc_place_cbs()` at each rung and reading its verdict.
 *
 * The decode family's depth 8 was found to be ENOMEM this way before a card arm for it
 * existed, and the alternative — modelling the arena — is how a rung gets swept that the
 * allocator was never going to accept. There is no card in this program and no environment
 * dependence in the page arithmetic: `ft_tt_op_gdnc_cb_pages_at_depth()` takes the depth as an
 * argument, and the one call that must go through the environment (`place_cbs`, which is the
 * SHIPPING path and reads the knob once per process) is re-checked here against it.
 *
 *   build: cc -std=c11 -O1 -Wall -Wextra -I.. -o build/price_gdnc_conv_depth \
 *            price_gdnc_conv_depth.c ../ft_tt_op_gdnc.c ../ft_tt_op_dispatch.c ../ft_tt_tilize.c -lm
 *   run:   for d in 1 2 4 8 16; do FT_TT_OP_GDNC_CONV_DEPTH=$d ./build/price_gdnc_conv_depth; done
 *
 * It prints ONE line per buffer and one verdict line, so a rung that does not fit is a
 * printed refusal and not an absence.
 *
 * WHAT IS STUBBED, AND WHY THE VERDICT IS STILL THE SHIPPING ALLOCATOR'S. `ft_tt_op_dispatch.c`
 * carries the whole one-shot launch path beside `ft_tt_op_cb_place()`, and linking it drags in
 * ELF loading, TLB windows and the vendor-free MMIO layer — none of which this question
 * touches. The functions below are stubbed so the link closes; NONE of them is on the path
 * being priced. `ft_tt_op_cb_place()` itself is the shipping one, unmodified, and the ENOMEM it
 * returns is the arena refusal a residency would get. The ONE stub it does call is
 * `ft_tt_launch_encode_cb_config()`, whose only failure is a config buffer too small for the
 * entries — so this program hands it a buffer of EXACTLY the size `pn_tt_residency.cb_buffer`
 * is, and returns what the real encoder's capacity check would return, rather than a blanket OK.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_op_gdnc.h"
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

static const char *cb_name(unsigned index)
{
    switch (index) {
    case FT_TT_OP_GDNC_CB_IN:      return "CB_IN";
    case FT_TT_OP_GDNC_CB_STATE:   return "CB_STATE";
    case FT_TT_OP_GDNC_CB_BAND:    return "CB_BAND";
    case FT_TT_OP_GDNC_CB_VECTOR:  return "CB_VECTOR";
    case FT_TT_OP_GDNC_CB_PROD:    return "CB_PROD";
    case FT_TT_OP_GDNC_CB_WORK:    return "CB_WORK";
    case FT_TT_OP_GDNC_CB_SCRATCH: return "CB_SCRATCH";
    case FT_TT_OP_GDNC_CB_DECAY:   return "CB_DECAY";
    case FT_TT_OP_GDNC_CB_CC:      return "CB_CC";
    case FT_TT_OP_GDNC_CB_OUT:     return "CB_OUT";
    default:                       return "?";
    }
}

int main(void)
{
    /* Ornith's own shape, read off ft_tt_op_gdnc.h rather than typed in. */
    const uint32_t taps = FT_TT_OP_GDNC_QWEN35_CONV_TAPS;
    const uint32_t state_tiles = FT_TT_OP_GDNC_QWEN35_STATE_TILES;
    const uint32_t arena_bytes = (uint32_t)(FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE);
    const uint32_t arena_pages = arena_bytes / FT_TT_OP_GDNC_FP32_TILE_BYTES;
    ft_tt_op_gdnc_request request;
    ft_tt_launch_cb_entry entries[FT_TT_OP_GDNC_CB_COUNT];
    /* EXACTLY `pn_tt_residency.cb_buffer`'s size, so the encoder's capacity question is asked
     * at the size the shipping path asks it at. */
    static unsigned char buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    uint32_t pages[FT_TT_OP_GDNC_CB_COUNT];
    static const unsigned indices[FT_TT_OP_GDNC_CB_COUNT] = {
        FT_TT_OP_GDNC_CB_IN,      FT_TT_OP_GDNC_CB_STATE,   FT_TT_OP_GDNC_CB_BAND,
        FT_TT_OP_GDNC_CB_VECTOR,  FT_TT_OP_GDNC_CB_PROD,    FT_TT_OP_GDNC_CB_WORK,
        FT_TT_OP_GDNC_CB_SCRATCH, FT_TT_OP_GDNC_CB_DECAY,   FT_TT_OP_GDNC_CB_CC,
        FT_TT_OP_GDNC_CB_OUT
    };
    uint64_t mask = 0;
    uint32_t total = 0;
    uint32_t depth;
    unsigned index;
    int rc;

    /* The one line the family prints for its own knob comes out first, on stderr. */
    depth = ft_tt_op_gdnc_conv_depth();
    if (depth == 0u) {
        printf("DEPTH REFUSED — the knob is not a usable depth, so nothing is priced\n");
        return 0;
    }

    if (ft_tt_op_gdnc_cb_pages_at_depth(taps, state_tiles, depth, pages) != FT_TT_OK) {
        printf("DEPTH %u — the page arithmetic REFUSES this shape\n", (unsigned)depth);
        return 1;
    }

    memset(&request, 0, sizeof(request));
    request.phase = FT_TT_OP_GDNC_PHASE_CONV;
    request.chunk_rows = FT_TT_OP_GDNC_MAX_CHUNK_ROWS;
    request.conv_dim = FT_TT_OP_GDNC_QWEN35_CONV_DIM;
    request.conv_taps = taps;
    request.state_tiles = state_tiles;
    request.v_heads = FT_TT_OP_GDNC_QWEN35_V_HEADS;
    request.k_heads = FT_TT_OP_GDNC_QWEN35_K_HEADS;
    request.activation_in.tile_bytes = FT_TT_OP_GDNC_FP32_TILE_BYTES;
    request.activation_out.tile_bytes = FT_TT_OP_GDNC_FP32_TILE_BYTES;
    request.constant.tile_bytes = FT_TT_OP_GDNC_FP32_TILE_BYTES;
    request.state_conv.tile_bytes = FT_TT_OP_GDNC_FP32_TILE_BYTES;

    rc = ft_tt_op_gdnc_place_cbs(&request, entries, buffer, sizeof(buffer), &mask);

    printf("=== gdnc CONV block depth %u — taps %u, state_tiles %u, arena %u B = %u pages\n",
           (unsigned)depth, (unsigned)taps, (unsigned)state_tiles, (unsigned)arena_bytes,
           (unsigned)arena_pages);
    for (index = 0; index < FT_TT_OP_GDNC_CB_COUNT; index++) {
        total += pages[index];
        printf("    %-11s idx %2u  %4u pages\n", cb_name(indices[index]), indices[index],
               (unsigned)pages[index]);
    }
    printf("    TOTAL %u pages of %u (%.1f%% of the arena), %u B\n", (unsigned)total,
           (unsigned)arena_pages, 100.0 * (double)total / (double)arena_pages,
           (unsigned)(total * FT_TT_OP_GDNC_FP32_TILE_BYTES));
    /* THE VERDICT IS THE ALLOCATOR'S, NOT THE SUM'S. `ft_tt_op_cb_place()` is what a residency
     * calls, and a total under the page count can still be refused by it — which is exactly
     * the difference between pricing a rung and modelling one. */
    printf("    place_cbs rc=%d -> %s\n", rc,
           rc == FT_TT_OK ? "PLACED" : (rc == FT_TT_ENOMEM ? "ENOMEM — the arena ends the "
                                                             "ladder here" : "REFUSED"));
    return 0;
}
