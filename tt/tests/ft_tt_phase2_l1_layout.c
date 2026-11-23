/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_phase2_l1_layout.c
 * @brief Derivation of the Phase 2 test L1 map from where the firmware images landed.
 *
 * @status Used by the on-card Phase 2 tests; the arithmetic is unit-tested in
 *   tests/test_launch.c, which needs no device.
 * @issues None
 * @todo None
 */

#include "ft_tt_phase2_l1_layout.h"

#include <string.h>

/** Highest offset any field of the map reaches, plus the two tiles the destination
 *  buffer holds. Used to prove a derived base still leaves the map inside L1. */
#define FT_TT_PHASE2_LAYOUT_SPAN_BYTES \
    (FT_TT_PHASE2_DESTINATION_OFFSET + 2u * FT_TT_PHASE2_TILE_BYTES)

int ft_tt_phase2_l1_layout_derive(const ft_tt_image_placement *placements, unsigned count,
                                  ft_tt_phase2_l1_layout *out_layout)
{
    uint32_t base;

    if (!out_layout)
        return FT_TT_EINVAL;
    memset(out_layout, 0, sizeof(*out_layout));

    base = ft_tt_image_placement_l1_floor(placements, count);
    if (base == 0u)
        return FT_TT_EINVAL;
    if (base < FT_TT_PHASE2_RESERVED_L1_END)
        base = FT_TT_PHASE2_RESERVED_L1_END;
    base = (base + (FT_TT_PHASE2_LAYOUT_ALIGNMENT - 1u)) & ~(FT_TT_PHASE2_LAYOUT_ALIGNMENT - 1u);

    if (base > FT_TT_MEM_L1_SIZE - FT_TT_PHASE2_LAYOUT_SPAN_BYTES) {
        ft_tt_log_error("phase2 layout: base 0x%05x plus a %u byte map does not fit L1", base,
                        (unsigned)FT_TT_PHASE2_LAYOUT_SPAN_BYTES);
        return FT_TT_EINVAL;
    }

    out_layout->kernel_config_base = base;
    out_layout->cb_config_address = base + FT_TT_PHASE2_CB_CONFIG_OFFSET;
    out_layout->runtime_args_base = base + FT_TT_PHASE2_RTA_OFFSET;
    out_layout->probe_address = base + FT_TT_PHASE2_PROBE_OFFSET;
    out_layout->cb_in0_address = base + FT_TT_PHASE2_CB_IN0_OFFSET;
    out_layout->cb_in1_address = base + FT_TT_PHASE2_CB_IN1_OFFSET;
    out_layout->cb_out_address = base + FT_TT_PHASE2_CB_OUT_OFFSET;
    out_layout->source0_address = base + FT_TT_PHASE2_SOURCE0_OFFSET;
    out_layout->source1_address = base + FT_TT_PHASE2_SOURCE1_OFFSET;
    out_layout->destination_address = base + FT_TT_PHASE2_DESTINATION_OFFSET;
    out_layout->allocation_floor = base;
    return FT_TT_OK;
}

uint32_t ft_tt_phase2_runtime_args_address(const ft_tt_phase2_l1_layout *layout,
                                           unsigned processor)
{
    if (!layout || processor >= FT_TT_PROC_COUNT)
        return 0u;
    return layout->runtime_args_base + (uint32_t)processor * FT_TT_PHASE2_RTA_STRIDE;
}
