/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_norm_role.c
 * @brief Host half of the persistent norm role: schedule tables, exchange targets, CB
 *        placement and the bridge parameter block (ft_tt_op_norm_role.h).
 *
 * @status P0 (persistent norm role, P0.1/P0.2).
 * @issues None
 * @todo None
 */

#include "ft_tt_op_norm_role.h"

#include <stdbool.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"

/** Bytes of the flags page (one word a member, then the writers' stage words 8 + member). */
#define NORM_ROLE_FLAGS_BYTES ((uint32_t)kNrFlagsBytes)

/** A rectangle of members, inclusive. */
typedef struct role_rect {
    uint32_t x0, y0, x1, y1;
} role_rect;

static bool schedule_shape_ok(const ft_tt_op_norm_role_schedule *schedule)
{
    uint32_t m, n;

    if (!schedule || schedule->members == 0u || schedule->members > kNrMaxMembers ||
        schedule->steps == 0u || schedule->steps > kNrMaxSteps || schedule->tile_cols == 0u ||
        schedule->tile_cols % schedule->members != 0u ||
        (schedule->exchange & ~(uint32_t)kNrExchangeModes) != 0u)
        return false;
    for (m = 0; m < schedule->members; m++) {
        if (schedule->member_x[m] > 0xFFu || schedule->member_y[m] > 0xFFu)
            return false;
        for (n = 0; n < m; n++)
            if (schedule->member_x[m] == schedule->member_x[n] &&
                schedule->member_y[m] == schedule->member_y[n])
                return false;
    }
    return true;
}

static bool member_at(const ft_tt_op_norm_role_schedule *schedule, uint32_t x, uint32_t y)
{
    for (uint32_t m = 0; m < schedule->members; m++)
        if (schedule->member_x[m] == x && schedule->member_y[m] == y)
            return true;
    return false;
}

/** Runs of consecutive x in each row, then merged down rows with the same x span. Returns
 *  the rectangle count, or kNrMaxTargets + 1 when there are too many. */
static uint32_t member_rects(const ft_tt_op_norm_role_schedule *schedule,
                             role_rect rects[kNrMaxTargets])
{
    bool covered[kNrMaxMembers] = { false };
    uint32_t count = 0u;

    for (uint32_t m = 0; m < schedule->members; m++) {
        role_rect r;
        uint32_t k;

        if (covered[m])
            continue;
        r.x0 = r.x1 = schedule->member_x[m];
        r.y0 = r.y1 = schedule->member_y[m];
        while (r.x0 > 0u && member_at(schedule, r.x0 - 1u, r.y0))
            r.x0--;
        while (member_at(schedule, r.x1 + 1u, r.y0))
            r.x1++;
        for (;;) {
            bool row = true;

            for (uint32_t x = r.x0; x <= r.x1; x++)
                row = row && member_at(schedule, x, r.y1 + 1u);
            if (!row || member_at(schedule, r.x0 - 1u, r.y1 + 1u) ||
                member_at(schedule, r.x1 + 1u, r.y1 + 1u))
                break;
            r.y1++;
        }
        for (k = 0; k < schedule->members; k++)
            if (schedule->member_x[k] >= r.x0 && schedule->member_x[k] <= r.x1 &&
                schedule->member_y[k] >= r.y0 && schedule->member_y[k] <= r.y1)
                covered[k] = true;
        if (count == kNrMaxTargets)
            return kNrMaxTargets + 1u;
        rects[count++] = r;
    }
    return count;
}

int ft_tt_op_norm_role_targets(const ft_tt_op_norm_role_schedule *schedule, uint32_t member,
                               uint32_t out_words[2u * kNrMaxTargets], uint32_t *out_count)
{
    role_rect rects[kNrMaxTargets];
    uint32_t count, t;

    if (!schedule_shape_ok(schedule) || member >= schedule->members || !out_words || !out_count)
        return FT_TT_EINVAL;
    count = member_rects(schedule, rects);
    if (count == 0u || count > kNrMaxTargets)
        return FT_TT_EINVAL;
    memset(out_words, 0, 2u * kNrMaxTargets * sizeof(uint32_t));
    for (t = 0; t < count; t++) {
        const role_rect *r = &rects[t];
        const uint32_t cells = (r->x1 - r->x0 + 1u) * (r->y1 - r->y0 + 1u);
        const bool inside = schedule->member_x[member] >= r->x0 &&
                            schedule->member_x[member] <= r->x1 &&
                            schedule->member_y[member] >= r->y0 &&
                            schedule->member_y[member] <= r->y1;

        out_words[2u * t] = r->x0 | (r->y0 << 8) | (r->x1 << 16) | (r->y1 << 24);
        out_words[2u * t + 1u] = (cells - (inside ? 1u : 0u)) |
                                 (inside ? (uint32_t)kNrTargetIncludesSender : 0u);
    }
    *out_count = count;
    return FT_TT_OK;
}

static uint32_t operand_xy(const ft_tt_op_norm_role_operand *operand)
{
    return operand->x | (operand->y << 8);
}

static bool step_ok(const ft_tt_op_norm_role_step *step, uint32_t index)
{
    const ft_tt_op_norm_role_operand *operands[3] = { &step->input, &step->constants,
                                                      &step->output };

    if (step->kind != (index == 0u ? (uint32_t)kNrKindLoad : (uint32_t)kNrKindAdd))
        return false;
    for (uint32_t i = 0; i < 3u; i++)
        if (operands[i]->x > 0xFFu || operands[i]->y > 0xFFu || operands[i]->base == 0u)
            return false;
    return true;
}

int ft_tt_op_norm_role_table(const ft_tt_op_norm_role_schedule *schedule, uint32_t member,
                             uint32_t out_words[FT_TT_OP_NORM_ROLE_TABLE_WORDS],
                             uint32_t *out_used)
{
    uint32_t targets = 0u, k;

    if (!out_words || !out_used || !schedule_shape_ok(schedule) || member >= schedule->members)
        return FT_TT_EINVAL;
    for (k = 0; k < schedule->steps; k++)
        if (!step_ok(&schedule->step[k], k))
            return FT_TT_EINVAL;
    memset(out_words, 0, FT_TT_OP_NORM_ROLE_TABLE_WORDS * sizeof(uint32_t));
    if (ft_tt_op_norm_role_targets(schedule, member, &out_words[kNrHeadTarget0], &targets) !=
        FT_TT_OK)
        return FT_TT_EINVAL;
    out_words[kNrHeadMagic] = kNrTableMagic;
    out_words[kNrHeadSteps] = schedule->steps;
    out_words[kNrHeadTileCols] = schedule->tile_cols;
    out_words[kNrHeadMembers] = schedule->members | (member << 8) | (schedule->exchange << 16);
    out_words[kNrHeadSliceTiles] = schedule->tile_cols / schedule->members;
    out_words[kNrHeadTargets] = targets;
    out_words[kNrHeadSelfXy] = schedule->member_x[member] | (schedule->member_y[member] << 8);
    for (k = 0; k < schedule->steps; k++) {
        const ft_tt_op_norm_role_step *step = &schedule->step[k];
        uint32_t *entry = &out_words[kNrHeadWords + k * kNrEntryWords];

        entry[kNrEntryKind] = step->kind;
        entry[kNrEntryInputXy] = operand_xy(&step->input);
        entry[kNrEntryInputBase] = step->input.base;
        entry[kNrEntryConstXy] = operand_xy(&step->constants);
        entry[kNrEntryConstBase] = step->constants.base;
        entry[kNrEntryOutXy] = operand_xy(&step->output);
        entry[kNrEntryOutBase] = step->output.base;
    }
    out_words[kNrHeadChecksum] = ftnr_checksum(out_words, schedule->steps);
    *out_used = kNrHeadWords + schedule->steps * kNrEntryWords;
    return FT_TT_OK;
}

int ft_tt_op_norm_role_parameters(const ft_tt_op_norm_role_schedule *schedule, uint32_t step,
                                  uint32_t out_words[kNrParamWords])
{
    if (!out_words || !schedule_shape_ok(schedule) || step >= schedule->steps)
        return FT_TT_EINVAL;
    memset(out_words, 0, kNrParamWords * sizeof(uint32_t));
    out_words[kNrParamMagic] = kNrMagic;
    out_words[kNrParamStep] = step;
    out_words[kNrParamKind] = schedule->step[step].kind;
    out_words[kNrParamTileCols] = schedule->tile_cols;
    out_words[kNrParamSliceTiles] = schedule->tile_cols / schedule->members;
    return FT_TT_OK;
}

int ft_tt_op_norm_role_place_cbs(uint32_t tile_cols, uint32_t slice_tiles,
                                 ft_tt_launch_cb_entry out_entries[FT_TT_OP_NORM_ROLE_CB_COUNT],
                                 void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    const uint32_t page = FT_TT_OP_TILE_BYTES_BF16;
    const uint32_t table_bytes = FT_TT_OP_NORM_ROLE_TABLE_WORDS * 4u;
    ft_tt_op_cb_request requests[FT_TT_OP_NORM_ROLE_CB_COUNT] = {
        { kNrCbIn, page, slice_tiles },        { kNrCbScaler, page, 2u },
        { kNrCbCol, page, 1u },                { kNrCbH, page, 2u * slice_tiles },
        { kNrCbSq, page, slice_tiles },        { kNrCbEps, page, 2u },
        { kNrCbGamma, page, 2u * slice_tiles }, { kNrCbSqAll, page, tile_cols },
        { kNrCbStage, page, slice_tiles },     { kNrCbTable, table_bytes, 1u },
        { kNrCbFlags, NORM_ROLE_FLAGS_BYTES, 1u }, { kNrCbOut, page, slice_tiles },
    };

    if (!out_entries || !buffer || !out_mask || slice_tiles == 0u || tile_cols == 0u ||
        tile_cols % slice_tiles != 0u || tile_cols / slice_tiles > kNrMaxMembers)
        return FT_TT_EINVAL;
    return ft_tt_op_cb_place(requests, FT_TT_OP_NORM_ROLE_CB_COUNT, out_entries, buffer,
                             buffer_bytes, out_mask);
}

uint32_t ft_tt_op_norm_role_cb_address(
    const ft_tt_launch_cb_entry entries[FT_TT_OP_NORM_ROLE_CB_COUNT], unsigned index)
{
    for (uint32_t i = 0; entries && i < FT_TT_OP_NORM_ROLE_CB_COUNT; i++)
        if (entries[i].index == index)
            return entries[i].address;
    return 0u;
}
