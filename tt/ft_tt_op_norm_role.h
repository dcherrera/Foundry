/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_norm_role.h
 * @brief Host half of the PERSISTENT NORM ROLE (kernels/resident_norm_role_protocol.h): the
 *        per-member schedule table, the exchange targets, the circular-buffer placement and
 *        the bridge descriptor's parameter block. Pure C, no card, so
 *        tests/test_norm_role.c proves all of it off hardware.
 *
 * A role is M members (cores) and a schedule of `steps` steps a token (Qwen3.8: 129 = one
 * LOAD + 128 ADD). The caller fills a ft_tt_op_norm_role_schedule once at arm; every member
 * gets its own table (the header names the member and its own NoC coordinate), written into
 * its kNrCbTable page, and every step is then one descriptor per member.
 *
 * @status P0 (persistent norm role, P0.1/P0.2).
 * @issues None
 * @todo None
 */
#ifndef FT_TT_OP_NORM_ROLE_H
#define FT_TT_OP_NORM_ROLE_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "kernels/resident_norm_role_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/** The role program's name in the embedded blob. */
#define FT_TT_OP_NORM_ROLE_PROGRAM "norm_role"
/** CBs the role places (kNrCb*). */
#define FT_TT_OP_NORM_ROLE_CB_COUNT 12u
/** Table words: the header plus kNrMaxSteps entries. */
#define FT_TT_OP_NORM_ROLE_TABLE_WORDS (kNrHeadWords + kNrMaxSteps * kNrEntryWords)

/** A DRAM operand: NoC coordinate of its bank and byte address. */
typedef struct ft_tt_op_norm_role_operand {
    uint32_t x, y;
    uint32_t base;
} ft_tt_op_norm_role_operand;

/** One step of the token's schedule. */
typedef struct ft_tt_op_norm_role_step {
    uint32_t kind;                         /**< kNrKindLoad (step 0) or kNrKindAdd */
    ft_tt_op_norm_role_operand input;      /**< the embedding row, or the projection */
    ft_tt_op_norm_role_operand constants;  /**< scaler, eps, then T gamma tiles */
    ft_tt_op_norm_role_operand output;     /**< the normed strip (its replica-map key) */
} ft_tt_op_norm_role_step;

/** The whole role, as the host arms it. */
typedef struct ft_tt_op_norm_role_schedule {
    uint32_t members;                      /**< M, 1 .. kNrMaxMembers */
    uint32_t member_x[kNrMaxMembers];      /**< member m's NoC coordinate */
    uint32_t member_y[kNrMaxMembers];
    uint32_t tile_cols;                    /**< T; must divide by M */
    uint32_t steps;                        /**< steps a token, 1 .. kNrMaxSteps */
    uint32_t exchange;                     /**< exchange mode bits (kNrExchange*), 0 = whole */
    ft_tt_op_norm_role_step step[kNrMaxSteps];
} ft_tt_op_norm_role_schedule;

/**
 * The exchange targets of member `member`: the members as at most kNrMaxTargets rectangles
 * (runs of consecutive x in one row, merged down consecutive rows), each with its receiver
 * count, the sender excluded.
 *
 * @param schedule   the role
 * @param member     the sender
 * @param out_words  2 * kNrMaxTargets words (rect, info) per target
 * @param out_count  receives the target count
 * @return FT_TT_OK; FT_TT_EINVAL when the members need more than kNrMaxTargets rectangles,
 *         a coordinate does not fit a byte, or two members share a core
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_norm_role_targets(const ft_tt_op_norm_role_schedule *schedule, uint32_t member,
                               uint32_t out_words[2u * kNrMaxTargets], uint32_t *out_count);

/**
 * Member `member`'s schedule table: header (checksum last) and one entry a step.
 *
 * @param out_words  FT_TT_OP_NORM_ROLE_TABLE_WORDS words; unused tail words are zeroed
 * @param out_used   receives the words the table occupies
 * @return FT_TT_OK, or FT_TT_EINVAL on a schedule the kernels would refuse
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_norm_role_table(const ft_tt_op_norm_role_schedule *schedule, uint32_t member,
                             uint32_t out_words[FT_TT_OP_NORM_ROLE_TABLE_WORDS],
                             uint32_t *out_used);

/**
 * The bridge descriptor's parameter block for step `step` (identical on every member and
 * every token).
 *
 * @return FT_TT_OK, or FT_TT_EINVAL for a step outside the schedule
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_norm_role_parameters(const ft_tt_op_norm_role_schedule *schedule, uint32_t step,
                                  uint32_t out_words[kNrParamWords]);

/**
 * Place the role's twelve CBs in the arena for slice S and row T (one placement serves
 * every member). SqAll is T pages; In, Sq, Stage and Out S; H and Gamma 2S; scaler and eps
 * 2; Col 1; the table one page of FT_TT_OP_NORM_ROLE_TABLE_WORDS words; flags kNrFlagsBytes (128 B).
 *
 * @return FT_TT_OK; FT_TT_EINVAL on a bad shape; FT_TT_ENOMEM when it does not fit
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_norm_role_place_cbs(uint32_t tile_cols, uint32_t slice_tiles,
                                 ft_tt_launch_cb_entry out_entries[FT_TT_OP_NORM_ROLE_CB_COUNT],
                                 void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/** The placed L1 address of CB `index` in `entries`, or 0. */
uint32_t ft_tt_op_norm_role_cb_address(
    const ft_tt_launch_cb_entry entries[FT_TT_OP_NORM_ROLE_CB_COUNT], unsigned index);

#ifdef __cplusplus
}
#endif

#endif /* FT_TT_OP_NORM_ROLE_H */
