/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_matvec_lane_split.h
 * @brief Shared grid arithmetic for the K-quant matvecs: how many of a SIMD
 *   group's 32 lanes one output row gets, and therefore how many rows share a
 *   threadgroup.  Host-side only; each family emits its own MSL reduction.
 *
 * @status Used by the q3_k, q4_k, q5_k and q6_k matvec families.  q8_0 does not
 *   need it — its lane unit is a single weight, so every row already fills the
 *   SIMD group at any cols % 32 == 0.
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_KERNEL_MATVEC_LANE_SPLIT_H
#define FOUNDRY_METAL_KERNEL_MATVEC_LANE_SPLIT_H

/* ================================================================
 * Why this exists
 *
 * Every K-quant matvec reduces its per-lane partials with SIMD-group
 * primitives, so one threadgroup is exactly one SIMD group of 32 lanes.  Give
 * that group a single output row and the row's work units — super-blocks for
 * q3_k/q4_k, 32-weight sub-blocks for q5_k/q6_k — have to fill all 32 lanes on
 * their own.  The MoE expert shapes cannot: a 512-column q3_k row is 2
 * super-blocks, so 30 of 32 lanes idle and the kernel runs at ~5 GB/s against a
 * ~90 GB/s roof.
 *
 * The fix that does NOT touch the decode path is to give the row only as many
 * lanes as it has units and put several rows in the same SIMD group.  Lane
 * occupancy goes to 100% while each lane still decodes a whole unit exactly
 * once, so the super-block header is not re-decoded — the measured trap that
 * pins the 32-weight-unit kernels at 44-52 GB/s.
 *
 * Lanes per row is a POWER OF TWO so the cross-lane reduction is a
 * simd_shuffle_xor butterfly that never crosses a row boundary: with a
 * row-group based at a multiple of `lanes`, `lane ^ offset` for every
 * offset < lanes stays inside that row's lanes.
 * ================================================================ */

/* One threadgroup is one SIMD group; the reductions require it. */
#define FM_MATVEC_SIMD_LANES 32

/**
 * Lanes to give one output row: the largest power of two that is <= the row's
 * work-unit count and <= the SIMD width.  A row with more units than lanes
 * keeps the whole group and strides, exactly as before this split existed.
 *
 * @param units_per_row  Lane-sized work units in one row (must be >= 1)
 * @return               Lanes per row, a power of two in [1, 32]
 *
 * Thread-safe: yes (pure)
 * Reentrant: yes
 * Signal-safe: yes
 */
static inline int fm_matvec_lanes_per_row(int units_per_row)
{
    int lanes = 1;

    while (lanes < FM_MATVEC_SIMD_LANES && lanes * 2 <= units_per_row)
        lanes *= 2;
    return lanes;
}

/**
 * Output rows one threadgroup covers, given the lanes each row gets.
 *
 * @param lanes_per_row  From fm_matvec_lanes_per_row
 * @return               Rows per threadgroup, a power of two in [1, 32]
 *
 * Thread-safe: yes (pure)
 * Reentrant: yes
 * Signal-safe: yes
 */
static inline int fm_matvec_rows_per_threadgroup(int lanes_per_row)
{
    return FM_MATVEC_SIMD_LANES / lanes_per_row;
}

/**
 * Threadgroups to dispatch for `rows` output rows.  Rounds up, so the last
 * group may carry row indices past the end — the kernels must accumulate zero
 * and store nothing for those, while still entering the reduction so every
 * lane of the SIMD group is active at the shuffles.
 *
 * @param rows           Output rows (must be >= 1)
 * @param lanes_per_row  From fm_matvec_lanes_per_row
 * @return               Threadgroup count for the 1-D grid
 *
 * Thread-safe: yes (pure)
 * Reentrant: yes
 * Signal-safe: yes
 */
static inline int fm_matvec_threadgroup_count(int rows, int lanes_per_row)
{
    int rows_per_group = fm_matvec_rows_per_threadgroup(lanes_per_row);

    return (rows + rows_per_group - 1) / rows_per_group;
}

#endif /* FOUNDRY_METAL_KERNEL_MATVEC_LANE_SPLIT_H */
