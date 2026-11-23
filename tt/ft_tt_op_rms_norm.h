/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_rms_norm.h
 * @brief Host driver for the RMS-norm triad — the fc_rms_norm equivalent (spec §5.1).
 *
 *     y[r][c] = x[r][c] * gamma[c] * rsqrt( mean_c(x[r][c]^2) + eps )
 *
 * Two things the host owns that are not obvious from the kernel:
 *
 *  1. **The scaler and eps tiles.** The row sum comes from reduce_tile<SUM, REDUCE_ROW>,
 *     which multiplies by a scaler tile; the host builds one holding 1/W. eps is then one
 *     more accumulated reduce against a second constant tile whose every element is
 *     eps*W/32, because each row of it contributes 32 * (eps*W/32) * (1/W) = eps. That
 *     buys eps for one extra reduce_tile call instead of an add_tiles_bcast_scalar, an
 *     extra CB round trip and a pack (see the kernel's header note 1).
 *
 *     ft_tt_op_rms_norm_build_constants() makes both, in the CB's own format, so nothing
 *     downstream has to know the trick.
 *
 *  2. **gamma is expanded to full tiles by the loader**, not broadcast in the kernel —
 *     ft_tt_fttw_upload_broadcast_rows(). The 32x inflation is 96 KiB for a 1536-wide
 *     model and it removes a broadcast reconfiguration from the inner loop.
 *
 * @status PCC and shape coverage in docs/tt_bringup_log.md §14.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_RMS_NORM_H
#define FT_TT_OP_RMS_NORM_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_kernel_blob.h"
#include "ft_tt_launch.h"
#include "ft_tt_op_matmul.h" /* ft_tt_op_operand */

/** One RMS norm over `row_blocks * 32` rows of `tile_cols * 32` columns. */
typedef struct ft_tt_op_rms_norm_request {
    uint32_t row_blocks; /**< rows / 32 */
    uint32_t tile_cols;  /**< W / 32 */
    ft_tt_op_operand activation;
    ft_tt_op_operand gamma;
    /** Two adjacent tiles: the 1/W scaler, then the eps tile. */
    ft_tt_op_operand constants;
    ft_tt_op_operand output;
} ft_tt_op_rms_norm_request;

/**
 * Build the two constant tiles into `out`, which must hold 2 * 2048 bytes (two Float16_b
 * tiles). Upload them contiguously and point `constants` at the first.
 *
 * @param width  W, the reduced dimension in ELEMENTS — not tiles. The eps tile's value
 *               depends on it, which is why it is not a fixed constant
 * @param eps    the epsilon the CPU reference uses
 * @return       FT_TT_OK, or FT_TT_EINVAL for a short buffer or a zero width
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_rms_norm_build_constants(uint32_t width, float eps, void *out, size_t out_bytes);

/**
 * Run one RMS norm on a core whose firmware is already booted.
 *
 * L1 cost is `(3 * tile_cols + 5)` tiles of circular buffer: the activation, gamma and the
 * x*x intermediate are all `tile_cols` pages deep, because the kernel needs a whole row
 * block resident to reduce over it and then to scale it. At W=1536 that is ~300 KiB of
 * L1's 1.5 MiB.
 *
 * @param timeout_ms  bounds the wait; there is no untimed path (spec §7.3)
 * @return            FT_TT_OK; FT_TT_EIO on timeout with the core dumped; FT_TT_EINVAL
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_op_rms_norm_run(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                          const ft_tt_kernel_program *program,
                          const ft_tt_op_rms_norm_request *request, uint32_t timeout_ms);

#endif /* FT_TT_OP_RMS_NORM_H */
