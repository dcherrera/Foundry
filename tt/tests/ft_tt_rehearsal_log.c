/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_rehearsal_log.c
 * @brief `ft_tt_log_error` for the rehearsal binary, so it need not link
 *   `ft_tt_device.c`.
 *
 *   `ft_tt_arch.c` and `ft_tt_coord.c` are pure tables and arithmetic and are
 *   exactly what the rehearsal wants — they are what makes the ARC tile and the
 *   Tensix coordinates DERIVED rather than literals. They call this one symbol,
 *   which lives in `ft_tt_device.c`, and that file carries the chardev path.
 *   Linking it would put `/dev/tenstorrent/%d` in a binary whose whole claim is
 *   that it cannot reach the vendor driver. Nine lines here keep that claim
 *   checkable with `strings`.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdarg.h>
#include <stdio.h>

void ft_tt_log_error(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

void ft_tt_log_error(const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    (void)fputs("[ft_tt] ", stderr);
    (void)vfprintf(stderr, fmt, args);
    (void)fputc('\n', stderr);
    va_end(args);
}
