/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_status_print.h
 * @brief The three human-readable forms of an `ft_tt_core_status`: a table header, one line
 *        per core, and the everything-decoded detail block.
 *
 * **Why these are not in the library beside the read.** `ft_tt_core_status.c` moved into
 * `tt/` when Pantheon's arming path became a second consumer of its verdict, and a file a
 * serve links has no business writing to stdout. The reading and the classification are
 * library; the presentation is this tool's.
 *
 * The detail block is also where the operational advice lives — which readings mean a card
 * reset and which mean "run probe" — and that advice is addressed to a person at a terminal,
 * not to a caller.
 *
 * @status Runs on the p150a test card.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_CORE_STATUS_PRINT_H
#define FT_TT_CORE_STATUS_PRINT_H

#include "ft_tt_core_status.h"

/** The header the one-line form is columned under. */
void ft_tt_core_status_print_header(void);

/** One line per core: index, coordinate, per-processor reset state, verdict. */
void ft_tt_core_status_print_line(const ft_tt_core_status *status);

/** Everything read, decoded, several lines. For a single core or a short list. */
void ft_tt_core_status_print_detail(const ft_tt_core_status *status);

#endif /* FT_TT_CORE_STATUS_PRINT_H */
