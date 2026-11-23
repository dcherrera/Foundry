/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_test_report.h
 * @brief One-line-per-check pass/fail reporting for the on-card Phase 2 checklists.
 *
 *   The hardware tests are read as a transcript by a human afterwards, not parsed, so
 *   every check prints its own line with the value it saw — a failing check that does
 *   not show the number it read is a failure you have to re-run to diagnose, and
 *   re-running costs a card.
 *
 * @status In use by every test_phase2_*.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_TEST_REPORT_H
#define FT_TT_TEST_REPORT_H

#include <stdbool.h>

/** Print a section heading. */
void ft_tt_test_step(const char *format, ...) __attribute__((format(printf, 1, 2)));

/**
 * Record one check. `detail` is what was actually observed and should never be omitted
 * for a check that reads a value.
 */
void ft_tt_test_check(const char *what, bool passed, const char *detail);

/** Same, with the detail formatted inline. */
void ft_tt_test_check_fmt(const char *what, bool passed, const char *format, ...)
    __attribute__((format(printf, 3, 4)));

/** How many checks have failed so far. */
int ft_tt_test_failure_count(void);

/** Print the summary line and return the process exit code (0 when all passed). */
int ft_tt_test_finish(void);

#endif /* FT_TT_TEST_REPORT_H */
