/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_test_report.c
 * @brief Implementation of the Phase 2 hardware-test transcript.
 *
 * @status In use by every test_phase2_*.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_test_report.h"

#include <stdarg.h>
#include <stdio.h>

static int failure_count;

void ft_tt_test_step(const char *format, ...)
{
    va_list arguments;

    printf("\n");
    va_start(arguments, format);
    vprintf(format, arguments);
    va_end(arguments);
    printf("\n");
    fflush(stdout);
}

void ft_tt_test_check(const char *what, bool passed, const char *detail)
{
    printf("  %-4s  %-52s %s\n", passed ? "ok" : "FAIL", what, detail ? detail : "");
    fflush(stdout);
    if (!passed)
        failure_count++;
}

void ft_tt_test_check_fmt(const char *what, bool passed, const char *format, ...)
{
    char detail[256];
    va_list arguments;

    va_start(arguments, format);
    (void)vsnprintf(detail, sizeof(detail), format, arguments);
    va_end(arguments);
    ft_tt_test_check(what, passed, detail);
}

int ft_tt_test_failure_count(void)
{
    return failure_count;
}

int ft_tt_test_finish(void)
{
    printf("\n%d checks failed\n", failure_count);
    fflush(stdout);
    return failure_count == 0 ? 0 : 1;
}
