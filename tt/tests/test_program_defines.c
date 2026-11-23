/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_program_defines.c
 * @brief Off-device gate for `ft_tt_program_defines_contain()` and for the SHIPPING blob's
 *   recorded `defines=`, the evidence the host's DST-section interlock keys on.
 *
 *   The build records a program's `-D` list COMMA-joined (`-DA=1,-DB=1`). A separator rule
 *   that knows only spaces reads every multi-define record as one token, so `rtriad_mixw`
 *   (three defines) reads as NOT SyncFull and the host refuses it on the eight-tile group.
 *   This test pins the token rule on hand-written strings AND on the real records, so a
 *   regression shows up here rather than as a refused artifact on the card.
 *
 *   Compiled against the SHIPPING `ft_tt_programs.h` and checks that it was: the record
 *   arm is about what this tree actually embeds, not about a fixture.
 *
 * Run: ./build-programblob/test_program_defines
 *
 * @status Green on the Mac; seen RED with the space/tab-only separator rule.
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_memmap.h"
#include "ft_tt_program_blob.h"
#include "ft_tt_program_resolve.h"
#include "ft_tt_test_report.h"

#define SYNC_FULL "-DFT_TT_MATMUL_DST_SYNC=1"

static void check_contains(const char *defines, const char *wanted, int expected)
{
    const int got = ft_tt_program_defines_contain(defines, wanted);
    char what[160];

    snprintf(what, sizeof(what), "\"%s\" %s %s", defines, expected ? "has" : "lacks", wanted);
    ft_tt_test_check_fmt(what, got == expected, "got %d", got);
}

static void check_record(const char *program, const char *wanted, int expected)
{
    char defines[512];
    char what[160];
    const int rc = ft_tt_program_recorded_defines(program, NULL, FT_TT_PROC_MATH0, defines,
                                                  sizeof(defines));

    snprintf(what, sizeof(what), "%s's MATH0 record is readable", program);
    ft_tt_test_check_fmt(what, rc == FT_TT_OK, "rc %d (%s)", rc, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;
    snprintf(what, sizeof(what), "%s %s %s", program, expected ? "records" : "does not record",
             wanted);
    ft_tt_test_check_fmt(what, ft_tt_program_defines_contain(defines, wanted) == expected,
                         "defines=%s", defines);
}

int main(void)
{
    const char *header = ft_tt_program_blob_generated_header();

    ft_tt_test_step("the token rule on hand-written strings");
    check_contains(SYNC_FULL, SYNC_FULL, 1);
    check_contains(SYNC_FULL ",-DFT_RTRIAD_MIXW=1", SYNC_FULL, 1);
    check_contains("-DFT_RTRIAD_MIXW=1," SYNC_FULL, SYNC_FULL, 1);
    check_contains("-DX=2," SYNC_FULL ",-DY=3", SYNC_FULL, 1);
    check_contains("-DX=2 " SYNC_FULL "\t-DY=3", SYNC_FULL, 1);
    check_contains("-DFT_TT_MATMUL_DST_SYNC=11", SYNC_FULL, 0);
    check_contains("-DFT_TT_MATMUL_DST_SYNC=11,-DFT_RTRIAD_MIXW=1", SYNC_FULL, 0);
    check_contains("-DX=2,-DFT_TT_MATMUL_DST_SYNC=11", SYNC_FULL, 0);
    check_contains("-DX=2,X" SYNC_FULL, SYNC_FULL, 0);
    check_contains("", SYNC_FULL, 0);
    check_contains(SYNC_FULL, "", 0);

    ft_tt_test_step("the shipping blob's records");
    ft_tt_test_check_fmt("compiled against the shipping header",
                         strcmp(header, "ft_tt_programs.h") == 0, "header is %s", header);
    check_record("rtriad_mixw", SYNC_FULL, 1);
    check_record("rtriad_mixw", "-DFT_RTRIAD_MIXW=1", 1);
    check_record("rtriad_mixw", "-DFT_RTRIAD_MIXW_ALT_TILE_BYTES=576", 1);
    check_record("rtriad_bfp8", SYNC_FULL, 1);
    check_record("rtriad_bfp8", "-DFT_RTRIAD_MIXW=1", 0);
    check_record("matmul_bfp4", SYNC_FULL, 0);

    return ft_tt_test_finish();
}
