/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_nx_staging_equivalence.c
 * @brief HOST-ONLY, NO GPU. Proves the two expansions of the batched matvec
 *   activation-staging macro (matvec_nx_staging.cuh: FC_NX_STAGE_X_FLAT, the
 *   shipped form, and FC_NX_STAGE_X_NESTED, the division-free rewrite) write the
 *   same values to the same shared addresses and issue the same global loads —
 *   for every (cols, n, NC) shape the _nx kernels are launched with, not just
 *   the deployed shapes.
 *
 * WHY A HOST TEST. The macro's only device-specific pieces are `threadIdx.x` and
 * `__ldg`, both trivially stubbed, so the *indexing* — which is the entire
 * content of the change — is provable on a laptop in one second. That closes the
 * "correct, or declines, for every shape" question without taking a die. NC is
 * passed as a plain int here where the kernels instantiate it as a template
 * parameter; the macro uses it only as a value, so the arithmetic under test is
 * identical either way.
 *
 * WHAT IT ASSERTS, per shape and per chunk, over a simulated 256-thread block:
 *   1. the staged shared buffer is byte-identical between the two forms
 *      (prefilled 0x5A, so an address one form leaves unwritten cannot pass);
 *   2. the number of __ldg calls is identical AND equals nc * cend, so neither
 *      form adds, drops or duplicates a global load;
 *   3. therefore every address in [0, NC) x [0, cend) is written exactly once by
 *      exactly one thread in every form, which is what makes the change of write
 *      ORDER unobservable across the caller's __syncthreads.
 *
 * ALL THREE forms are covered: FLAT (shipped), NESTED (column outer) and EOUTER
 * (element outer, columns unrolled — the measured-fastest one).
 *
 * Build and run (no CUDA needed, runs on the Mac):
 *   cc -std=c99 -O1 -Wall -Wextra -I. tests/test_nx_staging_equivalence.c \
 *      -o /tmp/test_nx_staging_equivalence && /tmp/test_nx_staging_equivalence
 *
 * PROVED ABLE TO FAIL, 2026-08-10: with the nested loop's `e` stride changed from
 * 256 to 255 it exits 1 with 0/342 shapes passing; with the zero-fill else branch
 * deleted it exits 1 with 120/342 passing and 1,332 mismatching chunks (exactly
 * the shapes where a partial column tile makes nc < NC).
 *
 * @status 342/342 (cols, n, NC) combinations identical for BOTH division-free forms,
 *   load counts exact.
 * @issues None
 * @todo None
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Device stubs — the only two things the macro needs from CUDA. */
static struct { int x; } threadIdx;
static long g_ldg_calls;

static float fc_nx_test_ldg(const float *p)
{
    g_ldg_calls++;
    return *p;
}
#define __ldg(p) fc_nx_test_ldg(p)

#include "matvec_nx_staging.cuh"

static void stage_flat(float *sh_x, const float *x, int col0, int cols,
                       int cbase, int cend, int nc_width, int nc, int tid)
{
    threadIdx.x = tid;
    FC_NX_STAGE_X_FLAT(sh_x, x, col0, cols, cbase, cend, nc_width, nc,
                       FC_NX_STAGE_KEEP);
}

static void stage_nested(float *sh_x, const float *x, int col0, int cols,
                         int cbase, int cend, int nc_width, int nc, int tid)
{
    threadIdx.x = tid;
    FC_NX_STAGE_X_NESTED(sh_x, x, col0, cols, cbase, cend, nc_width, nc,
                         FC_NX_STAGE_KEEP);
}

static void stage_eouter(float *sh_x, const float *x, int col0, int cols,
                         int cbase, int cend, int nc_width, int nc, int tid)
{
    threadIdx.x = tid;
    FC_NX_STAGE_X_EOUTER(sh_x, x, col0, cols, cbase, cend, nc_width, nc,
                         FC_NX_STAGE_KEEP);
}

/* One (cols, n, NC) shape: walk every column tile and every column chunk the
 * kernel would, and compare the two staging forms at each. 0 = identical. */
static int compare_shape(int cols, int n, int nc_width)
{
    int sh_floats = nc_width * FC_NX_CHUNK;
    float *flat = malloc((size_t)sh_floats * sizeof(float));
    float *nested = malloc((size_t)sh_floats * sizeof(float));
    float *eouter = malloc((size_t)sh_floats * sizeof(float));
    size_t n_x = (size_t)n * (size_t)cols;
    float *x = malloc(n_x * sizeof(float));
    int bad = 0;

    if (!flat || !nested || !eouter || !x) {
        printf("host alloc failed\n");
        exit(1);
    }
    for (size_t i = 0; i < n_x; i++)
        x[i] = (float)(i % 9973) * 0.001f - 4.0f;

    for (int col0 = 0; col0 < n; col0 += nc_width) {
        int nc = n - col0;
        if (nc > nc_width)
            nc = nc_width;
        for (int cbase = 0; cbase < cols; cbase += FC_NX_CHUNK) {
            int cend = cols - cbase < FC_NX_CHUNK ? cols - cbase : FC_NX_CHUNK;
            memset(flat, 0x5A, (size_t)sh_floats * sizeof(float));
            memset(nested, 0x5A, (size_t)sh_floats * sizeof(float));
            memset(eouter, 0x5A, (size_t)sh_floats * sizeof(float));

            long l0 = g_ldg_calls;
            for (int t = 0; t < 256; t++)
                stage_flat(flat, x, col0, cols, cbase, cend, nc_width, nc, t);
            long ldg_flat = g_ldg_calls - l0;
            for (int t = 0; t < 256; t++)
                stage_nested(nested, x, col0, cols, cbase, cend, nc_width, nc, t);
            long ldg_nested = g_ldg_calls - l0 - ldg_flat;
            for (int t = 0; t < 256; t++)
                stage_eouter(eouter, x, col0, cols, cbase, cend, nc_width, nc, t);
            long ldg_eouter = g_ldg_calls - l0 - ldg_flat - ldg_nested;

            if (memcmp(flat, eouter, (size_t)sh_floats * sizeof(float)) != 0) {
                printf("MISMATCH  form=eouter NC=%d cols=%d n=%d col0=%d cbase=%d "
                       "cend=%d nc=%d\n", nc_width, cols, n, col0, cbase, cend, nc);
                bad = 1;
            }
            if (ldg_eouter != (long)nc * cend) {
                printf("LDG COUNT form=eouter NC=%d cols=%d cend=%d nc=%d "
                       "eouter=%ld expected=%ld\n", nc_width, cols, cend, nc,
                       ldg_eouter, (long)nc * cend);
                bad = 1;
            }
            if (memcmp(flat, nested, (size_t)sh_floats * sizeof(float)) != 0) {
                printf("MISMATCH  NC=%d cols=%d n=%d col0=%d cbase=%d cend=%d "
                       "nc=%d\n", nc_width, cols, n, col0, cbase, cend, nc);
                bad = 1;
            }
            if (ldg_flat != ldg_nested || ldg_flat != (long)nc * cend) {
                printf("LDG COUNT NC=%d cols=%d cend=%d nc=%d flat=%ld "
                       "nested=%ld expected=%ld\n", nc_width, cols, cend, nc,
                       ldg_flat, ldg_nested, (long)nc * cend);
                bad = 1;
            }
        }
    }

    free(flat);
    free(nested);
    free(eouter);
    free(x);
    return bad;
}

int main(void)
{
    /* cols: every dense inner dimension saber carries (512, 2048, 4096), a
     * non-multiple of FC_NX_CHUNK (2560 -> a full chunk plus a partial one, so
     * cend < FC_NX_CHUNK is exercised), and a wide one.
     * n: 1..256 — every batched-decode width, the 128-token prefill chunk, the
     * n=12 tile-quantisation case, and values that leave a partial column tile
     * (nc < NC) so the zero-fill path runs. */
    static const int cols_list[] = { 512, 1024, 2048, 2560, 4096, 8192 };
    static const int n_list[] = { 1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 31, 32,
                                  33, 63, 127, 128, 129, 256 };
    static const int nc_list[] = { 2, 4, 8 };
    int shapes = 0, bad = 0;

    for (size_t ci = 0; ci < sizeof(cols_list) / sizeof(*cols_list); ci++) {
        for (size_t ni = 0; ni < sizeof(n_list) / sizeof(*n_list); ni++) {
            for (size_t wi = 0; wi < sizeof(nc_list) / sizeof(*nc_list); wi++) {
                bad += compare_shape(cols_list[ci], n_list[ni], nc_list[wi]);
                shapes++;
            }
        }
    }

    printf("%d/%d (cols, n, NC) shape combinations identical\n",
           shapes - bad, shapes);
    if (bad) {
        printf("VERDICT  FAIL — the nested staging is NOT equivalent; do not "
               "ship it\n");
        return 1;
    }
    printf("VERDICT  flat, nested and eouter staging are equivalent on every "
           "shape tested\n");
    return 0;
}
