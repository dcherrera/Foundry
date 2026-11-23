/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file calib_nx_stage_poison.c
 * @brief HOST-ONLY, NO GPU. Calibrates the negative control of
 *   tests/bench_matvec_nx_staging.cu: if every staged activation is scaled by
 *   (1 + 2^-23), how many of fc_matvec_q3k_nx8's fp32 outputs change their BIT
 *   PATTERN? Answers the one thing that decides whether `--poison` printing
 *   DIFFER means anything.
 *
 * WHY THIS EXISTS. `--poison` had never been run, so "bit-exact by construction"
 * rested on construction alone — and this project has shipped six green-but-
 * vacuous gates in a day. A negative control that *could* pass is worse than
 * none. The poison is deliberately calibrated to 1.2e-07 relative so that any
 * TOLERANCE-based check passes it and only a real bit comparison catches it;
 * that same calibration is what makes "will it actually move a bit?" a fair
 * question, and it cannot be answered by staring at the source.
 *
 * WHAT THIS IS. An exact-structure host replica of the kernel's per-column dot:
 * the magic-nibble Q3_K decode (FC_MAGIC_NIB / FC_CVT_BYTE), the per-lane
 * i = 0..7 fmaf chain, the per-super-block d scale, and the 5-step __shfl_down
 * butterfly — fed the benchmark's own deterministic operand generators
 * (fill_q3k_blocks seed 77, fill_deterministic seed 88) at the default shape.
 * Both arms run through the same replica, so the COMPARISON is exact even where
 * the replica is not a bit-exact oracle for the GPU.
 *
 * TWO-SIDED RESULT, MEASURED 2026-08-10 on 65,536 outputs (rows 0..511 x all 128
 * columns of the 8192 x 2048 default shape):
 *
 *     bit-DIFFER   55,608 / 65,536 = 84.85 %
 *     ulp distance 1: 13,213   2: 14,710   >2: 27,685
 *     relative      5.96e-08 min, 2.04e-02 max
 *
 * So `--poison` must print DIFFER with ~85 % of elements differing, i.e. about
 * 890,000 of the harness's 1,048,576 outputs. NOT 100 %: ~15 % of outputs are a
 * heavily cancelled sum of 2048 signed terms, where a 1-ulp shift of every input
 * still rounds back to the same fp32 result. That is exactly why a SINGLE-element
 * poison was rejected in design, and it is why the pass criterion is "any
 * element differs", never "all elements differ".
 *
 * PER-ELEMENT THE POISON CANNOT ROUND AWAY. For x = m * 2^E with m in [1, 2),
 * ulp(x) = 2^-23 * 2^E and x * 2^-23 = m * ulp(x) with m in [1, 2), so
 * fl(x * (1 + 2^-23)) is x + 1 ulp for m < 1.5 and x + 2 ulp for m >= 1.5 —
 * never x, for every finite normal x. Every staged element really changes; only
 * the summation can hide it, and it hides it 15 % of the time.
 *
 * Build and run (no CUDA needed, runs on the Mac in ~2 s):
 *   cc -std=c99 -O2 -ffp-contract=off -Wall -Wextra \
 *      tests/calib_nx_stage_poison.c -o /tmp/calib_nx_stage_poison -lm
 *   /tmp/calib_nx_stage_poison
 *
 * @status Calibration run and recorded above. -ffp-contract=off is REQUIRED: the
 *   replica must round `x * POISON` once, like __fmul_rn.
 * @issues The replica computes `p += d * a` as a multiply then an add, where
 *   ptxas may contract the kernel's identical source line into an FMA. That makes
 *   the replica a faithful-structure model, not a bit-exact oracle — it shifts
 *   both arms equally and so cannot change the DIFFER fraction materially, but do
 *   not quote these bit patterns as the GPU's.
 * @todo None
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RBLOCK_BYTES 152                          /* FC_Q3K_RBLOCK_BYTES */
#define SIM_COLS     2048                         /* bench default cols */
#define SIM_NCOL     128                          /* bench default n */
#define SIM_ROWS     512                          /* subset of the bench's 8192 */
#define POISON_SCALE 1.00000011920928955078125f    /* 1 + 2^-23 */

static float bits_to_float(unsigned u)
{
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static unsigned float_to_bits(float f)
{
    unsigned u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

/* FC_MAGIC_NIB: exact float (field - bias) from the 4-bit field at bit p. */
static float magic_nib(unsigned w, int p, int bias)
{
    return fmaf(bits_to_float(0x4B000000u | (w & (0xFu << p))),
                1.0f / (float)(1u << p),
                -(float)((1u << (23 - p)) + (unsigned)bias));
}

/* bench_matvec_nx_staging.cu fill_deterministic, verbatim. */
static void fill_activations(float *p, size_t n, unsigned seed, float lo,
                             float hi)
{
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)(st >> 8) / (float)(1u << 24);
        p[i] = lo + u * (hi - lo);
    }
}

/* bench_matvec_nx_staging.cu fill_q3k_blocks, verbatim. */
static void fill_q3k_blocks(unsigned char *buf, size_t nblocks, unsigned seed)
{
    unsigned st = seed | 1u;
    for (size_t b = 0; b < nblocks; b++) {
        unsigned char *blk = buf + b * RBLOCK_BYTES;
        unsigned *qw = (unsigned *)blk;
        for (int i = 0; i < 32; i++) {
            st = st * 1664525u + 1013904223u;
            qw[i] = st;
        }
        for (int i = 0; i < 16; i++) {
            st = st * 1664525u + 1013904223u;
            blk[128 + i] = (unsigned char)(st >> 26);
        }
        st = st * 1664525u + 1013904223u;
        float d = 1.0e-4f + 1.0e-4f * (float)(st >> 8) / (float)(1u << 24);
        memcpy(blk + 144, &d, sizeof(float));
        memset(blk + 148, 0, 4);
    }
}

/* fc_q3k_warp_sum: 5 __shfl_down steps. Kepler returns the lane's own value
 * when the source lane is out of range, which only affects lanes >= 16 — lane 0
 * carries the real tree and is the only one written out. */
static float warp_sum_lane0(const float *v)
{
    float cur[32], next[32];

    memcpy(cur, v, sizeof(cur));
    for (int d = 16; d > 0; d >>= 1) {
        for (int l = 0; l < 32; l++)
            next[l] = cur[l] + ((l + d < 32) ? cur[l + d] : cur[l]);
        memcpy(cur, next, sizeof(cur));
    }
    return cur[0];
}

int main(void)
{
    const int nsb = SIM_COLS / 256;                 /* super-blocks per row */
    const int row_stride = nsb * RBLOCK_BYTES;
    size_t nblocks = (size_t)SIM_ROWS * (size_t)nsb;
    unsigned char *w = malloc(nblocks * RBLOCK_BYTES);
    float *x = malloc((size_t)SIM_NCOL * SIM_COLS * sizeof(float));
    /* term[sb][i][lane] = fs * fq — the weight decode, shared by all columns. */
    static float term[SIM_COLS / 256][8][32];
    static float dsb[SIM_COLS / 256];
    long differ = 0, total = 0, ulp1 = 0, ulp2 = 0, ulp_big = 0;
    double rel_min = 1.0, rel_max = 0.0;

    if (!w || !x) {
        printf("host alloc failed\n");
        return 1;
    }
    fill_q3k_blocks(w, nblocks, 77u);
    fill_activations(x, (size_t)SIM_NCOL * SIM_COLS, 88u, -1.0f, 1.0f);

    for (int row = 0; row < SIM_ROWS; row++) {
        const unsigned char *base = w + (size_t)row * (size_t)row_stride;

        for (int sb = 0; sb < nsb; sb++) {
            const unsigned char *blk = base + (size_t)sb * RBLOCK_BYTES;
            memcpy(&dsb[sb], blk + 144, sizeof(float));
            for (int lane = 0; lane < 32; lane++) {
                unsigned wv, scx, scy;
                memcpy(&wv, blk + 4 * lane, sizeof(wv));
                memcpy(&scx, blk + 128 + 8 * (lane >> 4), sizeof(scx));
                memcpy(&scy, blk + 128 + 8 * (lane >> 4) + 4, sizeof(scy));
                unsigned wh = wv >> 16;
                for (int i = 0; i < 8; i++) {
                    int qp = (i <= 4) ? 4 * i : 4 * i - 16;
                    unsigned q = (i <= 4) ? wv : wh;
                    unsigned s = (i < 4) ? scx : scy;
                    float fq = magic_nib(q, qp, 4);
                    float fs = (float)((s >> (8 * (i & 3))) & 0xFFu) - 32.0f;
                    term[sb][i][lane] = fs * fq;
                }
            }
        }

        for (int col = 0; col < SIM_NCOL; col++) {
            const float *xc = x + (size_t)col * SIM_COLS;
            float clean[32], poisoned[32];

            for (int l = 0; l < 32; l++) {
                clean[l] = 0.0f;
                poisoned[l] = 0.0f;
            }
            for (int sb = 0; sb < nsb; sb++) {
                float ac[32], ap[32];
                for (int l = 0; l < 32; l++) {
                    ac[l] = 0.0f;
                    ap[l] = 0.0f;
                }
                for (int i = 0; i < 8; i++) {
                    for (int l = 0; l < 32; l++) {
                        float xv = xc[sb * 256 + 32 * i + l];
                        ac[l] = fmaf(term[sb][i][l], xv, ac[l]);
                        ap[l] = fmaf(term[sb][i][l], xv * POISON_SCALE, ap[l]);
                    }
                }
                for (int l = 0; l < 32; l++) {
                    clean[l] += dsb[sb] * ac[l];
                    poisoned[l] += dsb[sb] * ap[l];
                }
            }

            float rc = warp_sum_lane0(clean);
            float rp = warp_sum_lane0(poisoned);
            total++;
            if (float_to_bits(rc) != float_to_bits(rp)) {
                long du = labs((long)float_to_bits(rc) -
                               (long)float_to_bits(rp));
                double rel = fabs((double)rp - (double)rc) / fabs((double)rc);
                differ++;
                if (du == 1)
                    ulp1++;
                else if (du == 2)
                    ulp2++;
                else
                    ulp_big++;
                if (rel > rel_max)
                    rel_max = rel;
                if (rel < rel_min)
                    rel_min = rel;
            }
        }
    }

    printf("shape          rows=%d (of 8192) cols=%d n=%d, poison 1+2^-23\n",
           SIM_ROWS, SIM_COLS, SIM_NCOL);
    printf("outputs        %ld\n", total);
    printf("bit-DIFFER     %ld  (%.4f %%)\n", differ,
           100.0 * (double)differ / (double)total);
    printf("ulp distance   1:%ld  2:%ld  >2:%ld\n", ulp1, ulp2, ulp_big);
    printf("relative delta min %.3e  max %.3e\n", rel_min, rel_max);
    printf("VERDICT        %s\n", differ
           ? "the poison moves output bits — --poison MUST print DIFFER"
           : "POISON IS INERT — the negative control cannot fire, recalibrate");
    free(w);
    free(x);
    return differ ? 0 : 1;
}
