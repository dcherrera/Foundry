/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_pl_lora.c
 * @brief Tests for pl_lora forward/backward with fp64 FD check.
 *
 * @status Experimental
 * @issues None
 * @todo None
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lib/test.h"
#include "potluck/pl_lora.h"
#include "potluck/pl_rng.h"

/* Config */
#define DIN 12
#define DOUT 9
#define RANK 4
#define ALPHA 8
#define SCALE ((float)(ALPHA / RANK))
#define T 5

/* Helper: generate values in [-0.5, 0.5] from pl_rng */
static void fill_rng(float *data, int n, uint64_t seed, int stream)
{
    pl_rng r;
    pl_rng_seed(&r, seed, stream);
    for (int i = 0; i < n; i++) {
        data[i] = (float)(pl_rng_uniform(&r) - 0.5);
    }
}

/* Independent double-precision reference forward:
 * y[t][o] = W0[t][o] + s * Σ_r B[o][r] * Σ_j A[r][j] * x[t][j]
 * where W0 is a fixed [dout][din] base weight (zero here, caller adds base).
 */
static void ref_forward_double(const float *A, const float *B,
                                const float *x, double *y,
                                int nT, int ndin, int ndout, int nrank,
                                double s)
{
    memset(y, 0, (size_t)nT * (size_t)ndout * sizeof(double));
    for (int t = 0; t < nT; t++) {
        for (int o = 0; o < ndout; o++) {
            double sum = 0.0;
            for (int r = 0; r < nrank; r++) {
                double hr = 0.0;
                for (int j = 0; j < ndin; j++) {
                    hr += A[r * ndin + j] * x[t * ndin + j];
                }
                sum += B[o * nrank + r] * hr;
            }
            y[t * ndout + o] = s * sum;
        }
    }
}

/* Central FD of loss L = Σ_tΣ_o G[t][o]*y[t][o] w.r.t. A[r][j].
 * y[t][o] = s·Σ_r B[o][r]·Σ_k A[r][k]·x[t][k].
 * Only the A[r][j] entry changes, so y[t][o] changes by s·B[o][r]·x[t][j].
 */
static double fd_A(const pl_lora *L, const float *x, const float *G,
                   int r, int j, double h)
{
    float A_orig[RANK * DIN];
    memcpy(A_orig, L->A, sizeof(A_orig));

    double loss_plus = 0.0, loss_minus = 0.0;

    /* +h */
    L->A[r * DIN + j] += (float)h;
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < DOUT; o++) {
            double y = 0.0;
            for (int rr = 0; rr < RANK; rr++) {
                double hr = 0.0;
                for (int jj = 0; jj < DIN; jj++) {
                    hr += L->A[rr * DIN + jj] * x[t * DIN + jj];
                }
                y += L->B[o * RANK + rr] * hr;
            }
            loss_plus += G[t * DOUT + o] * L->scale * y;
        }
    }

    /* -h */
    L->A[r * DIN + j] -= (float)(2.0 * h);
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < DOUT; o++) {
            double y = 0.0;
            for (int rr = 0; rr < RANK; rr++) {
                double hr = 0.0;
                for (int jj = 0; jj < DIN; jj++) {
                    hr += L->A[rr * DIN + jj] * x[t * DIN + jj];
                }
                y += L->B[o * RANK + rr] * hr;
            }
            loss_minus += G[t * DOUT + o] * L->scale * y;
        }
    }

    /* Restore */
    L->A[r * DIN + j] = A_orig[r * DIN + j];

    return (loss_plus - loss_minus) / (2.0 * h);
}

/* Central FD of loss w.r.t. B[o][r] */
static double fd_B(const pl_lora *L, const float *x, const float *G,
                   int o, int r, double h)
{
    float B_orig[DOUT * RANK];
    memcpy(B_orig, L->B, sizeof(B_orig));

    double loss_plus = 0.0, loss_minus = 0.0;

    /* +h */
    L->B[o * RANK + r] += (float)h;
    for (int t = 0; t < T; t++) {
        for (int oo = 0; oo < DOUT; oo++) {
            double y = 0.0;
            for (int rr = 0; rr < RANK; rr++) {
                double hr = 0.0;
                for (int jj = 0; jj < DIN; jj++) {
                    hr += L->A[rr * DIN + jj] * x[t * DIN + jj];
                }
                y += L->B[oo * RANK + rr] * hr;
            }
            loss_plus += G[t * DOUT + oo] * L->scale * y;
        }
    }

    /* -h */
    L->B[o * RANK + r] -= (float)(2.0 * h);
    for (int t = 0; t < T; t++) {
        for (int oo = 0; oo < DOUT; oo++) {
            double y = 0.0;
            for (int rr = 0; rr < RANK; rr++) {
                double hr = 0.0;
                for (int jj = 0; jj < DIN; jj++) {
                    hr += L->A[rr * DIN + jj] * x[t * DIN + jj];
                }
                y += L->B[oo * RANK + rr] * hr;
            }
            loss_minus += G[t * DOUT + oo] * L->scale * y;
        }
    }

    /* Restore */
    L->B[o * RANK + r] = B_orig[o * RANK + r];

    return (loss_plus - loss_minus) / (2.0 * h);
}

/* Central FD of loss w.r.t. x[t][j] */
static double fd_x(const pl_lora *L, const float *x, const float *G,
                   int t, int j, double h)
{
    float x_orig[T * DIN];
    memcpy(x_orig, x, sizeof(x_orig));

    double loss_plus = 0.0, loss_minus = 0.0;
    float *xp = (float *)malloc((size_t)T * (size_t)DIN * sizeof(float));
    float *xm = (float *)malloc((size_t)T * (size_t)DIN * sizeof(float));
    if (!xp || !xm) { free(xp); free(xm); return 0.0; }

    /* +h */
    memcpy(xp, x_orig, sizeof(x_orig));
    xp[t * DIN + j] += (float)h;
    for (int tt = 0; tt < T; tt++) {
        for (int oo = 0; oo < DOUT; oo++) {
            double y = 0.0;
            for (int rr = 0; rr < RANK; rr++) {
                double hr = 0.0;
                for (int jj = 0; jj < DIN; jj++) {
                    hr += L->A[rr * DIN + jj] * xp[tt * DIN + jj];
                }
                y += L->B[oo * RANK + rr] * hr;
            }
            loss_plus += G[tt * DOUT + oo] * L->scale * y;
        }
    }

    /* -h */
    memcpy(xm, x_orig, sizeof(x_orig));
    xm[t * DIN + j] -= (float)(2.0 * h);
    for (int tt = 0; tt < T; tt++) {
        for (int oo = 0; oo < DOUT; oo++) {
            double y = 0.0;
            for (int rr = 0; rr < RANK; rr++) {
                double hr = 0.0;
                for (int jj = 0; jj < DIN; jj++) {
                    hr += L->A[rr * DIN + jj] * xm[tt * DIN + jj];
                }
                y += L->B[oo * RANK + rr] * hr;
            }
            loss_minus += G[tt * DOUT + oo] * L->scale * y;
        }
    }

    free(xp); free(xm);
    return (loss_plus - loss_minus) / (2.0 * h);
}

/* l1: backward FD check for A, B, x */
static void test_l1_backward_fd(void)
{
    pl_lora L;
    pl_lora_init(&L, DIN, DOUT, RANK, ALPHA, 11, 0);

    float x[T * DIN], G[T * DOUT];
    fill_rng(x, T * DIN, 11, 100);
    fill_rng(G, T * DOUT, 11, 101);

    double h = 1e-4;

    /* Check A: perturb each A[r][j] */
    for (int r = 0; r < RANK; r++) {
        for (int j = 0; j < DIN; j++) {
            double fd = fd_A(&L, x, G, r, j, h);
            pl_lora_zero_grad(&L);
            pl_lora_backward(&L, x, G, NULL, T);
            double g = L.dA[r * DIN + j];

            double maxv = fmax(fabs(fd), fabs(g));
            double rel = maxv > 1e-12 ? fabs(g - fd) / maxv : fabs(g - fd);
            ASSERT_TRUE(rel < 1e-4);
        }
    }

    /* Check B: perturb each B[o][r] */
    for (int o = 0; o < DOUT; o++) {
        for (int r = 0; r < RANK; r++) {
            double fd = fd_B(&L, x, G, o, r, h);
            pl_lora_zero_grad(&L);
            pl_lora_backward(&L, x, G, NULL, T);
            double g = L.dB[o * RANK + r];

            double maxv = fmax(fabs(fd), fabs(g));
            double rel = maxv > 1e-12 ? fabs(g - fd) / maxv : fabs(g - fd);
            ASSERT_TRUE(rel < 1e-4);
        }
    }

    /* Check x: perturb each x[t][j] */
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < DIN; j++) {
            double fd = fd_x(&L, x, G, t, j, h);
            pl_lora_zero_grad(&L);
            pl_lora_backward(&L, x, G, NULL, T);
            /* dx is NULL in backward, so we check the gradient via
             * the backward result stored in a temporary buffer */
            float dx[T * DIN];
            memset(dx, 0, sizeof(dx));
            pl_lora_zero_grad(&L);
            pl_lora_backward(&L, x, G, dx, T);
            double g = dx[t * DIN + j];

            double maxv = fmax(fabs(fd), fabs(g));
            double rel = maxv > 1e-12 ? fabs(g - fd) / maxv : fabs(g - fd);
            ASSERT_TRUE(rel < 1e-4);
        }
    }

    pl_lora_free(&L);
}

/* l2: forward comparison with double reference */
static void test_l2_forward(void)
{
    pl_lora L;
    pl_lora_init(&L, DIN, DOUT, RANK, ALPHA, 11, 0);

    float x[T * DIN];
    fill_rng(x, T * DIN, 11, 100);

    float y[T * DOUT];
    memset(y, 0, sizeof(y));
    pl_lora_forward(&L, x, y, T);

    /* Double reference */
    double yref[T * DOUT];
    ref_forward_double(L.A, L.B, x, yref, T, DIN, DOUT, RANK, (double)SCALE);

    for (int i = 0; i < T * DOUT; i++) {
        double diff = fabs((double)y[i] - yref[i]);
        ASSERT_TRUE(diff < 1e-5);
    }

    pl_lora_free(&L);
}

/* l3: accumulation test */
static void test_l3_accumulation(void)
{
    pl_lora L;
    pl_lora_init(&L, DIN, DOUT, RANK, ALPHA, 11, 0);

    float x[T * DIN], G[T * DOUT];
    fill_rng(x, T * DIN, 11, 100);
    fill_rng(G, T * DOUT, 11, 101);

    pl_lora_zero_grad(&L);
    pl_lora_backward(&L, x, G, NULL, T);

    /* Save first pass */
    float dA1[RANK * DIN], dB1[DOUT * RANK];
    memcpy(dA1, L.dA, sizeof(dA1));
    memcpy(dB1, L.dB, sizeof(dB1));

    /* Second pass */
    pl_lora_backward(&L, x, G, NULL, T);

    /* Check doubled */
    for (int i = 0; i < RANK * DIN; i++) {
        double expected = 2.0 * dA1[i];
        double rel = fabs(expected) > 1e-12 ? fabs((double)L.dA[i] - expected) / fabs(expected) : 0.0;
        ASSERT_TRUE(rel < 1e-6);
    }
    for (int i = 0; i < DOUT * RANK; i++) {
        double expected = 2.0 * dB1[i];
        double rel = fabs(expected) > 1e-12 ? fabs((double)L.dB[i] - expected) / fabs(expected) : 0.0;
        ASSERT_TRUE(rel < 1e-6);
    }

    /* pl_lora_zero_grad zeroes them */
    pl_lora_zero_grad(&L);
    for (int i = 0; i < RANK * DIN; i++) {
        ASSERT_TRUE(L.dA[i] == 0.0f);
    }
    for (int i = 0; i < DOUT * RANK; i++) {
        ASSERT_TRUE(L.dB[i] == 0.0f);
    }

    pl_lora_free(&L);
}

/* l4: init test */
static void test_l4_init(void)
{
    pl_lora L;
    pl_lora_init(&L, DIN, DOUT, RANK, ALPHA, 11, 0);

    /* B all zero after init */
    for (int i = 0; i < DOUT * RANK; i++) {
        ASSERT_TRUE(L.B[i] == 0.0f);
    }

    /* A equals pl_lora_init_a(11, 0, ...) bit for bit (all rank x din elements) */
    float A_ref[RANK * DIN];
    pl_lora_init_a(11, 0, A_ref, RANK, DIN);
    for (int i = 0; i < RANK * DIN; i++) {
        ASSERT_TRUE(L.A[i] == A_ref[i]);
    }

    pl_lora_free(&L);
}

int main(void)
{
    RUN_TEST(l1_backward_fd);
    RUN_TEST(l2_forward);
    RUN_TEST(l3_accumulation);
    RUN_TEST(l4_init);
    TEST_REPORT();
    return 0;
}