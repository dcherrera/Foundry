/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_ssm_conv1d_seq_golden.cu
 * @brief Golden-vector test for fc_ssm_conv1d_silu_seq's k == 4 specialisation
 *   (foundry 2ca7069, the commit that took hist[] out of local memory).
 *
 * THE GOLDEN VECTOR IS THE OLD KERNEL. A host reference cannot decide
 * bit-exactness here — device expf() inside SiLU does not agree with glibc's to
 * the last bit — so the authority is fc_ssm_conv1d_silu_seq_generic below: the
 * pre-2ca7069 body, copied verbatim, compiled into the same module from the same
 * source. If the specialisation is what it claims (the same mul + three fmas in
 * the same u = 0,1,2 order, only with the runtime index gone) the two kernels
 * produce identical BITS in all three outputs: the SiLU stream, the rolled
 * conv_state, and the per-position snaps.
 *
 * A host reference IS still run, to a stated tolerance, because bit-equality
 * between two kernels says nothing if both index the tensors the same wrong way.
 * That check is the one that would catch a layout error; the bit check is the one
 * that would catch a reassociation.
 *
 * NEGATIVE CONTROL (this harness must be shown able to FAIL before green means
 * anything). --poison runs fc_ssm_conv1d_silu_seq_poison: the k == 4 arithmetic
 * with the accumulation ORDER permuted — h0*w0 first, xv*w3 folded in last —
 * which is the same sum to ~1e-7 and passes any tolerance-based check. This test
 * MUST report it as DIFFER. If it reports EXACT, the comparison is not comparing
 * bits and the bit-exactness claim for the shipped kernel is unproven.
 *
 * k == 2 and k == 3 are also run, against the host reference only, as a
 * regression guard on the generic branch the specialisation left behind (no
 * shipped model uses them, and that is exactly why they would rot unnoticed).
 *
 * Build and run on a K80 host, ONE die, no sibling load:
 *   nvcc -arch=sm_37 -ccbin gcc-11 -Wno-deprecated-gpu-targets \
 *        -isystem ../nvcc_shim -O3 -Xptxas -v \
 *        -o test_ssm_conv1d_seq_golden test_ssm_conv1d_seq_golden.cu
 *   CUDA_VISIBLE_DEVICES=<free die> ./test_ssm_conv1d_seq_golden
 *   CUDA_VISIBLE_DEVICES=<free die> ./test_ssm_conv1d_seq_golden --poison
 * Exit 0 = pass, 2 = fail.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "../ssm_gated_delta.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shipped prefill shape for qwen35moe (nex-n2-mini / saber-orch). */
#define GOLD_CONV_DIM 8192
#define GOLD_NPOS     128

/*
 * THE GOLDEN REFERENCE: fc_ssm_conv1d_silu_seq exactly as it stood before the
 * k == 4 specialisation. hist[] here is indexed by loops bounded by the runtime
 * km1, which is why it lands in local memory — that is the defect under test,
 * not a mistake in this copy. Do not "fix" it.
 */
extern "C" __global__ void fc_ssm_conv1d_silu_seq_generic(
        const float *qkv, const float *weight, float *conv_state, float *out,
        int conv_dim, int conv_kernel, int n, float *snaps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim)
        return;

    const int k = conv_kernel;
    const int km1 = k - 1;
    const float *cw = weight + (size_t)c * k;
    float *cs = conv_state + (size_t)c * km1;

    float hist[3];
    for (int t = 0; t < km1; t++)
        hist[t] = cs[t];

    for (int t = 0; t < n; t++) {
        float xv = qkv[(size_t)t * conv_dim + c];
        float acc = xv * cw[km1];
        for (int u = 0; u < km1; u++)
            acc += hist[u] * cw[u];
        out[(size_t)t * conv_dim + c] = fc_ssm_silu(acc);
        for (int u = 0; u < km1 - 1; u++)
            hist[u] = hist[u + 1];
        hist[km1 - 1] = xv;
        if (snaps) {
            float *sn = snaps + ((size_t)t * conv_dim + c) * km1;
            for (int u = 0; u < km1; u++)
                sn[u] = hist[u];
        }
    }

    for (int t = 0; t < km1; t++)
        cs[t] = hist[t];
}

/*
 * NEGATIVE CONTROL. The k == 4 path with ONE defect: the taps are accumulated
 * newest-last instead of newest-first. Mathematically the same sum, numerically
 * within ~1e-7, different bits.
 */
extern "C" __global__ void fc_ssm_conv1d_silu_seq_poison(
        const float *qkv, const float *weight, float *conv_state, float *out,
        int conv_dim, int conv_kernel, int n, float *snaps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim)
        return;

    const float *cw = weight + (size_t)c * conv_kernel;
    float *cs = conv_state + (size_t)c * 3;
    const float w0 = cw[0], w1 = cw[1], w2 = cw[2], w3 = cw[3];
    float h0 = cs[0], h1 = cs[1], h2 = cs[2];

    for (int t = 0; t < n; t++) {
        float xv = qkv[(size_t)t * conv_dim + c];
        float acc = __fmul_rn(h0, w0);
        acc = __fmaf_rn(h1, w1, acc);
        acc = __fmaf_rn(h2, w2, acc);
        acc = __fmaf_rn(xv, w3, acc);
        out[(size_t)t * conv_dim + c] = fc_ssm_silu(acc);
        h0 = h1;
        h1 = h2;
        h2 = xv;
        if (snaps) {
            float *sn = snaps + ((size_t)t * conv_dim + c) * 3;
            sn[0] = h0;
            sn[1] = h1;
            sn[2] = h2;
        }
    }
    cs[0] = h0;
    cs[1] = h1;
    cs[2] = h2;
}

/* ---------------------------------------------------------------- host ---- */

#define CHECK(call)                                                           \
    do {                                                                      \
        cudaError_t rc_ = (call);                                             \
        if (rc_ != cudaSuccess) {                                             \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,    \
                    cudaGetErrorString(rc_));                                 \
            return 1;                                                        \
        }                                                                     \
    } while (0)

static void fill_deterministic(float *p, size_t n, unsigned seed, float lo,
                               float hi) {
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)(st >> 8) / (float)(1u << 24);
        p[i] = lo + u * (hi - lo);
    }
}

/* Host reference in the device's op order (fmaf mirrors __fmaf_rn); only SiLU's
 * expf can legitimately differ, hence a tolerance rather than a bit compare. */
static void conv1d_silu_host(const float *qkv, const float *weight,
                             const float *conv_state0, float *out,
                             float *cs_out, int conv_dim, int k, int n) {
    const int km1 = k - 1;
    for (int c = 0; c < conv_dim; c++) {
        const float *cw = weight + (size_t)c * k;
        float hist[3];
        for (int u = 0; u < km1; u++)
            hist[u] = conv_state0[(size_t)c * km1 + u];
        for (int t = 0; t < n; t++) {
            float xv = qkv[(size_t)t * conv_dim + c];
            float acc = xv * cw[km1];
            for (int u = 0; u < km1; u++)
                acc = fmaf(hist[u], cw[u], acc);
            out[(size_t)t * conv_dim + c] = acc / (1.0f + expf(-acc));
            for (int u = 0; u < km1 - 1; u++)
                hist[u] = hist[u + 1];
            hist[km1 - 1] = xv;
        }
        for (int u = 0; u < km1; u++)
            cs_out[(size_t)c * km1 + u] = hist[u];
    }
}

static size_t count_bit_diffs(const float *a, const float *b, size_t n,
                              double *max_abs) {
    size_t diff = 0;
    *max_abs = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            diff++;
            double d = fabs((double)a[i] - (double)b[i]);
            if (d > *max_abs)
                *max_abs = d;
        }
    }
    return diff;
}

static double max_rel_error(const float *a, const float *b, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double den = fabs((double)b[i]);
        if (den < 1e-6)
            den = 1e-6;
        double r = fabs((double)a[i] - (double)b[i]) / den;
        if (r > worst)
            worst = r;
    }
    return worst;
}

enum gold_arm { GOLD_SHIPPED = 0, GOLD_GENERIC, GOLD_POISON };

int main(int argc, char **argv) {
    int poison_mode = 0;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--poison"))
            poison_mode = 1;

    const int CD = GOLD_CONV_DIM, N = GOLD_NPOS;
    const size_t n_act = (size_t)N * CD;

    int failures = 0;

    /* Widest shape first (k == 4, the shipped case), then the generic-branch
     * regression guard at k == 3 and k == 2. */
    const int kernels[3] = { 4, 3, 2 };
    for (int ki = 0; ki < 3; ki++) {
        const int K = kernels[ki];
        const int KM1 = K - 1;
        const size_t n_w = (size_t)CD * K;
        const size_t n_cs = (size_t)CD * KM1;
        const size_t n_sn = n_act * KM1;

        float *h_x = (float *)malloc(n_act * sizeof(float));
        float *h_w = (float *)malloc(n_w * sizeof(float));
        float *h_cs0 = (float *)malloc(n_cs * sizeof(float));
        float *h_out_a = (float *)malloc(n_act * sizeof(float));
        float *h_out_b = (float *)malloc(n_act * sizeof(float));
        float *h_cs_a = (float *)malloc(n_cs * sizeof(float));
        float *h_cs_b = (float *)malloc(n_cs * sizeof(float));
        float *h_sn_a = (float *)malloc(n_sn * sizeof(float));
        float *h_sn_b = (float *)malloc(n_sn * sizeof(float));
        float *h_out_ref = (float *)malloc(n_act * sizeof(float));
        float *h_cs_ref = (float *)malloc(n_cs * sizeof(float));
        if (!h_x || !h_w || !h_cs0 || !h_out_a || !h_out_b || !h_cs_a ||
            !h_cs_b || !h_sn_a || !h_sn_b || !h_out_ref || !h_cs_ref) {
            fprintf(stderr, "host alloc failed\n");
            return 1;
        }

        fill_deterministic(h_x, n_act, 101u + (unsigned)K, -3.0f, 3.0f);
        fill_deterministic(h_w, n_w, 202u + (unsigned)K, -0.8f, 0.8f);
        fill_deterministic(h_cs0, n_cs, 303u + (unsigned)K, -2.0f, 2.0f);

        float *d_x, *d_w, *d_cs, *d_out, *d_sn;
        CHECK(cudaMalloc(&d_x, n_act * sizeof(float)));
        CHECK(cudaMalloc(&d_w, n_w * sizeof(float)));
        CHECK(cudaMalloc(&d_cs, n_cs * sizeof(float)));
        CHECK(cudaMalloc(&d_out, n_act * sizeof(float)));
        CHECK(cudaMalloc(&d_sn, n_sn * sizeof(float)));
        CHECK(cudaMemcpy(d_x, h_x, n_act * sizeof(float),
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_w, h_w, n_w * sizeof(float),
                         cudaMemcpyHostToDevice));

        const int threads = 256;
        const int blocks = (CD + threads - 1) / threads;

        /* Two device arms. Arm A is always the shipped kernel; arm B is the
         * pre-specialisation golden kernel, or the poison variant when asked. */
        for (int a = 0; a < 2; a++) {
            enum gold_arm arm = (a == 0) ? GOLD_SHIPPED
                              : (poison_mode && K == 4) ? GOLD_POISON
                                                        : GOLD_GENERIC;
            CHECK(cudaMemcpy(d_cs, h_cs0, n_cs * sizeof(float),
                             cudaMemcpyHostToDevice));
            /* 0xCD so an element the kernel never writes cannot pass as a
             * match. */
            CHECK(cudaMemset(d_out, 0xCD, n_act * sizeof(float)));
            CHECK(cudaMemset(d_sn, 0xCD, n_sn * sizeof(float)));

            if (arm == GOLD_SHIPPED)
                fc_ssm_conv1d_silu_seq<<<blocks, threads>>>(
                    d_x, d_w, d_cs, d_out, CD, K, N, d_sn);
            else if (arm == GOLD_GENERIC)
                fc_ssm_conv1d_silu_seq_generic<<<blocks, threads>>>(
                    d_x, d_w, d_cs, d_out, CD, K, N, d_sn);
            else
                fc_ssm_conv1d_silu_seq_poison<<<blocks, threads>>>(
                    d_x, d_w, d_cs, d_out, CD, K, N, d_sn);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());

            float *o = (a == 0) ? h_out_a : h_out_b;
            float *c = (a == 0) ? h_cs_a : h_cs_b;
            float *s = (a == 0) ? h_sn_a : h_sn_b;
            CHECK(cudaMemcpy(o, d_out, n_act * sizeof(float),
                             cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(c, d_cs, n_cs * sizeof(float),
                             cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(s, d_sn, n_sn * sizeof(float),
                             cudaMemcpyDeviceToHost));
        }

        /* Price it, too. The k==4 specialisation's whole claim is that it took
         * ~3 ld.local + 3 st.local per position per thread off the executed
         * path, so the shipped kernel should beat the pre-specialisation body at
         * the SAME k. Nothing but back-to-back launches inside the span. */
        const int reps = 50;
        float t_ms[2] = { 0.0f, 0.0f };
        cudaEvent_t ev0, ev1;
        CHECK(cudaEventCreate(&ev0));
        CHECK(cudaEventCreate(&ev1));
        for (int a = 0; a < 2; a++) {
            CHECK(cudaMemcpy(d_cs, h_cs0, n_cs * sizeof(float),
                             cudaMemcpyHostToDevice));
            for (int w = 0; w < 3; w++) {
                if (a == 0)
                    fc_ssm_conv1d_silu_seq<<<blocks, threads>>>(
                        d_x, d_w, d_cs, d_out, CD, K, N, NULL);
                else
                    fc_ssm_conv1d_silu_seq_generic<<<blocks, threads>>>(
                        d_x, d_w, d_cs, d_out, CD, K, N, NULL);
            }
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < reps; it++) {
                if (a == 0)
                    fc_ssm_conv1d_silu_seq<<<blocks, threads>>>(
                        d_x, d_w, d_cs, d_out, CD, K, N, NULL);
                else
                    fc_ssm_conv1d_silu_seq_generic<<<blocks, threads>>>(
                        d_x, d_w, d_cs, d_out, CD, K, N, NULL);
            }
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            float total = 0.0f;
            CHECK(cudaEventElapsedTime(&total, ev0, ev1));
            t_ms[a] = total / (float)reps;
        }
        CHECK(cudaEventDestroy(ev0));
        CHECK(cudaEventDestroy(ev1));

        conv1d_silu_host(h_x, h_w, h_cs0, h_out_ref, h_cs_ref, CD, K, N);

        double m_out = 0.0, m_cs = 0.0, m_sn = 0.0;
        size_t d_out_n = count_bit_diffs(h_out_a, h_out_b, n_act, &m_out);
        size_t d_cs_n = count_bit_diffs(h_cs_a, h_cs_b, n_cs, &m_cs);
        size_t d_sn_n = count_bit_diffs(h_sn_a, h_sn_b, n_sn, &m_sn);
        double rel_out = max_rel_error(h_out_a, h_out_ref, n_act);
        double rel_cs = max_rel_error(h_cs_a, h_cs_ref, n_cs);

        const char *bname = (poison_mode && K == 4) ? "POISON" : "generic(pre)";
        printf("k=%d conv_dim=%d npos=%d   arm B = %s\n", K, CD, N, bname);
        printf("  out    %-6s  %zu/%zu differ (max |d| %.3e)\n",
               d_out_n ? "DIFFER" : "EXACT", d_out_n, n_act, m_out);
        printf("  state  %-6s  %zu/%zu differ (max |d| %.3e)\n",
               d_cs_n ? "DIFFER" : "EXACT", d_cs_n, n_cs, m_cs);
        printf("  snaps  %-6s  %zu/%zu differ (max |d| %.3e)\n",
               d_sn_n ? "DIFFER" : "EXACT", d_sn_n, n_sn, m_sn);
        printf("  vs host reference: max rel err out %.3e, state %.3e "
               "(state must be 0 -- it is a copy of the input, no expf)\n",
               rel_out, rel_cs);
        printf("  timing shipped %8.4f ms  generic(pre) %8.4f ms  "
               "ratio %.4fx\n", t_ms[0], t_ms[1],
               t_ms[0] > 0.0f ? t_ms[1] / t_ms[0] : 0.0f);

        int ok;
        if (poison_mode && K == 4) {
            ok = (d_out_n != 0);
            printf("  VERDICT  control %s -- the poison arm MUST differ; EXACT "
                   "here means the comparison is not comparing bits\n",
                   ok ? "PASS" : "FAIL");
        } else {
            ok = (d_out_n == 0 && d_cs_n == 0 && d_sn_n == 0 &&
                  rel_out < 1e-5 && rel_cs == 0.0);
            printf("  VERDICT  %s -- shipped must be bit-identical to the "
                   "pre-specialisation kernel in out, state AND snaps\n",
                   ok ? "PASS" : "FAIL");
        }
        if (!ok)
            failures++;
        printf("\n");

        cudaFree(d_x); cudaFree(d_w); cudaFree(d_cs); cudaFree(d_out);
        cudaFree(d_sn);
        free(h_x); free(h_w); free(h_cs0); free(h_out_a); free(h_out_b);
        free(h_cs_a); free(h_cs_b); free(h_sn_a); free(h_sn_b);
        free(h_out_ref); free(h_cs_ref);
    }

    printf("%s (%d of 3 shapes failed)\n", failures ? "FAIL" : "PASS",
           failures);
    return failures ? 2 : 0;
}
