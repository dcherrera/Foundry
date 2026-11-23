/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_q2_0_half_round.c — exhaustive host check of the Q2_0 activation-scale rounding in
 * cuda_kernels/matvec_kquants.cu (fc_q2_0_round_via_half): float -> binary16 RNE -> float
 * for every non-negative binary32 up to +Inf, against the compiler's _Float16 conversion.
 *
 * The function body below mirrors the device helper line for line; keep them in sync.
 * --mutate breaks the tie rule (round half up) to show the check can fail.
 *
 * Build/run: make test-q2-0-half-round (no -ffast-math: it would fold the subnormal add).
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int mutate_ties = 0;

static float bits_to_float(uint32_t bits) {
    float f = 0.0f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint32_t float_to_bits(float f) {
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static float round_via_half(float v) {
    uint32_t bits = float_to_bits(v);

    if (bits >= 0x477FF000u)
        return bits_to_float(0x7F800000u);
    if (bits < 0x38800000u)
        return (v + 0.5f) - 0.5f;
    bits += 0x0FFFu + (mutate_ties ? 1u : ((bits >> 13) & 1u));
    return bits_to_float(bits & ~0x1FFFu);
}

int main(int argc, char **argv) {
    unsigned long long mismatches = 0;

    mutate_ties = argc > 1 && strcmp(argv[1], "--mutate") == 0;
    for (uint32_t bits = 0; bits <= 0x7F800000u; bits++) {
        const float v = bits_to_float(bits);
        const float want = (float)(_Float16)v;
        if (float_to_bits(round_via_half(v)) != float_to_bits(want) && mismatches++ < 5)
            printf("mismatch at 0x%08X\n", bits);
    }
    printf("%llu mismatches over all non-negative binary32 values up to +Inf\n", mismatches);
    return mismatches != 0;
}
