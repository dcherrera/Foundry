/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_test_files.h
 * @brief  Test helpers (header-only, one test binary each): a seeded Gaussian, file compare /
 *         copy / flip, a GGUF v3 writer, and ggml's reference Q8_0 quantizer computed in floats
 *         (independent of the codec's integer scale code).
 */
#ifndef FOUNDRY_INGOT_TEST_FILES_H
#define FOUNDRY_INGOT_TEST_FILES_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_bf16.h"

static uint64_t test_rng = 0x1234567887654321ull;

static inline uint64_t test_next(void) {
    test_rng ^= test_rng << 13;
    test_rng ^= test_rng >> 7;
    test_rng ^= test_rng << 17;
    return test_rng;
}

static inline double test_gauss(void) {
    double u = ((double)(test_next() >> 11) + 0.5) / 9007199254740992.0;
    double v = ((double)(test_next() >> 11) + 0.5) / 9007199254740992.0;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static inline long test_file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

static inline int test_files_equal(const char *a, const char *b) {
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    int eq = fa && fb;
    unsigned char ba[1 << 16], bb[1 << 16];
    while (eq) {
        size_t na = fread(ba, 1, sizeof(ba), fa), nb = fread(bb, 1, sizeof(bb), fb);
        eq = na == nb && !memcmp(ba, bb, na);
        if (na < sizeof(ba))
            break;
    }
    if (fa)
        fclose(fa);
    if (fb)
        fclose(fb);
    return eq;
}

/* Copy src to dst, XOR-ing byte `at` with `flip` (flip 0 = plain copy); keep only `keep` bytes
 * when keep >= 0 (truncation). */
static inline int test_copy_flipped(const char *src, const char *dst, long at, unsigned char flip,
                                    long keep) {
    long size = test_file_size(src);
    FILE *f = fopen(src, "rb");
    if (!f || size < 0)
        return -1;
    unsigned char *buf = malloc((size_t)size + 1);
    int ok = buf && fread(buf, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    if (ok && at >= 0 && at < size)
        buf[at] ^= flip;
    if (ok) {
        f = fopen(dst, "wb");
        size_t n = (size_t)(keep >= 0 && keep < size ? keep : size);
        ok = f && fwrite(buf, 1, n, f) == n;
        if (f)
            fclose(f);
    }
    free(buf);
    return ok ? 0 : -1;
}

/* fp16 bits of a float, round to nearest even, computed with float arithmetic. */
static inline uint16_t test_f32_to_f16(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u, mag = x & 0x7FFFFFFFu;
    if (mag >= 0x7F800000u)
        return (uint16_t)(sign | 0x7C00u | (mag > 0x7F800000u ? 0x200u : 0u));
    if (mag >= 0x477FF000u)
        return (uint16_t)(sign | 0x7C00u);
    if (mag < 0x38800000u) {
        float a = 0.0f;
        memcpy(&a, &mag, sizeof(a));
        return (uint16_t)(sign | (uint32_t)lrintf(a * 16777216.0f));
    }
    uint32_t r = mag - 0x38000000u;
    r += 0x0FFFu + ((r >> 13) & 1u);
    return (uint16_t)(sign | (r >> 13));
}

/* ggml quantize_row_q8_0_ref on BF16 weights: 34-byte blocks into out. */
static inline void test_q8_quantize(const uint16_t *w, size_t n, uint8_t *out) {
    for (size_t b = 0; b < n / 32; b++) {
        float amax = 0.0f;
        for (int k = 0; k < 32; k++) {
            float v = fabsf(ingot_bf16_to_f32(w[b * 32 + (size_t)k]));
            amax = v > amax ? v : amax;
        }
        float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint16_t d16 = test_f32_to_f16(d);
        uint8_t *blk = out + b * 34;
        blk[0] = (uint8_t)d16;
        blk[1] = (uint8_t)(d16 >> 8);
        for (int k = 0; k < 32; k++)
            blk[2 + k] = (uint8_t)(int8_t)roundf(ingot_bf16_to_f32(w[b * 32 + (size_t)k]) * id);
    }
}

/* Q8_0 blocks of rows x cols Gaussian weights (BF16-valued, sigma scaled per row). */
static inline uint8_t *test_q8_tensor(size_t rows, size_t cols, double sigma) {
    uint8_t *out = malloc(rows * cols / 32 * 34);
    uint16_t *w = malloc(cols * sizeof(*w));
    for (size_t r = 0; out && w && r < rows; r++) {
        double s = sigma * (0.5 + (double)(test_next() % 1000) / 1000.0);
        for (size_t c = 0; c < cols; c++)
            w[c] = ingot_f32_to_bf16((float)(s * test_gauss()));
        test_q8_quantize(w, cols, out + r * cols / 32 * 34);
    }
    free(w);
    return out;
}

/** One tensor for test_write_gguf: ggml type, GGUF dims (ne0 first), bytes. */
typedef struct {
    const char *name;
    uint32_t type;
    uint32_t ndim;
    uint64_t ne[4];
    const void *data;
    uint64_t size;
} test_gguf_tensor;

static inline void test_put(FILE *f, const void *p, size_t n) { fwrite(p, 1, n, f); }

static inline void test_put_string(FILE *f, const char *s) {
    uint64_t n = strlen(s);
    test_put(f, &n, 8);
    test_put(f, s, (size_t)n);
}

/* GGUF v3: one string KV, the tensors in order, each 32-aligned; `tail` extra bytes at the end
 * (0 for a normal file). */
static inline int test_write_gguf(const char *path, const test_gguf_tensor *t, size_t n,
                                  size_t tail) {
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    uint32_t version = 3, str_type = 8;
    uint64_t count = n, kv = 1, offset = 0;
    test_put(f, "GGUF", 4);
    test_put(f, &version, 4);
    test_put(f, &count, 8);
    test_put(f, &kv, 8);
    test_put_string(f, "general.architecture");
    test_put(f, &str_type, 4);
    test_put_string(f, "ingot-test");
    for (size_t i = 0; i < n; i++) {
        test_put_string(f, t[i].name);
        test_put(f, &t[i].ndim, 4);
        test_put(f, t[i].ne, 8 * t[i].ndim);
        test_put(f, &t[i].type, 4);
        test_put(f, &offset, 8);
        offset += (t[i].size + 31) / 32 * 32;
    }
    static const uint8_t zeros[32] = {0};
    long pos = ftell(f);
    test_put(f, zeros, (size_t)((32 - pos % 32) % 32));
    for (size_t i = 0; i < n; i++) {
        test_put(f, t[i].data, (size_t)t[i].size);
        test_put(f, zeros, (size_t)((32 - t[i].size % 32) % 32));
    }
    for (size_t i = 0; i < tail; i++)
        fputc((int)(i * 7 + 1), f);
    return fclose(f) ? -1 : 0;
}

#endif /* FOUNDRY_INGOT_TEST_FILES_H */
