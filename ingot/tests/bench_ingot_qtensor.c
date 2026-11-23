/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   bench_ingot_qtensor.c
 * @brief  Quantized-tensor coder benchmark on a real GGUF file: per ggml type, coded bits/weight
 *         (mode QUANT, tables included) and single-thread decode speed (best of N passes per
 *         tensor, so scheduler noise on a busy machine drops out). Every decode is checked byte
 *         for byte.
 *
 *   bench_ingot_qtensor FILE.gguf [passes (3)] [max tensors per type (all)]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ingot/codec/ingot_qtensor.h"
#include "ingot/container/ingot_layout.h"
#include "ingot/core/ingot_ggml.h"
#include "ingot/core/ingot_input.h"

enum { TYPES = 40 };

typedef struct {
    uint64_t weights, raw_bytes, coded_bytes, tensors;
    double decode_s; /* sum over tensors of the best pass */
} type_stats;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Decode every block of a payload into out; 0 or the first error. */
static int decode_all(ingot_dtype dtype, uint64_t rows, uint64_t cols, const ingot_sink *s,
                      uint8_t *out) {
    ingot_qtensor_decoder *d = NULL;
    uint64_t at = 0;
    int rc = ingot_qtensor_decoder_open(INGOT_MODE_QUANT, dtype, rows, cols, s->mem, s->pos, &d,
                                        &at);
    ingot_block_area area;
    if (!rc)
        rc = ingot_block_area_open(s->mem + at, s->pos - at, ingot_qtensor_decoder_blocks(d),
                                   &area);
    for (size_t b = 0; !rc && b < ingot_qtensor_decoder_blocks(d); b++) {
        ingot_block_view v;
        uint64_t lo = 0, hi = 0;
        rc = ingot_block_view_get(&area, b, &v);
        ingot_qtensor_decoder_range(d, b, &lo, &hi);
        if (!rc)
            rc = ingot_qtensor_decoder_block(d, b, &v, NULL, out + lo);
    }
    ingot_qtensor_decoder_close(d);
    return rc;
}

/* Encode one tensor, decode it `passes` times; adds to st. */
static int bench_tensor(const ingot_layout_tensor *t, const uint8_t *data, int passes,
                        type_stats *st) {
    uint64_t cols = t->shape[t->ndim - 1], rows = ingot_layout_elements(t) / cols;
    ingot_qtensor_plan *plan = NULL;
    ingot_sink s = {.fd = -1};
    int rc = ingot_qtensor_plan_create(INGOT_MODE_QUANT, t->dtype, data, NULL, rows, cols, 4,
                                       &plan);
    if (!rc)
        rc = ingot_qtensor_encode(plan, 4, &s);
    ingot_qtensor_plan_free(plan);
    uint8_t *out = rc ? NULL : malloc(t->size);
    double best = 1e30;
    for (int p = 0; out && !rc && p < passes; p++) {
        memset(out, 0, t->size);
        double t0 = now_s();
        rc = decode_all(t->dtype, rows, cols, &s, out);
        double dt = now_s() - t0;
        best = dt < best ? dt : best;
        if (!rc && memcmp(out, data, t->size))
            rc = -1;
    }
    if (!rc) {
        st->weights += ingot_layout_elements(t);
        st->raw_bytes += t->size;
        st->coded_bytes += s.pos;
        st->tensors++;
        st->decode_s += best;
    }
    free(out);
    free(s.mem);
    return rc;
}

static void report(const type_stats *stats) {
    printf("type\ttensors\tweights\traw_bpw\tcoded_bpw\tsaved_%%\tdecode_Mw_s\tdecode_MB_s\n");
    for (uint32_t ty = 0; ty < TYPES; ty++) {
        const type_stats *st = &stats[ty];
        if (!st->tensors)
            continue;
        printf("%s\t%llu\t%llu\t%.4f\t%.4f\t%.2f\t%.1f\t%.1f\n", ingot_ggml_name(ty),
               (unsigned long long)st->tensors, (unsigned long long)st->weights,
               8.0 * (double)st->raw_bytes / (double)st->weights,
               8.0 * (double)st->coded_bytes / (double)st->weights,
               100.0 * (1.0 - (double)st->coded_bytes / (double)st->raw_bytes),
               (double)st->weights / st->decode_s * 1e-6,
               (double)st->raw_bytes / st->decode_s * 1e-6);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s FILE.gguf [passes] [max_tensors_per_type]\n", argv[0]);
        return 2;
    }
    int passes = argc > 2 ? atoi(argv[2]) : 3;
    long max_per_type = argc > 3 ? atol(argv[3]) : -1;
    ingot_input *in = NULL;
    ingot_layout *lay = NULL;
    int rc = ingot_input_open_path(argv[1], &in);
    if (!rc)
        rc = ingot_layout_read(in, &lay);
    static type_stats stats[TYPES];
    for (size_t i = 0; !rc && i < ingot_layout_count(lay); i++) {
        const ingot_layout_tensor *t = ingot_layout_tensor_at(lay, i);
        uint32_t ty = 0;
        if (ingot_dtype_ggml_type(t->dtype, &ty) || ty >= TYPES ||
            !ingot_qtensor_supports(t->dtype))
            continue;
        if (max_per_type >= 0 && stats[ty].tensors >= (uint64_t)max_per_type)
            continue;
        uint8_t *data = malloc(t->size);
        rc = data ? ingot_input_read_at(in, t->offset, data, t->size) : -1;
        if (!rc)
            rc = bench_tensor(t, data, passes, &stats[ty]);
        if (rc)
            fprintf(stderr, "%s: failed (%d)\n", t->name, rc);
        free(data);
    }
    report(stats);
    ingot_layout_free(lay);
    ingot_input_close(in);
    return rc ? 1 : 0;
}
