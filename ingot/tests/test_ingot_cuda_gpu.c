/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_cuda_gpu.c
 * @brief  GPU decode on a K80 die: every synthetic case (plain, plain groups, small plain, delta,
 *         predict rows / columns / grouped heads, match rows with a self source, match columns)
 *         decoded on the GPU and compared bit for bit with the codec's CPU decoder; the matvec
 *         (alone and with the store) compared bit for bit with the CPU reference of the kernels;
 *         every block's status clean. A large plain tensor reports decode throughput.
 *
 * Build on the bank: make -C ingot cuda (PTX + this test). Run on one die:
 *   CUDA_VISIBLE_DEVICES=<die> ingot/build/test_ingot_cuda_gpu ingot/build/ingot_cuda.ptx
 * Red: make -C ingot cuda-red builds the PTX with INGOT_CUDA_RED_WORDS (a match lane reads its
 * symbol's word before its sign bit's); the match cases must fail with it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_error.h"
#include "ingot/cuda/ingot_cuda_runtime.h"
#include "ingot/tests/ingot_cuda_cases.h"

/** Device buffers of one case. */
typedef struct {
    uint64_t image, arrays[CASE_MAX_ARRAYS], base, out, x, y, partials, status;
} gpu_buffers;

static int gpu_buffers_alloc(ingot_cuda_runtime *rt, const cuda_case *k,
                             const ingot_cuda_image *img, const float *x, gpu_buffers *g) {
    memset(g, 0, sizeof(*g));
    const ingot_cuda_params *p = &img->params;
    int rc = ingot_cuda_image_upload(rt, img, &g->image);
    for (int i = 0; !rc && i < CASE_MAX_ARRAYS; i++)
        if (k->arrays[i]) {
            size_t bytes = k->array_entry[i].source_size;
            rc = ingot_cuda_alloc(rt, bytes, &g->arrays[i]);
            if (!rc)
                rc = ingot_cuda_upload(rt, g->arrays[i], k->arrays[i], bytes);
        }
    if (!rc && k->base) {
        rc = ingot_cuda_alloc(rt, k->e.source_size, &g->base);
        if (!rc)
            rc = ingot_cuda_upload(rt, g->base, k->base, k->e.source_size);
    }
    if (!rc)
        rc = ingot_cuda_alloc(rt, k->e.source_size, &g->out);
    if (!rc)
        rc = ingot_cuda_alloc(rt, p->cols * sizeof(float), &g->x);
    if (!rc)
        rc = ingot_cuda_upload(rt, g->x, x, p->cols * sizeof(float));
    if (!rc)
        rc = ingot_cuda_alloc(rt, p->rows * sizeof(float), &g->y);
    if (!rc)
        rc = ingot_cuda_alloc(rt, 2 * p->block_count * sizeof(float), &g->partials);
    if (!rc)
        rc = ingot_cuda_alloc(rt, p->block_count * sizeof(uint32_t), &g->status);
    return rc;
}

static void gpu_buffers_free(ingot_cuda_runtime *rt, gpu_buffers *g) {
    uint64_t *all = (uint64_t *)g;
    for (size_t i = 0; i < sizeof(*g) / sizeof(uint64_t); i++)
        ingot_cuda_free(rt, all[i]);
}

/* Blocks whose status is not clean. */
static uint32_t bad_blocks(ingot_cuda_runtime *rt, const gpu_buffers *g, uint64_t blocks) {
    uint32_t *st = calloc(blocks, sizeof(uint32_t)), bad = 0;
    ingot_cuda_download(rt, st, g->status, blocks * sizeof(uint32_t));
    for (uint64_t b = 0; b < blocks; b++)
        bad += st[b] != 0;
    free(st);
    return bad;
}

/** Result of one case on the GPU (failures counted; times of the store and matvec runs). */
typedef struct {
    int failures;
    float store_ms, matvec_ms;
} gpu_result;

/* One launch with the given flags; compares out with the codec and y with the reference. */
static int run_flags(ingot_cuda_runtime *rt, const cuda_case *k, const ingot_cuda_image *img,
                     const gpu_buffers *g, uint32_t flags, const uint8_t *codec,
                     const float *ref_y, float *ms) {
    ingot_cuda_params p;
    ingot_cuda_image_params(img, g->image, &p);
    int fail = case_bind_sources(k, img, &p, g->arrays) != 0;
    ingot_cuda_bind_self(&p, g->out);
    p.flags = flags;
    p.out = g->out;
    p.base = g->base;
    p.x = g->x;
    p.y = g->y;
    p.partials = g->partials;
    p.status = g->status;
    int rc = fail ? -1 : ingot_cuda_launch(rt, img, &p, ms);
    if (rc)
        return printf("    %s flags %u: launch rc %d\n", k->name, flags, rc), 1;
    uint32_t bad = bad_blocks(rt, g, p.block_count);
    fail += bad != 0;
    if ((flags & INGOT_CUDA_STORE) || img->self_source) {
        uint8_t *out = malloc(k->e.source_size);
        ingot_cuda_download(rt, out, g->out, k->e.source_size);
        fail += memcmp(out, codec, k->e.source_size) != 0;
        free(out);
    }
    if (flags & INGOT_CUDA_MATVEC) {
        float *y = malloc(p.rows * sizeof(float));
        ingot_cuda_download(rt, y, g->y, p.rows * sizeof(float));
        fail += memcmp(y, ref_y, p.rows * sizeof(float)) != 0;
        free(y);
    }
    printf("    %-16s flags %u  bad blocks %u  %s  %.3f ms\n", k->name, flags, bad,
           fail ? "FAIL" : "ok", (double)*ms);
    return fail;
}

static gpu_result run_case(ingot_cuda_runtime *rt, const cuda_case *k) {
    gpu_result res = {0};
    ingot_cuda_image img;
    if (ingot_cuda_image_build(&k->e, k->payload, k->size, &img))
        return printf("    %s: image refused\n", k->name), (gpu_result){1, 0, 0};
    uint8_t *codec = malloc(k->e.source_size);
    res.failures += case_codec_decode(k, codec) != 0;
    float *x = malloc(img.params.cols * sizeof(float));
    for (uint64_t c = 0; c < img.params.cols; c++)
        x[c] = (float)test_gauss();
    ref_result ref = case_reference_run(k, INGOT_CUDA_MATVEC, x);
    gpu_buffers g;
    float ms = 0;
    if (gpu_buffers_alloc(rt, k, &img, x, &g))
        res.failures++;
    else {
        res.failures += run_flags(rt, k, &img, &g, INGOT_CUDA_STORE, codec, ref.y, &res.store_ms);
        if (img.params.cols % ingot_cuda_step_weights(img.params.kind) == 0) {
            res.failures += run_flags(rt, k, &img, &g, INGOT_CUDA_MATVEC, codec, ref.y,
                                      &res.matvec_ms);
            res.failures += run_flags(rt, k, &img, &g, INGOT_CUDA_STORE | INGOT_CUDA_MATVEC,
                                      codec, ref.y, &ms);
        }
    }
    gpu_buffers_free(rt, &g);
    ref_free(&ref);
    free(x);
    free(codec);
    ingot_cuda_image_free(&img);
    return res;
}

/* Decode throughput on a large plain tensor (17408 x 5120, a Qwen3.8 MLP shape). */
static int run_plain_throughput(ingot_cuda_runtime *rt) {
    cuda_case k;
    test_rng = 77;
    case_plain(&k, "plain-17408x5120", 17408, 5120, 0);
    gpu_result r = run_case(rt, &k);
    double n = 17408.0 * 5120.0;
    printf("    plain decode %.2f G weights/s (store), %.2f G weights/s (matvec)\n",
           n / (r.store_ms * 1e6), n / (r.matvec_ms * 1e6));
    case_free(&k);
    return r.failures;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s KERNELS.ptx\n", argv[0]);
        return 2;
    }
    ingot_cuda_runtime *rt = NULL;
    if (ingot_cuda_runtime_open(0, argv[1], &rt))
        return 2;
    int failures = 0;
    for (int i = 0; i < CASE_COUNT; i++) {
        cuda_case k;
        case_build(&k, i);
        failures += run_case(rt, &k).failures;
        case_free(&k);
    }
    failures += run_plain_throughput(rt);
    printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    ingot_cuda_runtime_close(rt);
    return failures ? 1 : 0;
}
