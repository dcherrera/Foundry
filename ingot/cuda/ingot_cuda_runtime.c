/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cuda_runtime.c
 * @brief  CUDA driver API host side of the GPU decode (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_cuda_runtime.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot/core/ingot_error.h"
#include "ingot/cuda/ingot_cuda_reference.h"

enum {
    THREADS = 128,
    PLAIN_BLOCKS_PER_GROUP = THREADS / 32, /* one rANS block per warp */
    MATCH_BLOCKS_PER_GROUP = THREADS / 16, /* one rANS block per half-warp */
};

struct ingot_cuda_runtime {
    CUcontext ctx;
    CUmodule module;
    CUfunction plain, delta, match, combine;
    CUevent start, stop;
};

static int check(CUresult rc, const char *what) {
    if (rc == CUDA_SUCCESS)
        return 0;
    const char *msg = NULL;
    cuGetErrorString(rc, &msg);
    fprintf(stderr, "ingot cuda: %s: %s\n", what, msg ? msg : "unknown error");
    return -INGOT_EIO;
}

static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = n >= 0 ? malloc((size_t)n + 1) : NULL;
    if (s && fread(s, 1, (size_t)n, f) != (size_t)n) {
        free(s);
        s = NULL;
    }
    if (s)
        s[n] = 0;
    fclose(f);
    return s;
}

static int load_module(ingot_cuda_runtime *rt, const char *ptx_path) {
    char *ptx = read_text(ptx_path);
    if (!ptx) {
        fprintf(stderr, "ingot cuda: cannot read %s\n", ptx_path);
        return -INGOT_EIO;
    }
    int rc = check(cuModuleLoadData(&rt->module, ptx), "load PTX");
    free(ptx);
    if (!rc)
        rc = check(cuModuleGetFunction(&rt->plain, rt->module, "ingot_cuda_plain"), "plain");
    if (!rc)
        rc = check(cuModuleGetFunction(&rt->delta, rt->module, "ingot_cuda_delta"), "delta");
    if (!rc)
        rc = check(cuModuleGetFunction(&rt->match, rt->module, "ingot_cuda_match"), "match");
    if (!rc)
        rc = check(cuModuleGetFunction(&rt->combine, rt->module, "ingot_cuda_combine"), "combine");
    return rc;
}

int ingot_cuda_runtime_open(int device, const char *ptx_path, ingot_cuda_runtime **out) {
    ingot_cuda_runtime *rt = calloc(1, sizeof(*rt));
    if (!rt)
        return -INGOT_ENOMEM;
    CUdevice dev;
    int rc = check(cuInit(0), "init");
    if (!rc)
        rc = check(cuDeviceGet(&dev, device), "device");
    if (!rc)
        rc = check(cuCtxCreate(&rt->ctx, 0, dev), "context");
    if (!rc)
        rc = load_module(rt, ptx_path);
    if (!rc)
        rc = check(cuEventCreate(&rt->start, CU_EVENT_DEFAULT), "event");
    if (!rc)
        rc = check(cuEventCreate(&rt->stop, CU_EVENT_DEFAULT), "event");
    if (rc) {
        ingot_cuda_runtime_close(rt);
        return rc;
    }
    *out = rt;
    return 0;
}

void ingot_cuda_runtime_close(ingot_cuda_runtime *rt) {
    if (!rt)
        return;
    if (rt->start)
        cuEventDestroy(rt->start);
    if (rt->stop)
        cuEventDestroy(rt->stop);
    if (rt->module)
        cuModuleUnload(rt->module);
    if (rt->ctx)
        cuCtxDestroy(rt->ctx);
    free(rt);
}

int ingot_cuda_alloc(ingot_cuda_runtime *rt, size_t bytes, uint64_t *address) {
    CUdeviceptr d = 0;
    int rc = check(cuMemAlloc(&d, bytes ? bytes : 1), "alloc");
    if (!rc)
        rc = check(cuMemsetD8(d, 0, bytes ? bytes : 1), "memset");
    (void)rt;
    *address = (uint64_t)d;
    return rc;
}

void ingot_cuda_free(ingot_cuda_runtime *rt, uint64_t address) {
    (void)rt;
    if (address)
        cuMemFree((CUdeviceptr)address);
}

int ingot_cuda_upload(ingot_cuda_runtime *rt, uint64_t address, const void *src, size_t bytes) {
    (void)rt;
    return check(cuMemcpyHtoD((CUdeviceptr)address, src, bytes), "upload");
}

int ingot_cuda_download(ingot_cuda_runtime *rt, void *dst, uint64_t address, size_t bytes) {
    (void)rt;
    return check(cuMemcpyDtoH(dst, (CUdeviceptr)address, bytes), "download");
}

int ingot_cuda_image_upload(ingot_cuda_runtime *rt, const ingot_cuda_image *img,
                            uint64_t *address) {
    int rc = ingot_cuda_alloc(rt, img->image_bytes, address);
    if (!rc)
        rc = ingot_cuda_upload(rt, *address, img->image, img->image_bytes);
    return rc;
}

/* One kernel launch over the params at device address `params`. */
static int launch_one(CUfunction fn, uint64_t params, uint64_t blocks, uint32_t blocks_per_group) {
    CUdeviceptr arg = (CUdeviceptr)params;
    void *args[] = {&arg};
    unsigned grid = (unsigned)((blocks + blocks_per_group - 1) / blocks_per_group);
    unsigned threads = blocks == 1 && blocks_per_group == MATCH_BLOCKS_PER_GROUP ? 32 : THREADS;
    return check(cuLaunchKernel(fn, grid, 1, 1, threads, 1, 1, 0, NULL, args, NULL), "launch");
}

/* Params for every launch: one, or one per block of a serial payload. */
static int upload_params(ingot_cuda_runtime *rt, const ingot_cuda_params *p, int serial,
                         uint64_t *address, uint64_t *count) {
    *count = serial ? p->launch_blocks : 1;
    ingot_cuda_params *all = malloc(sizeof(*all) * (size_t)*count);
    if (!all)
        return -INGOT_ENOMEM;
    for (uint64_t i = 0; i < *count; i++) {
        all[i] = *p;
        if (serial) {
            all[i].block_first = p->block_first + i;
            all[i].launch_blocks = 1;
|= INGOT_CUDA_STORE;
        }
    }
    int rc = ingot_cuda_alloc(rt, sizeof(*all) * (size_t)*count, address);
    if (!rc)
        rc = ingot_cuda_upload(rt, *address, all, sizeof(*all) * (size_t)*count);
    free(all);
    return rc;
}

int ingot_cuda_launch(ingot_cuda_runtime *rt, const ingot_cuda_image *img,
                      const ingot_cuda_params *p, float *elapsed_ms) {
    uint32_t g = ingot_cuda_step_weights(p->kind);
    if (((p->flags & INGOT_CUDA_MATVEC) && p->cols % g) ||
        (img->self_source && p->unit_len < INGOT_CUDA_MATCH_GROUP))
        return -INGOT_EUNSUPPORTED;
    CUfunction fn = p->kind == INGOT_CUDA_KIND_PLAIN   ? rt->plain
                    : p->kind == INGOT_CUDA_KIND_DELTA ? rt->delta
                                                       : rt->match;
    uint32_t per = p->kind == INGOT_CUDA_KIND_MATCH ? MATCH_BLOCKS_PER_GROUP
                                                    : PLAIN_BLOCKS_PER_GROUP;
    uint64_t params = 0, count = 0;
    ingot_cuda_params q = *p;
    q.flags |= img->self_source ? INGOT_CUDA_STORE : 0; /* later rows read earlier ones */
    int rc = upload_params(rt, &q, img->serial, &params, &count);
    if (!rc)
        rc = check(cuEventRecord(rt->start, NULL), "event");
    for (uint64_t i = 0; !rc && i < count; i++)
        rc = launch_one(fn, params + i * sizeof(ingot_cuda_params),
                        img->serial ? 1 : p->launch_blocks, per);
    if (!rc && (p->flags & INGOT_CUDA_MATVEC) && p->block_first == 0 &&
        p->launch_blocks == p->block_count)
        rc = check(cuLaunchKernel(rt->combine, 1, 1, 1, 1, 1, 1, 0, NULL,
                                  (void *[]){&(CUdeviceptr){(CUdeviceptr)params}}, NULL),
                   "combine");
    if (!rc)
        rc = check(cuEventRecord(rt->stop, NULL), "event");
    if (!rc)
        rc = check(cuEventSynchronize(rt->stop), "synchronize");
    if (!rc && elapsed_ms)
        rc = check(cuEventElapsedTime(elapsed_ms, rt->start, rt->stop), "elapsed");
    ingot_cuda_free(rt, params);
    return rc;
}
