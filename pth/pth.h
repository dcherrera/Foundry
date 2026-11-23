/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * PyTorch .pth file reader
 *
 * Reads PyTorch state_dict files (ZIP + pickle format).
 * Zero-copy: tensor data points into mmap'd storage files.
 *
 * Supports: BFloat16Storage, FloatStorage, HalfStorage
 */

#ifndef FOUNDRY_PTH_H
#define FOUNDRY_PTH_H

#include "safetensors/safetensors.h"  /* for fs_dtype */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[256];        /* e.g. "blocks.0.att.x_r" */
    fs_dtype dtype;        /* FS_DTYPE_BF16, FS_DTYPE_F32, etc. */
    int64_t shape[8];
    int ndim;
    const void *data;      /* pointer into mmap'd storage (zero-copy) */
    size_t data_size;      /* total bytes */
} pt_tensor;

typedef struct pt_file pt_file;

/* Open a .pth file (mmap'd, zero-copy) */
pt_file*          pt_open(const char *path);

/* Number of tensors in the file */
int               pt_tensor_count(const pt_file *f);

/* Get tensor by index (0-based) */
const pt_tensor*  pt_tensor_at(const pt_file *f, int index);

/* Get all tensor names (caller frees the array, not the strings) */
const char**      pt_tensor_names(const pt_file *f, int *count);

/* Close and free all resources */
void              pt_close(pt_file *f);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_PTH_H */
