/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_base.h
 * @brief  The set of base tensors the delta modes code against: any mix of .safetensors and .gguf
 *         files, .ingot files (plain or raw entries) and directories holding them. Looked up by
 *         tensor name; read by byte range, so a decoder never holds a base tensor whole.
 *
 * Model files are read with pread when they are regular files. A pipe or FIFO base (e.g.
 * `--base <(curl ...)`) is read forward-only: tensors must be requested in file order, which holds
 * when target and base come from the same converter. .ingot bases decode a whole tensor into a
 * buffer owned by the set.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_BASE_H
#define FOUNDRY_INGOT_BASE_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/container/ingot_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ingot_base_set ingot_base_set;

/** A base tensor found by name. */
typedef struct {
    ingot_dtype dtype;
    uint64_t size;   /* bytes */
    size_t file;     /* internal: file index */
    size_t index;    /* internal: tensor index in that file */
} ingot_base_tensor;

/**
 * Open every base source under the given paths (directories are scanned one level for
 * *.safetensors and *.gguf files and .ingot files; .ingot files are recognised by magic).
 *
 * @param paths    Paths
 * @param count    Path count
 * @param threads  Threads used when decoding .ingot base entries
 * @param out      Out: set, closed with ingot_base_set_close
 * @return         0, -INGOT_EIO, -INGOT_EFORMAT or -INGOT_ENOMEM
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
int ingot_base_set_open(const char *const *paths, size_t count, int threads,
                        ingot_base_set **out);

/**
 * Close a set (NULL is a no-op).
 *
 * Thread-safe: no (no other call on the handle may run)   Reentrant: yes   Signal-safe: no
 */
void ingot_base_set_close(ingot_base_set *b);

/**
 * Find tensor `name`. @return 0 or -INGOT_ENOTFOUND
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_base_set_find(const ingot_base_set *b, const char *name, ingot_base_tensor *t);

/**
 * Read bytes [offset, offset + n) of a base tensor.
 *
 * @return 0, -INGOT_EINVAL (range outside the tensor), -INGOT_EUNSUPPORTED (forward-only file
 *         read backwards, or a delta-coded .ingot entry), an i/o or decode error
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int ingot_base_set_read(ingot_base_set *b, const ingot_base_tensor *t, uint64_t offset,
                        void *buf, size_t n);

/**
 * The whole tensor in a buffer owned by the set, valid until the next call (encoder side).
 *
 * @return 0, or an error as ingot_base_set_read
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int ingot_base_set_load(ingot_base_set *b, const ingot_base_tensor *t, const uint8_t **data);

/**
 * The single model file of the set, if it holds exactly one .safetensors / .gguf file and no
 * .ingot file: its size, and its SHA-256 (hashing the whole file, or for a forward-only file the
 * rest of it, which ends any further reads).
 *
 * @return 0, -INGOT_ENOTFOUND (not exactly one model file), or an i/o error
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int ingot_base_set_identity(ingot_base_set *b, uint64_t *size, uint8_t sha256[32]);

/**
 * Size of the single model file (as ingot_base_set_identity, without hashing; 0 if unknown,
 * e.g. a pipe).
 *
 * @return 0 or -INGOT_ENOTFOUND
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_base_set_single_size(const ingot_base_set *b, uint64_t *size);

/**
 * Shape of a found tensor (ndim, and INGOT_MAX_DIMS entries of shape).
 * @return 0 or -INGOT_EINVAL
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_base_set_shape(const ingot_base_set *b, const ingot_base_tensor *t, int *ndim,
                         uint64_t *shape);

/**
 * Number of opened source files.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_base_set_files(const ingot_base_set *b);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_BASE_H */
