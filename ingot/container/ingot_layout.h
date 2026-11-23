/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_layout.h
 * @brief  The layout of a source model file (safetensors or GGUF), read from the start of a
 *         stream: the header region verbatim and the tensors in file order. Only the header is
 *         read, so it works on pipes and on header-only Range fetches.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_LAYOUT_H
#define FOUNDRY_INGOT_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/container/ingot_format.h"
#include "ingot/core/ingot_input.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ingot_layout ingot_layout;

/** One tensor; offsets are absolute in the source file. */
typedef struct {
    const char *name;
    ingot_dtype dtype;          /* GGUF quantized types: INGOT_DTYPE_GGML + ggml type */
    int ndim;
    uint64_t shape[INGOT_MAX_DIMS]; /* row-major: the last dimension is contiguous */
    uint64_t offset;
    uint64_t size;
} ingot_layout_tensor;

/**
 * Read and parse the header region from the input's current position, which must be 0. Leaves
 * the input positioned at the end of the header region (GGUF: including the alignment padding
 * before the tensor data).
 *
 * @param in   Input at position 0
 * @param out  Out: layout, freed with ingot_layout_free
 * @return     0, -INGOT_EFORMAT (neither format, or a bad header), -INGOT_EUNSUPPORTED (GGUF
 *             version 1, or an unknown ggml type on the last tensor), -INGOT_ETRUNCATED,
 *             -INGOT_EIO or -INGOT_ENOMEM
 *
 * Thread-safe: yes (distinct inputs)   Reentrant: yes   Signal-safe: no
 */
int ingot_layout_read(ingot_input *in, ingot_layout **out);

/**
 * Free (NULL is a no-op).
 *
 * Thread-safe: no (no other call on the handle may run)   Reentrant: yes   Signal-safe: no
 */
void ingot_layout_free(ingot_layout *l);

/**
 * Source format.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
ingot_source_kind ingot_layout_kind(const ingot_layout *l);

/**
 * Header region bytes (safetensors: 8-byte length + JSON; GGUF: everything before the data).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const uint8_t *ingot_layout_header(const ingot_layout *l, uint64_t *size);

/**
 * End of the last tensor (the source size unless the file has trailing bytes).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_layout_data_end(const ingot_layout *l);

/**
 * Tensor count, and tensor i in file-offset order.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_layout_count(const ingot_layout *l);
const ingot_layout_tensor *ingot_layout_tensor_at(const ingot_layout *l, size_t i);

/**
 * Find a tensor by name. @return 0 or -INGOT_ENOTFOUND
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_layout_find(const ingot_layout *l, const char *name, size_t *index);

/**
 * Elements of a tensor (product of the shape; 1 for a scalar).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_layout_elements(const ingot_layout_tensor *t);

/**
 * Short dtype name ("BF16", "F32", "Q8_0", "Q4_K", ...).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const char *ingot_dtype_name(ingot_dtype dtype);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_LAYOUT_H */
