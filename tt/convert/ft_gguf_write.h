/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_gguf_write.h
 * @brief Write a GGUF v3 file: a key/value section (built here, or copied verbatim from
 *   another GGUF) and a list of tensors whose bytes are streamed from wherever they live.
 *
 * Two users. The metadata GGUF a card host keeps beside its FTTW artifacts (tokenizer,
 * config and the small non-matmul tensors, with the multi-GB weights left out), which copies
 * the source's key/value section byte for byte so no key is ever re-encoded. And the test
 * suites, which build synthetic checkpoints with ft_gguf_kv_buffer.
 *
 * Written from the GGUF layout as ft_gguf_map.c reads it: magic, version, tensor count,
 * key/value count, the key/value pairs, the tensor table (name, rank, dims, type, offset
 * from the data start), padding to the alignment, then the data, each tensor aligned.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_GGUF_WRITE_H
#define FT_GGUF_WRITE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    FT_GGUF_WRITE_OK = 0,
    FT_GGUF_WRITE_EINVAL = -1,
    FT_GGUF_WRITE_EIO = -2,
    FT_GGUF_WRITE_ENOMEM = -3
};

/** One tensor to write. `data` is read once, sequentially; it may be an mmap. */
typedef struct {
    const char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    const void *data;
    uint64_t byte_count;
} ft_gguf_write_tensor;

/** A growable key/value section, in GGUF's own encoding. */
typedef struct {
    unsigned char *bytes;
    size_t used;
    size_t capacity;
    uint64_t count; /**< pairs appended */
    int failed;     /**< sticky: an allocation failed, every later append is a no-op */
} ft_gguf_kv_buffer;

/** Append a u32 / u64 / f32 / string pair. Failures are sticky in `kv->failed`. */
void ft_gguf_kv_add_u32(ft_gguf_kv_buffer *kv, const char *key, uint32_t value);
void ft_gguf_kv_add_u64(ft_gguf_kv_buffer *kv, const char *key, uint64_t value);
void ft_gguf_kv_add_f32(ft_gguf_kv_buffer *kv, const char *key, float value);
void ft_gguf_kv_add_string(ft_gguf_kv_buffer *kv, const char *key, const char *value);

/** Free the buffer's storage and zero it. Safe on a zeroed buffer. */
void ft_gguf_kv_free(ft_gguf_kv_buffer *kv);

/**
 * Write a GGUF v3 file to `path` (truncating it).
 *
 * @param kv_bytes   the key/value section, `kv_count` pairs in `kv_byte_count` bytes
 * @param alignment  data alignment; must match `general.alignment` when the section sets
 *                   it, else 32 (GGUF's default). A power of two.
 * @return FT_GGUF_WRITE_OK, or a negative status (EIO names the path on stderr)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_gguf_write_file(const char *path, const void *kv_bytes, uint64_t kv_byte_count, uint64_t kv_count,
                       const ft_gguf_write_tensor *tensors, uint32_t tensor_count, uint64_t alignment);

#ifdef __cplusplus
}
#endif

#endif /* FT_GGUF_WRITE_H */
