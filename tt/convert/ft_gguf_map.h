/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ft_gguf_map.h
 * @brief  Read-only memory-mapped GGUF reader: header, metadata, tensor directory, splits.
 *
 * Written from a behaviour specification; no ggml or llama.cpp source was consulted.
 *
 * Every part file is mapped read-only and private; tensor `data` points straight
 * into the mapping (no copy). Failures print one line to stderr and return a
 * negative FT_GGUF_* code.
 *
 * @status None
 * @issues Kept as-is: keys and names truncate to 127 bytes; split filename digits
 *         are not compared with the metadata; split tensor totals sum in 32 bits;
 *         a zero-element tensor of any type is accepted; data_offset alignment is
 *         not checked.
 * @todo   None
 */

#ifndef FT_GGUF_MAP_H
#define FT_GGUF_MAP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FT_GGUF_MAX_NAME 128
#define FT_GGUF_MAX_PATH 1024
#define FT_GGUF_MAX_PARTS 256u

/** GGML element type ids this reader names in its API. */
enum {
    FT_GGML_TYPE_F32 = 0,
    FT_GGML_TYPE_F16 = 1,
    FT_GGML_TYPE_Q8_0 = 8,
    FT_GGML_TYPE_Q2_K = 10,
    FT_GGML_TYPE_Q3_K = 11,
    FT_GGML_TYPE_Q4_K = 12,
    FT_GGML_TYPE_Q5_K = 13,
    FT_GGML_TYPE_Q6_K = 14,
    FT_GGML_TYPE_BF16 = 30
};

/** Status codes returned by the reader. */
enum {
    FT_GGUF_OK = 0,
    FT_GGUF_ERROR = -1,
    FT_GGUF_ENOMEM = -2,
    FT_GGUF_EINVAL = -3,
    FT_GGUF_EIO = -4,
    FT_GGUF_ENOTFOUND = -5
};

/**
 * One tensor directory entry. `dims[0]` is the fastest axis; dims past `n_dims`
 * are 1. `data_offset` is relative to the owning part's tensor-data section,
 * `byte_count` is the on-disk size, `data` points into the mapping.
 */
typedef struct {
    char name[FT_GGUF_MAX_NAME];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    uint64_t data_offset;
    uint64_t byte_count;
    const void *data;
} ft_gguf_tensor;

/** An open GGUF file or split set. */
typedef struct ft_gguf_map ft_gguf_map;

/**
 * Map and parse a GGUF file. A file declaring split.count opens its whole set
 * (found by the `-NNNNN-of-NNNNN.gguf` name pattern).
 *
 * @param path GGUF path, shorter than FT_GGUF_MAX_PATH bytes
 * @param out  receives the handle on success (NULL on failure); the caller frees
 *             it with ft_gguf_map_close()
 * @return     FT_GGUF_OK, or FT_GGUF_EINVAL / EIO / ERROR / ENOMEM
 *
 * Thread-safe: yes (no shared state between handles)
 * Reentrant:   yes
 * Signal-safe: no
 */
int ft_gguf_map_open(const char *path, ft_gguf_map **out);

/**
 * Unmap every part and free the handle. NULL-safe.
 *
 * @param map handle from ft_gguf_map_open(), or NULL
 *
 * Thread-safe: no (not synchronised with readers of the same handle)
 * Reentrant:   no
 * Signal-safe: no
 */
void ft_gguf_map_close(ft_gguf_map *map);

/**
 * Number of tensors across all parts.
 *
 * @param map handle, or NULL
 * @return    tensor count; 0 for NULL
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
uint32_t ft_gguf_map_tensor_count(const ft_gguf_map *map);

/**
 * Tensor by directory index (part order, then file order).
 *
 * @param map   handle, or NULL
 * @param index directory index
 * @return      the entry, or NULL for a NULL handle or an index out of range
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
const ft_gguf_tensor *ft_gguf_map_tensor(const ft_gguf_map *map, uint32_t index);

/**
 * First tensor whose name equals `name`.
 *
 * @param map  handle, or NULL
 * @param name tensor name, or NULL
 * @return     the entry, or NULL when not found or on a NULL argument
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
const ft_gguf_tensor *ft_gguf_map_find(const ft_gguf_map *map, const char *name);

/**
 * GGUF header version (of part 1 for a split set).
 *
 * @param map handle, or NULL
 * @return    2 or 3; 0 for NULL
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
uint32_t ft_gguf_map_version(const ft_gguf_map *map);

/**
 * Number of part files behind the handle (1 for an unsplit file).
 *
 * @param map handle, or NULL
 * @return    part count; 0 for NULL
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
uint32_t ft_gguf_map_part_count(const ft_gguf_map *map);

/**
 * Path of one part file, as given or as formed from the split name pattern.
 *
 * @param map        handle, or NULL
 * @param part_index 0-based part index
 * @return           the path (owned by the handle), or NULL when out of range
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
const char *ft_gguf_map_part_path(const ft_gguf_map *map, uint32_t part_index);

/**
 * Which part file holds a tensor.
 *
 * @param map    handle, or NULL
 * @param tensor an entry returned by this handle
 * @return       0-based part index; UINT32_MAX for NULL or a foreign pointer
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
uint32_t ft_gguf_map_tensor_part(const ft_gguf_map *map, const ft_gguf_tensor *tensor);

/**
 * Absolute byte offset of a tensor's data within its part file.
 *
 * @param map    handle, or NULL
 * @param tensor an entry returned by this handle
 * @return       file offset; 0 for NULL or a foreign pointer
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
uint64_t ft_gguf_map_tensor_file_offset(const ft_gguf_map *map, const ft_gguf_tensor *tensor);

/**
 * Integer (or bool) metadata value; signed types come back sign-extended.
 *
 * @param map handle
 * @param key metadata key
 * @param out receives the value, written only on FT_GGUF_OK
 * @return    FT_GGUF_OK; FT_GGUF_ENOTFOUND; FT_GGUF_EINVAL for a NULL argument or
 *            a non-integer value
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
int ft_gguf_map_metadata_u64(const ft_gguf_map *map, const char *key, uint64_t *out);

/**
 * String metadata value, copied and NUL-terminated (truncated to fit).
 *
 * @param map       handle
 * @param key       metadata key
 * @param out       caller buffer
 * @param out_bytes size of `out`, at least 1
 * @return          FT_GGUF_OK; FT_GGUF_ENOTFOUND; FT_GGUF_EINVAL for a bad
 *                  argument or a non-string value
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
int ft_gguf_map_metadata_string(const ft_gguf_map *map, const char *key, char *out, size_t out_bytes);

/**
 * Advise the kernel that a tensor's whole pages are no longer needed
 * (MADV_DONTNEED). Pages shared with a neighbouring tensor are left alone.
 *
 * @param map    handle, or NULL (no-op)
 * @param tensor an entry returned by this handle, or NULL (no-op)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
void ft_gguf_map_release(const ft_gguf_map *map, const ft_gguf_tensor *tensor);

/**
 * Drop the resident pages of PART of a tensor: bytes `[byte_offset, byte_offset+byte_count)`
 * of its data, rounded inward to whole pages (MADV_DONTNEED). A converter streaming a 3-D
 * expert stack calls this behind each expert, so its RSS is one expert, not the stack.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
void ft_gguf_map_release_range(const ft_gguf_map *map, const ft_gguf_tensor *tensor,
                               uint64_t byte_offset, uint64_t byte_count);

/**
 * The first part's key/value section exactly as it sits in the file: `kv_count` pairs in
 * `byte_count` bytes. A writer copies it verbatim to carry every key (tokenizer arrays
 * included) into a new GGUF without re-encoding any of them.
 *
 * @return FT_GGUF_OK, or FT_GGUF_EINVAL for a NULL argument
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_gguf_map_metadata_raw(const ft_gguf_map *map, const void **bytes, uint64_t *byte_count,
                             uint64_t *kv_count);

/**
 * Where a key's VALUE sits inside the raw key/value section (ft_gguf_map_metadata_raw), and
 * its GGUF value type — so a writer can copy the section and patch one scalar in place
 * (a truncated checkpoint's `<arch>.block_count`).
 *
 * @return FT_GGUF_OK; FT_GGUF_ENOTFOUND; FT_GGUF_EINVAL for a NULL argument
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_gguf_map_metadata_value_offset(const ft_gguf_map *map, const char *key, uint32_t *type,
                                      uint64_t *raw_offset);

/** The data alignment the first part declares (`general.alignment`, default 32); 0 for NULL. */
uint64_t ft_gguf_map_alignment(const ft_gguf_map *map);

#ifdef __cplusplus
}
#endif

#endif /* FT_GGUF_MAP_H */
