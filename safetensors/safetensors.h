/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_SAFETENSORS_H
#define FOUNDRY_SAFETENSORS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Safe tensor serialization format.
 * Replaces HuggingFace safetensors.
 *
 * Format:
 *   - 8 bytes: header length (little-endian u64)
 *   - N bytes: JSON header with tensor metadata
 *   - Remaining: raw tensor data
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Safetensors data types (matches safetensors spec) */
typedef enum {
    FS_DTYPE_F32,
    FS_DTYPE_F64,
    FS_DTYPE_F16,
    FS_DTYPE_BF16,
    FS_DTYPE_I8,
    FS_DTYPE_I16,
    FS_DTYPE_I32,
    FS_DTYPE_I64,
    FS_DTYPE_U8,
    FS_DTYPE_U16,
    FS_DTYPE_U32,
    FS_DTYPE_U64,
    FS_DTYPE_BOOL,
    FS_DTYPE_UNKNOWN
} fs_dtype;

/* Tensor info */
typedef struct {
    char* name;
    fs_dtype dtype;
    int64_t* shape;
    int ndim;
    size_t data_offset;
    size_t data_size;
} fs_tensor_info;

/* Safetensors file handle */
typedef struct fs_file fs_file;

/* Writer for creating new safetensors files */
typedef struct fs_writer fs_writer;

/* ============================================================= */
/* Reading                                                       */
/* ============================================================= */

/* Open safetensors file (memory-maps the file) */
fs_file* fs_open(const char* path);

/* Open from memory buffer (does not copy) */
fs_file* fs_open_buffer(const void* data, size_t size);

/* Close file and free resources */
void fs_close(fs_file* f);

/* Get number of tensors in file */
int fs_num_tensors(fs_file* f);

/* Get list of tensor names (returns array of strings, caller must free array but not strings) */
const char** fs_tensor_names(fs_file* f, int* count);

/* Check if tensor exists */
bool fs_has_tensor(fs_file* f, const char* name);

/* Get tensor info (returns NULL if not found) */
fs_tensor_info* fs_get_info(fs_file* f, const char* name);

/* Free tensor info */
void fs_free_info(fs_tensor_info* info);

/* Get raw tensor data pointer (zero-copy, points into mmap'd region) */
const void* fs_get_data(fs_file* f, const char* name);

/* Get tensor data with type checking (returns NULL if dtype mismatch) */
const float* fs_get_f32(fs_file* f, const char* name);
const double* fs_get_f64(fs_file* f, const char* name);
const int32_t* fs_get_i32(fs_file* f, const char* name);
const int64_t* fs_get_i64(fs_file* f, const char* name);
const uint8_t* fs_get_u8(fs_file* f, const char* name);

/* ============================================================= */
/* Metadata                                                      */
/* ============================================================= */

/* Get metadata value (returns NULL if not found) */
const char* fs_get_metadata(fs_file* f, const char* key);

/* Get all metadata keys */
const char** fs_metadata_keys(fs_file* f, int* count);

/* ============================================================= */
/* Writing                                                       */
/* ============================================================= */

/* Create new safetensors writer */
fs_writer* fs_writer_new(void);

/* Free writer */
void fs_writer_free(fs_writer* w);

/* Add metadata */
void fs_writer_add_metadata(fs_writer* w, const char* key, const char* value);

/* Add tensor to writer */
void fs_writer_add_tensor(fs_writer* w, const char* name, fs_dtype dtype,
                          const int64_t* shape, int ndim, const void* data);

/* Add float32 tensor (convenience) */
void fs_writer_add_f32(fs_writer* w, const char* name,
                       const int64_t* shape, int ndim, const float* data);

/* Write to file */
bool fs_writer_save(fs_writer* w, const char* path);

/* Write to memory buffer (caller must free) */
void* fs_writer_to_buffer(fs_writer* w, size_t* size);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Get dtype name string */
const char* fs_dtype_name(fs_dtype dtype);

/* Get dtype size in bytes */
size_t fs_dtype_size(fs_dtype dtype);

/* Parse dtype from string */
fs_dtype fs_dtype_from_string(const char* s);

/* Print file summary */
void fs_print(fs_file* f);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_SAFETENSORS_H */
