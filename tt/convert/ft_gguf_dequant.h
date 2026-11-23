/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ft_gguf_dequant.h
 * @brief  Byte sizing, naming and fp32 decoding of GGUF tensor element types.
 *
 * Written from a behaviour specification; no ggml or llama.cpp source was consulted.
 *
 * Type ids are the FT_GGML_TYPE_* values in ft_gguf_map.h; this header does not
 * include it, so callers include both.
 *
 * @status None
 * @issues F32/F16/BF16 byte sizes wrap in 64-bit arithmetic (kept as-is).
 * @todo   None
 */

#ifndef FT_GGUF_DEQUANT_H
#define FT_GGUF_DEQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FT_GGML_Q8_0_BLOCK 32
#define FT_GGML_Q8_0_BLOCK_BYTES 34
#define FT_GGML_K_SUPERBLOCK 256
#define FT_GGML_Q2_K_BLOCK_BYTES 84
#define FT_GGML_Q3_K_BLOCK_BYTES 110
#define FT_GGML_Q4_K_BLOCK_BYTES 144
#define FT_GGML_Q5_K_BLOCK_BYTES 176
#define FT_GGML_Q6_K_BLOCK_BYTES 210

/**
 * On-disk byte size of `elements` values of `ggml_type`.
 *
 * @param ggml_type GGML type id
 * @param elements  element count
 * @return          byte count; 0 for an unknown type, for zero elements, or when a
 *                  block type's count is not a whole number of blocks
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
size_t ft_gguf_type_bytes(uint32_t ggml_type, uint64_t elements);

/**
 * Short display name of a GGML type id.
 *
 * @param ggml_type GGML type id
 * @return          a string literal; "?" for an id with no name
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
const char *ft_gguf_type_name(uint32_t ggml_type);

/**
 * Whether ft_gguf_dequant_to_float() can decode a type.
 *
 * @param ggml_type GGML type id
 * @return          1 for F32, F16, BF16, Q8_0, Q2_K..Q6_K; 0 otherwise
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
int ft_gguf_type_supported(uint32_t ggml_type);

/**
 * Decode `elements` values of `ggml_type` at `src` into fp32.
 *
 * @param ggml_type GGML type id
 * @param src       encoded bytes, any alignment
 * @param elements  value count; a whole number of blocks for block types
 * @param dst       caller-owned output of `elements` floats
 * @return          0 on success; -1 on NULL arguments, an unsupported type or a
 *                  partial block (nothing written in those cases)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
int ft_gguf_dequant_to_float(uint32_t ggml_type, const void *src, uint64_t elements, float *dst);

#ifdef __cplusplus
}
#endif

#endif /* FT_GGUF_DEQUANT_H */
