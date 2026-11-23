/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file synthetic_hybrid_checkpoint.h
 * @brief A small Qwen3.8-shaped GGUF for host tests: a GDN layer, a real-attention layer,
 *   an MTP block and a head, with the matmul weights stored as Q8_0 from realistic values.
 *
 * VALUES: per row a scale 0.02 * exp(0.3 g) (g standard normal), per input channel a gain
 * of 1 (98%) or 4 (2%, the outlier channels a shared block exponent is sensitive to), and
 * per value a standard normal draw, x6 with probability 0.5% (a heavy tail). Deterministic
 * in the seed. Q8_0 is ggml's layout: an fp16 scale (peak / 127) then 32 int8.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef SYNTHETIC_HYBRID_CHECKPOINT_H
#define SYNTHETIC_HYBRID_CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>

/** ggml type ids the fixture writes. */
#define SYNTHETIC_GGML_F32 0u
#define SYNTHETIC_GGML_Q8_0 8u

/** One tensor of the fixture, in GGML dimension order (dims[0] = input / fastest). */
typedef struct {
    const char *name;
    uint32_t n_dims;
    uint64_t dims[2];
    uint32_t ggml_type;
} synthetic_tensor_spec;

/** The fixture's tensors, in file order. */
extern const synthetic_tensor_spec synthetic_hybrid_tensors[];
extern const uint32_t synthetic_hybrid_tensor_count;

/** Realistic weight values, row-major rows x cols; caller frees. NULL on allocation failure. */
float *synthetic_realistic_weights(uint32_t seed, uint64_t rows, uint64_t cols);

/** Q8_0-encode `rows` rows of `cols` values (cols % 32 == 0); caller frees. */
unsigned char *synthetic_q8_0_encode(const float *values, uint64_t rows, uint64_t cols, uint64_t *bytes_out);

/**
 * Write the fixture GGUF (`general.architecture` qwen35, block_count 3; block 2 is the
 * MTP block). Returns 0, or -1 on any failure.
 */
int synthetic_hybrid_checkpoint_write(const char *path);

#endif /* SYNTHETIC_HYBRID_CHECKPOINT_H */
