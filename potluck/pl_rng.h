/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pl_rng.h
 * @brief Seeded per-adapter RNG using splitmix64 with LoRA A init helper.
 *
 * @status Experimental
 * @issues None
 * @todo None
 */
#ifndef PL_RNG_H
#define PL_RNG_H

#include <stdint.h>
#include <stddef.h>

typedef struct { uint64_t s; } pl_rng;

void     pl_rng_seed(pl_rng *r, uint64_t seed, uint64_t stream);
uint64_t pl_rng_u64(pl_rng *r);
double   pl_rng_uniform(pl_rng *r);
double   pl_rng_normal(pl_rng *r);
/* LoRA A init for adapter layer `layer_index`: stream = layer_index.
 * A[i] = normal * sqrt(1/din), float. */
void     pl_lora_init_a(uint64_t seed, int layer_index, float *A, int rank, int din);
/* FNV-1a 64 over bytes. */
uint64_t pl_fnv1a64(const void *data, size_t n, uint64_t h);
/* Base-model id: FNV-1a 64 over the file's first min(size, 1 MiB) bytes,
 * then over the 8-byte little-endian file size.
 * Writes 16 lowercase hex chars + NUL into out17. Returns 0, or -1 if the
 * file cannot be read. */
int      pl_base_hash(const char *path, char out17[17]);

#endif /* PL_RNG_H */