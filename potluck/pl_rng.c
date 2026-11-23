/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pl_rng.c
 * @brief Seeded per-adapter RNG using splitmix64 with LoRA A init helper.
 *
 * @status Experimental
 * @issues None
 * @todo None
 */
#include "pl_rng.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define PI 3.14159265358979323846

static uint64_t splitmix64(uint64_t *state)
{
	uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

void pl_rng_seed(pl_rng *r, uint64_t seed, uint64_t stream)
{
	/* s = splitmix64(seed ^ splitmix64(stream)) */
	r->s = splitmix64(&seed) ^ splitmix64(&stream);
	/* Re-seed from the combined value */
	r->s = splitmix64(&r->s);
}

uint64_t pl_rng_u64(pl_rng *r)
{
	return splitmix64(&r->s);
}

double pl_rng_uniform(pl_rng *r)
{
	/* (pl_rng_u64 >> 11) * 0x1.0p-53, in [0, 1) */
	return ((double)(pl_rng_u64(r) >> 11)) * 0x1.0p-53;
}

double pl_rng_normal(pl_rng *r)
{
	/* Box-Muller, u1 = 1 - uniform (never 0); no cached 2nd value */
	double u1 = 1.0 - pl_rng_uniform(r);
	double u2 = pl_rng_uniform(r);
	return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

void pl_lora_init_a(uint64_t seed, int layer_index, float *A, int rank, int din)
{
	pl_rng r;
	pl_rng_seed(&r, seed, (uint64_t)layer_index);
	double scale = sqrt(1.0 / (double)din);
	for (int i = 0; i < rank * din; i++)
		A[i] = (float)(pl_rng_normal(&r) * scale);
}

uint64_t pl_fnv1a64(const void *data, size_t n, uint64_t h)
{
	const unsigned char *p = data;
	for (size_t i = 0; i < n; i++) {
		h ^= (uint64_t)p[i];
		h *= 0x0100000001B3ULL;
	}
	return h;
}

int pl_base_hash(const char *path, char out17[17])
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return -1;
	struct stat st;
	if (fstat(fileno(f), &st) != 0) {
		fclose(f);
		return -1;
	}
	uint64_t hash = 0xcbf29ce484222325ULL;
	size_t remaining = (size_t)st.st_size;
	size_t buf[65536];
	while (remaining > 0) {
		size_t chunk = remaining < sizeof(buf) ? remaining : sizeof(buf);
		size_t got = fread(buf, 1, chunk, f);
		hash = pl_fnv1a64(buf, got, hash);
		remaining -= got;
	}
	fclose(f);
	hash = pl_fnv1a64(&st.st_size, sizeof(st.st_size), hash);
	for (int i = 0; i < 8; i++) {
		unsigned char byte = (unsigned char)(hash >> ((7 - i) * 8));
		unsigned char hi = (byte >> 4) & 0x0f;
		unsigned char lo = byte & 0x0f;
		out17[i * 2] = (char)(hi < 10 ? '0' + hi : 'a' + hi - 10);
		out17[i * 2 + 1] = (char)(lo < 10 ? '0' + lo : 'a' + lo - 10);
	}
	out17[16] = '\0';
	return 0;
}