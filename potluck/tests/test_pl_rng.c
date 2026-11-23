/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_pl_rng.c
 * @brief Tests for potluck/pl_rng.h — splitmix64 RNG, FNV-1a hash, LoRA A init.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

#include "lib/test.h"
#include "potluck/pl_rng.h"

/* ---------- r1: raw splitmix64 from s=0 ---------- */
static void test_r1(void)
{
	pl_rng r;
	uint64_t v0, v1, v2;

	r.s = 0;
	v0 = pl_rng_u64(&r);
	v1 = pl_rng_u64(&r);
	v2 = pl_rng_u64(&r);

	ASSERT_INT_EQ(v0, 0xe220a8397b1dcdafULL);
	ASSERT_INT_EQ(v1, 0x6e789e6aa1b965f4ULL);
	ASSERT_INT_EQ(v2, 0x06c45d188009454fULL);
}

/* ---------- r2: reproducibility ---------- */
static void test_r2(void)
{
	pl_rng a, b;
	int i;

	pl_rng_seed(&a, 42, 7);
	pl_rng_seed(&b, 42, 7);

	for (i = 0; i < 1000; i++)
		ASSERT_INT_EQ(pl_rng_u64(&a), pl_rng_u64(&b));

	pl_rng_seed(&a, 42, 0);
	pl_rng_seed(&b, 43, 0);
	ASSERT_INT_NE(pl_rng_u64(&a), pl_rng_u64(&b));
}

/* ---------- r3: lora_init_a + normal ---------- */
static void test_r3(void)
{
	float A1[4 * 64], A2[4 * 64], A3[4 * 64];
	double sum = 0, sumsq = 0;
	int i;

	pl_lora_init_a(7, 3, A1, 4, 64);
	pl_lora_init_a(7, 3, A2, 4, 64);
	for (i = 0; i < 4 * 64; i++)
		ASSERT_INT_EQ(*(uint32_t *)&A1[i], *(uint32_t *)&A2[i]);

	pl_lora_init_a(7, 3, A1, 4, 64);
	pl_lora_init_a(7, 4, A3, 4, 64);
	for (i = 0; i < 4 * 64; i++)
		ASSERT_INT_NE(*(uint32_t *)&A1[i], *(uint32_t *)&A3[i]);

	{
		pl_rng r;
		pl_rng_seed(&r, 99, 0);
		for (i = 0; i < 10000; i++) {
			double u = pl_rng_normal(&r);
			sum += u;
			sumsq += u * u;
		}
	}
	sum /= 10000;
	sumsq /= 10000;
	ASSERT_DBL_GT(sum, -0.02);
	ASSERT_DBL_LT(sum, 0.02);
	ASSERT_DBL_GT(sumsq, 0.98);
	ASSERT_DBL_LT(sumsq, 1.02);
}

/* ---------- r4: thread independence ---------- */
static void *lora_thread(void *arg)
{
	pl_rng *r = (pl_rng *)arg;
	for (int i = 0; i < 100; i++)
		pl_rng_u64(r);
	return NULL;
}

static void test_r4(void)
{
	pl_rng serial[2];
	pl_rng threads[2];
	pthread_t t[2];
	int i;

	pl_rng_seed(&serial[0], 7, 0);
	for (i = 0; i < 100; i++) pl_rng_u64(&serial[0]);
	pl_rng_seed(&serial[1], 7, 1);
	for (i = 0; i < 100; i++) pl_rng_u64(&serial[1]);

	pl_rng_seed(&threads[0], 7, 0);
	pl_rng_seed(&threads[1], 7, 1);
	pthread_create(&t[0], NULL, lora_thread, &threads[0]);
	pthread_create(&t[1], NULL, lora_thread, &threads[1]);
	pthread_join(t[0], NULL);
	pthread_join(t[1], NULL);

	for (i = 0; i < 2; i++)
		ASSERT_INT_EQ(serial[i].s, threads[i].s);
}

/* ---------- r5: FNV-1a + base_hash ---------- */
static void test_r5(void)
{
	uint64_t h;
	char hex[17];
	const char *path = "build/potluck_test/rng/f.bin";
	FILE *f;
	int ret;

	h = pl_fnv1a64("", 0, 0xcbf29ce484222325ULL);
	ASSERT_INT_EQ(h, 0xcbf29ce484222325ULL);

	h = pl_fnv1a64("a", 1, 0xcbf29ce484222325ULL);
	ASSERT_INT_EQ(h, 0xaf63dc4c8601ec8cULL);

	/* write test file */
	unlink(path);
	f = fopen(path, "wb");
	ASSERT_NOT_NULL(f);
	if (!f) return;
	fwrite("abc", 1, 3, f);
	fclose(f);

	ret = pl_base_hash(path, hex);
	ASSERT_INT_EQ(ret, 0);

	/* check all hex chars */
	for (int i = 0; i < 16; i++) {
		char c = hex[i];
		ASSERT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
	}

	/* compare with manual computation */
	uint64_t expected = pl_fnv1a64("abc", 3, 0xcbf29ce484222325ULL);
	uint64_t size_val = 3;
	expected = pl_fnv1a64(&size_val, sizeof(size_val), expected);
	char expected_hex[17];
	snprintf(expected_hex, 17, "%016llx", (unsigned long long)expected);
	ASSERT_STR_EQ(hex, expected_hex);

	/* cleanup */
	unlink(path);
}

int main(void)
{
	RUN_TEST(r1);
	RUN_TEST(r2);
	RUN_TEST(r3);
	RUN_TEST(r4);
	RUN_TEST(r5);
	TEST_REPORT();
	return 0;
}