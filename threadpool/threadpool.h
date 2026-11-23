/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * threadpool.h — Foundry worker thread pool
 *
 * Minimal pthread-based parallel_for for embarrassingly parallel loops.
 * Replaces OpenMP for the 3 parallel hotspots in rwkvx.c.
 *
 * Prefix: fw_ (foundry workers)
 */

#ifndef FOUNDRY_THREADPOOL_H
#define FOUNDRY_THREADPOOL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque pool handle */
typedef struct fw_pool fw_pool;

/* Callback: process range [start, end) with user data */
typedef void (*fw_for_fn)(int start, int end, void *arg);

/* Create a thread pool. num_threads=0 auto-detects core count.
 * Returns NULL on failure. */
fw_pool* fw_pool_new(int num_threads);

/* Destroy pool and join all worker threads. */
void fw_pool_free(fw_pool *pool);

/* Parallel for: splits [0, count) into chunks, dispatches to workers.
 * Main thread also participates. Returns when all chunks are done.
 * If pool is NULL, runs serially on the calling thread. */
void fw_for(fw_pool *pool, int count, fw_for_fn fn, void *arg);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_THREADPOOL_H */
