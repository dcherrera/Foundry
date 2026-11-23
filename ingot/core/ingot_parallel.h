/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_parallel.h
 * @brief  Minimal parallel-for over independent items (codec blocks) on pthreads.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_PARALLEL_H
#define FOUNDRY_INGOT_PARALLEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Work item callback: process item `index`; return 0 or a negated ingot_error. */
typedef int (*ingot_item_fn)(void *ctx, size_t index);

/**
 * Run fn(ctx, i) for i in [0, count) on up to `threads` threads (items are claimed in order from
 * an atomic counter). Stops claiming new items after the first failure.
 *
 * @param count    Number of items
 * @param threads  Thread count; <= 1 runs inline on the caller's thread
 * @param fn       Item callback; must be safe to call concurrently for different items
 * @param ctx      Passed through to fn
 * @return         0, the first error fn returned, or -INGOT_ENOMEM if threads cannot start
 *
 * Thread-safe: yes (if fn is)
 * Reentrant:   yes
 * Signal-safe: no
 */
int ingot_parallel_for(size_t count, int threads, ingot_item_fn fn, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_PARALLEL_H */
