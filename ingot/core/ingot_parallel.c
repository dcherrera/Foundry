/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_parallel.c
 * @brief  pthread parallel-for (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_parallel.h"

#include <pthread.h>
#include <stdatomic.h>

#include "ingot_error.h"

enum { MAX_THREADS = 64 };

typedef struct {
    size_t count;
    ingot_item_fn fn;
    void *ctx;
    atomic_size_t next;
    atomic_int error;
} parallel_job;

static void *parallel_worker(void *arg) {
    parallel_job *job = arg;
    for (;;) {
        if (atomic_load(&job->error))
            break;
        size_t i = atomic_fetch_add(&job->next, 1);
        if (i >= job->count)
            break;
        int rc = job->fn(job->ctx, i);
        if (rc) {
            int expected = 0;
            atomic_compare_exchange_strong(&job->error, &expected, rc);
        }
    }
    return NULL;
}

int ingot_parallel_for(size_t count, int threads, ingot_item_fn fn, void *ctx) {
    if (!fn)
        return -INGOT_EINVAL;
    if (threads > MAX_THREADS)
        threads = MAX_THREADS;
    if ((size_t)threads > count)
        threads = (int)count;
    parallel_job job = {.count = count, .fn = fn, .ctx = ctx};
    atomic_init(&job.next, 0);
    atomic_init(&job.error, 0);
    if (threads <= 1) {
        parallel_worker(&job);
        return atomic_load(&job.error);
    }
    pthread_t ids[MAX_THREADS];
    int started = 0;
    for (; started < threads - 1; started++)
        if (pthread_create(&ids[started], NULL, parallel_worker, &job))
            break;
    parallel_worker(&job);
    for (int t = 0; t < started; t++)
        pthread_join(ids[t], NULL);
    int rc = atomic_load(&job.error);
    if (!rc && started < threads - 1 && atomic_load(&job.next) < count)
        rc = -INGOT_ENOMEM;
    return rc;
}
