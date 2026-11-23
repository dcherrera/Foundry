/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * threadpool.c — Foundry worker thread pool
 *
 * Mutex + condvar based implementation for reliable synchronization.
 * Workers sleep on condvar when idle, wake on fw_for() signal.
 */

#include "threadpool/threadpool.h"

#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

struct fw_pool {
    pthread_t *threads;
    int        n_threads;       /* number of workers (excludes main) */

    /* Work descriptor — set by fw_for(), read by workers */
    fw_for_fn  fn;
    void      *arg;
    int        count;
    int        chunk_size;
    atomic_int next_chunk;      /* atomic counter for chunk dispatch */

    /* Synchronization */
    pthread_mutex_t mutex;
    pthread_cond_t  work_ready; /* signal: new work available */
    pthread_cond_t  work_done;  /* signal: all workers finished */
    atomic_int generation;      /* incremented each fw_for() call (lock-free spin) */
    int        workers_waiting; /* number of workers idle */
    atomic_int active;          /* threads still processing */
    atomic_int shutdown;
};

/* Bounded spin before parking on the condvar. Back-to-back fw_for() dispatches
 * (e.g. ~150 quantized matvecs per decode token) arrive within microseconds of
 * each other, so a short spin lets workers pick up the next generation via the
 * atomic counter and skip the ~20us condvar wake/park round-trip entirely. When
 * work stops arriving the spin is exhausted and the worker parks on the condvar,
 * preserving the original zero-CPU idle behaviour for latency-insensitive users
 * of this shared pool. */
#ifndef FW_SPIN_BUDGET
#define FW_SPIN_BUDGET 240000
#endif

static inline void fw_cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#endif
}

static int detect_cores(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

/* Execute chunks until none remain */
static void execute_chunks(fw_pool *pool) {
    int cs = pool->chunk_size;
    int total = pool->count;
    for (;;) {
        int ci = atomic_fetch_add(&pool->next_chunk, 1);
        int start = ci * cs;
        if (start >= total) break;
        int end = start + cs;
        if (end > total) end = total;
        pool->fn(start, end, pool->arg);
    }
}

static void *worker_loop(void *arg) {
    fw_pool *pool = (fw_pool *)arg;
    int my_gen = 0;

    for (;;) {
        /* Phase 1: lock-free spin for the next generation. */
        int spins = 0;
        int cur = atomic_load_explicit(&pool->generation, memory_order_acquire);
        while (cur == my_gen) {
            if (atomic_load_explicit(&pool->shutdown, memory_order_relaxed)) return NULL;
            if (++spins >= FW_SPIN_BUDGET) break;
            fw_cpu_relax();
            cur = atomic_load_explicit(&pool->generation, memory_order_acquire);
        }

        /* Phase 2: spin exhausted with no work — park on the condvar. */
        if (cur == my_gen) {
            pthread_mutex_lock(&pool->mutex);
            while (!atomic_load_explicit(&pool->shutdown, memory_order_relaxed) &&
                   atomic_load_explicit(&pool->generation, memory_order_acquire) == my_gen) {
                pool->workers_waiting++;
                pthread_cond_wait(&pool->work_ready, &pool->mutex);
                pool->workers_waiting--;
            }
            int done = atomic_load_explicit(&pool->shutdown, memory_order_relaxed);
            pthread_mutex_unlock(&pool->mutex);
            if (done) return NULL;
        } else if (atomic_load_explicit(&pool->shutdown, memory_order_relaxed)) {
            return NULL;
        }

        my_gen = atomic_load_explicit(&pool->generation, memory_order_acquire);

        /* Do work */
        execute_chunks(pool);

        /* Signal completion (main may be spinning on active or parked). */
        if (atomic_fetch_sub(&pool->active, 1) == 1) {
            pthread_mutex_lock(&pool->mutex);
            pthread_cond_signal(&pool->work_done);
            pthread_mutex_unlock(&pool->mutex);
        }
    }
}

fw_pool* fw_pool_new(int num_threads) {
    if (num_threads <= 0)
        num_threads = detect_cores();
    /* Workers = total - 1 (main thread participates) */
    int n_workers = num_threads - 1;
    if (n_workers < 0) n_workers = 0;

    fw_pool *pool = (fw_pool *)calloc(1, sizeof(fw_pool));
    if (!pool) return NULL;

    pool->n_threads = n_workers;
    atomic_init(&pool->next_chunk, 0);
    atomic_init(&pool->active, 0);
    atomic_init(&pool->generation, 0);
    pool->workers_waiting = 0;
    atomic_init(&pool->shutdown, 0);
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_ready, NULL);
    pthread_cond_init(&pool->work_done, NULL);

    if (n_workers > 0) {
        pool->threads = (pthread_t *)calloc(n_workers, sizeof(pthread_t));
        if (!pool->threads) {
            pthread_mutex_destroy(&pool->mutex);
            pthread_cond_destroy(&pool->work_ready);
            pthread_cond_destroy(&pool->work_done);
            free(pool);
            return NULL;
        }
        for (int i = 0; i < n_workers; i++) {
            if (pthread_create(&pool->threads[i], NULL, worker_loop, pool) != 0) {
                pthread_mutex_lock(&pool->mutex);
                atomic_store(&pool->shutdown, 1);
                pthread_cond_broadcast(&pool->work_ready);
                pthread_mutex_unlock(&pool->mutex);
                for (int j = 0; j < i; j++)
                    pthread_join(pool->threads[j], NULL);
                pthread_mutex_destroy(&pool->mutex);
                pthread_cond_destroy(&pool->work_ready);
                pthread_cond_destroy(&pool->work_done);
                free(pool->threads);
                free(pool);
                return NULL;
            }
        }
    }
    return pool;
}

void fw_pool_free(fw_pool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mutex);
    atomic_store(&pool->shutdown, 1);
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
    for (int i = 0; i < pool->n_threads; i++)
        pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_ready);
    pthread_cond_destroy(&pool->work_done);
    free(pool->threads);
    free(pool);
}

void fw_for(fw_pool *pool, int count, fw_for_fn fn, void *arg) {
    if (count <= 0) return;

    /* Serial fallback when no pool */
    if (!pool || pool->n_threads == 0) {
        fn(0, count, arg);
        return;
    }

    int total_threads = pool->n_threads + 1;
    /* Oversplit slightly for load balance */
    int cs = count / (total_threads * 4);
    if (cs < 1) cs = 1;

    /* Set up work descriptor */
    pool->fn = fn;
    pool->arg = arg;
    pool->count = count;
    pool->chunk_size = cs;
    atomic_store(&pool->next_chunk, 0);
    atomic_store(&pool->active, total_threads);

    /* Publish work: the release-store on generation hands the descriptor above
     * to spinning workers (they acquire-load generation). The broadcast then
     * wakes any worker that already parked on the condvar. Spinning workers
     * usually observe the new generation before the broadcast, skipping it. */
    atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);
    /* Always broadcast under the mutex so a worker that has just parked is never
     * missed (skipping it conditionally would race the park path and deadlock).
     * The lock is uncontended while workers spin, so this is a cheap ~100ns. */
    pthread_mutex_lock(&pool->mutex);
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);

    /* Main thread participates */
    execute_chunks(pool);

    /* Signal main's completion */
    if (atomic_fetch_sub(&pool->active, 1) == 1) {
        /* Main was the last — no need to wait */
        return;
    }

    /* Bounded spin-join before parking: workers were woken lock-free and finish
     * within microseconds, so the spin usually completes the join with no
     * condvar round-trip. */
    for (int s = 0; s < FW_SPIN_BUDGET; s++) {
        if (atomic_load_explicit(&pool->active, memory_order_acquire) == 0) return;
        fw_cpu_relax();
    }

    /* Fallback: park until all workers finish. */
    pthread_mutex_lock(&pool->mutex);
    while (atomic_load(&pool->active) > 0)
        pthread_cond_wait(&pool->work_done, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}
