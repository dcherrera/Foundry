/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * vram_mapping.c — Test BAR1 ↔ CUDA cross-domain VRAM access
 *
 * Instead of scanning all 16GB of BAR1 (which hangs on unmapped regions),
 * this test uses a targeted approach:
 *
 *   1. CUDA alloc + write a pattern
 *   2. Use cuMemGetInfo to understand VRAM layout
 *   3. Try reading specific BAR1 offsets near the CUDA dptr value
 *   4. Test BAR1 write → CUDA read (the critical cross-domain test)
 *
 * Usage:
 *   sudo LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64 ./cuda/tests/vram_mapping
 *
 * Requires: bar_swap loaded, die acquired, nvidia loaded, root.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <time.h>

#define TEST_SIZE    (1024 * 1024)    /* 1 MB */
#define PATTERN_A    0xDEADBEEF
#define PATTERN_B    0xCAFEBABE

/* Known BAR1 physical address for K80 on this server */
#define BAR1_PHYS    0xC00000000ULL
#define BAR1_FULL    (16ULL * 1024 * 1024 * 1024)

/* CUDA types */
typedef int CUresult;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
#define CUDA_SUCCESS 0

/* Signal handler for catching bus errors on BAR1 reads */
static sigjmp_buf jump_buf;
static volatile int got_signal = 0;

static void bus_handler(int sig) {
    (void)sig;
    got_signal = 1;
    siglongjmp(jump_buf, 1);
}

/* Safe read of a single word from BAR1. Returns 1 on success, 0 on bus error. */
static int safe_read32(volatile uint32_t *addr, uint32_t *out) {
    struct sigaction sa, old_bus, old_segv;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = bus_handler;
    sa.sa_flags = 0;
    sigaction(SIGBUS, &sa, &old_bus);
    sigaction(SIGSEGV, &sa, &old_segv);

    got_signal = 0;
    int ok = 0;
    if (sigsetjmp(jump_buf, 1) == 0) {
        *out = *addr;
        ok = 1;
    }

    sigaction(SIGBUS, &old_bus, NULL);
    sigaction(SIGSEGV, &old_segv, NULL);
    return ok;
}

/* Try to read n_words from BAR1 at given byte offset. Returns count read. */
static int try_read_bar1(volatile uint8_t *bar1, int64_t byte_offset,
                         uint32_t *buf, int n_words) {
    volatile uint32_t *src = (volatile uint32_t *)(bar1 + byte_offset);
    for (int i = 0; i < n_words; i++) {
        uint32_t val;
        if (!safe_read32(&src[i], &val)) return i;
        buf[i] = val;
    }
    return n_words;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    printf("=== VRAM Mapping Test v3 ===\n\n");

    int n_words = TEST_SIZE / sizeof(uint32_t);
    uint32_t *host_buf = malloc(TEST_SIZE);

    /* Step 1: mmap BAR1 */
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    /* Map full BAR1 — reads to unmapped regions will be caught by signal handler */
    volatile uint8_t *bar1 = mmap(NULL, BAR1_FULL, PROT_READ | PROT_WRITE,
                                   MAP_SHARED, mem_fd, BAR1_PHYS);
    if (bar1 == MAP_FAILED) {
        perror("mmap BAR1");
        close(mem_fd);
        return 1;
    }
    printf("BAR1 mapped at %p (16 GB)\n", (void *)bar1);

    /* Step 2: Init CUDA */
    void *libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!libcuda) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    CUresult (*cuInit)(unsigned int) = dlsym(libcuda, "cuInit");
    CUresult (*cuDeviceGet)(int *, int) = dlsym(libcuda, "cuDeviceGet");
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, int) =
        dlsym(libcuda, "cuCtxCreate_v2");
    CUresult (*cuCtxDestroy)(CUcontext) = dlsym(libcuda, "cuCtxDestroy_v2");
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t) = dlsym(libcuda, "cuMemAlloc_v2");
    CUresult (*cuMemFree)(CUdeviceptr) = dlsym(libcuda, "cuMemFree_v2");
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t) =
        dlsym(libcuda, "cuMemcpyHtoD_v2");
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t) =
        dlsym(libcuda, "cuMemcpyDtoH_v2");
    CUresult (*cuCtxSynchronize)(void) = dlsym(libcuda, "cuCtxSynchronize");
    CUresult (*cuMemGetInfo)(size_t *, size_t *) = dlsym(libcuda, "cuMemGetInfo_v2");

    CUresult err = cuInit(0);
    if (err) { fprintf(stderr, "cuInit: %d\n", err); return 1; }

    int dev;
    cuDeviceGet(&dev, 0);
    CUcontext cuda_ctx;
    err = cuCtxCreate(&cuda_ctx, 0, dev);
    if (err) { fprintf(stderr, "cuCtxCreate: %d\n", err); return 1; }

    size_t free_mem, total_mem;
    cuMemGetInfo(&free_mem, &total_mem);
    printf("CUDA: total=%zu MB, free=%zu MB\n",
           total_mem / (1024*1024), free_mem / (1024*1024));

    /* Step 3: Allocate and write pattern */
    CUdeviceptr dptr;
    err = cuMemAlloc(&dptr, TEST_SIZE);
    if (err) { fprintf(stderr, "cuMemAlloc: %d\n", err); return 1; }
    printf("cuMemAlloc: dptr=0x%llx\n\n", (unsigned long long)dptr);

    for (int i = 0; i < n_words; i++) host_buf[i] = PATTERN_A;
    cuMemcpyHtoD(dptr, host_buf, TEST_SIZE);
    cuCtxSynchronize();

    /* Verify CUDA side */
    memset(host_buf, 0, TEST_SIZE);
    cuMemcpyDtoH(host_buf, dptr, TEST_SIZE);
    int ok = 1;
    for (int i = 0; i < n_words; i++) if (host_buf[i] != PATTERN_A) { ok = 0; break; }
    printf("CUDA write/read verify: %s\n\n", ok ? "OK" : "FAIL");

    /* Step 4: Try to find the CUDA data in BAR1.
     *
     * On Kepler (GK210), GPU virtual addresses typically start around
     * 0x500000000. The BAR1 aperture maps physical VRAM starting at
     * physical address 0. CUDA's GPU VA might have a fixed offset from
     * BAR1 physical, or the mapping might be complex.
     *
     * Strategy: try a set of candidate BAR1 offsets derived from the dptr:
     *   - dptr itself (if GPU VA == BAR1 offset)
     *   - dptr & 0xFFFFFFFF (low 32 bits)
     *   - dptr - 0x500000000 (subtract typical Kepler VA base)
     *   - Small offsets near 0 (CUDA might alloc at start of VRAM)
     *   - Each try uses safe_read with signal handler to avoid hangs
     */
    printf("--- Probing BAR1 for CUDA data ---\n\n");

    int64_t candidates[] = {
        /* Direct mapping guesses */
        (int64_t)(dptr & 0xFFFFFFFF),                    /* low 32 bits */
        (int64_t)(dptr - 0x500000000ULL),                /* Kepler VA base */
        (int64_t)(dptr - 0x400000000ULL),
        (int64_t)(dptr - 0x300000000ULL),
        (int64_t)(dptr - 0x200000000ULL),
        (int64_t)(dptr - 0x100000000ULL),
        (int64_t)(dptr),
        /* Near start of VRAM */
        0,
        1 * 1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024,
        128 * 1024 * 1024,
        256 * 1024 * 1024,
        512 * 1024 * 1024,
        1024LL * 1024 * 1024,
    };
    int n_candidates = sizeof(candidates) / sizeof(candidates[0]);

    int64_t found_offset = -1;

    for (int c = 0; c < n_candidates; c++) {
        int64_t off = candidates[c];
        if (off < 0 || off + TEST_SIZE > (int64_t)BAR1_FULL) continue;

        /* Try reading first word safely */
        uint32_t val = 0;
        int readable = safe_read32((volatile uint32_t *)(bar1 + off), &val);

        if (!readable) {
            printf("  offset 0x%09llx: BUS ERROR (unmapped)\n",
                   (unsigned long long)off);
            continue;
        }

        if (val == PATTERN_A) {
            /* Possible match! Read more words to confirm */
            int match = 0;
            for (int i = 0; i < 16 && i < n_words; i++) {
                uint32_t v;
                if (safe_read32((volatile uint32_t *)(bar1 + off) + i, &v) && v == PATTERN_A)
                    match++;
            }
            printf("  offset 0x%09llx: FIRST WORD MATCH! (%d/16 confirmed)\n",
                   (unsigned long long)off, match);
            if (match >= 12) {
                found_offset = off;
                printf("  >>> FOUND IT! BAR1 offset = 0x%llx <<<\n",
                       (unsigned long long)off);
                break;
            }
        } else {
            printf("  offset 0x%09llx: readable, val=0x%08X (no match)\n",
                   (unsigned long long)off, val);
        }
    }

    if (found_offset < 0) {
        /* Try a broader search — read first word at 1MB intervals, skip on bus error */
        printf("\n  Broad search (1MB steps, first 2GB)...\n");
        for (int64_t off = 0; off < 2LL * 1024 * 1024 * 1024; off += 1024 * 1024) {
            uint32_t val = 0;
            if (!safe_read32((volatile uint32_t *)(bar1 + off), &val)) continue;
            if (val == PATTERN_A) {
                int match = 0;
                for (int i = 0; i < 16; i++) {
                    uint32_t v;
                    if (safe_read32((volatile uint32_t *)(bar1 + off) + i, &v) && v == PATTERN_A)
                        match++;
                }
                if (match >= 12) {
                    found_offset = off;
                    printf("  >>> FOUND at BAR1 offset 0x%llx (%lld MB) <<<\n",
                           (unsigned long long)off, (long long)(off / (1024*1024)));
                    break;
                }
            }
        }
    }

    if (found_offset < 0) {
        printf("\nCould not find CUDA data in BAR1.\n");
        printf("The GPU VA → BAR1 mapping may require page table inspection.\n");
        cuMemFree(dptr);
        cuCtxDestroy(cuda_ctx);
        goto done;
    }

    printf("\n  GPU VA 0x%llx → BAR1 offset 0x%llx (delta=0x%llx)\n\n",
           (unsigned long long)dptr, (unsigned long long)found_offset,
           (unsigned long long)((int64_t)dptr - found_offset));

    /* ================================================================
     * Experiment 2: BAR1 write → CUDA read
     * ================================================================ */
    printf("--- Exp 2: BAR1 write → CUDA read ---\n");

    volatile uint32_t *target = (volatile uint32_t *)(bar1 + found_offset);
    for (int i = 0; i < n_words; i++) target[i] = PATTERN_B;
    __sync_synchronize();
    printf("Wrote 0x%08X via BAR1\n", PATTERN_B);

    memset(host_buf, 0, TEST_SIZE);
    cuMemcpyDtoH(host_buf, dptr, TEST_SIZE);
    int m = 0;
    for (int i = 0; i < n_words; i++) if (host_buf[i] == PATTERN_B) m++;
    printf("CUDA readback: %d/%d match (%.0f%%)\n", m, n_words, 100.0f*m/n_words);

    if (m == n_words) {
        printf("\n  >>> BAR1 WRITE VISIBLE FROM CUDA! <<<\n");
        printf("  We can write VRAM via BAR1 and CUDA sees it.\n");

        /* Measure BAR1 write speed */
        double t0 = now_ms();
        int reps = 10;
        for (int r = 0; r < reps; r++) {
            for (int i = 0; i < n_words; i++) target[i] = PATTERN_A;
            __sync_synchronize();
        }
        double elapsed = now_ms() - t0;
        double mbps = (double)TEST_SIZE * reps / (elapsed / 1000.0) / (1024*1024);
        printf("\n  BAR1 write speed: %.1f MB/s (%d x 1MB in %.1fms)\n",
               mbps, reps, elapsed);
    }

    cuMemFree(dptr);
    cuCtxDestroy(cuda_ctx);

done:
    munmap((void *)bar1, BAR1_FULL);
    close(mem_fd);
    free(host_buf);
    dlclose(libcuda);
    printf("\nDone.\n");
    return 0;
}
