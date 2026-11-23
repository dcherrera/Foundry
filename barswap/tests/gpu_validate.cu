/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * gpu_validate.cu — Simple GPU validation test for K80 BAR swap
 *
 * Tests:
 *   1. CUDA device detection and properties
 *   2. GPU memory allocation (1 GB)
 *   3. Host-to-device memcpy
 *   4. Simple kernel execution (vector add)
 *   5. Device-to-host memcpy + verification
 *   6. Memory bandwidth measurement
 *
 * Build: nvcc -O3 -arch sm_37 gpu_validate.cu -o gpu_validate
 * Run:   ./gpu_validate
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

/* Simple vector add kernel */
__global__ void vector_add(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

/* Memory bandwidth test kernel — reads + writes */
__global__ void bandwidth_kernel(float *dst, const float *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[i] * 2.0f + 1.0f;
}

static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

int main(void)
{
    int device_count = 0;

    printf("=== K80 GPU Validation Test ===\n\n");

    /* Test 1: Device detection */
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    printf("[1] CUDA devices found: %d\n", device_count);
    if (device_count == 0) {
        printf("FAIL: No CUDA devices\n");
        return 1;
    }

    for (int d = 0; d < device_count; d++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, d));
        printf("    GPU %d: %s\n", d, prop.name);
        printf("      Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("      Global memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("      SMs: %d, Clock: %d MHz\n", prop.multiProcessorCount, prop.clockRate / 1000);
        printf("      Memory clock: %d MHz, Bus width: %d bit\n",
               prop.memoryClockRate / 1000, prop.memoryBusWidth);
        printf("      Peak bandwidth: %.1f GB/s\n",
               2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) / 1.0e9);
    }

    CHECK_CUDA(cudaSetDevice(0));
    printf("\n[1] PASS — device detected and properties read\n\n");

    /* Test 2: Memory allocation (1 GB) */
    size_t alloc_size = 256 * 1024 * 1024; /* 256M floats = 1 GB */
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;

    printf("[2] Allocating 3 GB GPU memory...\n");
    CHECK_CUDA(cudaMalloc(&d_a, alloc_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, alloc_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, alloc_size * sizeof(float)));
    printf("[2] PASS — 3 GB allocated on GPU\n\n");

    /* Test 3: Host-to-device transfer */
    int test_n = 1024 * 1024; /* 1M elements for correctness test */
    float *h_a = (float *)malloc(test_n * sizeof(float));
    float *h_b = (float *)malloc(test_n * sizeof(float));
    float *h_c = (float *)malloc(test_n * sizeof(float));

    for (int i = 0; i < test_n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(test_n - i);
    }

    printf("[3] Copying %d elements to GPU...\n", test_n);
    CHECK_CUDA(cudaMemcpy(d_a, h_a, test_n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, test_n * sizeof(float), cudaMemcpyHostToDevice));
    printf("[3] PASS — host-to-device memcpy\n\n");

    /* Test 4: Kernel execution */
    int threads = 256;
    int blocks = (test_n + threads - 1) / threads;

    printf("[4] Running vector_add kernel (%d blocks x %d threads)...\n", blocks, threads);
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, test_n);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("[4] PASS — kernel executed\n\n");

    /* Test 5: Device-to-host + verify */
    printf("[5] Copying result back and verifying...\n");
    CHECK_CUDA(cudaMemcpy(h_c, d_c, test_n * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < test_n; i++) {
        float expected = (float)i + (float)(test_n - i);
        if (h_c[i] != expected) {
            if (errors < 5)
                printf("    MISMATCH at [%d]: got %.1f, expected %.1f\n", i, h_c[i], expected);
            errors++;
        }
    }
    if (errors == 0)
        printf("[5] PASS — all %d results correct\n\n", test_n);
    else {
        printf("[5] FAIL — %d mismatches out of %d\n\n", errors, test_n);
        return 1;
    }

    /* Test 6: Memory bandwidth */
    printf("[6] Memory bandwidth test (1 GB transfers)...\n");
    size_t bw_n = alloc_size; /* 256M floats = 1 GB */
    int bw_blocks = (bw_n + threads - 1) / threads;

    /* Warm up */
    bandwidth_kernel<<<bw_blocks, threads>>>(d_b, d_a, bw_n);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Measure kernel bandwidth */
    double t0 = get_time_ms();
    int iters = 10;
    for (int i = 0; i < iters; i++) {
        bandwidth_kernel<<<bw_blocks, threads>>>(d_b, d_a, bw_n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double t1 = get_time_ms();
    double kernel_bw = (2.0 * bw_n * sizeof(float) * iters) / ((t1 - t0) / 1000.0) / 1.0e9;

    /* Measure H2D bandwidth */
    float *h_big = (float *)malloc(bw_n * sizeof(float));
    memset(h_big, 0, bw_n * sizeof(float));

    t0 = get_time_ms();
    CHECK_CUDA(cudaMemcpy(d_a, h_big, bw_n * sizeof(float), cudaMemcpyHostToDevice));
    t1 = get_time_ms();
    double h2d_bw = (bw_n * sizeof(float)) / ((t1 - t0) / 1000.0) / 1.0e9;

    /* Measure D2H bandwidth */
    t0 = get_time_ms();
    CHECK_CUDA(cudaMemcpy(h_big, d_a, bw_n * sizeof(float), cudaMemcpyDeviceToHost));
    t1 = get_time_ms();
    double d2h_bw = (bw_n * sizeof(float)) / ((t1 - t0) / 1000.0) / 1.0e9;

    printf("    Kernel read+write bandwidth: %.1f GB/s\n", kernel_bw);
    printf("    Host-to-Device bandwidth:    %.1f GB/s\n", h2d_bw);
    printf("    Device-to-Host bandwidth:    %.1f GB/s\n", d2h_bw);
    printf("[6] PASS — bandwidth measured\n\n");

    /* Cleanup */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_big);

    printf("=== ALL 6 TESTS PASSED ===\n");
    printf("GPU is fully functional.\n");

    return 0;
}
