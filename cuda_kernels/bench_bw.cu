/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_bw.cu
 * @brief Pure streaming-read bandwidth probes for one GK210 die (sm_37, K80).
 *
 * Phase 0 of quant_layout_plan.md. `bandwidthTest --dtod` is a COPY: it
 * alternates reads and writes and pays GDDR5 bus turnaround, so its 177.5 GB/s
 * is a FLOOR on the ceiling our near-pure-read matvecs actually race. These
 * kernels read and do nothing else.
 *
 * Three families, all driven by cuda/tests/bench_bandwidth.c:
 *
 *   fc_bw_read_{u32,u128}_{ldg,ld}_u{1,2,4,8}
 *       Flat coalesced grid-stride read of a large device buffer. `u32` =
 *       32-bit loads (the width fc_matvec_q3k issues for quant words), `u128`
 *       = uint4. `ldg` = __ldg (read-only/texture path, what the matvec uses),
 *       `ld` = an ordinary global load, so the texture path can be blamed or
 *       cleared. u{N} = N independent accumulators, i.e. N loads in flight per
 *       thread. Every thread runs EXACTLY `iters` iterations so the host can
 *       account bytes exactly rather than estimate a ragged grid-stride tail.
 *
 *   fc_bw_mvskel_r{1,2,4,8}
 *       fc_matvec_q3k's READ pattern with the dequant deleted: per lane per
 *       152-byte super-block, one u32 quant word (lane-indexed, coalesced),
 *       one uint2 of scales (lane>>4, two distinct per warp) and one broadcast
 *       f32 `d` — the exact load set of fc_q3k_sb_lane_dot. R rows per warp,
 *       grid = rows/(warps*R). Answers "what would this matvec run at if
 *       decode were free", in GB/s directly comparable to fc_matvec_q3k's.
 *
 *   fc_bw_read_u32_ldg_u4_dead
 *       ELISION CONTROL. Identical to fc_bw_read_u32_ldg_u4 except it never
 *       stores its accumulator, so nvcc deletes every load. It exists to be
 *       run and to report a physically impossible number — the demonstration
 *       that this harness can detect the failure mode its other numbers would
 *       otherwise silently sit in.
 *
 * Accumulators are XOR, not FP: one instruction per loaded word, no rounding,
 * and the XOR of an index-derived fill pattern is a checksum the host gates on.
 *
 * Compile (kernel-local; does NOT touch the shared embedded PTX header):
 *   cd cuda_kernels && make bench_bw.ptx
 * The C driver loads bench_bw.ptx from disk at runtime. bench_bw is
 * deliberately NOT in the Makefile's KERNELS list, so ../cuda/fc_cuda_ptx.h —
 * which production links against — is untouched by this benchmark.
 *
 * Templates (not macros) carry the unroll factor so N is compile-time and the
 * accumulator array is guaranteed register-promoted; a runtime bound would
 * leave it in local memory and quietly measure something else (see
 * -Xptxas -v: the STACK FRAME line must read 0 bytes).
 *
 * Kepler notes: pure sm_37, __shfl_down is the non-.sync variant.
 *
 * @status Phase 0 harness: flat + matvec-shaped read probes + elision control.
 * @issues None
 * @todo None
 */

/* Repacked Q3_K super-block stride, mirroring FC_Q3K_RBLOCK_BYTES in
 * matvec_q3k.cu. The skeleton must walk the same 152-byte grid or its DRAM
 * transaction pattern (and its 8-byte-aligned straddle) stops matching. */
#define FC_BW_RBLOCK_BYTES 152

/* Warp XOR reduction, same 5-round __shfl_down shape as fc_q3k_warp_sum, so
 * every lane's accumulator is live and no lane's loads can be dropped. */
__device__ __forceinline__ unsigned int fc_bw_warp_xor(unsigned int v)
{
    v ^= (unsigned int)__shfl_down((int)v, 16);
    v ^= (unsigned int)__shfl_down((int)v, 8);
    v ^= (unsigned int)__shfl_down((int)v, 4);
    v ^= (unsigned int)__shfl_down((int)v, 2);
    v ^= (unsigned int)__shfl_down((int)v, 1);
    return v;
}

/* ------------------------------------------------------------------
 * Flat streaming read
 *
 * Thread t starts at word t and each iteration consumes N words spaced
 * `stride` = total threads apart, so every warp's N loads are each a
 * fully coalesced 128-byte (u32) or 512-byte (u128) request.
 * ------------------------------------------------------------------ */

template <int N>
__device__ __forceinline__ unsigned int bw_read_u32_ldg(const unsigned int *src,
                                                        int iters)
{
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    const unsigned int *p = src + (unsigned long long)blockIdx.x * blockDim.x
                          + threadIdx.x;
    unsigned int a[N], s = 0;
    int m, k;

#pragma unroll
    for (k = 0; k < N; k++)
        a[k] = 0;
    for (m = 0; m < iters; m++) {
#pragma unroll
        for (k = 0; k < N; k++)
            a[k] ^= __ldg(p + (unsigned long long)k * stride);
        p += (unsigned long long)N * stride;
    }
#pragma unroll
    for (k = 0; k < N; k++)
        s ^= a[k];
    return s;
}

/* Non-const, non-restrict src: nvcc cannot prove read-only aliasing against
 * the dst store, so this emits ordinary LD.E instead of LDG.E.CI. */
template <int N>
__device__ __forceinline__ unsigned int bw_read_u32_ld(unsigned int *src,
                                                       int iters)
{
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned int *p = src + (unsigned long long)blockIdx.x * blockDim.x
                    + threadIdx.x;
    unsigned int a[N], s = 0;
    int m, k;

#pragma unroll
    for (k = 0; k < N; k++)
        a[k] = 0;
    for (m = 0; m < iters; m++) {
#pragma unroll
        for (k = 0; k < N; k++)
            a[k] ^= p[(unsigned long long)k * stride];
        p += (unsigned long long)N * stride;
    }
#pragma unroll
    for (k = 0; k < N; k++)
        s ^= a[k];
    return s;
}

template <int N>
__device__ __forceinline__ unsigned int bw_read_u128_ldg(const uint4 *src,
                                                         int iters)
{
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    const uint4 *p = src + (unsigned long long)blockIdx.x * blockDim.x
                   + threadIdx.x;
    unsigned int a[N], s = 0;
    int m, k;

#pragma unroll
    for (k = 0; k < N; k++)
        a[k] = 0;
    for (m = 0; m < iters; m++) {
#pragma unroll
        for (k = 0; k < N; k++) {
            uint4 v = __ldg(p + (unsigned long long)k * stride);
            a[k] ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        p += (unsigned long long)N * stride;
    }
#pragma unroll
    for (k = 0; k < N; k++)
        s ^= a[k];
    return s;
}

template <int N>
__device__ __forceinline__ unsigned int bw_read_u128_ld(uint4 *src, int iters)
{
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    uint4 *p = src + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a[N], s = 0;
    int m, k;

#pragma unroll
    for (k = 0; k < N; k++)
        a[k] = 0;
    for (m = 0; m < iters; m++) {
#pragma unroll
        for (k = 0; k < N; k++) {
            uint4 v = p[(unsigned long long)k * stride];
            a[k] ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        p += (unsigned long long)N * stride;
    }
#pragma unroll
    for (k = 0; k < N; k++)
        s ^= a[k];
    return s;
}

/* ------------------------------------------------------------------
 * Matvec-shaped read skeleton
 *
 * Same warp/row decomposition and same three loads per lane per super-block
 * as fc_q3k_sb_lane_dot, minus the 7-instruction-per-element decode. Rows are
 * required (host-side) to divide warps*R exactly, so the early-out below never
 * fires on the benched shapes and every byte of w is read exactly once.
 * ------------------------------------------------------------------ */

template <int R>
__device__ __forceinline__ void bw_mvskel(const unsigned char *w,
                                          unsigned int *dst, int rows,
                                          int cols, int row_stride)
{
    int lane = (int)(threadIdx.x & 31u);
    int warp = (int)(threadIdx.x >> 5);
    int warps = (int)(blockDim.x >> 5);
    int row0 = ((int)blockIdx.x * warps + warp) * R;
    int n_sb = cols >> 8;
    const unsigned char *base;
    unsigned int a[R];
    int sb, r;

    if (row0 + R > rows)
        return;

    base = w + (size_t)row0 * (size_t)row_stride;
#pragma unroll
    for (r = 0; r < R; r++)
        a[r] = 0;

    for (sb = 0; sb < n_sb; sb++) {
#pragma unroll
        for (r = 0; r < R; r++) {
            const unsigned char *blk = base + (size_t)r * (size_t)row_stride
                                     + (size_t)sb * FC_BW_RBLOCK_BYTES;
            unsigned int q = __ldg((const unsigned int *)blk + lane);
            unsigned int d = __ldg((const unsigned int *)(blk + 144));
            uint2 sc = __ldg((const uint2 *)(blk + 128) + (lane >> 4));
            a[r] ^= q ^ d ^ sc.x ^ sc.y;
        }
    }

#pragma unroll
    for (r = 0; r < R; r++) {
        unsigned int v = fc_bw_warp_xor(a[r]);
        if (lane == 0)
            dst[row0 + r] = v;
    }
}

/* ------------------------------------------------------------------
 * extern "C" entry points (the driver looks these up by name)
 * ------------------------------------------------------------------ */

extern "C" {

#define FC_BW_READ_ENTRY(kind, N)                                             \
    __global__ void fc_bw_read_##kind##_u##N(void *src, unsigned int *dst,    \
                                             int iters)                       \
    {                                                                         \
        dst[blockIdx.x * blockDim.x + threadIdx.x] =                          \
            bw_read_##kind<N>((FC_BW_SRC_##kind)src, iters);                  \
    }

#define FC_BW_SRC_u32_ldg  const unsigned int *
#define FC_BW_SRC_u32_ld   unsigned int *
#define FC_BW_SRC_u128_ldg const uint4 *
#define FC_BW_SRC_u128_ld  uint4 *

FC_BW_READ_ENTRY(u32_ldg, 1)
FC_BW_READ_ENTRY(u32_ldg, 2)
FC_BW_READ_ENTRY(u32_ldg, 4)
FC_BW_READ_ENTRY(u32_ldg, 8)
FC_BW_READ_ENTRY(u32_ld, 1)
FC_BW_READ_ENTRY(u32_ld, 2)
FC_BW_READ_ENTRY(u32_ld, 4)
FC_BW_READ_ENTRY(u32_ld, 8)
FC_BW_READ_ENTRY(u128_ldg, 1)
FC_BW_READ_ENTRY(u128_ldg, 2)
FC_BW_READ_ENTRY(u128_ldg, 4)
FC_BW_READ_ENTRY(u128_ldg, 8)
FC_BW_READ_ENTRY(u128_ld, 1)
FC_BW_READ_ENTRY(u128_ld, 2)
FC_BW_READ_ENTRY(u128_ld, 4)
FC_BW_READ_ENTRY(u128_ld, 8)

/* ELISION CONTROL — same loads, result thrown away. Must report an
 * impossible bandwidth; if it does not, the harness is not measuring what it
 * thinks it is. Never treat its number as a result. */
__global__ void fc_bw_read_u32_ldg_u4_dead(void *src, unsigned int *dst,
                                           int iters)
{
    (void)dst;
    (void)bw_read_u32_ldg<4>((const unsigned int *)src, iters);
}

#define FC_BW_MVSKEL_ENTRY(R)                                                 \
    __global__ void fc_bw_mvskel_r##R(const unsigned char *w,                 \
                                      unsigned int *dst, int rows, int cols,  \
                                      int row_stride)                         \
    {                                                                         \
        bw_mvskel<R>(w, dst, rows, cols, row_stride);                         \
    }

FC_BW_MVSKEL_ENTRY(1)
FC_BW_MVSKEL_ENTRY(2)
FC_BW_MVSKEL_ENTRY(4)
FC_BW_MVSKEL_ENTRY(8)

} /* extern "C" */
