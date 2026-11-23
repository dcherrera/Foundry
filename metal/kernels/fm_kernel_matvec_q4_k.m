/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Portions modelled on ggml (MIT License, Copyright (c) 2023-2026 The ggml
 * authors, https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file fm_kernel_matvec_q4_k.m
 * @brief Host dispatch and MSL source for the GGUF Q4_K kernels: fused
 *   dequant+matvec, the tiled simdgroup batched GEMM, the dequantize-to-F16
 *   helper, and the F16 MPS prefill GEMM built on top of it.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_matvec_q4_k.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_matvec_lane_split.h"
#include "metal/kernels/fm_kernel_registry.h"

/* Lanes one output row gets, for whichever of the two matvec kernels is being
 * encoded.  Both call this: their bit-exactness contract is that they share a
 * decomposition, so they must share the arithmetic that defines it. */
static int fm_q4_k_lanes_per_row(int cols) {
    return fm_matvec_lanes_per_row(cols / FM_Q4_K_BLOCK_WEIGHTS);
}

void fm_matvec_q4_k_at(fm_bufref w, fm_bufref x, fm_bufref out,
                       int rows, int cols, int row_stride_bytes) {
    if (!fm_metal_is_initialized() || !w.buffer || !x.buffer || !out.buffer) return;

    /* A row of `cols` weights is only cols/256 super-blocks, so at the MoE
     * expert shapes it cannot fill a SIMD group on its own.  Give the row just
     * the lanes it can keep busy and pack several rows into the group instead;
     * the decode path is untouched, so no lane re-reads another lane's header. */
    int lanes = fm_q4_k_lanes_per_row(cols);
    fm_fc_values fc;
    fm_fc_reset(&fc);
    if (fm_fc_set_int(&fc, FM_Q4_K_FC_LANES_PER_ROW, lanes) != 0) return;

    fm_bufref refs[] = { w, x, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols,
                          (uint32_t)row_stride_bytes };
    /* The weight offset is applied by the binding, so the kernel's own row
     * indexing is unchanged whether the tensor sits at the start of its own
     * buffer or deep inside a shared mmap region. */
    fm_dispatch_threadgroups_refs(@"matvec_q4_k", &fc, refs, 3, consts, 3,
                                  fm_matvec_threadgroup_count(rows, lanes),
                                  FM_MATVEC_SIMD_LANES);
}

void fm_matvec_q4_k(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                    int rows, int cols, int row_stride_bytes) {
    if (!w || !x || !out) return;
    fm_matvec_q4_k_at(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                      rows, cols, row_stride_bytes);
}

/* Encodes matvec_q4_k_nx for `tile_width` columns starting at `col_base`,
 * with the tile width bound as a function constant so the kernel's
 * accumulator loop has a compile-time bound. */
static void fm_matvec_q4_k_nx_tiles(fm_bufref w, fm_bufref x, fm_bufref out,
                                    int rows, int cols, int row_stride_bytes,
                                    int col_base, int tile_width,
                                    int tile_count) {
    int lanes = fm_q4_k_lanes_per_row(cols);
    fm_fc_values fc;
    fm_fc_reset(&fc);
    fm_fc_set_int(&fc, FM_Q4_K_NX_TILE_CONSTANT_INDEX, tile_width);
    if (fm_fc_set_int(&fc, FM_Q4_K_FC_LANES_PER_ROW, lanes) != 0) return;

    fm_bufref refs[] = { w, x, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols,
                          (uint32_t)row_stride_bytes,
                          (uint32_t)col_base, (uint32_t)tile_count };
    /* One threadgroup per (row group, column tile).  Splitting the batch
     * across threadgroups rather than looping inside one keeps the grid
     * large enough to fill the GPU when rows is small.  The row group is the
     * same lane split the single-column kernel uses — that shared
     * decomposition is what makes a batched column bit-identical to it. */
    fm_dispatch_threadgroups_refs(@"matvec_q4_k_nx", &fc, refs, 3, consts, 5,
                                  fm_matvec_threadgroup_count(rows, lanes) * tile_count,
                                  FM_MATVEC_SIMD_LANES);
}

void fm_matvec_q4_k_nx(fm_bufref w, fm_bufref x, fm_bufref out,
                       int rows, int cols, int row_stride_bytes, int n) {
    if (!fm_metal_is_initialized() || !w.buffer || !x.buffer || !out.buffer) return;
    if (rows <= 0 || cols <= 0 || n <= 0) return;

    /* At most two dispatches: the full-width tiles, then the remainder as
     * one narrower tile of its own width.  Specializing the remainder
     * rather than padding it means no lane ever accumulates a column that
     * is thrown away. */
    int full_tiles = n / FM_Q4_K_NX_MAX_TILE;
    int remainder  = n % FM_Q4_K_NX_MAX_TILE;

    if (full_tiles > 0)
        fm_matvec_q4_k_nx_tiles(w, x, out, rows, cols, row_stride_bytes,
                                0, FM_Q4_K_NX_MAX_TILE, full_tiles);
    if (remainder > 0)
        fm_matvec_q4_k_nx_tiles(w, x, out, rows, cols, row_stride_bytes,
                                full_tiles * FM_Q4_K_NX_MAX_TILE, remainder, 1);
}

void fm_sgemm_q4_k_at(fm_bufref w, fm_bufref x, fm_bufref out,
                      int rows, int cols, int row_stride_bytes, int B) {
    if (!fm_metal_is_initialized() || !w.buffer || !x.buffer || !out.buffer) return;
    if (B <= 0) return;

    fm_bufref refs[] = { w, x, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols,
                          (uint32_t)row_stride_bytes, (uint32_t)B };
    /* Tiled simdgroup_matrix GEMM: one threadgroup per 64-row tile, 256 threads
     * (8 simdgroups). Threadgroup memory is statically declared in the kernel. */
    int ntiles = (rows + 63) / 64;
    fm_dispatch_threadgroups_refs(@"gemm_q4_k", NULL, refs, 3, consts, 4, ntiles, 256);
}

void fm_sgemm_q4_k(fm_buffer *w, fm_buffer *x, fm_buffer *out,
                   int rows, int cols, int row_stride_bytes, int B) {
    if (!w || !x || !out) return;
    fm_sgemm_q4_k_at(fm_bufref_at(w, 0), fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                     rows, cols, row_stride_bytes, B);
}

/* Persistent F16 staging for the prefill GEMM (dequantized weight, converted
 * activations, F16 result), grown on demand and reused across calls. */
static fm_buffer *g_pf_w = NULL;  static size_t g_pf_w_cap = 0;
static fm_buffer *g_pf_x = NULL;  static size_t g_pf_x_cap = 0;
static fm_buffer *g_pf_o = NULL;  static size_t g_pf_o_cap = 0;
static fm_buffer *pf_stage(fm_buffer **b, size_t *cap, size_t bytes) {
    if (*b && *cap >= bytes) return *b;
    if (*b) fm_buffer_free(*b);
    *b = fm_buffer_alloc(bytes);
    *cap = *b ? bytes : 0;
    return *b;
}

void fm_gemm_q4_k_f16_at(fm_bufref w, fm_bufref x_f32, fm_bufref out_f32,
                         int M, int N, int K, int row_stride_bytes) {
    if (!fm_metal_is_initialized() || !w.buffer || !x_f32.buffer || !out_f32.buffer) return;
    if (M <= 0 || N <= 0 || K <= 0) return;

    fm_buffer *wf = pf_stage(&g_pf_w, &g_pf_w_cap, (size_t)N * K * sizeof(uint16_t));
    fm_buffer *xf = pf_stage(&g_pf_x, &g_pf_x_cap, (size_t)M * K * sizeof(uint16_t));
    fm_buffer *of = pf_stage(&g_pf_o, &g_pf_o_cap, (size_t)M * N * sizeof(uint16_t));
    if (!wf || !xf || !of) return;

    /* 1. Dequantize the Q4_K weight to F16 [N, K]. */
    { fm_bufref refs[] = { w, fm_bufref_at(wf, 0) };
      uint32_t c[] = { (uint32_t)N, (uint32_t)K, (uint32_t)row_stride_bytes };
      fm_dispatch_1d_refs(@"dequant_q4_k_f16", NULL, refs, 2, c, 3, N * (K / 256)); }

    /* 2. Convert activations F32 [M, K] -> F16.  Encoded here rather than
     * through fm_convert_f32_to_f16 only because that entry point binds at
     * offset 0, and x may be a slice of a larger staging buffer. */
    { fm_bufref refs[] = { x_f32, fm_bufref_at(xf, 0) };
      uint32_t c[] = { (uint32_t)(M * K) };
      fm_dispatch_1d_refs(@"cvt_f32_f16", NULL, refs, 2, c, 1, M * K); }

    /* 3. F16 MPS matmul: of[M, N] = xf[M, K] @ wf[N, K]^T. */
    MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
        rowBytes:K * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K
        rowBytes:K * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
        rowBytes:N * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:xf->mtl descriptor:dA];
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:wf->mtl descriptor:dB];
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:of->mtl descriptor:dC];
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:fm_metal_device()
        transposeLeft:NO transposeRight:YES resultRows:M resultColumns:N interiorColumns:K
        alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> cmdbuf = fm_get_command_buffer();
    [mm encodeToCommandBuffer:cmdbuf leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    fm_submit_if_not_batching();

    /* 4. Convert the F16 result back to F32 [M, N]. */
    { fm_bufref refs[] = { fm_bufref_at(of, 0), out_f32 };
      uint32_t c[] = { (uint32_t)(M * N) };
      fm_dispatch_1d_refs(@"cvt_f16_f32", NULL, refs, 2, c, 1, M * N); }
}

void fm_gemm_q4_k_f16(fm_buffer *w, fm_buffer *x_f32, fm_buffer *out_f32,
                      int M, int N, int K, int row_stride_bytes) {
    if (!w || !x_f32 || !out_f32) return;
    fm_gemm_q4_k_f16_at(fm_bufref_at(w, 0), fm_bufref_at(x_f32, 0),
                        fm_bufref_at(out_f32, 0), M, N, K, row_stride_bytes);
}

NSString *fm_kernel_msl_matvec_q4_k(void)
{
    return @
    "/* ============================================================\n"
    " * Q4_K fused dequant + matvec (no scratch roundtrip)\n"
    " *\n"
    " * One threadgroup is one SIMD group (32 lanes) and covers\n"
    " * 32/FMQ4K_LANES output rows: a row gets as many lanes as it has\n"
    " * super-blocks, capped at 32.  Weights stay GPU-resident as raw Q4_K\n"
    " * super-blocks (144 bytes / 256 weights), so only ~0.5625 bytes/weight are\n"
    " * read. Each lane dequantizes whole 256-element super-blocks (strided by\n"
    " * its row's lane count) and FMA-accumulates against x; a shuffle butterfly\n"
    " * reduces the row's partials.  Header decode stays once per super-block —\n"
    " * splitting a row across MORE lanes than it has blocks is what makes eight\n"
    " * lanes re-decode one header, and that is the cost this avoids.\n"
    " * Mirrors pn_dequantize_block_q4_k exactly (value = d*sc*q - dmin*mn).\n"
    " * ============================================================ */\n"
    "\n"
    "/* Lanes one output row gets; 32 (one row per SIMD group) unspecialized.\n"
    " * Shared by matvec_q4_k and matvec_q4_k_nx — the two are bit-identical\n"
    " * per column only while they split a row the same way. */\n"
    "constant int fmq4k_lanes_in [[function_constant(81)]];\n"
    "constant uint FMQ4K_LANES =\n"
    "    is_function_constant_defined(fmq4k_lanes_in) ? uint(fmq4k_lanes_in) : 32u;\n"
    "\n"
    "kernel void matvec_q4_k(\n"
    "    device const uchar* W        [[buffer(0)]],\n"
    "    device const float* x        [[buffer(1)]],\n"
    "    device float*       out      [[buffer(2)]],\n"
    "    constant uint& rows          [[buffer(3)]],\n"
    "    constant uint& cols          [[buffer(4)]],\n"
    "    constant uint& row_stride    [[buffer(5)]],\n"
    "    uint group [[threadgroup_position_in_grid]],\n"
    "    uint tid   [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    const uint lanes = FMQ4K_LANES;\n"
    "    uint lane_in_row = tid & (lanes - 1u);\n"
    "    uint row = group * (32u / lanes) + tid / lanes;\n"
    "    uint n_blocks = cols / 256u;\n"
    "\n"
    "    // Lanes owning a row past the end accumulate nothing but still enter the\n"
    "    // reduction, so all 32 lanes are active at the shuffles.\n"
    "    float acc = 0.0f;\n"
    "    if (row < rows) {\n"
    "        device const uchar* base = W + (ulong)row * row_stride;\n"
    "        for (uint b = lane_in_row; b < n_blocks; b += lanes) {\n"
    "            device const uchar* blk = base + (ulong)b * 144u;\n"
    "            device const half*  dh  = (device const half*)blk;\n"
    "            float d    = float(dh[0]);\n"
    "            float dmin = float(dh[1]);\n"
    "            device const uchar* sr = blk + 4;       // scales[12]\n"
    "            device const uchar* qs = blk + 16;      // qs[128]\n"
    "            uchar sc[8]; uchar mn[8];\n"
    "            for (uint j = 0; j < 4; j++) { sc[j] = sr[j] & 63; mn[j] = sr[j+4] & 63; }\n"
    "            for (uint j = 4; j < 8; j++) {\n"
    "                sc[j] = (sr[j+4] & 0xF) | ((sr[j-4] >> 6) << 4);\n"
    "                mn[j] = (sr[j+4] >> 4)  | ((sr[j]   >> 6) << 4);\n"
    "            }\n"
    "            device const float* xb = x + (ulong)b * 256u;\n"
    "            for (uint p = 0; p < 4; p++) {      // 4 pairs share 32 qs bytes\n"
    "                float d1 = d * float(sc[2*p]);\n"
    "                float m1 = dmin * float(mn[2*p]);\n"
    "                float d2 = d * float(sc[2*p+1]);\n"
    "                float m2 = dmin * float(mn[2*p+1]);\n"
    "                device const uchar* q = qs + p * 32u;\n"
    "                uint o = p * 64u;\n"
    "                for (uint l = 0; l < 32; l++) {\n"
    "                    uchar qb = q[l];\n"
    "                    acc += (d1 * float(qb & 0xF) - m1) * xb[o + l];\n"
    "                    acc += (d2 * float(qb >> 4)  - m2) * xb[o + 32u + l];\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    float row_sum = acc;\n"
    "    if (lanes == 32u) {\n"
    "        row_sum = simd_sum(acc);\n"
    "    } else {\n"
    "        for (uint off = lanes >> 1; off > 0u; off >>= 1)\n"
    "            row_sum += simd_shuffle_xor(row_sum, ushort(off));\n"
    "    }\n"
    "    if (lane_in_row == 0u && row < rows) out[row] = row_sum;\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * Q4_K matvec batched over n activation columns.\n"
    " *\n"
    " * out[t][rows] = dequant(W) . X[t], with X = [n, cols] and\n"
    " * out = [n, rows].  The batch axis is the ACTIVATION COLUMNS; rows are\n"
    " * already the parallel axis.  One threadgroup per (row group, column\n"
    " * tile); a lane decodes its super-blocks ONCE and FMAs each dequantized\n"
    " * weight into TILE accumulators, so the dequant amortizes across the\n"
    " * tile.\n"
    " *\n"
    " * TILE is a function constant, not an argument.  That is the whole\n"
    " * point: with a runtime bound the accumulator loop cannot unroll, so\n"
    " * acc[] would live in thread-local memory instead of registers and\n"
    " * the batching would cost more than it saves.  The host emits full\n"
    " * FM_Q4_K_NX_MAX_TILE tiles and, if n does not divide evenly, one\n"
    " * remainder tile specialized to its own exact width — so every column\n"
    " * a lane accumulates is a column that gets stored.\n"
    " *\n"
    " * The lane split, the block walk, the inner (p, l) order and the final\n"
    " * reduction are character-for-character the single-column kernel's, which\n"
    " * is what makes each column bit-identical to fm_matvec_q4_k_at on that\n"
    " * column.  FMQ4K_LANES is shared with it for exactly that reason.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fmq4k_nx_tile_in [[function_constant(80)]];\n"
    "constant int FMQ4K_NX_TILE =\n"
    "    is_function_constant_defined(fmq4k_nx_tile_in) ? fmq4k_nx_tile_in : 1;\n"
    "\n"
    "kernel void matvec_q4_k_nx(\n"
    "    device const uchar* W        [[buffer(0)]],\n"
    "    device const float* X        [[buffer(1)]],\n"   // [n, cols]
    "    device float*       out      [[buffer(2)]],\n"   // [n, rows]
    "    constant uint& rows          [[buffer(3)]],\n"
    "    constant uint& cols          [[buffer(4)]],\n"
    "    constant uint& row_stride    [[buffer(5)]],\n"
    "    constant uint& col_base      [[buffer(6)]],\n"
    "    constant uint& tile_count    [[buffer(7)]],\n"
    "    uint tgid  [[threadgroup_position_in_grid]],\n"
    "    uint tid   [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    // acc[] is declared at the widest tile so its size is a literal;\n"
    "    // every loop over it stops at the specialized FMQ4K_NX_TILE, which\n"
    "    // is a compile-time bound, so the array stays in registers.\n"
    "    const uint TILE  = (uint)FMQ4K_NX_TILE;\n"
    "    const uint lanes = FMQ4K_LANES;\n"
    "    uint lane_in_row = tid & (lanes - 1u);\n"
    "    uint row = (tgid / tile_count) * (32u / lanes) + tid / lanes;\n"
    "    uint t0  = col_base + (tgid % tile_count) * TILE;\n"
    "\n"
    "    uint n_blocks = cols / 256u;\n"
    "    device const float* xcol[8];\n"
    "    for (uint t = 0; t < TILE; t++) xcol[t] = X + (ulong)(t0 + t) * cols;\n"
    "\n"
    "    float acc[8];\n"
    "    for (uint t = 0; t < TILE; t++) acc[t] = 0.0f;\n"
    "\n"
    "    // Lanes owning a row past the end accumulate nothing but still enter the\n"
    "    // reduction, so all 32 lanes are active at the shuffles.\n"
    "    if (row < rows) {\n"
    "        device const uchar* base = W + (ulong)row * row_stride;\n"
    "        for (uint b = lane_in_row; b < n_blocks; b += lanes) {\n"
    "            device const uchar* blk = base + (ulong)b * 144u;\n"
    "            device const half*  dh  = (device const half*)blk;\n"
    "            float d    = float(dh[0]);\n"
    "            float dmin = float(dh[1]);\n"
    "            device const uchar* sr = blk + 4;       // scales[12]\n"
    "            device const uchar* qs = blk + 16;      // qs[128]\n"
    "            uchar sc[8]; uchar mn[8];\n"
    "            for (uint j = 0; j < 4; j++) { sc[j] = sr[j] & 63; mn[j] = sr[j+4] & 63; }\n"
    "            for (uint j = 4; j < 8; j++) {\n"
    "                sc[j] = (sr[j+4] & 0xF) | ((sr[j-4] >> 6) << 4);\n"
    "                mn[j] = (sr[j+4] >> 4)  | ((sr[j]   >> 6) << 4);\n"
    "            }\n"
    "            uint bcol = b * 256u;\n"
    "            for (uint p = 0; p < 4; p++) {      // 4 pairs share 32 qs bytes\n"
    "                float d1 = d * float(sc[2*p]);\n"
    "                float m1 = dmin * float(mn[2*p]);\n"
    "                float d2 = d * float(sc[2*p+1]);\n"
    "                float m2 = dmin * float(mn[2*p+1]);\n"
    "                device const uchar* q = qs + p * 32u;\n"
    "                uint o = bcol + p * 64u;\n"
    "                for (uint l = 0; l < 32; l++) {\n"
    "                    uchar qb = q[l];\n"
    "                    for (uint t = 0; t < TILE; t++) {\n"
    "                        acc[t] += (d1 * float(qb & 0xF) - m1) * xcol[t][o + l];\n"
    "                        acc[t] += (d2 * float(qb >> 4)  - m2) * xcol[t][o + 32u + l];\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "    for (uint t = 0; t < TILE; t++) {\n"
    "        float col_sum = acc[t];\n"
    "        if (lanes == 32u) {\n"
    "            col_sum = simd_sum(acc[t]);\n"
    "        } else {\n"
    "            for (uint off = lanes >> 1; off > 0u; off >>= 1)\n"
    "                col_sum += simd_shuffle_xor(col_sum, ushort(off));\n"
    "        }\n"
    "        if (lane_in_row == 0u && row < rows)\n"
    "            out[(ulong)(t0 + t) * rows + row] = col_sum;\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * Q4_K x B GEMM (batched decode). out[B, rows] = X[B, cols] @ dequant(W)^T.\n"
    " * One SIMD group (32 lanes) per output row; each lane dequantizes its\n"
    " * assigned 256-blocks ONCE and FMAs every element into B accumulators, so\n"
    " * the Q4_K dequant is amortized across the whole batch (the GPU analog of\n"
    " * the CPU AMX panel). B is capped at 32 (accumulator registers).\n"
    " * ============================================================ */\n"
    "/* Tiled simdgroup_matrix GEMM.\n"
    " *  - Each threadgroup owns ROWS_TILE=64 output rows and the full batch B.\n"
    " *  - 256 threads = 8 simdgroups; simdgroup sg owns rows [sg*8, sg*8+8).\n"
    " *  - Per KCHUNK=32 slice of cols: stage X into Xt[B][32] (f32, coalesced,\n"
    " *    reused by all 8 simdgroups) and dequantize W into Wt[32][64] (f32,\n"
    " *    transposed to [k][r], reused by all B tiles), then accumulate with\n"
    " *    simdgroup_float8x8 (F32 accumulation, exact enough for err/RMS<1e-3).\n"
    " *  - out[b,r] = sum_k X[b,k] * Wdeq[r,k];  C[b][r] = A[b][k] * Bm[k][r]. */\n"
    "kernel void gemm_q4_k(\n"
    "    device const uchar* W        [[buffer(0)]],\n"
    "    device const float* X        [[buffer(1)]],\n"   // [B, cols]
    "    device float*       out      [[buffer(2)]],\n"   // [B, rows]
    "    constant uint& rows          [[buffer(3)]],\n"
    "    constant uint& cols          [[buffer(4)]],\n"
    "    constant uint& row_stride    [[buffer(5)]],\n"
    "    constant uint& B             [[buffer(6)]],\n"
    "    uint tgid  [[threadgroup_position_in_grid]],\n"
    "    uint tid   [[thread_position_in_threadgroup]],\n"
    "    uint sgid  [[simdgroup_index_in_threadgroup]],\n"
    "    uint lane  [[thread_index_in_simdgroup]]\n"
    ") {\n"
    "    const uint ROWS_TILE = 64u;\n"
    "    const uint KCHUNK    = 64u;\n"
    "    threadgroup float Xt[32u * 64u];      // [b][k], ld = KCHUNK\n"
    "    threadgroup float Wt[64u * 64u];      // [k][r], ld = ROWS_TILE\n"
    "    threadgroup float scratch[8u * 64u];  // per-simdgroup 8x8 spill\n"
    "\n"
    "    uint rtile    = tgid * ROWS_TILE;\n"
    "    uint NB       = (B + 7u) / 8u;        // batch tiles (<=4)\n"
    "    uint nchunks  = cols / KCHUNK;\n"
    "\n"
    "    simdgroup_float8x8 C[4];\n"
    "    for (uint i = 0; i < NB; i++) C[i] = simdgroup_float8x8(0.0f);\n"
    "\n"
    "    for (uint c = 0; c < nchunks; c++) {\n"
    "        uint chunkbase = c * KCHUNK;\n"
    "        // --- stage activations Xt[b][k] (coalesced device reads) ---\n"
    "        for (uint i = tid; i < 32u * KCHUNK; i += 256u) {\n"
    "            uint b = i / KCHUNK; uint k = i % KCHUNK;\n"
    "            Xt[i] = (b < B) ? X[(ulong)b * cols + chunkbase + k] : 0.0f;\n"
    "        }\n"
    "        // --- dequantize W into Wt[k][r] (4 threads per row cover 64 k) ---\n"
    "        {\n"
    "            uint row = tid & 63u;             // 0..63 -> output row in tile\n"
    "            uint kg  = tid >> 6;              // 0..3  -> k group of 16\n"
    "            uint grow = rtile + row;\n"
    "            uint sb   = chunkbase / 256u;     // super-block index\n"
    "            uint p    = (chunkbase % 256u) / 64u;  // sub-block pair 0..3\n"
    "            if (grow < rows) {\n"
    "                device const uchar* blk = W + (ulong)grow * row_stride + (ulong)sb * 144u;\n"
    "                device const half*  dh  = (device const half*)blk;\n"
    "                float d    = float(dh[0]);\n"
    "                float dmin = float(dh[1]);\n"
    "                device const uchar* sr = blk + 4;\n"
    "                device const uchar* qs = blk + 16;\n"
    "                uint i0 = 2u * p, i1 = 2u * p + 1u;\n"
    "                uchar s0, n0, s1, n1;\n"
    "                if (i0 < 4u) { s0 = sr[i0] & 63u; n0 = sr[i0+4u] & 63u; }\n"
    "                else { s0 = (sr[i0+4u] & 0xFu) | ((sr[i0-4u] >> 6) << 4);\n"
    "                       n0 = (sr[i0+4u] >> 4)   | ((sr[i0]     >> 6) << 4); }\n"
    "                if (i1 < 4u) { s1 = sr[i1] & 63u; n1 = sr[i1+4u] & 63u; }\n"
    "                else { s1 = (sr[i1+4u] & 0xFu) | ((sr[i1-4u] >> 6) << 4);\n"
    "                       n1 = (sr[i1+4u] >> 4)   | ((sr[i1]     >> 6) << 4); }\n"
    "                float d1 = d * float(s0), m1 = dmin * float(n0);\n"
    "                float d2 = d * float(s1), m2 = dmin * float(n1);\n"
    "                device const uchar* q = qs + p * 32u;\n"
    "                for (uint kk = 0; kk < 16u; kk++) {\n"
    "                    uint k = kg * 16u + kk;\n"
    "                    float val;\n"
    "                    if (k < 32u) { uchar qb = q[k];      val = d1 * float(qb & 0xF) - m1; }\n"
    "                    else         { uchar qb = q[k - 32u]; val = d2 * float(qb >> 4)  - m2; }\n"
    "                    Wt[k * 64u + row] = val;\n"
    "                }\n"
    "            } else {\n"
    "                for (uint kk = 0; kk < 16u; kk++)\n"
    "                    Wt[(kg * 16u + kk) * 64u + row] = 0.0f;\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        // --- simdgroup matmul: accumulate this chunk ---\n"
    "        for (uint kk = 0; kk < KCHUNK; kk += 8u) {\n"
    "            simdgroup_float8x8 bm;\n"
    "            simdgroup_load(bm, &Wt[kk * 64u + sgid * 8u], 64);\n"
    "            for (uint nb = 0; nb < NB; nb++) {\n"
    "                simdgroup_float8x8 a;\n"
    "                simdgroup_load(a, &Xt[nb * 8u * KCHUNK + kk], KCHUNK);\n"
    "                simdgroup_multiply_accumulate(C[nb], a, bm, C[nb]);\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    // --- store C[b][r] -> out[b*rows + r] ---\n"
    "    uint rbase = rtile + sgid * 8u;\n"
    "    for (uint nb = 0; nb < NB; nb++) {\n"
    "        uint bbase = nb * 8u;\n"
    "        if (bbase + 8u <= B && rbase + 8u <= rows) {\n"
    "            simdgroup_store(C[nb], out + (ulong)bbase * rows + rbase, rows);\n"
    "        } else {\n"
    "            simdgroup_store(C[nb], &scratch[sgid * 64u], 8);\n"
    "            simdgroup_barrier(mem_flags::mem_threadgroup);\n"
    "            for (uint idx = lane; idx < 64u; idx += 32u) {\n"
    "                uint bi = idx / 8u, rj = idx % 8u;\n"
    "                if (bbase + bi < B && rbase + rj < rows)\n"
    "                    out[(ulong)(bbase + bi) * rows + rbase + rj] = scratch[sgid * 64u + idx];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "/* Dequantize a Q4_K weight [rows, cols] to F16 (one thread per super-block). */\n"
    "kernel void dequant_q4_k_f16(\n"
    "    device const uchar* W        [[buffer(0)]],\n"
    "    device half*        out      [[buffer(1)]],\n"
    "    constant uint& rows          [[buffer(2)]],\n"
    "    constant uint& cols          [[buffer(3)]],\n"
    "    constant uint& row_stride    [[buffer(4)]],\n"
    "    uint gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    uint n_blocks = cols / 256u;\n"
    "    if (gid >= rows * n_blocks) return;\n"
    "    uint row = gid / n_blocks, b = gid % n_blocks;\n"
    "    device const uchar* blk = W + (ulong)row * row_stride + (ulong)b * 144u;\n"
    "    device const half*  dh  = (device const half*)blk;\n"
    "    float d = float(dh[0]), dmin = float(dh[1]);\n"
    "    device const uchar* sr = blk + 4; device const uchar* qs = blk + 16;\n"
    "    uchar sc[8], mn[8];\n"
    "    for (uint j=0;j<4;j++){ sc[j]=sr[j]&63; mn[j]=sr[j+4]&63; }\n"
    "    for (uint j=4;j<8;j++){ sc[j]=(sr[j+4]&0xF)|((sr[j-4]>>6)<<4); mn[j]=(sr[j+4]>>4)|((sr[j]>>6)<<4); }\n"
    "    device half* o = out + (ulong)row * cols + (ulong)b * 256u;\n"
    "    for (uint p=0;p<4;p++){\n"
    "        float d1=d*float(sc[2*p]), m1=dmin*float(mn[2*p]);\n"
    "        float d2=d*float(sc[2*p+1]), m2=dmin*float(mn[2*p+1]);\n"
    "        device const uchar* q = qs + p*32u; uint o0 = p*64u;\n"
    "        for (uint l=0;l<32;l++){ uchar qb=q[l];\n"
    "            o[o0+l]      = half(d1*float(qb & 0xF) - m1);\n"
    "            o[o0+32u+l]  = half(d2*float(qb >> 4)  - m2);\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    ;
}
