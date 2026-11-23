/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_attn.m
 * @brief Host dispatch and MSL source for the attn kernel family: fp32 KV
 *   append, split-K flash decode plus its combine, and tiled causal GQA
 *   flash prefill.
 *
 * @status All four kernels implemented and tested (metal/tests/test_metal_attn.c).
 * @issues The tile loop is shared by decode and prefill through the MSL helper
 *   fmattn_accumulate_range, which takes the per-thread `acc` and `qreg`
 *   arrays as thread-address-space pointers. Both loops over them are bounded
 *   by function constants, so after inlining the indices are constants and the
 *   arrays promote to registers — but that promotion is the compiler's choice,
 *   not something the source guarantees. If this ever shows up as a perf cliff,
 *   the fix is to duplicate the body into each kernel, not to add a runtime
 *   bound.
 * @todo
 *   - [ ] fm_attn_blk — per-KV-block mask classification (fully masked → skip).
 *
 * The CUDA sibling (cuda_kernels/attention.cu) reduces each q.k dot across a
 * 16-lane sub-warp with __shfl_down_sync(..., 16). There is no 16-wide
 * sub-group on a 32-wide Metal SIMD group, so the mapping is different here:
 * one SIMD GROUP owns one query head of the GQA group and reduces a whole
 * 256-element dot with a single simd_sum, and the query row lives in
 * REGISTERS (8 floats per lane) instead of threadgroup memory. Same math,
 * different decomposition.
 */

#include "metal/kernels/fm_kernel_attn.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

#include <string.h>

/* The MSL below sizes its threadgroup arrays with literals, because
 * fm_metal's dispatch helpers never call setThreadgroupMemoryLength and a
 * threadgroup array therefore has to be statically sized. These assertions
 * are what keeps those literals honest. */
_Static_assert(FM_ATTN_TILE_KEYS == 16, "MSL k_tile/score_s/p_s assume 16");
_Static_assert(FM_ATTN_MAX_HEAD_DIM == 256, "MSL k_tile assumes 256");
_Static_assert(FM_ATTN_MAX_GROUPS == 8, "MSL score_s/p_s/m_s assume 8");
_Static_assert(FM_ATTN_THREADS == 256, "MSL assumes 8 SIMD groups of 32");

/* ================================================================
 * Host dispatch
 * ================================================================ */

/* Shapes the flash kernels accept. head_dim must be a multiple of the SIMD
 * width because the QK dot splits one head across 32 lanes with no tail, and
 * it must not exceed FM_ATTN_MAX_HEAD_DIM because one thread owns one output
 * dimension and the workgroup is FM_ATTN_THREADS wide. */
static bool fm_attn_shape_is_valid(int kv_groups, int head_dim)
{
    if (kv_groups < 1 || kv_groups > FM_ATTN_MAX_GROUPS) return false;
    if (head_dim < 32 || head_dim > FM_ATTN_MAX_HEAD_DIM) return false;
    return head_dim % 32 == 0;
}

/* kv_groups and head_dim are function constants, not kernel arguments: they
 * bound the per-thread acc[] and qreg[] loops, and a runtime bound would
 * leave those arrays in thread-local memory instead of registers. */
static int fm_attn_specialize(fm_fc_values *fc, int kv_groups, int head_dim)
{
    fm_fc_reset(fc);
    if (fm_fc_set_int(fc, FM_ATTN_FC_KV_GROUPS, kv_groups) != 0) return -1;
    if (fm_fc_set_int(fc, FM_ATTN_FC_HEAD_DIM, head_dim) != 0) return -1;
    return 0;
}

/* The dispatch helpers carry uint32 arguments only, so a float rides across
 * as its bit pattern and the shader reads it back with as_type<float>. */
static uint32_t fm_attn_float_bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int fm_attn_decode_plan(int seq_len, int n_head_kv, int target_workgroups,
                        int *split_len_out)
{
    if (split_len_out) *split_len_out = 0;
    if (seq_len < 1 || n_head_kv < 1) return 0;
    if (target_workgroups <= 0)
        target_workgroups = FM_ATTN_DECODE_TARGET_WORKGROUPS;

    int n_splits = target_workgroups / n_head_kv;
    if (n_splits < 1) n_splits = 1;

    /* Round the slice up to a whole tile so every workgroup starts on a tile
     * boundary, then recompute the count from the rounded length — otherwise
     * the last split would be empty whenever the rounding overshoots. This
     * is also what keeps a short context from fanning out into slivers: at
     * seq_len 128 the requested 16 splits round to 8 tiles of 16 keys. */
    int split_len = (seq_len + n_splits - 1) / n_splits;
    split_len = ((split_len + FM_ATTN_TILE_KEYS - 1) / FM_ATTN_TILE_KEYS) *
                FM_ATTN_TILE_KEYS;
    if (split_len_out) *split_len_out = split_len;
    return (seq_len + split_len - 1) / split_len;
}

void fm_attn_decode_scratch_floats(int n_head, int head_dim, int n_splits,
                                   size_t *m_floats, size_t *acc_floats)
{
    size_t partials = (size_t)(n_head > 0 ? n_head : 0) *
                      (size_t)(n_splits > 0 ? n_splits : 0);
    if (m_floats) *m_floats = partials;
    if (acc_floats) *acc_floats = partials * (size_t)(head_dim > 0 ? head_dim : 0);
}

void fm_attn_kv_append_at(fm_bufref k_new, fm_bufref v_new,
                          fm_bufref k_cache, fm_bufref v_cache,
                          int n, int n_head_kv, int head_dim, int max_seq,
                          int pos0)
{
    if (!fm_metal_is_initialized()) return;
    if (!k_new.buffer || !v_new.buffer || !k_cache.buffer || !v_cache.buffer) return;
    if (n <= 0 || n_head_kv <= 0 || head_dim <= 0 || max_seq <= 0 || pos0 < 0) return;

    int total = n * n_head_kv * head_dim;
    fm_bufref refs[] = { k_new, v_new, k_cache, v_cache };
    uint32_t consts[] = { (uint32_t)n_head_kv, (uint32_t)head_dim,
                          (uint32_t)max_seq, (uint32_t)pos0, (uint32_t)total };
    fm_dispatch_1d_refs(@"fmattn_kv_append", NULL, refs, 4, consts, 5, total);
}

void fm_attn_kv_append(fm_buffer *k_new, fm_buffer *v_new,
                       fm_buffer *k_cache, fm_buffer *v_cache,
                       int n, int n_head_kv, int head_dim, int max_seq,
                       int pos0)
{
    if (!k_new || !v_new || !k_cache || !v_cache) return;
    fm_attn_kv_append_at(fm_bufref_at(k_new, 0), fm_bufref_at(v_new, 0),
                         fm_bufref_at(k_cache, 0), fm_bufref_at(v_cache, 0),
                         n, n_head_kv, head_dim, max_seq, pos0);
}

void fm_attn_flash_decode_at(fm_bufref q, fm_bufref k_cache, fm_bufref v_cache,
                             fm_bufref m_partial, fm_bufref l_partial,
                             fm_bufref acc_partial,
                             int kv_groups, int n_head_kv, int head_dim,
                             int max_seq, int seq_len, int n_splits,
                             int split_len, float scale)
{
    if (!fm_metal_is_initialized()) return;
    if (!q.buffer || !k_cache.buffer || !v_cache.buffer) return;
    if (!m_partial.buffer || !l_partial.buffer || !acc_partial.buffer) return;
    if (!fm_attn_shape_is_valid(kv_groups, head_dim)) return;
    if (n_head_kv <= 0 || max_seq <= 0 || seq_len < 1) return;
    if (n_splits < 1 || split_len < 1) return;

    fm_fc_values fc;
    if (fm_attn_specialize(&fc, kv_groups, head_dim) != 0) return;

    fm_bufref refs[] = { q, k_cache, v_cache, m_partial, l_partial, acc_partial };
    uint32_t consts[] = { (uint32_t)max_seq, (uint32_t)seq_len,
                          (uint32_t)n_splits, (uint32_t)split_len,
                          fm_attn_float_bits(scale) };
    /* One workgroup per (kv head, split): the split-K fan-out. At batch 1 a
     * single workgroup per kv head leaves a 10-core M4 almost idle. */
    fm_dispatch_threadgroups_refs(@"fmattn_flash_decode", &fc, refs, 6,
                                  consts, 5, n_head_kv * n_splits,
                                  FM_ATTN_THREADS);
}

void fm_attn_flash_decode(fm_buffer *q, fm_buffer *k_cache, fm_buffer *v_cache,
                          fm_buffer *m_partial, fm_buffer *l_partial,
                          fm_buffer *acc_partial,
                          int kv_groups, int n_head_kv, int head_dim,
                          int max_seq, int seq_len, int n_splits,
                          int split_len, float scale)
{
    if (!q || !k_cache || !v_cache || !m_partial || !l_partial || !acc_partial)
        return;
    fm_attn_flash_decode_at(fm_bufref_at(q, 0), fm_bufref_at(k_cache, 0),
                            fm_bufref_at(v_cache, 0),
                            fm_bufref_at(m_partial, 0),
                            fm_bufref_at(l_partial, 0),
                            fm_bufref_at(acc_partial, 0),
                            kv_groups, n_head_kv, head_dim, max_seq, seq_len,
                            n_splits, split_len, scale);
}

void fm_attn_flash_decode_reduce_at(fm_bufref m_partial, fm_bufref l_partial,
                                    fm_bufref acc_partial, fm_bufref out,
                                    int n_head, int head_dim, int n_splits)
{
    if (!fm_metal_is_initialized()) return;
    if (!m_partial.buffer || !l_partial.buffer || !acc_partial.buffer || !out.buffer)
        return;
    if (n_head <= 0 || head_dim <= 0 || n_splits < 1) return;

    fm_bufref refs[] = { m_partial, l_partial, acc_partial, out };
    uint32_t consts[] = { (uint32_t)head_dim, (uint32_t)n_splits };
    fm_dispatch_threadgroups_refs(@"fmattn_flash_decode_reduce", NULL, refs, 4,
                                  consts, 2, n_head, head_dim);
}

void fm_attn_flash_decode_reduce(fm_buffer *m_partial, fm_buffer *l_partial,
                                 fm_buffer *acc_partial, fm_buffer *out,
                                 int n_head, int head_dim, int n_splits)
{
    if (!m_partial || !l_partial || !acc_partial || !out) return;
    fm_attn_flash_decode_reduce_at(fm_bufref_at(m_partial, 0),
                                   fm_bufref_at(l_partial, 0),
                                   fm_bufref_at(acc_partial, 0),
                                   fm_bufref_at(out, 0),
                                   n_head, head_dim, n_splits);
}

void fm_attn_flash_prefill_at(fm_bufref q, fm_bufref k_cache, fm_bufref v_cache,
                              fm_bufref out, int n, int kv_groups,
                              int n_head_kv, int head_dim, int max_seq,
                              int pos0, float scale)
{
    if (!fm_metal_is_initialized()) return;
    if (!q.buffer || !k_cache.buffer || !v_cache.buffer || !out.buffer) return;
    if (!fm_attn_shape_is_valid(kv_groups, head_dim)) return;
    if (n <= 0 || n_head_kv <= 0 || max_seq <= 0 || pos0 < 0) return;
    if (pos0 + n > max_seq) return;   /* would wrap inside the attended window */

    fm_fc_values fc;
    if (fm_attn_specialize(&fc, kv_groups, head_dim) != 0) return;

    fm_bufref refs[] = { q, k_cache, v_cache, out };
    uint32_t consts[] = { (uint32_t)n_head_kv, (uint32_t)max_seq,
                          (uint32_t)pos0, fm_attn_float_bits(scale) };
    /* One workgroup per (query row, kv head). The whole GQA group shares one
     * workgroup so each staged K tile is read once and scored by all of it. */
    fm_dispatch_threadgroups_refs(@"fmattn_flash_prefill", &fc, refs, 4,
                                  consts, 4, n * n_head_kv, FM_ATTN_THREADS);
}

void fm_attn_flash_prefill(fm_buffer *q, fm_buffer *k_cache, fm_buffer *v_cache,
                           fm_buffer *out, int n, int kv_groups,
                           int n_head_kv, int head_dim, int max_seq,
                           int pos0, float scale)
{
    if (!q || !k_cache || !v_cache || !out) return;
    fm_attn_flash_prefill_at(fm_bufref_at(q, 0), fm_bufref_at(k_cache, 0),
                             fm_bufref_at(v_cache, 0), fm_bufref_at(out, 0),
                             n, kv_groups, n_head_kv, head_dim, max_seq, pos0,
                             scale);
}

/* ================================================================
 * MSL fragment
 * ================================================================ */

NSString *fm_kernel_msl_attn(void)
{
    return @
    "/* ============================================================\n"
    " * Attention: causal GQA over an fp32 ring KV cache.\n"
    " *\n"
    " * Layouts (all fp32, all row-major):\n"
    " *   q         [rows][n_head][head_dim], global head = kvh*G + g\n"
    " *   k_cache   [n_head_kv][max_seq][head_dim]\n"
    " *   v_cache   same as k_cache\n"
    " *\n"
    " * Decomposition. A workgroup is 8 SIMD groups of 32 lanes and owns ONE\n"
    " * kv head. SIMD group g scores query head g of the GQA group; its 32\n"
    " * lanes each hold head_dim/32 elements of that query row in REGISTERS\n"
    " * for the whole kernel, and one simd_sum turns a lane-partial into a\n"
    " * whole q.k dot. Thread tid owns output dimension tid during the PV\n"
    " * accumulate. Only K is staged in threadgroup memory, because it is the\n"
    " * one operand every SIMD group of the workgroup re-reads.\n"
    " *\n"
    " * Online softmax. Per tile: fold the tile's scores into the running max\n"
    " * m, rescale the running sum l and the output accumulator by\n"
    " * exp(m_old - m_new), add the tile. The [seq][seq] score matrix is never\n"
    " * materialized, and the result differs from a single-pass softmax only\n"
    " * by reassociation.\n"
    " *\n"
    " * Causality is a key COUNT, not a -inf fill: the loop stops at the\n"
    " * query's own position and the final tile is clamped to the live key\n"
    " * count, so a future key is never read, never scored and never weighted.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fmattn_tile_keys    = 16;\n"
    "constant int fmattn_max_head_dim = 256;\n"
    "constant int fmattn_max_groups   = 8;\n"
    "constant int fmattn_threads      = 256;\n"
    "constant int fmattn_simd_width   = 32;\n"
    "\n"
    "/* 'No key has been folded in yet.' NOT -INFINITY: fm_init compiles this\n"
    " * library with MTLMathModeFast, under which an operand may be assumed\n"
    " * finite, so a real infinity here is undefined behaviour. A finite floor\n"
    " * behaves identically for every use it has — a real score can never reach\n"
    " * it, max() picks the real score, and exp(floor - anything_real)\n"
    " * underflows to exactly 0, which is the weight an empty range needs. */\n"
    "constant float fmattn_score_floor = -1.0e30f;\n"
    "\n"
    "/* Shapes as function constants so every loop over acc[] and qreg[] has a\n"
    " * compile-time bound and both arrays stay in registers. */\n"
    "constant int fmattn_kv_groups_in [[function_constant(260)]];\n"
    "constant int fmattn_kv_groups =\n"
    "    is_function_constant_defined(fmattn_kv_groups_in) ? fmattn_kv_groups_in\n"
    "                                                      : fmattn_max_groups;\n"
    "constant int fmattn_head_dim_in [[function_constant(261)]];\n"
    "constant int fmattn_head_dim =\n"
    "    is_function_constant_defined(fmattn_head_dim_in) ? fmattn_head_dim_in\n"
    "                                                     : fmattn_max_head_dim;\n"
    "constant int fmattn_q_per_lane = fmattn_head_dim / fmattn_simd_width;\n"
    "\n"
    "/* Append one post-RoPE K/V element to the ring cache. One thread per\n"
    " * element of k_new; row t of the batch lands at (pos0 + t) % max_seq. */\n"
    "kernel void fmattn_kv_append(\n"
    "    device const float* k_new     [[buffer(0)]],\n"
    "    device const float* v_new     [[buffer(1)]],\n"
    "    device float*       k_cache   [[buffer(2)]],\n"
    "    device float*       v_cache   [[buffer(3)]],\n"
    "    constant uint&      n_head_kv [[buffer(4)]],\n"
    "    constant uint&      head_dim  [[buffer(5)]],\n"
    "    constant uint&      max_seq   [[buffer(6)]],\n"
    "    constant uint&      pos0      [[buffer(7)]],\n"
    "    constant uint&      total     [[buffer(8)]],\n"
    "    uint i [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (i >= total) return;\n"
    "    uint d   = i % head_dim;\n"
    "    uint th  = i / head_dim;          // flattened (row, kv head)\n"
    "    uint h   = th % n_head_kv;\n"
    "    uint t   = th / n_head_kv;\n"
    "    uint cache_pos = (pos0 + t) % max_seq;\n"
    "    ulong dst = ((ulong)h * max_seq + cache_pos) * head_dim + d;\n"
    "    k_cache[dst] = k_new[i];\n"
    "    v_cache[dst] = v_new[i];\n"
    "}\n"
    "\n"
    "/* Online-softmax pass over keys [k_lo, k_hi) for the whole GQA group.\n"
    " *\n"
    " * On entry m_s/l_s hold the running state (-inf / 0 for a fresh pass) and\n"
    " * acc[] the running output; on exit all three have absorbed the range.\n"
    " * Shared verbatim by decode (which passes one split's slice) and prefill\n"
    " * (which passes [0, query position]), because they differ only in which\n"
    " * range they hand over and what they do with the result. */\n"
    "static void fmattn_accumulate_range(\n"
    "    device const float* k_head, device const float* v_head,\n"
    "    thread float* qreg, thread float* acc,\n"
    "    threadgroup float* k_tile, threadgroup float* score_s,\n"
    "    threadgroup float* p_s, threadgroup float* m_s,\n"
    "    threadgroup float* l_s, threadgroup float* corr_s,\n"
    "    uint k_lo, uint k_hi, float scale, uint tid, uint sgid, uint lane)\n"
    "{\n"
    "    const uint HD = uint(fmattn_head_dim);\n"
    "    const uint G  = uint(fmattn_kv_groups);\n"
    "    const uint TK = uint(fmattn_tile_keys);\n"
    "    const uint MG = uint(fmattn_max_groups);\n"
    "\n"
    "    for (uint j0 = k_lo; j0 < k_hi; j0 += TK) {\n"
    "        uint tk = min(TK, k_hi - j0);\n"
    "\n"
    "        /* Stage the tile's K rows once for the whole group. Flattened so\n"
    "         * consecutive threads read consecutive addresses whatever HD is. */\n"
    "        for (uint e = tid; e < tk * HD; e += uint(fmattn_threads))\n"
    "            k_tile[e] = k_head[(ulong)(j0 + e / HD) * HD + (e % HD)];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "        /* q.k for every (head, key) in the tile. sgid and tk are uniform\n"
    "         * across the SIMD group, so simd_sum sees all 32 lanes. */\n"
    "        if (sgid < G) {\n"
    "            for (uint j = 0; j < tk; j++) {\n"
    "                float part = 0.0f;\n"
    "                threadgroup const float* krow = k_tile + (ulong)j * HD;\n"
    "                for (int i = 0; i < fmattn_q_per_lane; i++)\n"
    "                    part += qreg[i] * krow[lane + uint(i) * 32u];\n"
    "                float dot = simd_sum(part);\n"
    "                if (lane == 0) score_s[sgid * TK + j] = dot * scale;\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "        /* Fold the tile into (m, l), one thread per head. p_s is stored\n"
    "         * key-major so the PV loop below reads a contiguous run. */\n"
    "        if (tid < G) {\n"
    "            threadgroup const float* sc = score_s + tid * TK;\n"
    "            float m_new = m_s[tid];\n"
    "            for (uint j = 0; j < tk; j++) m_new = max(m_new, sc[j]);\n"
    "            float corr = exp(m_s[tid] - m_new);\n"
    "            float sum = 0.0f;\n"
    "            for (uint j = 0; j < tk; j++) {\n"
    "                float pj = exp(sc[j] - m_new);\n"
    "                p_s[j * MG + tid] = pj;\n"
    "                sum += pj;\n"
    "            }\n"
    "            l_s[tid] = l_s[tid] * corr + sum;\n"
    "            m_s[tid] = m_new;\n"
    "            corr_s[tid] = corr;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "        /* Rescale to the tile's max, then accumulate PV. Thread tid owns\n"
    "         * output dim tid, so the V read is coalesced across the group. */\n"
    "        if (tid < HD) {\n"
    "            for (int g = 0; g < fmattn_kv_groups; g++) acc[g] *= corr_s[g];\n"
    "            for (uint j = 0; j < tk; j++) {\n"
    "                float vv = v_head[(ulong)(j0 + j) * HD + tid];\n"
    "                threadgroup const float* pj = p_s + j * MG;\n"
    "                for (int g = 0; g < fmattn_kv_groups; g++)\n"
    "                    acc[g] += pj[g] * vv;\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* Loads this SIMD group's query head into registers: lane holds elements\n"
    " * lane, lane+32, ... of head `sgid` of the group based at q_base. */\n"
    "static void fmattn_load_query(device const float* q_base, thread float* qreg,\n"
    "                              uint sgid, uint lane)\n"
    "{\n"
    "    for (int i = 0; i < fmattn_q_per_lane; i++) qreg[i] = 0.0f;\n"
    "    if (sgid >= uint(fmattn_kv_groups)) return;\n"
    "    device const float* head = q_base + (ulong)sgid * uint(fmattn_head_dim);\n"
    "    for (int i = 0; i < fmattn_q_per_lane; i++)\n"
    "        qreg[i] = head[lane + uint(i) * 32u];\n"
    "}\n"
    "\n"
    "/* One split of the decode key range for one kv head.\n"
    " *\n"
    " * Writes the split's UNNORMALIZED accumulator plus its (m, l) so\n"
    " * fmattn_flash_decode_reduce can put every split on a common scale. The\n"
    " * partial index is (kvh*G + g)*n_splits + split — head-major, so the\n"
    " * reduce reads one head's splits contiguously.\n"
    " *\n"
    " * A split whose slice starts past seq_len writes m = -inf, l = 0 and a\n"
    " * zero accumulator; the reduce weights that by exp(-inf - M) = 0. */\n"
    "kernel void fmattn_flash_decode(\n"
    "    device const float* q          [[buffer(0)]],\n"
    "    device const float* k_cache    [[buffer(1)]],\n"
    "    device const float* v_cache    [[buffer(2)]],\n"
    "    device float*       m_out      [[buffer(3)]],\n"
    "    device float*       l_out      [[buffer(4)]],\n"
    "    device float*       acc_out    [[buffer(5)]],\n"
    "    constant uint&      max_seq    [[buffer(6)]],\n"
    "    constant uint&      seq_len    [[buffer(7)]],\n"
    "    constant uint&      n_splits   [[buffer(8)]],\n"
    "    constant uint&      split_len  [[buffer(9)]],\n"
    "    constant uint&      scale_bits [[buffer(10)]],\n"
    "    uint tgid [[threadgroup_position_in_grid]],\n"
    "    uint tid  [[thread_position_in_threadgroup]],\n"
    "    uint sgid [[simdgroup_index_in_threadgroup]],\n"
    "    uint lane [[thread_index_in_simdgroup]]\n"
    ") {\n"
    "    threadgroup float k_tile[16 * 256];\n"
    "    threadgroup float score_s[8 * 16];\n"
    "    threadgroup float p_s[16 * 8];\n"
    "    threadgroup float m_s[8];\n"
    "    threadgroup float l_s[8];\n"
    "    threadgroup float corr_s[8];\n"
    "\n"
    "    const uint HD  = uint(fmattn_head_dim);\n"
    "    const uint G   = uint(fmattn_kv_groups);\n"
    "    const uint kvh   = tgid / n_splits;\n"
    "    const uint split = tgid % n_splits;\n"
    "\n"
    "    uint k_lo = split * split_len;\n"
    "    uint k_hi = k_lo + split_len;\n"
    "    if (k_lo > seq_len) k_lo = seq_len;\n"
    "    if (k_hi > seq_len) k_hi = seq_len;\n"
    "\n"
    "    float qreg[8];\n"
    "    float acc[8];\n"
    "    fmattn_load_query(q + (ulong)kvh * G * HD, qreg, sgid, lane);\n"
    "    for (int g = 0; g < fmattn_kv_groups; g++) acc[g] = 0.0f;\n"
    "    if (tid < G) { m_s[tid] = fmattn_score_floor; l_s[tid] = 0.0f; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    fmattn_accumulate_range(k_cache + (ulong)kvh * max_seq * HD,\n"
    "                            v_cache + (ulong)kvh * max_seq * HD,\n"
    "                            qreg, acc, k_tile, score_s, p_s, m_s, l_s,\n"
    "                            corr_s, k_lo, k_hi, as_type<float>(scale_bits),\n"
    "                            tid, sgid, lane);\n"
    "\n"
    "    if (tid < HD) {\n"
    "        for (int g = 0; g < fmattn_kv_groups; g++) {\n"
    "            ulong po = (ulong)(kvh * G + uint(g)) * n_splits + split;\n"
    "            acc_out[po * HD + tid] = acc[g];\n"
    "        }\n"
    "    }\n"
    "    if (tid < G) {\n"
    "        ulong po = (ulong)(kvh * G + tid) * n_splits + split;\n"
    "        m_out[po] = m_s[tid];\n"
    "        l_out[po] = l_s[tid];\n"
    "    }\n"
    "}\n"
    "\n"
    "/* Combine the per-split partials of one query head.\n"
    " *\n"
    " * Take M = max of the split maxima, weight each split by exp(m_s - M),\n"
    " * sum the weighted l's and accumulators, divide. Identical to a\n"
    " * single-pass softmax over the union of the slices. One workgroup per\n"
    " * head, thread tid owns output dim tid. */\n"
    "kernel void fmattn_flash_decode_reduce(\n"
    "    device const float* m_in     [[buffer(0)]],\n"
    "    device const float* l_in     [[buffer(1)]],\n"
    "    device const float* acc_in   [[buffer(2)]],\n"
    "    device float*       out      [[buffer(3)]],\n"
    "    constant uint&      head_dim [[buffer(4)]],\n"
    "    constant uint&      n_splits [[buffer(5)]],\n"
    "    uint head [[threadgroup_position_in_grid]],\n"
    "    uint tid  [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    if (tid >= head_dim) return;\n"
    "    device const float* m_h = m_in + (ulong)head * n_splits;\n"
    "    device const float* l_h = l_in + (ulong)head * n_splits;\n"
    "    device const float* a_h = acc_in + (ulong)head * n_splits * head_dim;\n"
    "\n"
    "    float peak = fmattn_score_floor;\n"
    "    for (uint s = 0; s < n_splits; s++) peak = max(peak, m_h[s]);\n"
    "\n"
    "    float o = 0.0f;\n"
    "    float total = 0.0f;\n"
    "    for (uint s = 0; s < n_splits; s++) {\n"
    "        float w = exp(m_h[s] - peak);\n"
    "        total += l_h[s] * w;\n"
    "        o += a_h[(ulong)s * head_dim + tid] * w;\n"
    "    }\n"
    "    out[(ulong)head * head_dim + tid] = o / total;\n"
    "}\n"
    "\n"
    "/* Causal GQA attention for one query row against the cache.\n"
    " *\n"
    " * Workgroup (t, kvh): query row t sits at absolute position pos0 + t and\n"
    " * attends cache rows [0, pos0 + t]. Unlike decode this normalizes in\n"
    " * place — there is no split to combine. */\n"
    "kernel void fmattn_flash_prefill(\n"
    "    device const float* q          [[buffer(0)]],\n"
    "    device const float* k_cache    [[buffer(1)]],\n"
    "    device const float* v_cache    [[buffer(2)]],\n"
    "    device float*       out        [[buffer(3)]],\n"
    "    constant uint&      n_head_kv  [[buffer(4)]],\n"
    "    constant uint&      max_seq    [[buffer(5)]],\n"
    "    constant uint&      pos0       [[buffer(6)]],\n"
    "    constant uint&      scale_bits [[buffer(7)]],\n"
    "    uint tgid [[threadgroup_position_in_grid]],\n"
    "    uint tid  [[thread_position_in_threadgroup]],\n"
    "    uint sgid [[simdgroup_index_in_threadgroup]],\n"
    "    uint lane [[thread_index_in_simdgroup]]\n"
    ") {\n"
    "    threadgroup float k_tile[16 * 256];\n"
    "    threadgroup float score_s[8 * 16];\n"
    "    threadgroup float p_s[16 * 8];\n"
    "    threadgroup float m_s[8];\n"
    "    threadgroup float l_s[8];\n"
    "    threadgroup float corr_s[8];\n"
    "\n"
    "    const uint HD = uint(fmattn_head_dim);\n"
    "    const uint G  = uint(fmattn_kv_groups);\n"
    "    const uint t   = tgid / n_head_kv;\n"
    "    const uint kvh = tgid % n_head_kv;\n"
    "    const uint n_head = n_head_kv * G;\n"
    "    const ulong row_base = ((ulong)t * n_head + (ulong)kvh * G) * HD;\n"
    "\n"
    "    float qreg[8];\n"
    "    float acc[8];\n"
    "    fmattn_load_query(q + row_base, qreg, sgid, lane);\n"
    "    for (int g = 0; g < fmattn_kv_groups; g++) acc[g] = 0.0f;\n"
    "    if (tid < G) { m_s[tid] = fmattn_score_floor; l_s[tid] = 0.0f; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    fmattn_accumulate_range(k_cache + (ulong)kvh * max_seq * HD,\n"
    "                            v_cache + (ulong)kvh * max_seq * HD,\n"
    "                            qreg, acc, k_tile, score_s, p_s, m_s, l_s,\n"
    "                            corr_s, 0u, pos0 + t + 1u,\n"
    "                            as_type<float>(scale_bits), tid, sgid, lane);\n"
    "\n"
    "    if (tid < HD) {\n"
    "        device float* o = out + row_base;\n"
    "        for (int g = 0; g < fmattn_kv_groups; g++)\n"
    "            o[(ulong)uint(g) * HD + tid] = acc[g] / l_s[g];\n"
    "    }\n"
    "}\n"
    "\n"
    ;
}
