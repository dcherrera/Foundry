/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_ssm.m
 * @brief Host dispatch and MSL source for the gated-delta SSM family: the
 *   decay gate, the causal depthwise conv1d + SiLU (single step and scan),
 *   the fused gated RMSNorm, and the delta-rule recurrence in both its
 *   register-resident decode form and its permanent sequential reference form.
 *
 * @status All six kernels validated against the qwen35moe golden set.
 * @issues None
 * @todo None
 *
 * The MSL fragment defines no preprocessor macro at all and prefixes every
 * shader identifier — kernels included — with fmssm_: the families are
 * concatenated into ONE MSL translation unit, so an unfenced macro or a bare
 * helper name would collide with whatever is assembled after this fragment.
 */

#include "metal/kernels/fm_kernel_ssm.h"

#include <string.h>

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

/* Lanes in an Apple GPU SIMD group. Spelled as a constant on both sides
 * rather than read from [[threads_per_simdgroup]], because the shader's
 * per-lane register count is state_size / this and a runtime value there
 * would put the register array back into a runtime-indexed loop. */
#define FM_SSM_SIMD_WIDTH 32

/* SIMD groups per recurrence threadgroup. Each owns one (head, row) pair
 * independently, so this only trades threadgroup count against threadgroup
 * size; 8 gives the 256-thread groups the dispatcher is happiest with. */
#define FM_SSM_RECUR_GROUPS_PER_TG 8

/* The dispatch helpers carry runtime scalars as uint32; a float rides
 * through as its bit pattern and the kernel reads it back with as_type.
 * Done explicitly rather than by punning the argument type so the trip
 * through an integer channel is visible on both sides. */
static uint32_t fmssm_float_bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* A state row is split across the lanes of one SIMD group with each lane
 * holding state_size / 32 elements in registers, so state_size has to be a
 * whole number of SIMD widths and small enough for the register array. */
static bool fmssm_state_size_is_valid(int state_size)
{
    if (state_size <= 0 || state_size > FM_SSM_MAX_STATE_SIZE) return false;
    return state_size % FM_SSM_SIMD_WIDTH == 0;
}

/* ================================================================
 * Host dispatch — decay gate
 * ================================================================ */

void fm_ssm_decay_gate_at(fm_bufref alpha, fm_bufref dt, fm_bufref a,
                          fm_bufref g, int n_heads, int rows)
{
    if (!fm_metal_is_initialized()) return;
    if (!alpha.buffer || !dt.buffer || !a.buffer || !g.buffer) return;
    if (n_heads <= 0 || rows <= 0) return;

    fm_bufref refs[] = { alpha, dt, a, g };
    uint32_t consts[] = { (uint32_t)n_heads, (uint32_t)(rows * n_heads) };
    fm_dispatch_1d_refs(@"fmssm_decay_gate", NULL, refs, 4, consts, 2,
                        rows * n_heads);
}

void fm_ssm_decay_gate(fm_buffer *alpha, fm_buffer *dt, fm_buffer *a,
                       fm_buffer *g, int n_heads, int rows)
{
    if (!alpha || !dt || !a || !g) return;
    fm_ssm_decay_gate_at(fm_bufref_at(alpha, 0), fm_bufref_at(dt, 0),
                         fm_bufref_at(a, 0), fm_bufref_at(g, 0), n_heads, rows);
}

/* ================================================================
 * Host dispatch — causal conv1d + SiLU
 * ================================================================ */

/* Both conv forms are the same dispatch shape — one thread per channel,
 * conv_kernel bound as a function constant so the tap loop unrolls and the
 * history array stays in registers — and differ only in the position count
 * they scan, so they share one launcher. */
static void fmssm_dispatch_conv(NSString *kernel_name, fm_bufref qkv,
                                fm_bufref weight, fm_bufref conv_state,
                                fm_bufref out, int conv_dim, int conv_kernel,
                                int n_tokens)
{
    if (!fm_metal_is_initialized()) return;
    if (!qkv.buffer || !weight.buffer || !conv_state.buffer || !out.buffer) return;
    if (conv_dim <= 0 || n_tokens <= 0) return;
    if (conv_kernel < 2 || conv_kernel > FM_SSM_MAX_CONV_KERNEL) return;

    fm_fc_values fc;
    fm_fc_reset(&fc);
    if (fm_fc_set_int(&fc, FM_SSM_FC_CONV_KERNEL, conv_kernel) != 0) return;

    fm_bufref refs[] = { qkv, weight, conv_state, out };
    uint32_t consts[] = { (uint32_t)conv_dim, (uint32_t)n_tokens };
    fm_dispatch_1d_refs(kernel_name, &fc, refs, 4, consts, 2, conv_dim);
}

void fm_ssm_conv1d_silu_at(fm_bufref qkv, fm_bufref weight, fm_bufref conv_state,
                           fm_bufref out, int conv_dim, int conv_kernel)
{
    fmssm_dispatch_conv(@"fmssm_conv1d_silu", qkv, weight, conv_state, out,
                        conv_dim, conv_kernel, 1);
}

void fm_ssm_conv1d_silu(fm_buffer *qkv, fm_buffer *weight, fm_buffer *conv_state,
                        fm_buffer *out, int conv_dim, int conv_kernel)
{
    if (!qkv || !weight || !conv_state || !out) return;
    fm_ssm_conv1d_silu_at(fm_bufref_at(qkv, 0), fm_bufref_at(weight, 0),
                          fm_bufref_at(conv_state, 0), fm_bufref_at(out, 0),
                          conv_dim, conv_kernel);
}

void fm_ssm_conv1d_silu_seq_at(fm_bufref qkv, fm_bufref weight,
                               fm_bufref conv_state, fm_bufref out,
                               int conv_dim, int conv_kernel, int n_tokens)
{
    fmssm_dispatch_conv(@"fmssm_conv1d_silu_seq", qkv, weight, conv_state, out,
                        conv_dim, conv_kernel, n_tokens);
}

void fm_ssm_conv1d_silu_seq(fm_buffer *qkv, fm_buffer *weight,
                            fm_buffer *conv_state, fm_buffer *out,
                            int conv_dim, int conv_kernel, int n_tokens)
{
    if (!qkv || !weight || !conv_state || !out) return;
    fm_ssm_conv1d_silu_seq_at(fm_bufref_at(qkv, 0), fm_bufref_at(weight, 0),
                              fm_bufref_at(conv_state, 0), fm_bufref_at(out, 0),
                              conv_dim, conv_kernel, n_tokens);
}

/* ================================================================
 * Host dispatch — gated RMSNorm
 * ================================================================ */

void fm_ssm_gated_norm_at(fm_bufref y, fm_bufref weight, fm_bufref z,
                          fm_bufref gated, int n_heads, int head_dim, float eps)
{
    if (!fm_metal_is_initialized()) return;
    if (!y.buffer || !weight.buffer || !z.buffer || !gated.buffer) return;
    if (n_heads <= 0 || head_dim <= 0 || head_dim % FM_SSM_SIMD_WIDTH != 0) return;

    /* head_dim is a function constant so the per-lane loop bound is known at
     * shader-compile time; one SIMD group per head means the sum of squares
     * is a single simd_sum with no threadgroup memory and no barrier. */
    fm_fc_values fc;
    fm_fc_reset(&fc);
    if (fm_fc_set_int(&fc, FM_SSM_FC_HEAD_DIM, head_dim) != 0) return;

    fm_bufref refs[] = { y, weight, z, gated };
    uint32_t consts[] = { fmssm_float_bits(eps) };
    fm_dispatch_threadgroups_refs(@"fmssm_gated_norm", &fc, refs, 4, consts, 1,
                                  n_heads, FM_SSM_SIMD_WIDTH);
}

void fm_ssm_gated_norm(fm_buffer *y, fm_buffer *weight, fm_buffer *z,
                       fm_buffer *gated, int n_heads, int head_dim, float eps)
{
    if (!y || !weight || !z || !gated) return;
    fm_ssm_gated_norm_at(fm_bufref_at(y, 0), fm_bufref_at(weight, 0),
                         fm_bufref_at(z, 0), fm_bufref_at(gated, 0),
                         n_heads, head_dim, eps);
}

/* ================================================================
 * Host dispatch — delta-rule recurrence
 * ================================================================ */

/* The decode kernel and the scan share this launcher: identical operands and
 * identical mapping of (head, state row) pairs onto SIMD groups, differing
 * only in how many positions the kernel walks before writing its registers
 * back. Keeping them on one launcher is what stops the reference form and
 * the production form from drifting apart in shape. */
static void fmssm_dispatch_recurrence(NSString *kernel_name, fm_bufref state,
                                      fm_bufref q, fm_bufref k, fm_bufref v,
                                      fm_bufref g, fm_bufref beta, fm_bufref y,
                                      int n_v_heads, int state_size,
                                      int n_tokens, float scale)
{
    if (!fm_metal_is_initialized()) return;
    if (!state.buffer || !q.buffer || !k.buffer || !v.buffer) return;
    if (!g.buffer || !beta.buffer || !y.buffer) return;
    if (n_v_heads <= 0 || n_tokens <= 0) return;
    if (!fmssm_state_size_is_valid(state_size)) return;

    fm_fc_values fc;
    fm_fc_reset(&fc);
    if (fm_fc_set_int(&fc, FM_SSM_FC_STATE_SIZE, state_size) != 0) return;

    const int pairs = n_v_heads * state_size;
    const int groups = (pairs + FM_SSM_RECUR_GROUPS_PER_TG - 1) /
                       FM_SSM_RECUR_GROUPS_PER_TG;

    fm_bufref refs[] = { state, q, k, v, g, beta, y };
    uint32_t consts[] = { (uint32_t)pairs, fmssm_float_bits(scale),
                          (uint32_t)n_tokens };
    fm_dispatch_threadgroups_refs(kernel_name, &fc, refs, 7, consts, 3, groups,
                                  FM_SSM_RECUR_GROUPS_PER_TG * FM_SSM_SIMD_WIDTH);
}

void fm_ssm_recurrence_at(fm_bufref state, fm_bufref q, fm_bufref k, fm_bufref v,
                          fm_bufref g, fm_bufref beta, fm_bufref y,
                          int n_v_heads, int state_size, float scale)
{
    fmssm_dispatch_recurrence(@"fmssm_recurrence", state, q, k, v, g, beta, y,
                              n_v_heads, state_size, 1, scale);
}

void fm_ssm_recurrence(fm_buffer *state, fm_buffer *q, fm_buffer *k, fm_buffer *v,
                       fm_buffer *g, fm_buffer *beta, fm_buffer *y,
                       int n_v_heads, int state_size, float scale)
{
    if (!state || !q || !k || !v || !g || !beta || !y) return;
    fm_ssm_recurrence_at(fm_bufref_at(state, 0), fm_bufref_at(q, 0),
                         fm_bufref_at(k, 0), fm_bufref_at(v, 0),
                         fm_bufref_at(g, 0), fm_bufref_at(beta, 0),
                         fm_bufref_at(y, 0), n_v_heads, state_size, scale);
}

void fm_ssm_recurrence_seq_at(fm_bufref state, fm_bufref q, fm_bufref k,
                              fm_bufref v, fm_bufref g, fm_bufref beta,
                              fm_bufref y, int n_v_heads, int state_size,
                              int n_tokens, float scale)
{
    fmssm_dispatch_recurrence(@"fmssm_recurrence_seq", state, q, k, v, g, beta,
                              y, n_v_heads, state_size, n_tokens, scale);
}

void fm_ssm_recurrence_seq(fm_buffer *state, fm_buffer *q, fm_buffer *k,
                           fm_buffer *v, fm_buffer *g, fm_buffer *beta,
                           fm_buffer *y, int n_v_heads, int state_size,
                           int n_tokens, float scale)
{
    if (!state || !q || !k || !v || !g || !beta || !y) return;
    fm_ssm_recurrence_seq_at(fm_bufref_at(state, 0), fm_bufref_at(q, 0),
                             fm_bufref_at(k, 0), fm_bufref_at(v, 0),
                             fm_bufref_at(g, 0), fm_bufref_at(beta, 0),
                             fm_bufref_at(y, 0), n_v_heads, state_size,
                             n_tokens, scale);
}

/* ================================================================
 * MSL fragment
 * ================================================================ */

NSString *fm_kernel_msl_ssm(void)
{
    return @
    "/* ============================================================\n"
    " * Gated-delta SSM family\n"
    " *\n"
    " * Every identifier here, kernels included, is prefixed fmssm_, and the\n"
    " * fragment defines no preprocessor macro at all: the families are\n"
    " * concatenated into ONE MSL translation unit, so an unfenced macro or a\n"
    " * bare helper name would collide with whatever is assembled after this\n"
    " * fragment.\n"
    " *\n"
    " * eps and scale arrive as uint bit patterns because the dispatch\n"
    " * helpers carry runtime scalars as uint32; as_type<float> reads them\n"
    " * back.\n"
    " *\n"
    " * exp is spelled precise::exp throughout. The decay multiplies the\n"
    " * recurrent state on EVERY token, so a fast-math exp's error would\n"
    " * compound along the sequence rather than stay a per-token rounding\n"
    " * difference, and the sequential reference below would stop being an\n"
    " * oracle for the chunked path.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fmssm_fc_state_size_in  [[function_constant(200)]];\n"
    "constant int fmssm_fc_conv_kernel_in [[function_constant(201)]];\n"
    "constant int fmssm_fc_head_dim_in    [[function_constant(202)]];\n"
    "\n"
    "/* The host always specializes these; the defaults exist only because MSL\n"
    " * requires the unspecialized pipeline to still build, and they are the\n"
    " * qwen35moe shapes. */\n"
    "constant int fmssm_state_size =\n"
    "    is_function_constant_defined(fmssm_fc_state_size_in)\n"
    "        ? fmssm_fc_state_size_in : 128;\n"
    "constant int fmssm_conv_kernel =\n"
    "    is_function_constant_defined(fmssm_fc_conv_kernel_in)\n"
    "        ? fmssm_fc_conv_kernel_in : 4;\n"
    "constant int fmssm_head_dim =\n"
    "    is_function_constant_defined(fmssm_fc_head_dim_in)\n"
    "        ? fmssm_fc_head_dim_in : 128;\n"
    "\n"
    "/* State elements each lane of a 32-wide SIMD group holds in registers,\n"
    " * and taps of conv history each thread holds. Both are compile-time, so\n"
    " * the loops over them unroll and every element of the arrays below\n"
    " * becomes a named register instead of a thread-local memory slot. */\n"
    "constant int fmssm_regs_per_lane = fmssm_state_size / 32;\n"
    "constant int fmssm_conv_taps = fmssm_conv_kernel - 1;\n"
    "constant int fmssm_head_regs_per_lane = fmssm_head_dim / 32;\n"
    "\n"
    "/* Declared sizes of those arrays. A function constant is not a valid\n"
    " * array bound in MSL, so the arrays are sized for the largest shape the\n"
    " * host accepts (FM_SSM_MAX_STATE_SIZE / 32 and FM_SSM_MAX_CONV_KERNEL - 1)\n"
    " * and the loops above bound the part that is actually touched. After\n"
    " * unrolling the untouched tail is dead and costs no register. */\n"
    "constant int fmssm_max_regs_per_lane = 8;\n"
    "constant int fmssm_max_conv_taps = 7;\n"
    "\n"
    "/* SiLU(x) = x * sigmoid(x), written as the divisor form the CPU\n"
    " * reference uses so the two round identically. */\n"
    "inline float fmssm_silu(float x) {\n"
    "    return x / (1.0f + precise::exp(-x));\n"
    "}\n"
    "\n"
    "/* softplus(v) = log(1 + exp(v)), guarded at 20.\n"
    " *\n"
    " * MSL HAS NO log1p — the CPU reference's log1pf(expf(v)) cannot be\n"
    " * transcribed, and log(1.0f + exp(v)) is not a substitute: once exp(v)\n"
    " * approaches a float32 ulp of 1.0 that addition absorbs it and softplus\n"
    " * loses all of its low-order accuracy, collapsing to 0 outright below an\n"
    " * ulp. On its own that is a sub-ulp error in one token's decay, but the\n"
    " * chunked prefill path takes CUMULATIVE prefix sums of g along a chunk,\n"
    " * so a decay biased toward zero accumulates over CS positions instead of\n"
    " * staying a per-token rounding difference. So the small branch uses the\n"
    " * cancellation-free identity log(1+x) = 2*atanh(x/(2+x)), summed as an\n"
    " * odd series in s = x/(2+x). For x in (0, 1] that is s <= 1/3, where the\n"
    " * terms kept below leave a truncation error near 1e-9 relative — under a\n"
    " * float32 ulp — and no step in it ever subtracts nearby quantities, so\n"
    " * the library-wide fast-math mode cannot reassociate the accuracy away.\n"
    " * Above x = 1 the addition has nothing to absorb and a plain log is\n"
    " * already correct to an ulp. */\n"
    "inline float fmssm_softplus(float v) {\n"
    "    if (v > 20.0f) return v;\n"
    "    float x = precise::exp(v);\n"
    "    if (x > 1.0f) return precise::log(1.0f + x);\n"
    "    float s = x / (2.0f + x);\n"
    "    float s2 = s * s;\n"
    "    float poly = 1.0f / 15.0f;\n"
    "    poly = poly * s2 + 1.0f / 13.0f;\n"
    "    poly = poly * s2 + 1.0f / 11.0f;\n"
    "    poly = poly * s2 + 1.0f / 9.0f;\n"
    "    poly = poly * s2 + 1.0f / 7.0f;\n"
    "    poly = poly * s2 + 1.0f / 5.0f;\n"
    "    poly = poly * s2 + 1.0f / 3.0f;\n"
    "    poly = poly * s2 + 1.0f;\n"
    "    return 2.0f * s * poly;\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * Decay gate: g[r][h] = a[h] * softplus(alpha[r][h] + dt[h])\n"
    " *\n"
    " * The guard at 20 is not a micro-optimization: exp(89) already\n"
    " * overflows float32, so without it a large pre-activation makes\n"
    " * softplus infinite, g minus-infinite and exp(g) exactly zero — the\n"
    " * head's entire recurrent state is wiped on the next token. Above 20,\n"
    " * softplus(v) - v is under 2e-9, far below a float32 ulp there.\n"
    " *\n"
    " * dt and a are per-head and broadcast across rows; alpha and g carry\n"
    " * the row axis. One thread per (row, head).\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void fmssm_decay_gate(\n"
    "    device const float* alpha [[buffer(0)]],\n"
    "    device const float* dt    [[buffer(1)]],\n"
    "    device const float* a     [[buffer(2)]],\n"
    "    device float* g           [[buffer(3)]],\n"
    "    constant uint& n_heads    [[buffer(4)]],\n"
    "    constant uint& total      [[buffer(5)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (idx >= total) return;\n"
    "    uint h = idx % n_heads;\n"
    "    g[idx] = a[h] * fmssm_softplus(alpha[idx] + dt[h]);\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * Causal depthwise conv1d + SiLU\n"
    " *\n"
    " * Depthwise, so channels never interact and one thread owns a whole\n"
    " * channel: its (conv_kernel - 1)-deep history, its conv_kernel weights\n"
    " * and its output. The window is the history followed by this position's\n"
    " * value, and the state then drops its oldest tap and appends the RAW\n"
    " * input — not the convolved result.\n"
    " *\n"
    " * The accumulation starts from the newest tap and then walks the\n"
    " * history oldest-first, which is the CPU reference's order; a\n"
    " * conv1d is short enough that reassociating it would show up in the\n"
    " * golden comparison.\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void fmssm_conv1d_silu(\n"
    "    device const float* qkv    [[buffer(0)]],\n"
    "    device const float* weight [[buffer(1)]],\n"
    "    device float* conv_state   [[buffer(2)]],\n"
    "    device float* out          [[buffer(3)]],\n"
    "    constant uint& conv_dim    [[buffer(4)]],\n"
    "    constant uint& n_tokens    [[buffer(5)]],\n"
    "    uint c [[thread_position_in_grid]]\n"
    ") {\n"
    "    /* Bound because the single-step and scan forms share one host\n"
    "     * launcher and therefore one argument list; a single step is the\n"
    "     * n_tokens == 1 case and never reads it. */\n"
    "    (void)n_tokens;\n"
    "    if (c >= conv_dim) return;\n"
    "\n"
    "    const uint taps = uint(fmssm_conv_taps);\n"
    "    device const float* cw = weight + (ulong)c * uint(fmssm_conv_kernel);\n"
    "    device float* cs = conv_state + (ulong)c * taps;\n"
    "\n"
    "    float x = qkv[c];\n"
    "    float acc = x * cw[taps];\n"
    "    for (uint t = 0; t < taps; t++)\n"
    "        acc += cs[t] * cw[t];\n"
    "    out[c] = fmssm_silu(acc);\n"
    "\n"
    "    for (uint t = 0; t + 1 < taps; t++)\n"
    "        cs[t] = cs[t + 1];\n"
    "    cs[taps - 1] = x;\n"
    "}\n"
    "\n"
    "/* The scan over n_tokens positions. The history is read once into\n"
    " * registers, advanced n_tokens times and written back once, so the state\n"
    " * never round-trips through device memory mid-sequence. Per position the\n"
    " * arithmetic is fmssm_conv1d_silu's term for term, which is what makes\n"
    " * the scan and n single-step dispatches bit-identical. */\n"
    "kernel void fmssm_conv1d_silu_seq(\n"
    "    device const float* qkv    [[buffer(0)]],\n"
    "    device const float* weight [[buffer(1)]],\n"
    "    device float* conv_state   [[buffer(2)]],\n"
    "    device float* out          [[buffer(3)]],\n"
    "    constant uint& conv_dim    [[buffer(4)]],\n"
    "    constant uint& n_tokens    [[buffer(5)]],\n"
    "    uint c [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (c >= conv_dim) return;\n"
    "\n"
    "    const uint taps = uint(fmssm_conv_taps);\n"
    "    device const float* cw = weight + (ulong)c * uint(fmssm_conv_kernel);\n"
    "    device float* cs = conv_state + (ulong)c * taps;\n"
    "\n"
    "    float hist[fmssm_max_conv_taps];\n"
    "    for (uint t = 0; t < taps; t++)\n"
    "        hist[t] = cs[t];\n"
    "\n"
    "    for (uint p = 0; p < n_tokens; p++) {\n"
    "        float x = qkv[(ulong)p * conv_dim + c];\n"
    "        float acc = x * cw[taps];\n"
    "        for (uint t = 0; t < taps; t++)\n"
    "            acc += hist[t] * cw[t];\n"
    "        out[(ulong)p * conv_dim + c] = fmssm_silu(acc);\n"
    "        for (uint t = 0; t + 1 < taps; t++)\n"
    "            hist[t] = hist[t + 1];\n"
    "        hist[taps - 1] = x;\n"
    "    }\n"
    "\n"
    "    for (uint t = 0; t < taps; t++)\n"
    "        cs[t] = hist[t];\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * Gated RMSNorm: gated = RMSNorm(y) * weight * silu(z), per V-head\n"
    " *\n"
    " * One threadgroup of exactly ONE SIMD group per head, so the sum of\n"
    " * squares is a single simd_sum: no threadgroup memory, no barriers.\n"
    " * head_dim is a function constant, so the lane loop's trip count is\n"
    " * compile-time and unrollable.\n"
    " *\n"
    " * The weight is SHARED ACROSS HEADS: it is head_dim long and head h\n"
    " * indexes weight[d], NOT weight[h * head_dim + d], matching the single\n"
    " * pre-baked norm weight GGUF stores per tensor. Indexing it per head\n"
    " * reads past the buffer for h > 0.\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void fmssm_gated_norm(\n"
    "    device const float* y      [[buffer(0)]],\n"
    "    device const float* weight [[buffer(1)]],\n"
    "    device const float* z      [[buffer(2)]],\n"
    "    device float* gated        [[buffer(3)]],\n"
    "    constant uint& eps_bits    [[buffer(4)]],\n"
    "    uint head [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_index_in_simdgroup]]\n"
    ") {\n"
    "    const uint head_dim = uint(fmssm_head_dim);\n"
    "    const ulong base = (ulong)head * head_dim;\n"
    "\n"
    "    float local_sum_sq = 0.0f;\n"
    "    for (int r = 0; r < fmssm_head_regs_per_lane; r++) {\n"
    "        float val = y[base + uint(r) * 32u + lane];\n"
    "        local_sum_sq += val * val;\n"
    "    }\n"
    "    float inv_rms = precise::rsqrt(simd_sum(local_sum_sq) / float(head_dim)\n"
    "                                   + as_type<float>(eps_bits));\n"
    "\n"
    "    for (int r = 0; r < fmssm_head_regs_per_lane; r++) {\n"
    "        uint d = uint(r) * 32u + lane;\n"
    "        gated[base + d] = y[base + d] * inv_rms * weight[d]\n"
    "                          * fmssm_silu(z[base + d]);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * Delta-rule recurrence — register-resident\n"
    " *\n"
    " * The state is TRANSPOSED: state[h*S*S + j*S + i] is S_h[i][j], so row j\n"
    " * of head h is S contiguous floats and ONE SIMD GROUP owns it. Lane L\n"
    " * holds elements L, 32+L, 64+L, ... in registers — S/32 of them, four at\n"
    " * the model's S = 128 — so every access to the row is a fully coalesced\n"
    " * 128-byte transaction and no lane is idle.\n"
    " *\n"
    " * Per token, per (h, j):\n"
    " *   row  *= exp(g[h])\n"
    " *   pred  = sum_i row[i]*k[h][i]           first simd_sum\n"
    " *   delta = (v[h][j] - pred) * beta[h]\n"
    " *   row  += k[h] * delta                   rank-1 update\n"
    " *   y[h][j] = scale * sum_i row[i]*q[h][i] second simd_sum, read-after-write\n"
    " *\n"
    " * Two simd_sum reductions and nothing else: no threadgroup_barrier, no\n"
    " * shared-memory state array. The state reaches device memory exactly\n"
    " * once per dispatch however many tokens are scanned.\n"
    " *\n"
    " * A threadgroup packs several independent SIMD groups; the pair index is\n"
    " * uniform within a SIMD group, so the out-of-range early return never\n"
    " * splits one and leaves simd_sum with absent lanes.\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void fmssm_recurrence(\n"
    "    device float* state      [[buffer(0)]],\n"
    "    device const float* q    [[buffer(1)]],\n"
    "    device const float* k    [[buffer(2)]],\n"
    "    device const float* v    [[buffer(3)]],\n"
    "    device const float* g    [[buffer(4)]],\n"
    "    device const float* beta [[buffer(5)]],\n"
    "    device float* y          [[buffer(6)]],\n"
    "    constant uint& n_pairs    [[buffer(7)]],\n"
    "    constant uint& scale_bits [[buffer(8)]],\n"
    "    constant uint& n_tokens   [[buffer(9)]],\n"
    "    uint tg    [[threadgroup_position_in_grid]],\n"
    "    uint sg    [[simdgroup_index_in_threadgroup]],\n"
    "    uint sg_pg [[simdgroups_per_threadgroup]],\n"
    "    uint lane  [[thread_index_in_simdgroup]]\n"
    ") {\n"
    "    /* Bound because the decode and scan forms share one host launcher\n"
    "     * and therefore one argument list; decode is the n_tokens == 1 case\n"
    "     * and never reads it. */\n"
    "    (void)n_tokens;\n"
    "    const uint pair = tg * sg_pg + sg;\n"
    "    if (pair >= n_pairs) return;\n"
    "\n"
    "    const uint S = uint(fmssm_state_size);\n"
    "    const uint j = pair % S;                 /* row within the head */\n"
    "    const uint h = pair / S;                 /* V-head */\n"
    "    const ulong head_base = (ulong)h * S;\n"
    "    device float* row = state + (ulong)pair * S;\n"
    "\n"
    "    const float decay = precise::exp(g[h]);\n"
    "    const float beta_h = beta[h];\n"
    "\n"
    "    float ri[fmssm_max_regs_per_lane];\n"
    "    float ki[fmssm_max_regs_per_lane];\n"
    "    float partial = 0.0f;\n"
    "    for (int r = 0; r < fmssm_regs_per_lane; r++) {\n"
    "        uint i = uint(r) * 32u + lane;\n"
    "        ri[r] = row[i] * decay;\n"
    "        ki[r] = k[head_base + i];\n"
    "        partial += ri[r] * ki[r];\n"
    "    }\n"
    "    const float delta = (v[head_base + j] - simd_sum(partial)) * beta_h;\n"
    "\n"
    "    float readout = 0.0f;\n"
    "    for (int r = 0; r < fmssm_regs_per_lane; r++) {\n"
    "        uint i = uint(r) * 32u + lane;\n"
    "        ri[r] += ki[r] * delta;\n"
    "        readout += ri[r] * q[head_base + i];\n"
    "    }\n"
    "    const float acc = simd_sum(readout);\n"
    "\n"
    "    for (int r = 0; r < fmssm_regs_per_lane; r++)\n"
    "        row[uint(r) * 32u + lane] = ri[r];\n"
    "    if (lane == 0)\n"
    "        y[head_base + j] = as_type<float>(scale_bits) * acc;\n"
    "}\n"
    "\n"
    "/* The sequential scan — the PERMANENT correctness reference for the\n"
    " * chunked prefill path, not scaffolding. The state row is read into\n"
    " * registers once, advanced through all n_tokens positions and written\n"
    " * back once; q/k/v/y advance by n_pairs per position and g/beta by\n"
    " * n_pairs/S. Per position the arithmetic is fmssm_recurrence's term for\n"
    " * term and reduction for reduction, so the scan and n single-step\n"
    " * dispatches agree bit for bit. */\n"
    "kernel void fmssm_recurrence_seq(\n"
    "    device float* state      [[buffer(0)]],\n"
    "    device const float* q    [[buffer(1)]],\n"
    "    device const float* k    [[buffer(2)]],\n"
    "    device const float* v    [[buffer(3)]],\n"
    "    device const float* g    [[buffer(4)]],\n"
    "    device const float* beta [[buffer(5)]],\n"
    "    device float* y          [[buffer(6)]],\n"
    "    constant uint& n_pairs    [[buffer(7)]],\n"
    "    constant uint& scale_bits [[buffer(8)]],\n"
    "    constant uint& n_tokens   [[buffer(9)]],\n"
    "    uint tg    [[threadgroup_position_in_grid]],\n"
    "    uint sg    [[simdgroup_index_in_threadgroup]],\n"
    "    uint sg_pg [[simdgroups_per_threadgroup]],\n"
    "    uint lane  [[thread_index_in_simdgroup]]\n"
    ") {\n"
    "    const uint pair = tg * sg_pg + sg;\n"
    "    if (pair >= n_pairs) return;\n"
    "\n"
    "    const uint S = uint(fmssm_state_size);\n"
    "    const uint j = pair % S;\n"
    "    const uint h = pair / S;\n"
    "    const uint n_heads = n_pairs / S;\n"
    "    device float* row = state + (ulong)pair * S;\n"
    "    const float scale = as_type<float>(scale_bits);\n"
    "\n"
    "    float ri[fmssm_max_regs_per_lane];\n"
    "    for (int r = 0; r < fmssm_regs_per_lane; r++)\n"
    "        ri[r] = row[uint(r) * 32u + lane];\n"
    "\n"
    "    for (uint p = 0; p < n_tokens; p++) {\n"
    "        const ulong head_base = (ulong)p * n_pairs + (ulong)h * S;\n"
    "        const float decay = precise::exp(g[(ulong)p * n_heads + h]);\n"
    "        const float beta_h = beta[(ulong)p * n_heads + h];\n"
    "\n"
    "        float ki[fmssm_max_regs_per_lane];\n"
    "        float partial = 0.0f;\n"
    "        for (int r = 0; r < fmssm_regs_per_lane; r++) {\n"
    "            uint i = uint(r) * 32u + lane;\n"
    "            ri[r] *= decay;\n"
    "            ki[r] = k[head_base + i];\n"
    "            partial += ri[r] * ki[r];\n"
    "        }\n"
    "        const float delta = (v[head_base + j] - simd_sum(partial)) * beta_h;\n"
    "\n"
    "        float readout = 0.0f;\n"
    "        for (int r = 0; r < fmssm_regs_per_lane; r++) {\n"
    "            uint i = uint(r) * 32u + lane;\n"
    "            ri[r] += ki[r] * delta;\n"
    "            readout += ri[r] * q[head_base + i];\n"
    "        }\n"
    "        const float acc = simd_sum(readout);\n"
    "        if (lane == 0)\n"
    "            y[head_base + j] = scale * acc;\n"
    "    }\n"
    "\n"
    "    for (int r = 0; r < fmssm_regs_per_lane; r++)\n"
    "        row[uint(r) * 32u + lane] = ri[r];\n"
    "}\n"
    "\n"
    ;
}
