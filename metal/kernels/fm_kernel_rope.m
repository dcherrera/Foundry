/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_rope.m
 * @brief Host dispatch and MSL source for the rotary-position-embedding kernels.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_rope.h"

#include <string.h>

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

/* MSL function-constant indices owned by this family (see the header).
 * The shader spells the same indices literally in its
 * [[function_constant(n)]] attributes — change one and change both. */
#define FM_ROPE_FC_HEAD_DIM 100
#define FM_ROPE_FC_ROT_DIM  101

/* eps/theta ride through the uint32 constant channel as bit patterns; the
 * kernel reads them back with as_type<float>. */
static uint32_t fmrope_float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* Dims actually rotated. Anything outside (0, head_dim) means the whole
 * head, which is what the CPU reference does. */
static int fmrope_rotated_dims(int head_dim, int rope_dim) {
    return (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
}

/* head_dim and the rotated span are function constants so the kernel's
 * head/pair index split and its frequency exponent fold at shader-compile
 * time; only the head count and position stay runtime values. */
static void fmrope_fill_constants(fm_fc_values *fc, int head_dim, int rot) {
    fm_fc_reset(fc);
    fm_fc_set_int(fc, FM_ROPE_FC_HEAD_DIM, head_dim);
    fm_fc_set_int(fc, FM_ROPE_FC_ROT_DIM, rot);
}

void fm_rope_neox_partial(fm_buffer *data, int n_heads, int head_dim,
                          int rope_dim, int pos, float theta_base) {
    if (!fm_metal_is_initialized() || !data || n_heads <= 0 || head_dim <= 0)
        return;

    int rot = fmrope_rotated_dims(head_dim, rope_dim);
    fm_fc_values fc;
    fmrope_fill_constants(&fc, head_dim, rot);

    /* One thread per rotated pair: n_heads * rot/2 threads, each touching
     * exactly two elements. Nothing is shared, so there is no reduction,
     * no threadgroup memory and no barrier. */
    fm_buffer *bufs[] = { data };
    uint32_t consts[] = { (uint32_t)(n_heads * (rot / 2)), (uint32_t)pos,
                          fmrope_float_bits(theta_base) };
    fm_dispatch_1d_specialized(@"rope_neox_partial", &fc, bufs, 1, consts, 3,
                               n_heads * (rot / 2));
}

void fm_rope_neox_partial_pos(fm_buffer *data, fm_buffer *positions,
                              int n_rows, int n_heads, int head_dim,
                              int rope_dim, float theta_base) {
    if (!fm_metal_is_initialized() || !data || !positions ||
        n_rows <= 0 || n_heads <= 0 || head_dim <= 0)
        return;

    int rot = fmrope_rotated_dims(head_dim, rope_dim);
    fm_fc_values fc;
    fmrope_fill_constants(&fc, head_dim, rot);

    int pairs_per_row = n_heads * (rot / 2);
    fm_buffer *bufs[] = { data, positions };
    uint32_t consts[] = { (uint32_t)(n_rows * pairs_per_row),
                          (uint32_t)pairs_per_row,
                          fmrope_float_bits(theta_base) };
    fm_dispatch_1d_specialized(@"rope_neox_partial_pos", &fc, bufs, 2, consts, 3,
                               n_rows * pairs_per_row);
}

NSString *fm_kernel_msl_rope(void)
{
    return @
    "/* ============================================================\n"
    " * Partial NEOX rotary position embedding\n"
    " *\n"
    " * Embarrassingly parallel: one thread owns one (head, j) pair and\n"
    " * touches exactly the two elements j and j + pair_span. No threadgroup\n"
    " * memory, no shuffles, no barriers.\n"
    " *\n"
    " * PARTIAL means only the first fmrope_fc_rot_dim dims of each head are\n"
    " * rotated. Dims [rot_dim, head_dim) are never written — they pass\n"
    " * through byte-identical, which is a correctness property a test that\n"
    " * only inspects the rotated span would not notice.\n"
    " *\n"
    " * NEOX pairing is j against j + rot_dim/2, NOT adjacent j / j+1.\n"
    " *\n"
    " * precise:: trig and division are deliberate: the library is compiled\n"
    " * with fast math on, and the fast reciprocal/trig drift far enough\n"
    " * from the CPU reference at large positions to matter.\n"
    " *\n"
    " * Names are prefixed fmrope_ and the fragment defines no preprocessor\n"
    " * macro: all fragments share ONE MSL translation unit, so a macro or\n"
    " * an unprefixed helper would leak into every family after this one.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fmrope_fc_head_dim_in [[function_constant(100)]];\n"
    "constant int fmrope_fc_rot_dim_in  [[function_constant(101)]];\n"
    "/* The host always specializes both; the defaults exist only because MSL\n"
    " * requires the unspecialized pipeline to still build, and are the\n"
    " * qwen35moe head_dim / rope_dim. */\n"
    "constant int fmrope_fc_head_dim =\n"
    "    is_function_constant_defined(fmrope_fc_head_dim_in) ? fmrope_fc_head_dim_in : 256;\n"
    "constant int fmrope_fc_rot_dim =\n"
    "    is_function_constant_defined(fmrope_fc_rot_dim_in) ? fmrope_fc_rot_dim_in : 64;\n"
    "\n"
    "/* Rotates the single pair (j, j + pair_span) of `head` in place. Mirrors the\n"
    " * CPU reference term for term, including taking the reciprocal of pow\n"
    " * rather than raising to a negated exponent. */\n"
    "inline void fmrope_rotate_pair(device float* head, uint j, float pos,\n"
    "                                float theta_base) {\n"
    "    const uint rot       = uint(fmrope_fc_rot_dim);\n"
    "    const uint pair_span = rot / 2u;  // 'half' is a reserved MSL type\n"
    "\n"
    "    float freq = precise::divide(1.0f,\n"
    "        precise::pow(theta_base, float(2u * j) / float(rot)));\n"
    "    float angle = pos * freq;\n"
    "    float cos_a = precise::cos(angle);\n"
    "    float sin_a = precise::sin(angle);\n"
    "\n"
    "    float x0 = head[j];\n"
    "    float x1 = head[j + pair_span];\n"
    "    head[j]             = x0 * cos_a - x1 * sin_a;\n"
    "    head[j + pair_span] = x0 * sin_a + x1 * cos_a;\n"
    "}\n"
    "\n"
    "kernel void rope_neox_partial(\n"
    "    device float* data        [[buffer(0)]],\n"
    "    constant uint& n_pairs    [[buffer(1)]],\n"
    "    constant uint& pos        [[buffer(2)]],\n"
    "    constant uint& theta_bits [[buffer(3)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (idx >= n_pairs) return;\n"
    "\n"
    "    const uint pair_span = uint(fmrope_fc_rot_dim) / 2u;\n"
    "    uint head = idx / pair_span;\n"
    "    uint j    = idx % pair_span;\n"
    "    fmrope_rotate_pair(data + head * uint(fmrope_fc_head_dim), j,\n"
    "                        float(pos), as_type<float>(theta_bits));\n"
    "}\n"
    "\n"
    "/* Same rotation, but each row of heads carries its own token position —\n"
    " * one dispatch rotates a whole prefill chunk. */\n"
    "kernel void rope_neox_partial_pos(\n"
    "    device float* data           [[buffer(0)]],\n"
    "    device const int* positions  [[buffer(1)]],\n"
    "    constant uint& n_pairs       [[buffer(2)]],\n"
    "    constant uint& pairs_per_row [[buffer(3)]],\n"
    "    constant uint& theta_bits    [[buffer(4)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (idx >= n_pairs) return;\n"
    "\n"
    "    const uint pair_span = uint(fmrope_fc_rot_dim) / 2u;\n"
    "    uint row       = idx / pairs_per_row;\n"
    "    uint in_row    = idx % pairs_per_row;\n"
    "    uint head      = row * (pairs_per_row / pair_span) + in_row / pair_span;\n"
    "    uint j         = in_row % pair_span;\n"
    "    fmrope_rotate_pair(data + head * uint(fmrope_fc_head_dim), j,\n"
    "                        float(positions[row]), as_type<float>(theta_bits));\n"
    "}\n"
    "\n"
    ;
}
