/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_weight_convert.c
 * @brief Implementation of the streaming GGUF -> Blackhole weight conversion.
 *
 * The whole file is one loop shape:
 *
 *     for each tensor
 *         for each strip of 32 rows
 *             dequantize the strip to fp32          (ft_gguf_dequant)
 *             lay it out as tiles                   (ft_tt_tilize / swizzle)
 *             encode to the target format           (ft_bfp / ft_float_convert)
 *             write it
 *             optionally decode it back and fold the error into a PCC
 *
 * so the peak memory is the strip buffers, which are sized once for the widest
 * tensor in the file and then reused. Nothing here allocates per strip.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_tt_weight_convert.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ft_bfp.h"
#include "ft_float_convert.h"
#include "ft_gguf_dequant.h"
#include "ft_gguf_map.h"
#include "ft_tt_precision_ledger.h"

#include "../ft_tt_tilize.h"

/** Running Pearson accumulator over the fp32 source and its round trip. */
typedef struct {
    double sum_x;
    double sum_y;
    double sum_xx;
    double sum_yy;
    double sum_xy;
    double max_abs_error;
    double sum_sq_error;
    uint64_t count;
} pearson_accumulator;

/** Buffers reused across every strip of every tensor. */
typedef struct {
    float *source_strip;   /**< 32 x cols fp32, dequantized */
    float *tile_strip;     /**< tile_cols tiles, row-major within each tile */
    unsigned char *packed; /**< the encoded strip, ready to write */
    float *decoded;        /**< round trip, only when PCC is being measured */
    float *reference_strip; /**< the same rows of a reference checkpoint, or NULL */
    uint32_t max_cols;
    uint32_t max_tile_cols;
} strip_buffers;

static void pearson_reset(pearson_accumulator *acc)
{
    memset(acc, 0, sizeof(*acc));
}

static void pearson_add(pearson_accumulator *acc, double x, double y)
{
    double error = y - x;
    double magnitude = error < 0 ? -error : error;

    acc->sum_x += x;
    acc->sum_y += y;
    acc->sum_xx += x * x;
    acc->sum_yy += y * y;
    acc->sum_xy += x * y;
    acc->sum_sq_error += error * error;
    if (magnitude > acc->max_abs_error)
        acc->max_abs_error = magnitude;
    acc->count++;
}

/**
 * Pearson correlation of the two accumulated series.
 *
 * A constant series has zero variance and no defined correlation. That is not
 * a corner case here — a zero-initialized tensor is exactly that — so it
 * reports 1.0 when the two series are identical and 0.0 when they are not,
 * rather than a NaN that would silently poison a table of results.
 */
static double pearson_value(const pearson_accumulator *acc)
{
    double n = (double)acc->count;
    double cov, var_x, var_y;

    if (acc->count == 0)
        return 1.0;

    cov = n * acc->sum_xy - acc->sum_x * acc->sum_y;
    var_x = n * acc->sum_xx - acc->sum_x * acc->sum_x;
    var_y = n * acc->sum_yy - acc->sum_y * acc->sum_y;

    if (var_x <= 0.0 || var_y <= 0.0)
        return acc->sum_sq_error == 0.0 ? 1.0 : 0.0;

    return cov / sqrt(var_x * var_y);
}

static double pearson_rms(const pearson_accumulator *acc)
{
    if (acc->count == 0)
        return 0.0;
    return sqrt(acc->sum_sq_error / (double)acc->count);
}

/**
 * Error RMS as a fraction of the source's own RMS.
 *
 * `sum_xx` is already the sum of the squared source values, so this costs
 * nothing beyond a divide — and it is the only error figure that can be
 * compared between two tensors of different scale.
 */
static double pearson_relative_rms(const pearson_accumulator *acc)
{
    if (acc->count == 0 || acc->sum_xx <= 0.0)
        return 0.0;
    return sqrt(acc->sum_sq_error / acc->sum_xx);
}

/* ------------------------------------------------------------------------- */
/* Geometry                                                                  */
/* ------------------------------------------------------------------------- */

/**
 * A GGUF tensor's logical shape as a matrix. dims[0] is the fastest axis, so
 * it is the column count; everything above it collapses into rows, which is
 * correct for the 2-D weight matrices this converter handles and keeps a
 * 3-D tensor from silently being tiled along the wrong axis.
 */
static void tensor_shape(const ft_gguf_tensor *tensor, uint64_t *rows, uint64_t *cols)
{
    *cols = tensor->dims[0];
    *rows = tensor->dims[1] * tensor->dims[2] * tensor->dims[3];
}

static size_t format_element_bytes(ft_tt_tensor_format format)
{
    return format == FT_TT_FORMAT_FP32 ? 4u : 2u;
}

static ft_bfp_format bfp_format_for(ft_tt_tensor_format format)
{
    switch (format) {
    case FT_TT_FORMAT_BFP4_B:
        return FT_BFP_BFP4_B;
    case FT_TT_FORMAT_BFP6_B:
        return FT_BFP_BFP6_B;
    default:
        return FT_BFP_BFP8_B;
    }
}

static int format_is_block_float(ft_tt_tensor_format format)
{
    return format == FT_TT_FORMAT_BFP8_B || format == FT_TT_FORMAT_BFP4_B ||
           format == FT_TT_FORMAT_BFP6_B;
}

int ft_tt_convert_is_expert_stack(const ft_gguf_tensor *tensor)
{
    ft_tt_tensor_role role;

    if (!tensor || tensor->n_dims != 3)
        return 0;
    role = ft_tt_ledger_role(tensor->name);
    return role == FT_TT_ROLE_EXPERT_UP || role == FT_TT_ROLE_EXPERT_DOWN;
}

/**
 * Refuse an expert stack whose experts would not each be whole tiles.
 *
 * The per-expert layout is FREE only because a conversion strip is 32 source rows and an
 * expert is `dims[1]` source rows: when 32 divides `dims[1]` no strip straddles two experts,
 * so the strips of expert e are exactly expert e's Nt tile columns, contiguous, in N-major
 * order. Any other N would interleave two experts inside one tile column — every index in
 * range, every expert slice wrong — so it is refused rather than padded.
 */
static int expert_stack_geometry_checked(const ft_gguf_tensor *tensor)
{
    if (tensor->dims[1] == 0u || tensor->dims[1] % FT_TT_TILE_HEIGHT != 0u || tensor->dims[2] == 0u ||
        tensor->dims[3] != 1u) {
        fprintf(stderr, "ft_tt_convert: expert stack '%s' is [%llu, %llu, %llu, %llu]; its per-expert "
                        "N (dims[1]) must be a non-zero multiple of 32 so each expert is whole tiles\n",
                tensor->name, (unsigned long long)tensor->dims[0], (unsigned long long)tensor->dims[1],
                (unsigned long long)tensor->dims[2], (unsigned long long)tensor->dims[3]);
        return FT_TT_CONVERT_EEXPERTGEOMETRY;
    }
    return FT_TT_CONVERT_OK;
}

int ft_tt_convert_layout_for_checked(ft_tt_tensor_role role, uint32_t n_dims, ft_tt_tensor_layout *out)
{
    if (!out)
        return FT_TT_CONVERT_EINVAL;
    if (n_dims < 2) {
        *out = FT_TT_LAYOUT_SOURCE;
        return FT_TT_CONVERT_OK;
    }

    switch (role) {
    case FT_TT_ROLE_ATTENTION:
    case FT_TT_ROLE_FFN_UP:
    case FT_TT_ROLE_FFN_DOWN:
    case FT_TT_ROLE_LM_HEAD:
    case FT_TT_ROLE_SSM_IN:
    case FT_TT_ROLE_SSM_OUT:
    case FT_TT_ROLE_SSM_GATE:
    case FT_TT_ROLE_PROJECTION:
    case FT_TT_ROLE_ROUTER:
    case FT_TT_ROLE_EXPERT_UP:
    case FT_TT_ROLE_EXPERT_DOWN:
        *out = FT_TT_LAYOUT_MATMUL_NMAJOR;
        return FT_TT_CONVERT_OK;
    case FT_TT_ROLE_EMBEDDING:
        /* Indexed BY ROW by a token id. Transposing it turns a lookup into a
         * gather across the whole tensor. */
        *out = FT_TT_LAYOUT_SOURCE;
        return FT_TT_CONVERT_OK;
    case FT_TT_ROLE_SSM_CONV:
        /* A depthwise convolution over `taps` positions, not a matmul: there is
         * no K axis to put first. */
        *out = FT_TT_LAYOUT_SOURCE;
        return FT_TT_CONVERT_OK;
    case FT_TT_ROLE_NORM:
    case FT_TT_ROLE_BIAS:
    case FT_TT_ROLE_SSM_DECAY:
        /* Reached only by a 2-D tensor carrying one of these names, which is
         * not a shape we have seen; keeping the source order is the answer that
         * matches the 1-D case these always are. */
        *out = FT_TT_LAYOUT_SOURCE;
        return FT_TT_CONVERT_OK;
    default:
        /* ROLE_UNKNOWN. See the header: both defaults are a guess about a
         * kernel, so neither is taken. */
        return FT_TT_CONVERT_EUNCLASSIFIED;
    }
}

/* ------------------------------------------------------------------------- */
/* Rotary row permutation                                                    */
/* ------------------------------------------------------------------------- */

/**
 * The head geometry one tensor's rows are permuted under.
 *
 * `head_rows` is the whole head; `rot_rows` is the rotated prefix of it. Rows at
 * or past `rot_rows` inside a head are not rotated by either form of RoPE and so
 * are not moved.
 */
typedef struct {
    uint64_t head_rows;
    uint64_t rot_rows;
    /** Reflect the rotary PAIR INDEX: slot p carries the model's pair half-1-p. */
    int reverse_pairs;
    /** Keep GGUF's INTERLEAVED grouping — a pair's partner stays adjacent — instead of
     *  undoing it to split-half. Only meaningful with `reverse_pairs`, because interleaved
     *  without a reflection is just the source order and needs no permutation at all. */
    int keep_interleaved;
} rope_row_permutation;

/**
 * Which SOURCE row supplies target row `row`, undoing llama.cpp's HF->GGUF
 * permutation.
 *
 * llama.cpp maps split-half index `(a, b)` — `a` in {0,1}, `b` in [0, rot/2) —
 * to interleaved index `2b + a`. Inverting it: target `n` below `rot/2` is
 * `(a=0, b=n)` and comes from source `2n`; target `n` at or above `rot/2` is
 * `(a=1, b=n-rot/2)` and comes from source `2(n - rot/2) + 1`.
 *
 * With `reverse_pairs` the pair index is reflected FIRST — target slot `(a, b)`
 * takes what split-half would have put at `(a, half-1-b)` — and the split-half
 * inverse then runs unchanged. Composing in that order is what preserves the
 * property everything downstream rests on: slots `(0, b)` and `(1, b)` reflect to
 * the same `half-1-b`, so a partner is still exactly `half` rows away.
 */
static uint64_t rope_permuted_source_row(const rope_row_permutation *permutation, uint64_t row)
{
    uint64_t head = row / permutation->head_rows;
    uint64_t within = row % permutation->head_rows;
    uint64_t half = permutation->rot_rows / 2u;

    if (within >= permutation->rot_rows)
        return row;
    if (permutation->keep_interleaved) {
        /* The partner stays ADJACENT, so a slot is (pair, component) = (within/2, within%2)
         * and the model's pair `half-1-p` sits at GGUF row `2*(half-1-p) + component`. No
         * split-half undo runs at all: this permutation stays inside GGUF's own order. */
        uint64_t pair = within / 2u;
        uint64_t component = within % 2u;

        return head * permutation->head_rows + (half - 1u - pair) * 2u + component;
    }
    if (permutation->reverse_pairs) {
        uint64_t upper = within / half; /* 0 = first component of the pair, 1 = second */
        uint64_t pair = within % half;  /* the slot's pair index */

        within = upper * half + (half - 1u - pair);
    }
    within = within < half ? within * 2u : (within - half) * 2u + 1u;
    return head * permutation->head_rows + within;
}

/* ------------------------------------------------------------------------- */
/* Tile transpose                                                            */
/* ------------------------------------------------------------------------- */

/**
 * Where element (row, column) of a 32x32 tile sits, in elements from the tile base.
 *
 * Two layouts, because the converter produces both. A block-float strip is SWIZZLED —
 * row-major inside the tile — because tt-metal's packer does the face reordering itself,
 * while fp32 and bf16 strips are already FACE-ORDERED, four 16x16 faces in the order
 * (0,0) (0,1) (1,0) (1,1). Transposing a face-ordered tile is therefore not a transpose
 * of a 32x32 array: it transposes each face AND swaps the two off-diagonal faces, and
 * doing only the first half is the kind of mistake PCC would not see.
 */
static size_t tile_element_index(uint32_t row, uint32_t column, int face_ordered)
{
    size_t face;

    if (!face_ordered)
        return (size_t)row * FT_TT_TILE_WIDTH + column;
    face = (size_t)(row / FT_TT_FACE_HEIGHT) * 2u + (column / FT_TT_FACE_WIDTH);
    return face * FT_TT_FACE_VALUES + (size_t)(row % FT_TT_FACE_HEIGHT) * FT_TT_FACE_WIDTH +
           (column % FT_TT_FACE_WIDTH);
}

/** Transpose one 32x32 tile in place. `element_bytes` is 2 (bf16) or 4 (fp32 or float). */
static void transpose_tile(void *tile, size_t element_bytes, int face_ordered)
{
    unsigned char scratch[FT_TT_TILE_VALUES * 4];
    unsigned char *bytes = tile;
    uint32_t row;

    memcpy(scratch, bytes, (size_t)FT_TT_TILE_VALUES * element_bytes);
    for (row = 0; row < FT_TT_TILE_HEIGHT; row++) {
        uint32_t column;

        for (column = 0; column < FT_TT_TILE_WIDTH; column++)
            memcpy(bytes + tile_element_index(row, column, face_ordered) * element_bytes,
                   scratch + tile_element_index(column, row, face_ordered) * element_bytes,
                   element_bytes);
    }
}

/** Transpose every tile of one strip in place. */
static void transpose_strip_tiles(void *tiles, uint32_t tile_count, size_t element_bytes,
                                  int face_ordered)
{
    unsigned char *bytes = tiles;
    uint32_t index;

    for (index = 0; index < tile_count; index++)
        transpose_tile(bytes + (size_t)index * FT_TT_TILE_VALUES * element_bytes, element_bytes,
                       face_ordered);
}

/** Bytes one strip of `tile_cols` tiles occupies in `format`. */
static size_t strip_bytes_for(ft_tt_tensor_format format, uint32_t tile_cols)
{
    if (format_is_block_float(format))
        return (size_t)tile_cols * ft_bfp_tile_bytes(bfp_format_for(format));
    return (size_t)tile_cols * FT_TT_TILE_VALUES * format_element_bytes(format);
}

/* ------------------------------------------------------------------------- */
/* Buffers                                                                   */
/* ------------------------------------------------------------------------- */

static void strip_buffers_free(strip_buffers *buffers)
{
    free(buffers->source_strip);
    free(buffers->tile_strip);
    free(buffers->packed);
    free(buffers->decoded);
    free(buffers->reference_strip);
    memset(buffers, 0, sizeof(*buffers));
}

/** Add the reference strip, for a measurement scored against a second file. */
static int strip_buffers_add_reference(strip_buffers *buffers)
{
    buffers->reference_strip = calloc((size_t)FT_TT_TILE_HEIGHT * buffers->max_cols, sizeof(float));
    return buffers->reference_strip ? FT_TT_CONVERT_OK : FT_TT_CONVERT_ENOMEM;
}

/**
 * Size the buffers once, for the widest tensor in the file. `measure_pcc`
 * adds the decode buffer; without it that allocation is skipped entirely.
 */
static int strip_buffers_alloc(strip_buffers *buffers, const ft_gguf_map *map, int measure_pcc)
{
    uint32_t max_cols = 0;
    size_t strip_values;

    memset(buffers, 0, sizeof(*buffers));

    for (uint32_t i = 0; i < ft_gguf_map_tensor_count(map); i++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_tensor(map, i);
        uint64_t rows, cols;

        tensor_shape(tensor, &rows, &cols);
        if (cols > UINT32_MAX)
            return FT_TT_CONVERT_EINVAL;
        if ((uint32_t)cols > max_cols)
            max_cols = (uint32_t)cols;
    }
    if (max_cols == 0)
        return FT_TT_CONVERT_EINVAL;

    buffers->max_cols = max_cols;
    buffers->max_tile_cols = ft_tt_tile_count_for(max_cols);
    strip_values = (size_t)buffers->max_tile_cols * FT_TT_TILE_VALUES;

    buffers->source_strip = calloc((size_t)FT_TT_TILE_HEIGHT * max_cols, sizeof(float));
    buffers->tile_strip = calloc(strip_values, sizeof(float));
    /* fp32 tiles are the largest encoding, so sizing on them covers every
     * format without a per-format branch here. */
    buffers->packed = calloc(strip_values, 4u);
    if (measure_pcc)
        buffers->decoded = calloc(strip_values, sizeof(float));

    if (!buffers->source_strip || !buffers->tile_strip || !buffers->packed || (measure_pcc && !buffers->decoded)) {
        strip_buffers_free(buffers);
        return FT_TT_CONVERT_ENOMEM;
    }

    return FT_TT_CONVERT_OK;
}

/* ------------------------------------------------------------------------- */
/* One strip                                                                 */
/* ------------------------------------------------------------------------- */

/**
 * A COLUMN WINDOW of a source tensor: each logical row is `cols` values starting
 * `byte_offset` bytes into a source row that is `row_bytes` long. The MTP eh_proj split
 * reads its two halves through one; NULL everywhere else (whole rows).
 */
typedef struct {
    uint64_t row_bytes;
    uint64_t byte_offset;
} source_window;

/**
 * Dequantize `valid_rows` source rows starting at `first_row` into the strip
 * buffer, one row at a time so the mapping is touched strictly sequentially.
 *
 * `permutation` is NULL for every tensor but a permuted `attn_q`/`attn_k`, and
 * that is the only case where the mapping is NOT walked in order: the split-half
 * partner of a row sits `rot/2` rows away, which for MiniCPM5-1B is two strips
 * off, so the gather cannot be made strip-local. It stays inside ONE HEAD —
 * 128 rows, ~192 KiB of a Q8_0 tensor — so it is a walk over a few pages, not a
 * walk over the tensor.
 */
static int load_source_strip(const ft_gguf_tensor *tensor, uint64_t first_row, uint32_t valid_rows, uint64_t cols,
                             const rope_row_permutation *permutation, const source_window *window, float *strip)
{
    size_t row_bytes = window ? (size_t)window->row_bytes : ft_gguf_type_bytes(tensor->ggml_type, cols);
    const unsigned char *base = (const unsigned char *)tensor->data + (window ? window->byte_offset : 0u);

    if (row_bytes == 0)
        return FT_TT_CONVERT_EUNSUPPORTED;

    for (uint32_t row = 0; row < valid_rows; row++) {
        uint64_t target = first_row + row;
        uint64_t source = permutation ? rope_permuted_source_row(permutation, target) : target;
        const unsigned char *src = base + (size_t)source * row_bytes;

        if (ft_gguf_dequant_to_float(tensor->ggml_type, src, cols, strip + (size_t)row * cols) != 0)
            return FT_TT_CONVERT_EUNSUPPORTED;
    }

    return FT_TT_CONVERT_OK;
}

/**
 * Encode one strip into `buffers->packed` and report its size.
 *
 * Block-float takes the SWIZZLED layout (row-major within each tile) because
 * tt-metal's packer does the face reordering itself; fp32 and bf16 take the
 * face-ordered layout because nothing downstream will reorder them.
 *
 * `transposed` is FT_TT_LAYOUT_MATMUL_NMAJOR's half of the work that is not
 * free: each 32x32 tile is transposed before it is encoded, so the strip's
 * tiles come out as target column `nt`'s Kt tiles, in kt order — which IS
 * N-major order for that column, so the write stays sequential. The transpose
 * has to happen before ft_bfp packs: a packed tile's exponent is shared along
 * a row, so transposing afterwards would mean unpack and repack.
 */
static int encode_strip(strip_buffers *buffers, ft_tt_tensor_format format, uint32_t valid_rows, uint64_t cols,
                        uint32_t tile_cols, int transposed, size_t *bytes_out)
{
    size_t src_stride = (size_t)cols * sizeof(float);
    int rc;

    if (format_is_block_float(format)) {
        rc = ft_tt_tile_swizzle_strip(buffers->source_strip, sizeof(float), valid_rows, (uint32_t)cols, src_stride,
                                      buffers->tile_strip);
        if (rc != FT_TT_TILIZE_OK)
            return FT_TT_CONVERT_ERROR;
        if (transposed)
            transpose_strip_tiles(buffers->tile_strip, tile_cols, sizeof(float), 0);
        if (ft_bfp_pack_tiles(bfp_format_for(format), buffers->tile_strip, tile_cols, buffers->packed) != FT_BFP_OK)
            return FT_TT_CONVERT_ERROR;
        *bytes_out = strip_bytes_for(format, tile_cols);
        return FT_TT_CONVERT_OK;
    }

    if (format == FT_TT_FORMAT_FP32) {
        rc = ft_tt_tilize_strip(buffers->source_strip, sizeof(float), valid_rows, (uint32_t)cols, src_stride,
                                buffers->packed);
        if (rc != FT_TT_TILIZE_OK)
            return FT_TT_CONVERT_ERROR;
        if (transposed)
            transpose_strip_tiles(buffers->packed, tile_cols, sizeof(float), 1);
        *bytes_out = strip_bytes_for(format, tile_cols);
        return FT_TT_CONVERT_OK;
    }

    /* bf16: narrow first, into the tile-strip buffer reused as bf16 scratch,
     * then tilize at 2 bytes per element. */
    ft_bf16_from_float_array(buffers->source_strip, (uint16_t *)buffers->tile_strip,
                             (size_t)valid_rows * (size_t)cols);
    rc = ft_tt_tilize_strip(buffers->tile_strip, sizeof(uint16_t), valid_rows, (uint32_t)cols,
                            (size_t)cols * sizeof(uint16_t), buffers->packed);
    if (rc != FT_TT_TILIZE_OK)
        return FT_TT_CONVERT_ERROR;
    if (transposed)
        transpose_strip_tiles(buffers->packed, tile_cols, sizeof(uint16_t), 1);
    *bytes_out = strip_bytes_for(format, tile_cols);
    return FT_TT_CONVERT_OK;
}

/**
 * Decode the strip that was just encoded, back into `buffers->decoded`.
 *
 * The PCC this feeds is against the SOURCE strip, so a transposed strip has to
 * be transposed back first — otherwise the round trip would score a transpose
 * against an untransposed original and report a number near zero for a
 * conversion that is exactly right. This runs after the strip has been
 * written, so mutating `packed` in place is free.
 */
static int decode_strip(strip_buffers *buffers, ft_tt_tensor_format format, uint32_t valid_rows, uint64_t cols,
                        uint32_t tile_cols, int transposed)
{
    size_t dst_stride = (size_t)cols * sizeof(float);

    if (format_is_block_float(format)) {
        if (ft_bfp_unpack_tiles(bfp_format_for(format), buffers->packed, tile_cols, buffers->tile_strip) != FT_BFP_OK)
            return FT_TT_CONVERT_ERROR;
        if (transposed)
            transpose_strip_tiles(buffers->tile_strip, tile_cols, sizeof(float), 0);
        if (ft_tt_tile_unswizzle_strip(buffers->tile_strip, sizeof(float), valid_rows, (uint32_t)cols, dst_stride,
                                       buffers->decoded) != FT_TT_TILIZE_OK)
            return FT_TT_CONVERT_ERROR;
        return FT_TT_CONVERT_OK;
    }

    if (format == FT_TT_FORMAT_FP32) {
        if (transposed)
            transpose_strip_tiles(buffers->packed, tile_cols, sizeof(float), 1);
        if (ft_tt_untilize_strip(buffers->packed, sizeof(float), valid_rows, (uint32_t)cols, dst_stride,
                                 buffers->decoded) != FT_TT_TILIZE_OK)
            return FT_TT_CONVERT_ERROR;
        return FT_TT_CONVERT_OK;
    }

    if (transposed)
        transpose_strip_tiles(buffers->packed, tile_cols, sizeof(uint16_t), 1);
    if (ft_tt_untilize_strip(buffers->packed, sizeof(uint16_t), valid_rows, (uint32_t)cols,
                             (size_t)cols * sizeof(uint16_t), buffers->tile_strip) != FT_TT_TILIZE_OK)
        return FT_TT_CONVERT_ERROR;
    ft_float_from_bf16_array((const uint16_t *)buffers->tile_strip, buffers->decoded,
                             (size_t)valid_rows * (size_t)cols);
    return FT_TT_CONVERT_OK;
}

/** Fold one strip into an accumulator. `truth` is the x series, `estimate` the y. */
static void accumulate_strip_error(pearson_accumulator *acc, const float *truth, const float *estimate,
                                   uint32_t valid_rows, uint64_t cols)
{
    for (uint32_t row = 0; row < valid_rows; row++) {
        const float *x = truth + (size_t)row * cols;
        const float *y = estimate + (size_t)row * cols;

        for (uint64_t col = 0; col < cols; col++)
            pearson_add(acc, (double)x[col], (double)y[col]);
    }
}

/**
 * Score one strip, against the reference when there is one.
 *
 * With a reference the truth is the reference, NOT the source — which is the
 * whole point of supplying one — and `source_acc` picks up how far the source
 * itself already was from it.
 */
static int score_strip(strip_buffers *buffers, const ft_gguf_tensor *reference, uint64_t first, uint32_t valid,
                       uint64_t cols, pearson_accumulator *acc, pearson_accumulator *source_acc)
{
    if (!reference) {
        accumulate_strip_error(acc, buffers->source_strip, buffers->decoded, valid, cols);
        return FT_TT_CONVERT_OK;
    }

    {
        int rc = load_source_strip(reference, first, valid, cols, NULL, NULL, buffers->reference_strip);

        if (rc != FT_TT_CONVERT_OK)
            return rc;
    }
    accumulate_strip_error(acc, buffers->reference_strip, buffers->decoded, valid, cols);
    accumulate_strip_error(source_acc, buffers->reference_strip, buffers->source_strip, valid, cols);
    return FT_TT_CONVERT_OK;
}

/* ------------------------------------------------------------------------- */
/* One tensor                                                               */
/* ------------------------------------------------------------------------- */

/**
 * How one tensor is written: what it is packed as, how its tiles are arranged,
 * and whether its output rows are permuted on the way in.
 *
 * Three fields that always travel together and are always decided together, so
 * they are one argument rather than three that could be passed inconsistently.
 */
typedef struct {
    ft_tt_tensor_format format;
    ft_tt_tensor_layout layout;
    /** NULL for every tensor except a permuted `attn_q`/`attn_k`. */
    const rope_row_permutation *rope;
    /** NULL for every tensor except one half of the MTP eh_proj split. */
    const source_window *window;
} tensor_encoding;

/** A 1-D tensor is written row-major and untiled; see the ledger header. */
static int write_untiled(FILE *out, const ft_gguf_tensor *tensor, uint64_t cols, strip_buffers *buffers,
                         uint64_t *bytes_written)
{
    int rc = load_source_strip(tensor, 0, 1, cols, NULL, NULL, buffers->source_strip);

    if (rc != FT_TT_CONVERT_OK)
        return rc;
    if (out && fwrite(buffers->source_strip, sizeof(float), (size_t)cols, out) != cols)
        return FT_TT_CONVERT_EIO;

    *bytes_written = cols * sizeof(float);
    return FT_TT_CONVERT_OK;
}

/**
 * Source rows read between two releases of the pages behind the cursor: one expert of an
 * expert stack (2048 x 512 Q8_0 = 1.1 MB on Qwen3.6), else 1024 rows. Without it a tensor's
 * pages stay resident until the WHOLE tensor is done — 285 MB for one Qwen3.6 expert stack,
 * 540 MB for its LM head — and the converter's RSS is the largest tensor, not a strip.
 */
static uint64_t release_rows_for(const ft_gguf_tensor *tensor)
{
    return ft_tt_convert_is_expert_stack(tensor) ? tensor->dims[1] : 1024u;
}

/**
 * Drop the source pages of rows `[from_row, to_row)`. Correctness never depends on it: a
 * re-read of a released page (a rope permutation reaching back inside its head) faults the
 * same bytes back in from the file.
 */
static void release_source_rows(const ft_gguf_map *map, const ft_gguf_tensor *tensor, uint64_t cols,
                                uint64_t from_row, uint64_t to_row)
{
    size_t row_bytes = ft_gguf_type_bytes(tensor->ggml_type, cols);

    if (map && row_bytes && to_row > from_row)
        ft_gguf_map_release_range(map, tensor, from_row * row_bytes, (to_row - from_row) * row_bytes);
}

/**
 * Convert one tensor, writing to `out` when it is non-NULL. With `out` NULL
 * this is the measure-only path — same arithmetic, no artifact. `map` (may be NULL)
 * is the source's, for releasing pages behind the cursor.
 */
static int convert_tensor(FILE *out, const ft_gguf_map *map, const ft_gguf_tensor *tensor,
                          const ft_gguf_tensor *reference, const tensor_encoding *encoding,
                          strip_buffers *buffers, int measure_pcc, ft_tt_convert_tensor_report *report)
{
    uint64_t release_rows = release_rows_for(tensor);
    uint64_t released_through = 0;
    ft_tt_tensor_format format = encoding->format;
    int transposed = encoding->layout == FT_TT_LAYOUT_MATMUL_NMAJOR;
    uint64_t rows, cols;
    uint32_t tile_cols;
    pearson_accumulator acc;
    pearson_accumulator source_acc;
    uint64_t written = 0;
    int rc;

    tensor_shape(tensor, &rows, &cols);
    pearson_reset(&acc);
    pearson_reset(&source_acc);

    memset(report, 0, sizeof(*report));
    snprintf(report->name, sizeof(report->name), "%s", tensor->name);
    report->role = ft_tt_ledger_role(tensor->name);
    report->format = format;
    report->rows = rows;
    report->cols = cols;
    report->source_bytes = tensor->byte_count;
    report->pcc = 1.0;

    if (!ft_gguf_type_supported(tensor->ggml_type))
        return FT_TT_CONVERT_EUNSUPPORTED;

    if (tensor->n_dims < 2) {
        rc = write_untiled(out, tensor, cols, buffers, &written);
        report->converted_bytes = written;
        return rc;
    }

    tile_cols = ft_tt_tile_count_for((uint32_t)cols);

    for (uint64_t first = 0; first < rows; first += FT_TT_TILE_HEIGHT) {
        uint32_t valid = rows - first < FT_TT_TILE_HEIGHT ? (uint32_t)(rows - first) : FT_TT_TILE_HEIGHT;
        size_t bytes = 0;

        rc = load_source_strip(tensor, first, valid, cols, encoding->rope, encoding->window,
                               buffers->source_strip);
        if (rc != FT_TT_CONVERT_OK)
            return rc;

        rc = encode_strip(buffers, format, valid, cols, tile_cols, transposed, &bytes);
        if (rc != FT_TT_CONVERT_OK)
            return rc;

        if (out && fwrite(buffers->packed, 1, bytes, out) != bytes)
            return FT_TT_CONVERT_EIO;
        written += bytes;

        if (measure_pcc) {
            rc = decode_strip(buffers, format, valid, cols, tile_cols, transposed);
            if (rc != FT_TT_CONVERT_OK)
                return rc;
            rc = score_strip(buffers, reference, first, valid, cols, &acc, &source_acc);
            if (rc != FT_TT_CONVERT_OK)
                return rc;
        }
        if (!encoding->window && first + valid - released_through >= release_rows) {
            release_source_rows(map, tensor, cols, released_through, first + valid);
            released_through = first + valid;
        }
    }

    report->converted_bytes = written;
    if (measure_pcc) {
        report->pcc = pearson_value(&acc);
        report->max_abs_error = acc.max_abs_error;
        report->rms_error = pearson_rms(&acc);
        report->relative_rms_error = pearson_relative_rms(&acc);
        report->source_relative_rms_error = pearson_relative_rms(&source_acc);
    }

    return FT_TT_CONVERT_OK;
}

/* ------------------------------------------------------------------------- */
/* Artifact                                                                  */
/* ------------------------------------------------------------------------- */

static int write_padding(FILE *out, uint64_t bytes)
{
    static const unsigned char zeros[FT_TT_ARTIFACT_ALIGNMENT] = {0};

    while (bytes > 0) {
        size_t chunk = bytes < sizeof(zeros) ? (size_t)bytes : sizeof(zeros);

        if (fwrite(zeros, 1, chunk, out) != chunk)
            return FT_TT_CONVERT_EIO;
        bytes -= chunk;
    }
    return FT_TT_CONVERT_OK;
}

static uint64_t align_up(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

/* ------------------------------------------------------------------------- */
/* The rotary permutation plan                                               */
/* ------------------------------------------------------------------------- */

/**
 * The head geometry the whole model is permuted under, read once from metadata.
 *
 * `head_rows` is 0 when the metadata does not pin it, which is a refusal and not
 * a default: deriving it from a tensor's own row count would silently accept
 * `attn_q` tensors that carry more than one projection per head — Qwen3.8's
 * do — and permute rows that are not rotary features at all.
 */
typedef struct {
    int enabled;
    int reverse_pairs;
    int keep_interleaved;
    uint64_t n_head;
    uint64_t n_head_kv;
    uint64_t head_rows;
    uint64_t rot_rows;
} rope_plan;

static int metadata_u64_for_arch(const ft_gguf_map *map, const char *architecture, const char *suffix,
                                 uint64_t *out)
{
    char key[128];

    if (snprintf(key, sizeof(key), "%s.%s", architecture, suffix) >= (int)sizeof(key))
        return FT_TT_CONVERT_EINVAL;
    return ft_gguf_map_metadata_u64(map, key, out) == FT_GGUF_OK ? FT_TT_CONVERT_OK
                                                                 : FT_TT_CONVERT_EROPEGEOMETRY;
}

/**
 * Build the permutation plan, or refuse and say why.
 *
 * Every refusal here is a case where permuting would produce an artifact that is
 * wrong in a way nothing downstream can detect, so none of them fall back.
 */
static int rope_plan_build(const ft_gguf_map *map, const char *architecture, ft_tt_model_family family,
                           ft_tt_rope_pairing pairing, rope_plan *plan)
{
    memset(plan, 0, sizeof(*plan));
    if (pairing == FT_TT_ROPE_PAIRING_SOURCE)
        return FT_TT_CONVERT_OK;

    if (family == FT_TT_FAMILY_QWEN) {
        fprintf(stderr, "ft_tt_convert: --rope-pairing split-half refused: architecture '%s' already "
                        "stores q/k in split-half (NeoX) order, so permuting would BREAK it\n",
                architecture);
        return FT_TT_CONVERT_EROPEGEOMETRY;
    }
    if (metadata_u64_for_arch(map, architecture, "attention.head_count", &plan->n_head) != FT_TT_CONVERT_OK ||
        metadata_u64_for_arch(map, architecture, "attention.head_count_kv", &plan->n_head_kv) !=
            FT_TT_CONVERT_OK ||
        metadata_u64_for_arch(map, architecture, "attention.key_length", &plan->head_rows) !=
            FT_TT_CONVERT_OK ||
        metadata_u64_for_arch(map, architecture, "rope.dimension_count", &plan->rot_rows) !=
            FT_TT_CONVERT_OK) {
        fprintf(stderr, "ft_tt_convert: --rope-pairing split-half refused: '%s' does not declare all of "
                        "attention.head_count, attention.head_count_kv, attention.key_length and "
                        "rope.dimension_count\n",
                architecture);
        return FT_TT_CONVERT_EROPEGEOMETRY;
    }
    if (plan->n_head == 0u || plan->n_head_kv == 0u || plan->head_rows == 0u || plan->rot_rows == 0u ||
        plan->rot_rows > plan->head_rows || (plan->rot_rows % 2u) != 0u) {
        fprintf(stderr, "ft_tt_convert: --rope-pairing split-half refused: head geometry heads=%llu kv=%llu "
                        "head_dim=%llu rope_dim=%llu is not a pairing this permutation expresses\n",
                (unsigned long long)plan->n_head, (unsigned long long)plan->n_head_kv,
                (unsigned long long)plan->head_rows, (unsigned long long)plan->rot_rows);
        return FT_TT_CONVERT_EROPEGEOMETRY;
    }

    /* The reversal reflects `p -> half-1-p`, so it needs `half` to exist as an integer.
     * `rot_rows % 2 == 0` is already checked above, which is the whole requirement. */
    plan->enabled = 1;
    plan->reverse_pairs = (pairing == FT_TT_ROPE_PAIRING_SPLIT_HALF_REVERSED ||
                           pairing == FT_TT_ROPE_PAIRING_INTERLEAVED_REVERSED);
    plan->keep_interleaved = (pairing == FT_TT_ROPE_PAIRING_INTERLEAVED_REVERSED);
    if (plan->reverse_pairs)
        fprintf(stderr, "ft_tt_convert: --rope-pairing %s is an INSTRUMENT, not a checkpoint "
                        "format: slot p carries the model's rotary pair %llu-1-p, so a consumer "
                        "that does not read FT_TT_ARTIFACT_FLAG_ROPE_PAIRS_REVERSED will apply "
                        "every rotary frequency to the wrong feature\n",
                plan->keep_interleaved ? "interleaved-reversed" : "split-half-reversed",
                (unsigned long long)(plan->rot_rows / 2u));
    return FT_TT_CONVERT_OK;
}

/** Whether `name` is one of the two tensors whose output rows carry rotary features. */
static int name_is_rotary_projection(const char *name)
{
    /* `attn_q_norm`/`attn_k_norm` are ROLE_NORM and never reach here; `attn_qkv`
     * is a GDN in-projection and is ROLE_SSM_IN. So matching the stem is enough
     * once the caller has already established the role is ATTENTION. */
    return strstr(name, "attn_q") != NULL || strstr(name, "attn_k") != NULL;
}

/**
 * The permutation for one tensor, or 0 when this tensor is not permuted.
 *
 * @return 1 and fills `out` when the tensor is permuted; 0 when it is not;
 *         FT_TT_CONVERT_EROPEGEOMETRY when it should be but its shape disagrees
 *         with the declared head geometry
 */
static int rope_plan_for_tensor(const rope_plan *plan, const char *name, ft_tt_tensor_role role, uint64_t rows,
                                rope_row_permutation *out)
{
    uint64_t heads;

    if (!plan->enabled || role != FT_TT_ROLE_ATTENTION || !name_is_rotary_projection(name))
        return 0;

    heads = strstr(name, "attn_q") != NULL ? plan->n_head : plan->n_head_kv;
    if (rows != heads * plan->head_rows) {
        fprintf(stderr, "ft_tt_convert: '%s' has %llu output rows but the declared geometry says %llu heads "
                        "x %llu — refusing to permute rows whose meaning is not established\n",
                name, (unsigned long long)rows, (unsigned long long)heads,
                (unsigned long long)plan->head_rows);
        return FT_TT_CONVERT_EROPEGEOMETRY;
    }
    out->head_rows = plan->head_rows;
    out->rot_rows = plan->rot_rows;
    out->reverse_pairs = plan->reverse_pairs;
    out->keep_interleaved = plan->keep_interleaved;
    return 1;
}

/**
 * Decide format, layout and rotary permutation for one tensor.
 *
 * The refusals are the point of the function: an unclassified 2-D tensor and a
 * shape that contradicts the declared head geometry both stop the conversion
 * rather than producing an artifact whose bytes mean something other than what
 * the directory says.
 */
static int tensor_encoding_for(const ft_gguf_tensor *tensor, const char *source_name,
                               const ft_tt_convert_options *opts, ft_tt_model_family family, const rope_plan *plan,
                               uint64_t rows, ft_tt_tensor_role *role_out, rope_row_permutation *rope_out,
                               tensor_encoding *encoding)
{
    ft_tt_tensor_role role = ft_tt_ledger_role(tensor->name);
    const ft_tt_format_map_entry *mapped = ft_tt_format_map_find(opts->format_map, source_name);
    int rc;

    *role_out = role;
    encoding->format = ft_tt_ledger_format_for(opts->mode, family, role, tensor->n_dims);
    /* The experiment override, applied AFTER the ledger so the ledger remains the one
     * description of what production is. 1-D tensors keep the fp32 every rung gives them:
     * overriding those would measure a norm, not a matmul. */
    if (opts->format_override_name_contains && tensor->n_dims >= 2 && role != FT_TT_ROLE_NORM &&
        role != FT_TT_ROLE_BIAS && strstr(tensor->name, opts->format_override_name_contains)) {
        fprintf(stderr, "ft_tt_convert: OVERRIDE '%s' %s -> %s (matched '%s')\n", tensor->name,
                ft_tt_ledger_format_name(encoding->format),
                ft_tt_ledger_format_name(opts->format_override),
                opts->format_override_name_contains);
        encoding->format = opts->format_override;
    }
    /* The map, also AFTER the ledger, by SOURCE name; format_map_checked() has already
     * refused every entry that does not name a mappable 2-D matmul weight. */
    if (mapped) {
        if (mapped->format != encoding->format)
            fprintf(stderr, "ft_tt_convert: MAP '%s' %s -> %s (%s:%u)\n", tensor->name,
                    ft_tt_ledger_format_name(encoding->format), ft_tt_ledger_format_name(mapped->format),
                    opts->format_map->source, mapped->line);
        encoding->format = mapped->format;
    }
    encoding->rope = NULL;
    encoding->window = NULL;

    rc = ft_tt_convert_layout_for_checked(role, tensor->n_dims, &encoding->layout);
    if (rc != FT_TT_CONVERT_OK) {
        fprintf(stderr, "ft_tt_convert: '%s' is %u-D and matches no role rule, so neither layout can be "
                        "chosen for it — both would be a guess about a kernel, and a wrong one is "
                        "silent. Add a rule to ft_tt_ledger_role().\n",
                tensor->name, tensor->n_dims);
        return rc;
    }

    rc = rope_plan_for_tensor(plan, tensor->name, role, rows, rope_out);
    if (rc < 0)
        return rc;
    if (rc == 1)
        encoding->rope = rope_out;
    return FT_TT_CONVERT_OK;
}

const ft_gguf_tensor *ft_tt_convert_tied_head_source(const ft_gguf_map *map)
{
    const ft_gguf_tensor *embedding = NULL;
    uint32_t count = ft_gguf_map_tensor_count(map);

    for (uint32_t i = 0; i < count; i++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_tensor(map, i);

        if (!tensor)
            continue;
        if (strcmp(tensor->name, "output.weight") == 0)
            return NULL;
        if (strcmp(tensor->name, "token_embd.weight") == 0 && tensor->n_dims == 2)
            embedding = tensor;
    }
    return embedding;
}

/**
 * Fill a directory entry's STORED tile grid. A transposed weight is Kt x Nt =
 * ceil(cols/32) x ceil(rows/32), the swap of the source grid; an expert stack's grid is ONE
 * expert's (rows = dims[1], not dims[1] * dims[2]) — format v3.
 */
static void entry_tile_grid(const ft_gguf_tensor *tensor, ft_tt_tensor_layout layout, uint64_t rows,
                            uint64_t cols, ft_tt_artifact_entry *entry)
{
    if (ft_tt_convert_is_expert_stack(tensor))
        rows = tensor->dims[1];
    if (tensor->n_dims < 2) {
        entry->tile_rows = 0;
        entry->tile_cols = 0;
    } else if (layout == FT_TT_LAYOUT_MATMUL_NMAJOR) {
        entry->tile_rows = ft_tt_tile_count_for((uint32_t)cols);
        entry->tile_cols = ft_tt_tile_count_for((uint32_t)rows);
    } else {
        entry->tile_rows = ft_tt_tile_count_for((uint32_t)rows);
        entry->tile_cols = ft_tt_tile_count_for((uint32_t)cols);
    }
}

/**
 * Convert one tensor into the artifact at `*cursor` and fill its directory entry. The
 * caller owns `tensor` and releases its pages; `*cursor` advances past the padded bytes.
 * `tensor->name` is the name WRITTEN (the MTP scope passes a renamed view) and
 * `source_name` the GGUF name a format map is matched against; `window` is NULL except for
 * one half of the MTP eh_proj split.
 */
static int convert_entry(FILE *out, const ft_gguf_map *map, const ft_gguf_tensor *tensor, const char *source_name,
                         const source_window *window, const ft_tt_convert_options *opts,
                         ft_tt_model_family family, const rope_plan *plan, strip_buffers *buffers,
                         ft_tt_artifact_entry *entry, uint64_t *cursor)
{
    ft_tt_convert_tensor_report report;
    ft_tt_tensor_role role;
    rope_row_permutation permutation;
    tensor_encoding encoding;
    uint64_t padded;
    uint64_t rows, cols;
    int rc;

    if (ft_tt_convert_is_expert_stack(tensor)) {
        rc = expert_stack_geometry_checked(tensor);
        if (rc != FT_TT_CONVERT_OK)
            return rc;
    }
    tensor_shape(tensor, &rows, &cols);
    rc = tensor_encoding_for(tensor, source_name, opts, family, plan, rows, &role, &permutation, &encoding);
    if (rc != FT_TT_CONVERT_OK)
        return rc;
    encoding.window = window;

    rc = convert_tensor(out, map, tensor, NULL, &encoding, buffers, opts->measure_pcc, &report);
    if (rc != FT_TT_CONVERT_OK) {
        fprintf(stderr, "ft_tt_convert: tensor '%s' (%s) failed: %d\n", tensor->name,
                ft_gguf_type_name(tensor->ggml_type), rc);
        return rc;
    }

    snprintf(entry->name, sizeof(entry->name), "%s", tensor->name);
    entry->format = (uint32_t)encoding.format;
    entry->role = (uint32_t)role;
    entry->n_dims = tensor->n_dims;
    entry->layout = (uint32_t)encoding.layout;
    entry_tile_grid(tensor, encoding.layout, rows, cols, entry);
    for (int d = 0; d < 4; d++)
        entry->dims[d] = tensor->dims[d];
    entry->data_offset = *cursor;
    entry->byte_count = report.converted_bytes;

    *cursor += report.converted_bytes;
    padded = align_up(*cursor, FT_TT_ARTIFACT_ALIGNMENT);
    rc = write_padding(out, padded - *cursor);
    if (rc != FT_TT_CONVERT_OK)
        return rc;
    *cursor = padded;

    if (opts->progress && opts->progress(&report, opts->progress_user) != 0)
        return FT_TT_CONVERT_ERROR;
    return FT_TT_CONVERT_OK;
}

/* ------------------------------------------------------------------------- */
/* Scope: the trunk, or the MTP block as its own artifact                    */
/* ------------------------------------------------------------------------- */

int ft_tt_convert_mtp_block(const ft_gguf_map *map, uint32_t *block)
{
    char architecture[64] = "";
    char key[128];
    uint64_t block_count = 0;

    if (!map || !block)
        return 0;
    (void)ft_gguf_map_metadata_string(map, "general.architecture", architecture, sizeof(architecture));
    if (snprintf(key, sizeof(key), "%s.block_count", architecture) >= (int)sizeof(key) ||
        ft_gguf_map_metadata_u64(map, key, &block_count) != FT_GGUF_OK || block_count == 0u ||
        block_count > UINT32_MAX)
        return 0;
    snprintf(key, sizeof(key), "blk.%llu.nextn.eh_proj.weight", (unsigned long long)(block_count - 1u));
    if (!ft_gguf_map_find(map, key))
        return 0;
    *block = (uint32_t)(block_count - 1u);
    return 1;
}

/** Whether `name` belongs to block `block` ("blk.<block>." prefix); `*rest` is what follows. */
static int name_in_block(const char *name, uint32_t block, const char **rest)
{
    char prefix[32];
    size_t length;

    snprintf(prefix, sizeof(prefix), "blk.%u.", block);
    length = strlen(prefix);
    if (strncmp(name, prefix, length) != 0)
        return 0;
    *rest = name + length;
    return 1;
}

/**
 * The name a tensor is written under in an MTP artifact: `blk.L.X` -> `mtp.X`, with a
 * leading `nextn.` dropped from X (`blk.40.nextn.enorm.weight` -> `mtp.enorm.weight`).
 */
static void mtp_target_name(const char *rest, char *out, size_t out_bytes)
{
    if (strncmp(rest, "nextn.", 6) == 0)
        rest += 6;
    snprintf(out, out_bytes, "mtp.%s", rest);
}

/** Whether `tensor` is the MTP block's eh_proj, the one tensor the MTP scope splits in two. */
static int is_mtp_eh_proj(const char *rest)
{
    return strcmp(rest, "nextn.eh_proj.weight") == 0;
}

/**
 * Convert the MTP eh_proj [2n -> n] as two entries, `mtp.eh_proj_e` (input columns [0, n))
 * and `mtp.eh_proj_h` ([n, 2n)), each read through a column window of the source rows — byte
 * slices, so a block-quantized source is never requantized. Refused unless n is a whole
 * number of source blocks.
 */
static int convert_mtp_eh_proj(FILE *out, const ft_gguf_map *map, const ft_gguf_tensor *tensor,
                               const ft_tt_convert_options *opts, ft_tt_model_family family,
                               const rope_plan *plan, strip_buffers *buffers, ft_tt_artifact_entry *entries,
                               uint64_t *cursor)
{
    static const char *const names[2] = {"mtp.eh_proj_e.weight", "mtp.eh_proj_h.weight"};
    uint64_t half = tensor->dims[0] / 2u;
    size_t half_bytes = ft_gguf_type_bytes(tensor->ggml_type, half);
    size_t row_bytes = ft_gguf_type_bytes(tensor->ggml_type, tensor->dims[0]);

    if (tensor->n_dims != 2 || tensor->dims[0] % 2u != 0u || half_bytes == 0u || half_bytes * 2u != row_bytes) {
        fprintf(stderr, "ft_tt_convert: '%s' [%llu, %llu] does not split into two whole-block input halves\n",
                tensor->name, (unsigned long long)tensor->dims[0], (unsigned long long)tensor->dims[1]);
        return FT_TT_CONVERT_EINVAL;
    }
    for (int part = 0; part < 2; part++) {
        ft_gguf_tensor view = *tensor;
        source_window window;
        int rc;

        snprintf(view.name, sizeof(view.name), "%s", names[part]);
        view.dims[0] = half;
        window.row_bytes = row_bytes;
        window.byte_offset = part ? half_bytes : 0u;
        rc = convert_entry(out, map, &view, tensor->name, &window, opts, family, plan, buffers, &entries[part],
                           cursor);
        if (rc != FT_TT_CONVERT_OK)
            return rc;
    }
    return FT_TT_CONVERT_OK;
}

/**
 * How one source tensor is written under a scope: 0 = skipped, 1 = one entry (possibly
 * renamed into `target`), 2 = the MTP eh_proj split.
 */
static int scope_disposition(const ft_gguf_tensor *tensor, ft_tt_convert_scope scope, int has_mtp,
                             uint32_t mtp_block, char *target, size_t target_bytes)
{
    const char *rest = NULL;
    int in_mtp = has_mtp && name_in_block(tensor->name, mtp_block, &rest);

    snprintf(target, target_bytes, "%s", tensor->name);
    if (scope == FT_TT_CONVERT_SCOPE_ALL)
        return 1;
    if (scope == FT_TT_CONVERT_SCOPE_TRUNK)
        return in_mtp ? 0 : 1;
    if (!in_mtp)
        return 0;
    if (is_mtp_eh_proj(rest))
        return 2;
    mtp_target_name(rest, target, target_bytes);
    return 1;
}

/**
 * Check a format map against the checkpoint before anything is written: every entry must
 * name a tensor of `map` that ft_tt_ledger_map_refusal() accepts, and the map cannot be
 * combined with the substring override. Entries outside the scope are legal and counted.
 */
static int format_map_checked(const ft_gguf_map *map, const ft_tt_convert_options *opts)
{
    const ft_tt_format_map *formats = opts->format_map;
    char target[FT_GGUF_MAX_NAME];
    uint32_t mtp_block = 0, outside = 0;
    int has_mtp = ft_tt_convert_mtp_block(map, &mtp_block);

    if (!formats)
        return FT_TT_CONVERT_OK;
    if (opts->format_override_name_contains) {
        fprintf(stderr, "ft_tt_convert: a format map and a format override cannot be combined\n");
        return FT_TT_CONVERT_EINVAL;
    }
    for (uint32_t i = 0; i < formats->count; i++) {
        const ft_tt_format_map_entry *entry = &formats->entries[i];
        const ft_gguf_tensor *tensor = ft_gguf_map_find(map, entry->name);
        const char *refusal;

        if (!tensor) {
            fprintf(stderr, "ft_tt_convert: format map %s:%u names '%s', which this checkpoint does not have\n",
                    formats->source, entry->line, entry->name);
            return FT_TT_CONVERT_EFORMATMAP;
        }
        refusal = ft_tt_ledger_map_refusal(ft_tt_ledger_role(entry->name), tensor->n_dims);
        if (refusal) {
            fprintf(stderr, "ft_tt_convert: format map %s:%u names '%s', which a map may not move: %s\n",
                    formats->source, entry->line, entry->name, refusal);
            return FT_TT_CONVERT_EFORMATMAP;
        }
        if (scope_disposition(tensor, opts->scope, has_mtp, mtp_block, target, sizeof(target)) == 0)
            outside++;
    }
    if (outside)
        fprintf(stderr, "ft_tt_convert: format map %s: %u of %u entries are outside this scope and not applied\n",
                formats->source, outside, formats->count);
    return FT_TT_CONVERT_OK;
}

static const ft_tt_convert_options *options_or_defaults(const ft_tt_convert_options *options)
{
    static const ft_tt_convert_options defaults = {FT_TT_LEDGER_PRODUCTION, 0, NULL, NULL,
                                                   FT_TT_ROPE_PAIRING_SOURCE, NULL,
                                                   FT_TT_FORMAT_FP32, FT_TT_CONVERT_SCOPE_TRUNK, NULL};
    return options ? options : &defaults;
}

uint32_t ft_tt_convert_tensor_count_for(const ft_gguf_map *map, const ft_tt_convert_options *options)
{
    const ft_tt_convert_options *opts = options_or_defaults(options);
    char target[FT_GGUF_MAX_NAME];
    uint32_t mtp_block = 0;
    int has_mtp = ft_tt_convert_mtp_block(map, &mtp_block);
    uint32_t total = 0;

    if (!map || (opts->scope == FT_TT_CONVERT_SCOPE_MTP && !has_mtp))
        return 0;
    for (uint32_t i = 0; i < ft_gguf_map_tensor_count(map); i++)
        total += (uint32_t)scope_disposition(ft_gguf_map_tensor(map, i), opts->scope, has_mtp, mtp_block, target,
                                             sizeof(target));
    if (opts->scope != FT_TT_CONVERT_SCOPE_MTP && ft_tt_convert_tied_head_source(map))
        total++;
    return total;
}

uint32_t ft_tt_convert_artifact_version_for(const ft_gguf_map *map, const ft_tt_convert_options *options)
{
    const ft_tt_convert_options *opts = options_or_defaults(options);
    char architecture[64] = "";
    char target[FT_GGUF_MAX_NAME];
    ft_tt_model_family family;
    uint32_t mtp_block = 0;
    int has_mtp = ft_tt_convert_mtp_block(map, &mtp_block);

    if (!map)
        return FT_TT_ARTIFACT_VERSION_DENSE;
    (void)ft_gguf_map_metadata_string(map, "general.architecture", architecture, sizeof(architecture));
    family = ft_tt_ledger_family_from_architecture(architecture);
    for (uint32_t i = 0; i < ft_gguf_map_tensor_count(map); i++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_tensor(map, i);
        ft_tt_tensor_format format;

        if (scope_disposition(tensor, opts->scope, has_mtp, mtp_block, target, sizeof(target)) == 0)
            continue;
        format = ft_tt_ledger_format_for(opts->mode, family, ft_tt_ledger_role(target), tensor->n_dims);
        if (ft_tt_convert_is_expert_stack(tensor) || format == FT_TT_FORMAT_BFP6_B)
            return FT_TT_ARTIFACT_VERSION;
    }
    return FT_TT_ARTIFACT_VERSION_DENSE;
}

/** Convert every in-scope tensor, in GGUF order, filling `directory` from `*filled`. */
static int convert_scoped_tensors(FILE *out, ft_gguf_map *map, const ft_tt_convert_options *opts,
                                  ft_tt_model_family family, const rope_plan *plan, strip_buffers *buffers,
                                  ft_tt_artifact_entry *directory, uint32_t *filled, uint64_t *cursor)
{
    uint32_t mtp_block = 0;
    int has_mtp = ft_tt_convert_mtp_block(map, &mtp_block);

    for (uint32_t i = 0; i < ft_gguf_map_tensor_count(map); i++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_tensor(map, i);
        ft_gguf_tensor view = *tensor;
        int how = scope_disposition(tensor, opts->scope, has_mtp, mtp_block, view.name, sizeof(view.name));
        int rc = FT_TT_CONVERT_OK;

        if (how == 1)
            rc = convert_entry(out, map, &view, tensor->name, NULL, opts, family, plan, buffers, &directory[*filled],
                               cursor);
        else if (how == 2)
            rc = convert_mtp_eh_proj(out, map, tensor, opts, family, plan, buffers, &directory[*filled], cursor);
        ft_gguf_map_release(map, tensor);
        if (rc != FT_TT_CONVERT_OK)
            return rc;
        *filled += (uint32_t)how;
    }
    return FT_TT_CONVERT_OK;
}

int ft_tt_convert_gguf(const char *gguf_path, const char *output_path, const ft_tt_convert_options *options)
{
    const ft_tt_convert_options *opts = options_or_defaults(options);
    ft_gguf_map *map = NULL;
    FILE *out = NULL;
    strip_buffers buffers;
    ft_tt_artifact_entry *directory = NULL;
    ft_tt_artifact_header header;
    ft_tt_model_family family;
    rope_plan plan;
    char architecture[64] = "";
    const ft_gguf_tensor *tied;
    uint64_t cursor;
    uint32_t total, filled = 0;
    int rc;

    memset(&buffers, 0, sizeof(buffers));

    if (!gguf_path || !output_path)
        return FT_TT_CONVERT_EINVAL;

    rc = ft_gguf_map_open(gguf_path, &map);
    if (rc != FT_GGUF_OK)
        return FT_TT_CONVERT_ERROR;

    tied = opts->scope == FT_TT_CONVERT_SCOPE_MTP ? NULL : ft_tt_convert_tied_head_source(map);
    total = ft_tt_convert_tensor_count_for(map, opts);
    if (total == 0u) {
        fprintf(stderr, "ft_tt_convert: '%s' has no MTP / NextN block (blk.L.nextn.eh_proj.weight in "
                        "its last block), so there is no MTP artifact to write\n", gguf_path);
        rc = FT_TT_CONVERT_ENOMTP;
        goto cleanup;
    }
    (void)ft_gguf_map_metadata_string(map, "general.architecture", architecture, sizeof(architecture));
    family = ft_tt_ledger_family_from_architecture(architecture);

    /* Before the output file is created, so a refused geometry or map leaves no
     * half-written artifact behind to be mistaken for a real one. */
    rc = rope_plan_build(map, architecture, family, opts->rope_pairing, &plan);
    if (rc == FT_TT_CONVERT_OK)
        rc = format_map_checked(map, opts);
    if (rc != FT_TT_CONVERT_OK)
        goto cleanup;

    rc = strip_buffers_alloc(&buffers, map, opts->measure_pcc);
    if (rc != FT_TT_CONVERT_OK)
        goto cleanup;

    directory = calloc(total, sizeof(*directory));
    if (!directory) {
        rc = FT_TT_CONVERT_ENOMEM;
        goto cleanup;
    }

    out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "ft_tt_convert: fopen('%s'): %s\n", output_path, strerror(errno));
        rc = FT_TT_CONVERT_EIO;
        goto cleanup;
    }

    memset(&header, 0, sizeof(header));
    header.magic = FT_TT_ARTIFACT_MAGIC;
    header.version = ft_tt_convert_artifact_version_for(map, opts);
    header.tensor_count = total;
    /* The two bits are independent: SPLIT_HALF says WHERE a partner is, PAIRS_REVERSED says
     * WHICH pair a slot holds. A reader that knows only the first must refuse the second
     * rather than serve it as plain split-half, which is what FT_TT_ARTIFACT_FLAGS_KNOWN
     * and ft_tt_fttw_open()'s refusal enforce. */
    header.flags = (plan.enabled && !plan.keep_interleaved) ? FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF : 0u;
    if (plan.reverse_pairs)
        header.flags |= FT_TT_ARTIFACT_FLAG_ROPE_PAIRS_REVERSED;
    header.data_offset = align_up(sizeof(header), FT_TT_ARTIFACT_ALIGNMENT);

    if (fwrite(&header, sizeof(header), 1, out) != 1) {
        rc = FT_TT_CONVERT_EIO;
        goto cleanup;
    }
    rc = write_padding(out, header.data_offset - sizeof(header));
    if (rc != FT_TT_CONVERT_OK)
        goto cleanup;

    cursor = header.data_offset;
    rc = convert_scoped_tensors(out, map, opts, family, &plan, &buffers, directory, &filled, &cursor);
    if (rc != FT_TT_CONVERT_OK)
        goto cleanup;
    if (tied) {
        ft_gguf_tensor head = *tied;

        snprintf(head.name, sizeof(head.name), "%s", "output.weight");
        rc = convert_entry(out, map, &head, tied->name, NULL, opts, family, &plan, &buffers, &directory[filled],
                           &cursor);
        ft_gguf_map_release(map, tied);
        if (rc != FT_TT_CONVERT_OK)
            goto cleanup;
    }

    header.directory_offset = cursor;
    if (fwrite(directory, sizeof(*directory), total, out) != total) {
        rc = FT_TT_CONVERT_EIO;
        goto cleanup;
    }
    header.total_bytes = cursor + (uint64_t)total * sizeof(*directory);

    /* Rewrite the header now that the directory is placed. This is the only
     * seek in the whole path. */
    if (fseek(out, 0, SEEK_SET) != 0 || fwrite(&header, sizeof(header), 1, out) != 1) {
        rc = FT_TT_CONVERT_EIO;
        goto cleanup;
    }

    rc = FT_TT_CONVERT_OK;

cleanup:
    if (out && fclose(out) != 0 && rc == FT_TT_CONVERT_OK)
        rc = FT_TT_CONVERT_EIO;
    free(directory);
    strip_buffers_free(&buffers);
    ft_gguf_map_close(map);
    return rc;
}

/**
 * The reference tensor matching `tensor`, checked for shape agreement.
 *
 * A missing or differently-shaped tensor is refused rather than skipped: a
 * measurement that silently covered 90% of a model is how a regression hides.
 */
static const ft_gguf_tensor *matching_reference(const ft_gguf_map *reference, const ft_gguf_tensor *tensor)
{
    const ft_gguf_tensor *found = ft_gguf_map_find(reference, tensor->name);

    if (!found) {
        fprintf(stderr, "ft_tt_convert_measure: the reference has no tensor '%s'\n", tensor->name);
        return NULL;
    }
    if (found->n_dims != tensor->n_dims || memcmp(found->dims, tensor->dims, sizeof(found->dims)) != 0) {
        fprintf(stderr, "ft_tt_convert_measure: '%s' is %llux%llu here and %llux%llu in the reference\n", tensor->name,
                (unsigned long long)tensor->dims[1], (unsigned long long)tensor->dims[0],
                (unsigned long long)found->dims[1], (unsigned long long)found->dims[0]);
        return NULL;
    }
    return found;
}

int ft_tt_convert_measure_against(const char *gguf_path, const char *reference_gguf_path,
                                  ft_tt_tensor_format format, ft_tt_convert_progress_fn progress, void *user)
{
    ft_gguf_map *map = NULL;
    ft_gguf_map *reference = NULL;
    strip_buffers buffers;
    int rc;

    memset(&buffers, 0, sizeof(buffers));

    if (!gguf_path || !progress)
        return FT_TT_CONVERT_EINVAL;

    rc = ft_gguf_map_open(gguf_path, &map);
    if (rc != FT_GGUF_OK)
        return FT_TT_CONVERT_ERROR;
    if (reference_gguf_path && ft_gguf_map_open(reference_gguf_path, &reference) != FT_GGUF_OK) {
        ft_gguf_map_close(map);
        return FT_TT_CONVERT_ERROR;
    }

    rc = strip_buffers_alloc(&buffers, map, 1);
    if (rc == FT_TT_CONVERT_OK && reference)
        rc = strip_buffers_add_reference(&buffers);
    if (rc != FT_TT_CONVERT_OK)
        goto cleanup;

    for (uint32_t i = 0; i < ft_gguf_map_tensor_count(map); i++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_tensor(map, i);
        const ft_gguf_tensor *truth = NULL;
        ft_tt_convert_tensor_report report;

        /* Score every tensor the caller could actually place in this format.
         * 1-D tensors are fp32 on every rung, so scoring them at BFP would
         * report a number no configuration can produce. */
        tensor_encoding encoding;

        encoding.format = tensor->n_dims < 2 ? FT_TT_FORMAT_FP32 : format;
        encoding.rope = NULL;
        /* The gate scores the format AS SHIPPED, so it must use the shipped layout: the
         * transpose changes which 16 elements share a block-float exponent, and a gate
         * that measured the other grouping would be scoring a tensor nobody stores.
         * An unclassified tensor has no shipped layout, so it is refused here too — a
         * gate that quietly scored the wrong grouping is worse than one that stops. */
        rc = ft_tt_convert_layout_for_checked(ft_tt_ledger_role(tensor->name), tensor->n_dims,
                                              &encoding.layout);
        if (rc != FT_TT_CONVERT_OK) {
            fprintf(stderr, "ft_tt_convert_measure: '%s' matches no role rule, so it has no shipped "
                            "layout to score\n",
                    tensor->name);
            goto cleanup;
        }

        if (reference) {
            truth = matching_reference(reference, tensor);
            if (!truth) {
                rc = FT_TT_CONVERT_EINVAL;
                goto cleanup;
            }
        }

        encoding.window = NULL;
        rc = convert_tensor(NULL, map, tensor, truth, &encoding, &buffers, 1, &report);
        if (rc != FT_TT_CONVERT_OK) {
            fprintf(stderr, "ft_tt_convert_measure: tensor '%s' failed: %d\n", tensor->name, rc);
            goto cleanup;
        }

        ft_gguf_map_release(map, tensor);
        if (truth)
            ft_gguf_map_release(reference, truth);

        if (progress(&report, user) != 0) {
            rc = FT_TT_CONVERT_ERROR;
            goto cleanup;
        }
    }

    rc = FT_TT_CONVERT_OK;

cleanup:
    strip_buffers_free(&buffers);
    ft_gguf_map_close(reference);
    ft_gguf_map_close(map);
    return rc;
}

int ft_tt_convert_measure(const char *gguf_path, ft_tt_tensor_format format, ft_tt_convert_progress_fn progress,
                          void *user)
{
    return ft_tt_convert_measure_against(gguf_path, NULL, format, progress, user);
}
