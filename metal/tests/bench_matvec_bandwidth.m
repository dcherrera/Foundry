/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file bench_matvec_bandwidth.m
 * @brief Effective weight-read bandwidth of the four K-quant matvec kernels
 *   (Q3_K, Q4_K, Q5_K, Q6_K) across the shapes the Nex-N2-mini GGUF actually
 *   dispatches, so Phase 1.5's lane-starvation hypothesis is decided by a
 *   sweep rather than by one microbench shape.
 *
 *   Wall-clock is not comparable across quants — each reads a different number
 *   of bytes for the same shape — so every arm reports GB/s computed from that
 *   quant's real block size, alongside the lane occupancy (units_per_row / 32)
 *   the hypothesis says drives it.
 *
 *   The shape list is not hand-written: the GGUF tensor table is parsed and the
 *   distinct (cols, rows) of every quantized weight that decode dispatches is
 *   used, plus row-scaled variants that force the small ones out of cache.
 *
 *   Rigour guards, in the order they run:
 *     - Every arm streams distinct weight bytes (the bind offset walks the
 *       buffer between repetitions), so a repeated arm is not measuring cache.
 *       Explicit cache-resident twins for the small real shapes make the
 *       DRAM-vs-issue question visible instead of assumed.
 *     - Elision defence: the output is read back and must change when the
 *       activation changes, when the LAST super-block of row 0 changes, and
 *       when the last row's bytes change — so no prefix of the row and no
 *       suffix of the grid can have been skipped.
 *     - `report_elision_guard_can_fail` mutates the weights and then does NOT
 *       dispatch; the same comparator must report "unchanged". A green that was
 *       never shown to fail is not a gate.
 *     - Replica arms give the noise floor on the reported statistic; nothing
 *       smaller than it is interpreted.
 *     - Arms are interleaved and rotated between rounds, and the pooled MINIMUM
 *       is reported — this machine shares its GPU with the window server, so
 *       the mean tracks desktop contention rather than kernel cost. p50/p95 are
 *       printed beside it to keep the contention band visible.
 *
 *   Results and verdict: metal/tests/RESULTS_matvec_bandwidth.md
 *
 * @status Complete and run; verdict recorded in RESULTS_matvec_bandwidth.md.
 * @issues None
 * @todo
 *   - [ ] Add the `bench-matvec-bandwidth` Makefile stanza. It is not added
 *         here because concurrent work owns the Makefile; the exact stanza and
 *         the standalone compile line are in RESULTS_matvec_bandwidth.md.
 */

#import <Foundation/Foundation.h>

#include "metal/fm_metal.h"
#include "metal/fm_metal_internal.h"
#include "metal/tests/gguf_test_reader.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================
 * Configuration
 * ================================================================ */

#define DEFAULT_MODEL_PATH \
    "models/Nex-N2-mini/nex-agi_Nex-N2-mini-Q3_K_M.gguf"

/* Every matvec here runs one SIMD group per output row. */
#define SIMD_LANES_PER_THREADGROUP   32

/* ggml type ids for the quants this benchmark covers. */
#define GGML_TYPE_Q3_K   11
#define GGML_TYPE_Q4_K   12
#define GGML_TYPE_Q5_K   13
#define GGML_TYPE_Q6_K   14
#define GGML_TYPE_Q8_0    8

/* Sampling. Rounds are interleaved and rotated; all samples pool. */
#define MEASUREMENT_ROUND_COUNT      5
#define SAMPLES_PER_ROUND            10
#define WARMUP_SAMPLES_PER_ARM       2
#define MAX_POOLED_SAMPLE_COUNT      (MEASUREMENT_ROUND_COUNT * SAMPLES_PER_ROUND)

/* Coprime with the arm count, so each arm follows a different predecessor in
 * every round instead of one arm always inheriting the heaviest one's wake. */
#define ARM_ORDER_ROTATION_STRIDE    7

/* Weight bytes one timed sample should move, by repeating the dispatch over
 * successive slabs of the weight buffer. Large enough that per-commit overhead
 * is a rounding error, small enough that a full sweep stays under a minute. */
#define TARGET_BYTES_PER_SAMPLE      (64u * 1024u * 1024u)
#define MAX_REPETITIONS_PER_SAMPLE   256

/* Row counts used by the row-scaled arms, chosen so the smallest quant (Q3_K)
 * still spans >= 128 MB and cannot sit in any level of cache. */
#define SCALED_ROWS_AT_512_COLS      589824
#define SCALED_ROWS_AT_4096_COLS      73728

/* Footprint under which an arm's weights plausibly live in cache. */
#define CACHE_RESIDENT_FOOTPRINT_LIMIT  (16u * 1024u * 1024u)

#define MAX_CENSUS_ENTRIES   32
#define MAX_SHAPES           16
#define MAX_ARMS            128

/* ================================================================
 * Types
 * ================================================================ */

typedef void (*matvec_dispatch_fn)(fm_bufref w, fm_bufref x, fm_bufref out,
                                   int rows, int cols, int row_stride_bytes);

/* One quant's kernel plus the buffer of real blocks it reads. */
typedef struct {
    const char        *name;
    uint32_t           ggml_type;
    int                block_bytes;         /* bytes per 256-weight super-block */
    int                lane_unit_weights;   /* weights one lane owns per step  */
    matvec_dispatch_fn dispatch;
    fm_buffer         *weights;
    size_t             weight_bytes;
    char               source_tensor[GGUF_MAX_TENSOR_NAME];
} quant_matvec_kernel;

/* A weight shape a matvec is asked to run. */
typedef struct {
    int         cols;
    int         rows;
    const char *origin;          /* where in the model this shape comes from */
    bool        from_model;      /* false for the row-scaled residency arms  */
} matvec_shape;

typedef struct {
    double mean_ms;
    double min_ms;
    double median_ms;
    double p95_ms;
} timing_statistics;

/* One measured configuration: a kernel at a shape, cached or streaming. */
typedef struct {
    quant_matvec_kernel *kernel;
    matvec_shape         shape;
    bool                 stream_distinct_slabs;
    bool                 is_noise_replica;
    int                  repetitions;
    size_t               slab_stride_bytes;
    int                  slab_count;
    double               pooled_samples[MAX_POOLED_SAMPLE_COUNT];
    int                  pooled_count;
} measurement_arm;

/* Distinct (type, cols, rows) seen in the model's tensor table. */
typedef struct {
    uint32_t ggml_type;
    int      cols;
    int      rows;
    int      tensor_count;
    long     matrix_count;       /* tensors x expert slots */
    char     example[GGUF_MAX_TENSOR_NAME];
} census_entry;

/* ================================================================
 * Shared benchmark buffers
 * ================================================================ */

static fm_buffer *activation_a;      /* two distinct activations, so a repeat */
static fm_buffer *activation_b;      /* dispatch is never bit-identical       */
static fm_buffer *output_rows;
static int        max_rows_allocated;
static int        max_cols_allocated;

/* ================================================================
 * Timing and statistics
 * ================================================================ */

static double monotonic_milliseconds(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1.0e6;
}

static int compare_doubles_ascending(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;
    if (a < b)
        return -1;
    if (a > b)
        return 1;
    return 0;
}

/**
 * Summarises latency samples. Sorts `samples` in place.
 *
 * @param samples  Per-sample wall times in milliseconds
 * @param count    Sample count, must be > 0
 * @return         Mean, min, median and p95
 */
static timing_statistics summarise_latency_samples(double *samples, int count)
{
    timing_statistics stats = {0};
    double sum = 0.0;

    for (int i = 0; i < count; i++)
        sum += samples[i];
    stats.mean_ms = sum / (double)count;

    qsort(samples, (size_t)count, sizeof(*samples), compare_doubles_ascending);
    stats.min_ms = samples[0];
    stats.median_ms = samples[count / 2];
    stats.p95_ms = samples[(int)((double)(count - 1) * 0.95)];
    return stats;
}

/* ================================================================
 * Shape arithmetic — the axis the hypothesis is about
 * ================================================================ */

/* Every K-quant here packs 256 weights per super-block, whatever its size. */
static int blocks_per_row(int cols)
{
    return cols / 256;
}

static int row_stride_bytes_for(const quant_matvec_kernel *kernel, int cols)
{
    return blocks_per_row(cols) * kernel->block_bytes;
}

static size_t weight_bytes_for(const quant_matvec_kernel *kernel, int cols, int rows)
{
    return (size_t)rows * (size_t)row_stride_bytes_for(kernel, cols);
}

/** Lane-sized work units in one row: what the hypothesis says starves lanes. */
static int units_per_row(const quant_matvec_kernel *kernel, int cols)
{
    return cols / kernel->lane_unit_weights;
}

/** Fraction of a SIMD group's 32 lanes that own work; 1.0 once units >= 32. */
static double lane_occupancy(const quant_matvec_kernel *kernel, int cols)
{
    int units = units_per_row(kernel, cols);
    return units >= SIMD_LANES_PER_THREADGROUP
        ? 1.0 : (double)units / (double)SIMD_LANES_PER_THREADGROUP;
}

/* ================================================================
 * GGUF census — the shapes decode actually dispatches
 * ================================================================ */

static bool ggml_type_is_quantized_matvec(uint32_t ggml_type)
{
    return ggml_type == GGML_TYPE_Q3_K || ggml_type == GGML_TYPE_Q4_K ||
           ggml_type == GGML_TYPE_Q5_K || ggml_type == GGML_TYPE_Q6_K ||
           ggml_type == GGML_TYPE_Q8_0;
}

/* token_embd is Q3_K but is dequantized at load and never dispatched, so its
 * shape must not enter the sweep. Everything else quantized is a matvec. */
static bool tensor_is_dispatched_weight(const char *name, uint32_t ggml_type,
                                        uint32_t n_dims)
{
    return ggml_type_is_quantized_matvec(ggml_type) && n_dims >= 2 &&
           strstr(name, "token_embd") == NULL;
}

static int find_census_slot(census_entry *entries, int count,
                            uint32_t ggml_type, int cols, int rows)
{
    for (int i = 0; i < count; i++) {
        if (entries[i].ggml_type == ggml_type && entries[i].cols == cols &&
            entries[i].rows == rows)
            return i;
    }
    return -1;
}

/**
 * Walks the tensor table and aggregates every dispatched quantized weight by
 * (type, cols, rows). GGUF stores dims fastest-varying first, so dims[0] is the
 * column count, dims[1] the row count, and dims[2] (experts) multiplies how
 * many matrices of that shape exist.
 *
 * @param file      Handle from gguf_test_open
 * @param entries   Receives the distinct (type, shape) rows
 * @param capacity  Capacity of `entries`
 * @return          Number of entries filled, or -1 on a malformed table
 */
static int collect_dispatched_weight_census(gguf_test_file *file,
                                            census_entry *entries, int capacity)
{
    if (fseeko(file->fp, (off_t)file->tensor_info_start, SEEK_SET) != 0)
        return -1;

    int count = 0;
    for (uint64_t i = 0; i < file->tensor_count; i++) {
        char name[GGUF_MAX_TENSOR_NAME];
        gguf_tensor_location location;
        if (gguf_read_tensor_info(file->fp, name, sizeof(name), &location) != 0)
            return -1;
        if (!tensor_is_dispatched_weight(name, location.ggml_type, location.n_dims))
            continue;

        int cols = (int)location.dims[0];
        int rows = (int)location.dims[1];
        long experts = location.n_dims >= 3 ? (long)location.dims[2] : 1;
        int slot = find_census_slot(entries, count, location.ggml_type, cols, rows);
        if (slot < 0) {
            if (count >= capacity)
                continue;
            slot = count++;
            memset(&entries[slot], 0, sizeof(entries[slot]));
            entries[slot].ggml_type = location.ggml_type;
            entries[slot].cols = cols;
            entries[slot].rows = rows;
            snprintf(entries[slot].example, sizeof(entries[slot].example), "%s", name);
        }
        entries[slot].tensor_count++;
        entries[slot].matrix_count += experts;
    }
    return count;
}

static const char *ggml_type_name(uint32_t ggml_type)
{
    switch (ggml_type) {
    case GGML_TYPE_Q3_K: return "Q3_K";
    case GGML_TYPE_Q4_K: return "Q4_K";
    case GGML_TYPE_Q5_K: return "Q5_K";
    case GGML_TYPE_Q6_K: return "Q6_K";
    case GGML_TYPE_Q8_0: return "Q8_0";
    default:             return "?";
    }
}

static void print_census(const census_entry *entries, int count)
{
    printf("\nDispatched quantized weights in the model (token_embd excluded — "
           "dequantized at load)\n");
    printf("%-6s %7s %8s %9s %10s  %s\n",
           "type", "cols", "rows", "tensors", "matrices", "example");
    printf("%-6s %7s %8s %9s %10s  %s\n",
           "------", "-------", "--------", "---------", "----------",
           "------------------------------");
    for (int i = 0; i < count; i++) {
        printf("%-6s %7d %8d %9d %10ld  %s\n",
               ggml_type_name(entries[i].ggml_type), entries[i].cols,
               entries[i].rows, entries[i].tensor_count,
               entries[i].matrix_count, entries[i].example);
    }
}

/** Distinct (cols, rows) across the census, in first-seen order. */
static int collect_distinct_shapes(const census_entry *entries, int count,
                                   matvec_shape *shapes, int capacity)
{
    int shape_count = 0;

    for (int i = 0; i < count; i++) {
        bool seen = false;
        for (int s = 0; s < shape_count; s++) {
            if (shapes[s].cols == entries[i].cols && shapes[s].rows == entries[i].rows)
                seen = true;
        }
        if (seen || shape_count >= capacity)
            continue;
        if (entries[i].cols % 256 != 0)
            continue;
        shapes[shape_count].cols = entries[i].cols;
        shapes[shape_count].rows = entries[i].rows;
        shapes[shape_count].origin = entries[i].example;
        shapes[shape_count].from_model = true;
        shape_count++;
    }
    return shape_count;
}

/* ================================================================
 * Real weight bytes
 * ================================================================ */

/* Largest chunk of a real tensor copied out of the model file and tiled to
 * fill a weight buffer. Real blocks matter: random bytes put NaN in the half
 * scales, which would make every integrity check vacuous. */
#define WEIGHT_SOURCE_CHUNK_LIMIT   (8u * 1024u * 1024u)

/** Stored bytes of a quantized tensor, expert dimensions included. */
static size_t quantized_tensor_bytes(const gguf_tensor_location *location,
                                     int block_bytes)
{
    size_t bytes = (size_t)location->dims[1] *
                   ((size_t)location->dims[0] / 256) * (size_t)block_bytes;

    for (uint32_t d = 2; d < location->n_dims; d++)
        bytes *= (size_t)location->dims[d];
    return bytes;
}

/** Largest tensor of `ggml_type` in the file, so the tile period is as long
 *  as possible; reports its absolute offset and stored size. */
static int find_weight_source_tensor(gguf_test_file *file, uint32_t ggml_type,
                                     int block_bytes, gguf_tensor_location *out,
                                     size_t *bytes_out, char *name_out,
                                     size_t name_out_size)
{
    size_t best_bytes = 0;

    if (fseeko(file->fp, (off_t)file->tensor_info_start, SEEK_SET) != 0)
        return -1;

    for (uint64_t i = 0; i < file->tensor_count; i++) {
        char name[GGUF_MAX_TENSOR_NAME];
        gguf_tensor_location location;
        if (gguf_read_tensor_info(file->fp, name, sizeof(name), &location) != 0)
            return -1;
        if (location.ggml_type != ggml_type || location.n_dims < 2)
            continue;

        size_t bytes = quantized_tensor_bytes(&location, block_bytes);
        if (bytes <= best_bytes)
            continue;

        best_bytes = bytes;
        location.file_offset += file->tensor_data_start;
        *out = location;
        *bytes_out = bytes;
        snprintf(name_out, name_out_size, "%s", name);
    }
    return best_bytes > 0 ? 0 : -1;
}

/**
 * Fills a kernel's weight buffer by tiling a chunk of one real tensor. The
 * chunk is a whole number of super-blocks so every tile boundary lands on a
 * block boundary and the kernel decodes valid scales everywhere.
 *
 * @return 0 on success, -1 if no suitable tensor exists or the read failed
 */
static int fill_weight_buffer_from_model(gguf_test_file *file,
                                         quant_matvec_kernel *kernel)
{
    gguf_tensor_location location;
    size_t available = 0;

    if (find_weight_source_tensor(file, kernel->ggml_type, kernel->block_bytes,
                                  &location, &available, kernel->source_tensor,
                                  sizeof(kernel->source_tensor)) != 0)
        return -1;

    size_t budget = available < WEIGHT_SOURCE_CHUNK_LIMIT
        ? available : WEIGHT_SOURCE_CHUNK_LIMIT;
    size_t chunk_blocks = budget / (size_t)kernel->block_bytes;
    size_t chunk_bytes = chunk_blocks * (size_t)kernel->block_bytes;

    if (chunk_bytes == 0)
        return -1;

    unsigned char *chunk = (unsigned char *)malloc(chunk_bytes);
    if (!chunk)
        return -1;
    if (gguf_test_read_at(file, location.file_offset, chunk, chunk_bytes) != 0) {
        free(chunk);
        return -1;
    }

    unsigned char *dst = (unsigned char *)fm_buffer_data(kernel->weights);
    for (size_t written = 0; written < kernel->weight_bytes; written += chunk_bytes) {
        size_t span = kernel->weight_bytes - written;
        if (span > chunk_bytes)
            span = chunk_bytes;
        memcpy(dst + written, chunk, span);
    }
    free(chunk);
    return 0;
}

/* ================================================================
 * Slab walking — keeps a repeated dispatch off the cache it just filled
 * ================================================================ */

static size_t greatest_common_divisor(size_t a, size_t b)
{
    while (b != 0) {
        size_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

/**
 * Byte stride between successive repetitions of one arm: the arm's own weight
 * footprint rounded up to a multiple of both the bind alignment and the block
 * size, so every repetition starts on a real super-block at a legal offset.
 */
static size_t slab_stride_for(const quant_matvec_kernel *kernel, int cols, int rows)
{
    size_t alignment = fm_buffer_offset_alignment();
    size_t block = (size_t)kernel->block_bytes;
    size_t granule = alignment / greatest_common_divisor(alignment, block) * block;
    size_t footprint = weight_bytes_for(kernel, cols, rows);

    return (footprint + granule - 1) / granule * granule;
}

/** How many distinct slabs of this shape the kernel's buffer holds. */
static int slab_count_for(const quant_matvec_kernel *kernel, int cols, int rows,
                          size_t slab_stride)
{
    size_t footprint = weight_bytes_for(kernel, cols, rows);

    if (kernel->weight_bytes < footprint || slab_stride == 0)
        return 1;
    return (int)((kernel->weight_bytes - footprint) / slab_stride) + 1;
}

/* ================================================================
 * Measurement
 * ================================================================ */

/** Encodes `repetitions` dispatches into one command buffer and waits. */
static double time_one_sample(const measurement_arm *arm)
{
    quant_matvec_kernel *kernel = arm->kernel;
    int stride = row_stride_bytes_for(kernel, arm->shape.cols);
    double started;

    fm_batch_begin();
    started = monotonic_milliseconds();
    for (int r = 0; r < arm->repetitions; r++) {
        size_t offset = arm->stream_distinct_slabs
            ? (size_t)(r % arm->slab_count) * arm->slab_stride_bytes : 0;
        fm_buffer *x = (r & 1) ? activation_b : activation_a;
        kernel->dispatch(fm_bufref_at(kernel->weights, offset),
                         fm_bufref_at(x, 0), fm_bufref_at(output_rows, 0),
                         arm->shape.rows, arm->shape.cols, stride);
    }
    fm_batch_end();
    return monotonic_milliseconds() - started;
}

static void run_measurement_arms(measurement_arm *arms, int count)
{
    for (int arm = 0; arm < count; arm++) {
        for (int w = 0; w < WARMUP_SAMPLES_PER_ARM; w++)
            (void)time_one_sample(&arms[arm]);
    }

    for (int round = 0; round < MEASUREMENT_ROUND_COUNT; round++) {
        for (int slot = 0; slot < count; slot++) {
            int index = (slot + round * ARM_ORDER_ROTATION_STRIDE) % count;
            measurement_arm *arm = &arms[index];
            for (int s = 0; s < SAMPLES_PER_ROUND; s++)
                arm->pooled_samples[arm->pooled_count++] = time_one_sample(arm);
        }
    }
}

/** Weight bytes one timed sample moves. */
static double sample_weight_bytes(const measurement_arm *arm)
{
    return (double)arm->repetitions *
           (double)weight_bytes_for(arm->kernel, arm->shape.cols, arm->shape.rows);
}

static double effective_gigabytes_per_second(const measurement_arm *arm,
                                             double milliseconds)
{
    if (milliseconds <= 0.0)
        return 0.0;
    return sample_weight_bytes(arm) / (milliseconds / 1000.0) / 1.0e9;
}

/** Milliseconds one single dispatch of this arm's shape costs. */
static double milliseconds_per_dispatch(const measurement_arm *arm,
                                        double sample_milliseconds)
{
    return sample_milliseconds / (double)arm->repetitions;
}

/* ================================================================
 * Elision and dependence guards
 * ================================================================ */

/* Shape used for the integrity probes: big enough that a skipped tail would be
 * a large fraction of the work, small enough to check quickly. */
#define GUARD_COLS   2048
#define GUARD_ROWS   4096

static void dispatch_once_and_capture(quant_matvec_kernel *kernel, fm_buffer *x,
                                      int rows, int cols, float *snapshot)
{
    memset(fm_buffer_data(output_rows), 0, (size_t)rows * sizeof(float));
    fm_batch_begin();
    kernel->dispatch(fm_bufref_at(kernel->weights, 0), fm_bufref_at(x, 0),
                     fm_bufref_at(output_rows, 0), rows, cols,
                     row_stride_bytes_for(kernel, cols));
    fm_batch_end();
    memcpy(snapshot, fm_buffer_data(output_rows), (size_t)rows * sizeof(float));
}

/** True if the two snapshots differ anywhere outside float noise. */
static bool snapshots_differ(const float *left, const float *right, int count)
{
    for (int i = 0; i < count; i++) {
        double a = left[i];
        double b = right[i];
        double scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
        if (fabs(a - b) > 1.0e-6 * (scale > 1.0 ? scale : 1.0))
            return true;
    }
    return false;
}

static bool snapshot_is_live(const float *values, int count)
{
    int nonzero = 0;

    for (int i = 0; i < count; i++) {
        if (!isfinite(values[i]))
            return false;
        if (values[i] != 0.0f)
            nonzero++;
    }
    return nonzero > count / 2;
}

/**
 * Proves the timed kernel really reads every weight byte it is credited with.
 * Three dependences are required, each of which a partially-elided or
 * early-exiting kernel would fail: the activation, the LAST super-block of
 * row 0, and the last row's bytes.
 *
 * @param kernel    Kernel under test (its weight buffer is mutated and restored)
 * @param scratch_a Snapshot buffer, >= GUARD_ROWS floats
 * @param scratch_b Second snapshot buffer
 * @return          true if every dependence held
 */
static bool probe_full_read_dependence(quant_matvec_kernel *kernel,
                                       float *scratch_a, float *scratch_b)
{
    unsigned char *weights = (unsigned char *)fm_buffer_data(kernel->weights);
    int stride = row_stride_bytes_for(kernel, GUARD_COLS);
    size_t last_block_of_row0 = (size_t)stride - (size_t)kernel->block_bytes;
    size_t last_row_first_byte = (size_t)(GUARD_ROWS - 1) * (size_t)stride;
    bool ok = true;

    dispatch_once_and_capture(kernel, activation_a, GUARD_ROWS, GUARD_COLS, scratch_a);
    if (!snapshot_is_live(scratch_a, GUARD_ROWS)) {
        printf("  %-5s FAIL  output is not finite/non-trivial\n", kernel->name);
        return false;
    }

    dispatch_once_and_capture(kernel, activation_b, GUARD_ROWS, GUARD_COLS, scratch_b);
    if (!snapshots_differ(scratch_a, scratch_b, GUARD_ROWS)) {
        printf("  %-5s FAIL  output does not depend on the activation\n", kernel->name);
        ok = false;
    }

    weights[last_block_of_row0] ^= 0xFFu;
    dispatch_once_and_capture(kernel, activation_a, GUARD_ROWS, GUARD_COLS, scratch_b);
    weights[last_block_of_row0] ^= 0xFFu;
    if (!snapshots_differ(scratch_a, scratch_b, 1)) {
        printf("  %-5s FAIL  row 0 ignores its LAST super-block\n", kernel->name);
        ok = false;
    }

    weights[last_row_first_byte] ^= 0xFFu;
    dispatch_once_and_capture(kernel, activation_a, GUARD_ROWS, GUARD_COLS, scratch_b);
    weights[last_row_first_byte] ^= 0xFFu;
    if (!snapshots_differ(&scratch_a[GUARD_ROWS - 1], &scratch_b[GUARD_ROWS - 1], 1)) {
        printf("  %-5s FAIL  the last row was never dispatched\n", kernel->name);
        ok = false;
    }

    if (ok)
        printf("  %-5s pass  activation, last super-block of row 0, last row\n",
               kernel->name);
    return ok;
}

/**
 * Shows the dependence comparator can report red: mutate the weights and then
 * skip the dispatch. The snapshot must be unchanged, which is exactly what a
 * silently-elided dispatch would look like.
 *
 * @return true if the comparator correctly reported "unchanged"
 */
static bool report_elision_guard_can_fail(quant_matvec_kernel *kernel,
                                          float *scratch_a, float *scratch_b)
{
    unsigned char *weights = (unsigned char *)fm_buffer_data(kernel->weights);
    int stride = row_stride_bytes_for(kernel, GUARD_COLS);
    size_t last_block_of_row0 = (size_t)stride - (size_t)kernel->block_bytes;
    bool detected;

    dispatch_once_and_capture(kernel, activation_a, GUARD_ROWS, GUARD_COLS, scratch_a);
    memcpy(scratch_b, scratch_a, (size_t)GUARD_ROWS * sizeof(float));

    weights[last_block_of_row0] ^= 0xFFu;   /* mutate, then deliberately do not dispatch */
    detected = snapshots_differ(scratch_a, scratch_b, GUARD_ROWS);
    weights[last_block_of_row0] ^= 0xFFu;

    printf("  %-5s %s  comparator on a mutation with NO dispatch reported %s\n",
           kernel->name, detected ? "FAIL" : "pass",
           detected ? "changed (bogus)" : "unchanged");
    return !detected;
}

/* ================================================================
 * Arm construction
 * ================================================================ */

static int repetitions_for(const quant_matvec_kernel *kernel, int cols, int rows)
{
    size_t footprint = weight_bytes_for(kernel, cols, rows);
    size_t wanted = TARGET_BYTES_PER_SAMPLE / (footprint > 0 ? footprint : 1);

    if (wanted < 1)
        wanted = 1;
    if (wanted > MAX_REPETITIONS_PER_SAMPLE)
        wanted = MAX_REPETITIONS_PER_SAMPLE;
    return (int)wanted;
}

static void configure_arm(measurement_arm *arm, quant_matvec_kernel *kernel,
                          matvec_shape shape, bool stream_distinct_slabs)
{
    memset(arm, 0, sizeof(*arm));
    arm->kernel = kernel;
    arm->shape = shape;
    arm->stream_distinct_slabs = stream_distinct_slabs;
    arm->repetitions = repetitions_for(kernel, shape.cols, shape.rows);
    arm->slab_stride_bytes = slab_stride_for(kernel, shape.cols, shape.rows);
    arm->slab_count = slab_count_for(kernel, shape.cols, shape.rows,
                                     arm->slab_stride_bytes);
}

/**
 * Builds the sweep: every kernel at every shape, streaming distinct slabs, plus
 * a cache-resident twin wherever the shape's footprint is small enough for the
 * distinction to mean anything.
 */
static int build_sweep_arms(measurement_arm *arms, int capacity,
                            quant_matvec_kernel *kernels, int kernel_count,
                            const matvec_shape *shapes, int shape_count)
{
    int count = 0;

    for (int s = 0; s < shape_count; s++) {
        for (int k = 0; k < kernel_count; k++) {
            if (count >= capacity)
                return count;
            configure_arm(&arms[count++], &kernels[k], shapes[s], true);
        }
    }

    for (int s = 0; s < shape_count; s++) {
        size_t smallest = weight_bytes_for(&kernels[0], shapes[s].cols, shapes[s].rows);
        if (smallest >= CACHE_RESIDENT_FOOTPRINT_LIMIT)
            continue;
        for (int k = 0; k < kernel_count; k++) {
            if (count >= capacity)
                return count;
            configure_arm(&arms[count++], &kernels[k], shapes[s], false);
        }
    }
    return count;
}

/* Indices replicated byte-for-byte to establish the noise floor. A replica
 * runs through the same interleaved rounds as its original, so the gap between
 * the two is pure measurement noise on the statistic actually reported. */
static const int NOISE_FLOOR_REPLICATED_ARMS[] = { 0, 5 };
#define NOISE_FLOOR_REPLICA_COUNT \
    ((int)(sizeof(NOISE_FLOOR_REPLICATED_ARMS) / sizeof(NOISE_FLOOR_REPLICATED_ARMS[0])))

static int append_noise_floor_replicas(measurement_arm *arms, int count, int capacity)
{
    for (int i = 0; i < NOISE_FLOOR_REPLICA_COUNT; i++) {
        int source = NOISE_FLOOR_REPLICATED_ARMS[i];
        if (source >= count || count >= capacity)
            continue;
        arms[count] = arms[source];
        arms[count].is_noise_replica = true;
        arms[count].pooled_count = 0;
        count++;
    }
    return count;
}

/* ================================================================
 * Reporting
 * ================================================================ */

static void print_results_header(const char *title)
{
    printf("\n%s\n", title);
    printf("  (every kernel dispatches ONE threadgroup of %d threads per output "
           "row, so threadgroups == rows)\n", SIMD_LANES_PER_THREADGROUP);
    printf("%-5s %6s %8s %9s %6s %5s %10s %9s %9s %8s %8s\n",
           "quant", "cols", "rows/tg", "footprnt", "units", "lane%", "resident",
           "min ms/d", "p50 ms/d", "GB/s", "p50GB/s");
    printf("%-5s %6s %8s %9s %6s %5s %10s %9s %9s %8s %8s\n",
           "-----", "------", "--------", "---------", "------", "-----",
           "----------", "---------", "---------", "--------", "--------");
}

static void print_arm_row(measurement_arm *arm)
{
    timing_statistics stats = summarise_latency_samples(arm->pooled_samples,
                                                        arm->pooled_count);
    size_t footprint = weight_bytes_for(arm->kernel, arm->shape.cols, arm->shape.rows);

    printf("%-5s %6d %8d %8.1fM %6d %4.0f%% %10s %9.3f %9.3f %8.2f %8.2f%s\n",
           arm->kernel->name, arm->shape.cols, arm->shape.rows,
           (double)footprint / (1024.0 * 1024.0),
           units_per_row(arm->kernel, arm->shape.cols),
           100.0 * lane_occupancy(arm->kernel, arm->shape.cols),
           arm->stream_distinct_slabs ? "streaming" : "cache-hot",
           milliseconds_per_dispatch(arm, stats.min_ms),
           milliseconds_per_dispatch(arm, stats.median_ms),
           effective_gigabytes_per_second(arm, stats.min_ms),
           effective_gigabytes_per_second(arm, stats.median_ms),
           arm->is_noise_replica ? "   (noise replica)" : "");
}

/**
 * Reports the noise floor as the GB/s gap between each replicated arm and its
 * byte-identical replica. Nothing smaller than this is a difference.
 *
 * @return The largest observed replica gap, as a fraction of the arm's GB/s
 */
static double report_noise_floor(measurement_arm *arms, int replica_start)
{
    double worst = 0.0;

    printf("\nNoise floor — byte-identical replica arms, same interleaved rounds\n");
    printf("%-5s %6s %8s %12s %12s %10s\n",
           "quant", "cols", "rows", "GB/s", "replica GB/s", "gap");
    for (int i = 0; i < NOISE_FLOOR_REPLICA_COUNT; i++) {
        measurement_arm *original = &arms[NOISE_FLOOR_REPLICATED_ARMS[i]];
        measurement_arm *replica = &arms[replica_start + i];
        timing_statistics a = summarise_latency_samples(original->pooled_samples,
                                                        original->pooled_count);
        timing_statistics b = summarise_latency_samples(replica->pooled_samples,
                                                        replica->pooled_count);
        double left = effective_gigabytes_per_second(original, a.min_ms);
        double right = effective_gigabytes_per_second(replica, b.min_ms);
        double gap = left > 0.0 ? fabs(left - right) / left : 0.0;

        if (gap > worst)
            worst = gap;
        printf("%-5s %6d %8d %12.2f %12.2f %9.2f%%\n",
               original->kernel->name, original->shape.cols, original->shape.rows,
               left, right, 100.0 * gap);
    }
    printf("\nNOISE FLOOR: %.2f%% of GB/s. No difference below this is real.\n",
           100.0 * worst);
    return worst;
}

/* ================================================================
 * The kernels under test
 *
 * lane_unit_weights is the axis Phase 1.5 is about: q3_k and q4_k give one
 * lane a whole 256-weight super-block, q5_k and q6_k give it a 32-weight
 * sub-block, so at the same shape the latter two have 8x the work units to
 * spread over the same 32 lanes.
 * ================================================================ */

static quant_matvec_kernel KERNELS[] = {
    { "q3_k", GGML_TYPE_Q3_K, 110, 256, fm_matvec_q3k_at,  NULL, 0, {0} },
    { "q4_k", GGML_TYPE_Q4_K, 144, 256, fm_matvec_q4_k_at, NULL, 0, {0} },
    { "q5_k", GGML_TYPE_Q5_K, 176,  32, fm_matvec_q5k,     NULL, 0, {0} },
    { "q6_k", GGML_TYPE_Q6_K, 210,  32, fm_matvec_q6k,     NULL, 0, {0} },
};
#define KERNEL_COUNT ((int)(sizeof(KERNELS) / sizeof(KERNELS[0])))

/* ================================================================
 * Commit cost — what a one-dispatch-per-commit measurement really times
 *
 * Phase 1.5 was raised on a single-shape number of 1.04 ms (Q3_K) / 0.97 ms
 * (Q4_K) at rows=4096, cols=2048. Every arm above amortizes the command-buffer
 * commit across many dispatches, so this stage re-measures the same shape with
 * exactly one dispatch per commit-and-wait and prints both, making visible how
 * much of that original number was the kernel and how much was the round trip.
 * ================================================================ */

#define COMMIT_COST_COLS   2048
#define COMMIT_COST_ROWS   4096

static void report_commit_cost_at_microbench_shape(measurement_arm *scratch)
{
    matvec_shape shape = { COMMIT_COST_COLS, COMMIT_COST_ROWS,
                           "Phase 1.5 microbench shape", true };

    printf("\nCommit cost at the Phase 1.5 microbench shape "
           "(rows=%d, cols=%d)\n", COMMIT_COST_ROWS, COMMIT_COST_COLS);
    printf("%-5s %12s %14s %14s %12s\n", "quant", "amortized ms",
           "1-per-commit ms", "commit ms", "amortized GB/s");
    for (int k = 0; k < KERNEL_COUNT; k++) {
        configure_arm(&scratch[0], &KERNELS[k], shape, true);
        configure_arm(&scratch[1], &KERNELS[k], shape, true);
        scratch[1].repetitions = 1;
        run_measurement_arms(scratch, 2);

        timing_statistics amortized =
            summarise_latency_samples(scratch[0].pooled_samples, scratch[0].pooled_count);
        timing_statistics single =
            summarise_latency_samples(scratch[1].pooled_samples, scratch[1].pooled_count);
        double per_dispatch = milliseconds_per_dispatch(&scratch[0], amortized.min_ms);

        printf("%-5s %12.3f %14.3f %14.3f %12.2f\n", KERNELS[k].name, per_dispatch,
               single.min_ms, single.min_ms - per_dispatch,
               effective_gigabytes_per_second(&scratch[0], amortized.min_ms));
    }
}

/* ================================================================
 * Threads-per-threadgroup probe — the OTHER occupancy axis
 *
 * Lane occupancy (units per row vs 32 lanes) and core occupancy (how the grid
 * maps onto the M4's 10 cores) are independent, and every matvec here is
 * pinned at 32 threads per threadgroup because it reduces with simd_sum. That
 * pin cannot be varied without rewriting the kernels, so it is measured on a
 * kernel whose geometry IS free: elementwise_add, dispatched through the same
 * threadgroup API at a range of group sizes over a fixed, DRAM-sized vector.
 *
 * If a 32-thread group reaches the same bandwidth as a 256-thread group, then
 * 32-wide threadgroups are not what holds the matvecs down and the remaining
 * suspect is inside the threadgroup. If it does not, the matvecs inherit a
 * ceiling that no lane-unit rework can lift.
 * ================================================================ */

/* 16M floats per operand: 192 MB touched per pass, far past any cache. */
#define ROOF_ELEMENT_COUNT        (16 * 1024 * 1024)
#define ROOF_BYTES_PER_ELEMENT    12          /* two reads + one write */
#define ROOF_SAMPLES_PER_ROUND    4
#define ROOF_ROUND_COUNT          5
#define ROOF_MAX_SAMPLES          (ROOF_SAMPLES_PER_ROUND * ROOF_ROUND_COUNT)

static void fill_activation(fm_buffer *buffer, int cols, unsigned seed);

static const int ROOF_THREADS_PER_GROUP[] = { 32, 64, 128, 256, 512, 1024 };
#define ROOF_ARM_COUNT \
    ((int)(sizeof(ROOF_THREADS_PER_GROUP) / sizeof(ROOF_THREADS_PER_GROUP[0])))

typedef struct {
    int    threads_per_group;
    int    threadgroups;
    double samples[ROOF_MAX_SAMPLES];
    int    sample_count;
} threadgroup_size_arm;

static double time_elementwise_add(fm_buffer *a, fm_buffer *b, fm_buffer *out,
                                   int threads_per_group)
{
    fm_bufref refs[] = { fm_bufref_at(a, 0), fm_bufref_at(b, 0),
                         fm_bufref_at(out, 0) };
    uint32_t consts[] = { (uint32_t)ROOF_ELEMENT_COUNT };
    int groups = (ROOF_ELEMENT_COUNT + threads_per_group - 1) / threads_per_group;
    double started;

    fm_batch_begin();
    started = monotonic_milliseconds();
    fm_dispatch_threadgroups_refs(@"elementwise_add", NULL, refs, 3, consts, 1,
                                  groups, threads_per_group);
    fm_batch_end();
    return monotonic_milliseconds() - started;
}

/**
 * Sweeps threads-per-threadgroup on elementwise_add and prints the achieved
 * bandwidth, establishing both the machine's memory roof and whether a
 * 32-thread group — the shape every matvec is pinned to — costs anything.
 *
 * @return The best GB/s observed, i.e. the measured memory roof
 */
static double report_threadgroup_size_roof(void)
{
    size_t bytes = (size_t)ROOF_ELEMENT_COUNT * sizeof(float);
    fm_buffer *a = fm_buffer_alloc(bytes);
    fm_buffer *b = fm_buffer_alloc(bytes);
    fm_buffer *out = fm_buffer_alloc(bytes);
    threadgroup_size_arm arms[ROOF_ARM_COUNT] = {{0, 0, {0}, 0}};
    double roof = 0.0;

    if (!a || !b || !out) {
        printf("  (skipped: could not allocate %.0f MB x 3)\n",
               (double)bytes / (1024.0 * 1024.0));
        return 0.0;
    }
    fill_activation(a, ROOF_ELEMENT_COUNT, 4242u);
    fill_activation(b, ROOF_ELEMENT_COUNT, 777u);

    for (int i = 0; i < ROOF_ARM_COUNT; i++) {
        arms[i].threads_per_group = ROOF_THREADS_PER_GROUP[i];
        arms[i].threadgroups = (ROOF_ELEMENT_COUNT + arms[i].threads_per_group - 1) /
                               arms[i].threads_per_group;
        (void)time_elementwise_add(a, b, out, arms[i].threads_per_group);
    }
    for (int round = 0; round < ROOF_ROUND_COUNT; round++) {
        for (int slot = 0; slot < ROOF_ARM_COUNT; slot++) {
            int i = (slot + round) % ROOF_ARM_COUNT;
            for (int s = 0; s < ROOF_SAMPLES_PER_ROUND; s++)
                arms[i].samples[arms[i].sample_count++] =
                    time_elementwise_add(a, b, out, arms[i].threads_per_group);
        }
    }

    printf("\nThreads-per-threadgroup roof — elementwise_add, n=%d (%d MB/pass)\n",
           ROOF_ELEMENT_COUNT,
           (int)((size_t)ROOF_ELEMENT_COUNT * ROOF_BYTES_PER_ELEMENT / (1024 * 1024)));
    printf("%10s %13s %10s %10s\n", "threads/tg", "threadgroups", "min ms", "GB/s");
    for (int i = 0; i < ROOF_ARM_COUNT; i++) {
        timing_statistics stats = summarise_latency_samples(arms[i].samples,
                                                            arms[i].sample_count);
        double gbps = (double)ROOF_ELEMENT_COUNT * ROOF_BYTES_PER_ELEMENT /
                      (stats.min_ms / 1000.0) / 1.0e9;
        if (gbps > roof)
            roof = gbps;
        printf("%10d %13d %10.3f %10.2f\n", arms[i].threads_per_group,
               arms[i].threadgroups, stats.min_ms, gbps);
    }

    fm_buffer_free(a);
    fm_buffer_free(b);
    fm_buffer_free(out);
    return roof;
}

/* ================================================================
 * Setup
 * ================================================================ */

static void fill_activation(fm_buffer *buffer, int cols, unsigned seed)
{
    float *values = (float *)fm_buffer_data(buffer);
    unsigned state = seed;

    for (int i = 0; i < cols; i++) {
        state = state * 1664525u + 1013904223u;
        values[i] = ((float)(state >> 8) / 8388608.0f - 1.0f) * 0.05f;
    }
}

/**
 * Allocates one weight buffer per quant, large enough for the largest arm, and
 * fills it with tiled real blocks of that quant from the model file.
 *
 * @return 0 on success, -1 if allocation or the model read failed
 */
static int allocate_kernel_weight_buffers(gguf_test_file *file,
                                          quant_matvec_kernel *kernels,
                                          int kernel_count,
                                          const matvec_shape *shapes,
                                          int shape_count)
{
    for (int k = 0; k < kernel_count; k++) {
        size_t largest = 0;
        for (int s = 0; s < shape_count; s++) {
            size_t bytes = weight_bytes_for(&kernels[k], shapes[s].cols, shapes[s].rows);
            if (bytes > largest)
                largest = bytes;
        }
        kernels[k].weight_bytes = largest;
        kernels[k].weights = fm_buffer_alloc(largest);
        if (!kernels[k].weights) {
            printf("FAIL: could not allocate %.1f MB for %s\n",
                   (double)largest / (1024.0 * 1024.0), kernels[k].name);
            return -1;
        }
        if (fill_weight_buffer_from_model(file, &kernels[k]) != 0) {
            printf("FAIL: no usable %s tensor in the model file\n", kernels[k].name);
            return -1;
        }
        printf("  %-5s %8.1f MB weight buffer, blocks tiled from %s\n",
               kernels[k].name, (double)largest / (1024.0 * 1024.0),
               kernels[k].source_tensor);
    }
    return 0;
}

static int allocate_shared_buffers(const matvec_shape *shapes, int shape_count)
{
    for (int s = 0; s < shape_count; s++) {
        if (shapes[s].rows > max_rows_allocated)
            max_rows_allocated = shapes[s].rows;
        if (shapes[s].cols > max_cols_allocated)
            max_cols_allocated = shapes[s].cols;
    }

    activation_a = fm_buffer_alloc((size_t)max_cols_allocated * sizeof(float));
    activation_b = fm_buffer_alloc((size_t)max_cols_allocated * sizeof(float));
    output_rows = fm_buffer_alloc((size_t)max_rows_allocated * sizeof(float));
    if (!activation_a || !activation_b || !output_rows)
        return -1;

    fill_activation(activation_a, max_cols_allocated, 12345u);
    fill_activation(activation_b, max_cols_allocated, 987654321u);
    return 0;
}

/** Appends the row-scaled shapes that force the small real ones out of cache. */
static int append_residency_shapes(matvec_shape *shapes, int count, int capacity)
{
    const matvec_shape scaled[] = {
        { 512,  SCALED_ROWS_AT_512_COLS,  "row-scaled cols=512 (DRAM-resident)",  false },
        { 4096, SCALED_ROWS_AT_4096_COLS, "row-scaled cols=4096 (DRAM-resident)", false },
    };

    for (int i = 0; i < (int)(sizeof(scaled) / sizeof(scaled[0])); i++) {
        if (count >= capacity)
            break;
        shapes[count++] = scaled[i];
    }
    return count;
}

static void describe_shapes(const matvec_shape *shapes, int count)
{
    printf("\nShapes measured (%d)\n", count);
    for (int i = 0; i < count; i++) {
        printf("  cols=%-5d rows=%-7d %-8s %s\n", shapes[i].cols, shapes[i].rows,
               shapes[i].from_model ? "model" : "scaled", shapes[i].origin);
    }
}

/* ================================================================
 * Entry point
 * ================================================================ */

static int run_integrity_guards(void)
{
    float *scratch_a = (float *)malloc((size_t)GUARD_ROWS * sizeof(float));
    float *scratch_b = (float *)malloc((size_t)GUARD_ROWS * sizeof(float));
    int failures = 0;

    if (!scratch_a || !scratch_b) {
        free(scratch_a);
        free(scratch_b);
        return 1;
    }

    printf("\nElision guard — the timed kernel must read every byte it is credited with\n");
    for (int k = 0; k < KERNEL_COUNT; k++) {
        if (!probe_full_read_dependence(&KERNELS[k], scratch_a, scratch_b))
            failures++;
    }

    printf("\nCan-fail demonstration — the same comparator, mutation without dispatch\n");
    for (int k = 0; k < KERNEL_COUNT; k++) {
        if (!report_elision_guard_can_fail(&KERNELS[k], scratch_a, scratch_b))
            failures++;
    }

    free(scratch_a);
    free(scratch_b);
    return failures;
}

int main(int argc, char **argv)
{
    const char *model_path = argc > 1 ? argv[1] : DEFAULT_MODEL_PATH;
    gguf_test_file model;
    census_entry census[MAX_CENSUS_ENTRIES];
    matvec_shape shapes[MAX_SHAPES];
    static measurement_arm arms[MAX_ARMS];
    int census_count, shape_count, arm_count, replica_start, failures;

    printf("=== K-quant matvec effective bandwidth ===\n");
    printf("  model                : %s\n", model_path);
    printf("  rounds x samples     : %d x %d (%d pooled per arm)\n",
           MEASUREMENT_ROUND_COUNT, SAMPLES_PER_ROUND, MAX_POOLED_SAMPLE_COUNT);
    printf("  statistic            : pooled MIN (p50 shown for the contention band)\n");
    printf("  target bytes/sample  : %u MB\n", TARGET_BYTES_PER_SAMPLE / (1024u * 1024u));

    if (fm_init() != 0 || !fm_available()) {
        printf("FAIL: Metal unavailable\n");
        return 1;
    }
    if (gguf_test_open(model_path, &model) != 0) {
        printf("FAIL: cannot open %s\n", model_path);
        return 1;
    }

    census_count = collect_dispatched_weight_census(&model, census, MAX_CENSUS_ENTRIES);
    if (census_count <= 0) {
        printf("FAIL: could not read the tensor table\n");
        gguf_test_close(&model);
        return 1;
    }
    print_census(census, census_count);

    shape_count = collect_distinct_shapes(census, census_count, shapes, MAX_SHAPES);
    shape_count = append_residency_shapes(shapes, shape_count, MAX_SHAPES);
    describe_shapes(shapes, shape_count);

    printf("\nBuffers\n");
    if (allocate_shared_buffers(shapes, shape_count) != 0 ||
        allocate_kernel_weight_buffers(&model, KERNELS, KERNEL_COUNT,
                                       shapes, shape_count) != 0) {
        gguf_test_close(&model);
        return 1;
    }
    gguf_test_close(&model);

    failures = run_integrity_guards();

    arm_count = build_sweep_arms(arms, MAX_ARMS, KERNELS, KERNEL_COUNT,
                                 shapes, shape_count);
    replica_start = arm_count;
    arm_count = append_noise_floor_replicas(arms, arm_count, MAX_ARMS);
    printf("\nMeasuring %d arms...\n", arm_count);
    run_measurement_arms(arms, arm_count);

    print_results_header("Effective weight-read bandwidth (weight bytes only; "
                         "x and out are <2% of traffic)");
    for (int i = 0; i < arm_count; i++)
        print_arm_row(&arms[i]);
    report_noise_floor(arms, replica_start);

    static measurement_arm commit_cost_arms[2];
    report_commit_cost_at_microbench_shape(commit_cost_arms);

    double roof = report_threadgroup_size_roof();
    printf("\nMeasured memory roof on this machine: %.1f GB/s "
           "(elementwise_add, best threadgroup size).\n", roof);

    printf("\n%s\n", failures == 0
        ? "Integrity guards: all green (and shown able to fail)."
        : "Integrity guards: FAILURES — the timings above are not trustworthy.");
    return failures == 0 ? 0 : 1;
}
