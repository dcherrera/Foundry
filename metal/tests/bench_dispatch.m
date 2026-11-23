/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_dispatch.m
 * @brief Phase-0 dispatch-overhead probe for the qwen35moe Metal backend.
 *   Encodes a decode step's worth of trivial kernel dispatches (40 layers x K
 *   dispatches) and measures the tok/s ceiling that dispatch + encode + commit
 *   cost imposes, independent of any real compute.
 *
 *   Measurement paths:
 *     1. The `fm_` API (fm_add / fm_mul / fm_silu) under four batching
 *        strategies — this is what the planned forward would actually pay.
 *     2. A raw-Metal reference on a private device/queue that isolates
 *        *encoder creation* and *pipeline switching* from *dispatch*, because
 *        `fm_metal.m` opens a new MTLComputeCommandEncoder per dispatch.
 *
 *   Rigour guards, in the order they run:
 *     - Elision defence: every dispatch is a link in a data-dependent chain
 *       (`accumulator += ones`, in place), and the final value must equal the
 *       dispatch count predicted from the CONFIGURATION, not from what ran.
 *     - `report_can_fail_demonstration` drops every 7th dispatch and requires
 *       the elision check to report a mismatch — a green that was never shown
 *       to fail is not a gate.
 *     - `report_batch_begin_sync_pairing` checks whether fm_sync() commits a
 *       fm_batch_begin() batch (it does not — see RESULTS_dispatch_probe.md).
 *     - Replica arms give a noise floor on the reported statistic; the
 *       compute-sensitivity sweep proves the kernels are overhead, not work.
 *
 *   Results and verdict: metal/tests/RESULTS_dispatch_probe.md
 *
 * @status Complete. Run: make USE_METAL=1 bench-dispatch
 * @issues Reports the pooled MINIMUM: this machine shares its GPU with the
 *   window server, so the mean and p50 track desktop contention rather than
 *   dispatch cost. p50/p95 are printed alongside to keep that visible.
 * @todo None
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/fm_metal.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================
 * Probe configuration
 * ================================================================ */

/* The planned qwen35moe forward is 40 transformer layers per decoded token. */
#define TRANSFORMER_LAYER_COUNT   40

/* Buffers are allocated at the largest size the probe ever uses; each arm
 * dispatches over `element_count` of them. The default is deliberately small so
 * the kernels are pure overhead — but that is an assumption, not a fact, so the
 * compute-sensitivity stage sweeps the size and proves the default sits on the
 * flat part of the curve. Never assume a kernel is free; show it. */
#define MAX_PROBE_ELEMENT_COUNT   65536
#define PROBE_ELEMENT_COUNT       4096

/* Element counts swept to prove the timing is dispatch-bound, not compute-bound. */
static const int ELEMENT_COUNT_SWEEP[] = { 256, 1024, 4096, 16384, 65536 };
#define ELEMENT_COUNT_SWEEP_COUNT \
    ((int)(sizeof(ELEMENT_COUNT_SWEEP) / sizeof(ELEMENT_COUNT_SWEEP[0])))

/* Per-token samples collected for each configuration, plus discarded warmup. */
#define WARMUP_TOKEN_COUNT        4
#define MEASURED_TOKEN_COUNT      25

/* Seed for the repeatedly-applied silu in the kernel-mix arm. */
#define SILU_STABLE_OPERAND       20.0f

/* Dispatches-per-layer values swept. 0 measures the fixed per-token cost. */
static const int DISPATCHES_PER_LAYER_SWEEP[] = { 0, 5, 10, 20, 40 };
#define DISPATCHES_PER_LAYER_SWEEP_COUNT \
    ((int)(sizeof(DISPATCHES_PER_LAYER_SWEEP) / sizeof(DISPATCHES_PER_LAYER_SWEEP[0])))

/* ================================================================
 * Types
 * ================================================================ */

typedef enum {
    BATCH_STRATEGY_PER_TOKEN,   /* fm_batch_begin .. fm_batch_end once per token */
    BATCH_STRATEGY_PER_LAYER,   /* fm_batch_begin .. fm_batch_end once per layer */
    BATCH_STRATEGY_AUTO_SYNC,   /* fm_set_auto_batch(true), fm_sync() once per token */
    BATCH_STRATEGY_NONE         /* auto-batch off: every dispatch commits and waits */
} fm_batch_strategy;

typedef enum {
    OP_MIX_ADD_CHAIN,           /* every dispatch is fm_add — count-verifiable */
    OP_MIX_ROUND_ROBIN          /* fm_add / fm_mul / fm_silu round-robin */
} fm_op_mix;

typedef struct {
    int               dispatches_per_layer;
    int               element_count;           /* elements each kernel touches */
    fm_batch_strategy strategy;
    fm_op_mix         op_mix;
    int               dropped_dispatch_period; /* 0 = none; else drop every Nth */
} dispatch_probe_config;

typedef struct {
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double median_ms;
    double p95_ms;
} timing_statistics;

typedef struct {
    timing_statistics timing;
    bool              work_verified;
    double            observed_accumulator;
    double            expected_accumulator;
} probe_result;

/* ================================================================
 * Timing and statistics helpers
 * ================================================================ */

/** Monotonic wall clock in milliseconds. Immune to wall-clock adjustment. */
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
 * Summarises per-token latency samples.
 *
 * @param samples  Latency samples in milliseconds (reordered in place)
 * @param count    Number of samples, must be > 0
 * @return         Mean, sample stddev, min, median and p95
 */
static timing_statistics summarise_latency_samples(double *samples, int count)
{
    timing_statistics stats = {0};
    double sum = 0.0;
    double sum_squared_deviation = 0.0;

    for (int i = 0; i < count; i++)
        sum += samples[i];
    stats.mean_ms = sum / (double)count;

    for (int i = 0; i < count; i++) {
        double deviation = samples[i] - stats.mean_ms;
        sum_squared_deviation += deviation * deviation;
    }
    stats.stddev_ms = count > 1 ? sqrt(sum_squared_deviation / (double)(count - 1)) : 0.0;

    qsort(samples, (size_t)count, sizeof(*samples), compare_doubles_ascending);
    stats.min_ms = samples[0];
    stats.median_ms = samples[count / 2];
    stats.p95_ms = samples[(int)((double)(count - 1) * 0.95)];
    return stats;
}

/* Ceiling reported for configurations that do no GPU work at all (K=0 with no
 * batching): the wall time is pure loop overhead, so the ratio is meaningless.
 * -ffast-math is on, so INFINITY is not usable as a sentinel. */
#define UNBOUNDED_TOKENS_PER_SECOND  (-1.0)

/** Implied decode ceiling in tokens/second from a per-token wall time. */
static double implied_tokens_per_second(double milliseconds_per_token)
{
    if (milliseconds_per_token <= 1.0e-3)
        return UNBOUNDED_TOKENS_PER_SECOND;
    return 1000.0 / milliseconds_per_token;
}

/* ================================================================
 * fm_ path — probe buffers
 * ================================================================ */

typedef struct {
    fm_buffer *accumulator;   /* chain target: accumulator += ones, in place */
    fm_buffer *ones;          /* all 1.0f */
    fm_buffer *silu_scratch;  /* target for the round-robin fm_silu dispatches */
} fm_probe_buffers;

static bool allocate_fm_probe_buffers(fm_probe_buffers *buffers)
{
    size_t bytes = (size_t)MAX_PROBE_ELEMENT_COUNT * sizeof(float);

    buffers->accumulator = fm_buffer_alloc(bytes);
    buffers->ones = fm_buffer_alloc(bytes);
    buffers->silu_scratch = fm_buffer_alloc(bytes);
    if (!buffers->accumulator || !buffers->ones || !buffers->silu_scratch)
        return false;

    float *ones_data = (float *)fm_buffer_data(buffers->ones);
    float *scratch_data = (float *)fm_buffer_data(buffers->silu_scratch);
    for (int i = 0; i < MAX_PROBE_ELEMENT_COUNT; i++) {
        ones_data[i] = 1.0f;
        /* Near-fixed-point of silu: x*sigmoid(x) ~= x for x >> 0, so repeated
         * application keeps the operand in a normal float range. Seeding with
         * 1.0 instead halves the value every pass and reaches denormals within
         * ~150 dispatches, which would time denormal handling, not dispatch. */
        scratch_data[i] = SILU_STABLE_OPERAND;
    }
    memset(fm_buffer_data(buffers->accumulator), 0, bytes);
    return true;
}

static void free_fm_probe_buffers(fm_probe_buffers *buffers)
{
    fm_buffer_free(buffers->accumulator);
    fm_buffer_free(buffers->ones);
    fm_buffer_free(buffers->silu_scratch);
    memset(buffers, 0, sizeof(*buffers));
}

/** Zeroes the chain accumulator. Caller must guarantee the GPU is idle. */
static void reset_fm_accumulator(fm_probe_buffers *buffers)
{
    fm_sync();
    memset(fm_buffer_data(buffers->accumulator), 0,
           (size_t)MAX_PROBE_ELEMENT_COUNT * sizeof(float));
}

/**
 * Number of accumulator-incrementing dispatches one simulated token should
 * issue, derived from the configuration alone — never from what actually ran.
 * This is what makes the elision check meaningful.
 */
static long expected_increments_per_token(const dispatch_probe_config *config)
{
    long per_layer = 0;

    for (int index = 0; index < config->dispatches_per_layer; index++) {
        if (config->op_mix == OP_MIX_ADD_CHAIN || index % 3 == 0)
            per_layer++;
    }
    return per_layer * TRANSFORMER_LAYER_COUNT;
}

/* ================================================================
 * fm_ path — encoding one simulated token
 * ================================================================ */

/** Issues one dispatch of the configured op mix. Returns false if dropped. */
static bool issue_fm_dispatch(fm_probe_buffers *buffers,
                              const dispatch_probe_config *config,
                              int index_within_layer, long global_index)
{
    if (config->dropped_dispatch_period > 0 &&
        global_index % config->dropped_dispatch_period == 0)
        return false;

    if (config->op_mix == OP_MIX_ADD_CHAIN || index_within_layer % 3 == 0) {
        fm_add(buffers->accumulator, buffers->ones, buffers->accumulator,
               config->element_count);
    } else if (index_within_layer % 3 == 1) {
        /* Multiply by ones: keeps the chain dependency without changing value. */
        fm_mul(buffers->accumulator, buffers->ones, buffers->accumulator,
               config->element_count);
    } else {
        fm_silu(buffers->silu_scratch, buffers->silu_scratch, config->element_count);
    }
    return true;
}

/**
 * Encodes and completes one simulated decode token through the fm_ API.
 *
 * @param buffers       Probe buffers
 * @param config        Layer count is fixed; K, strategy, op mix, fault period
 * @param global_index  Running dispatch counter, advanced in place
 */
static void run_fm_simulated_token(fm_probe_buffers *buffers,
                                   const dispatch_probe_config *config,
                                   long *global_index)
{
    if (config->strategy == BATCH_STRATEGY_PER_TOKEN)
        fm_batch_begin();

    for (int layer = 0; layer < TRANSFORMER_LAYER_COUNT; layer++) {
        if (config->strategy == BATCH_STRATEGY_PER_LAYER)
            fm_batch_begin();
        for (int index = 0; index < config->dispatches_per_layer; index++) {
            issue_fm_dispatch(buffers, config, index, *global_index);
            (*global_index)++;
        }
        if (config->strategy == BATCH_STRATEGY_PER_LAYER)
            fm_batch_end();
    }

    if (config->strategy == BATCH_STRATEGY_PER_TOKEN)
        fm_batch_end();
    else if (config->strategy == BATCH_STRATEGY_AUTO_SYNC)
        fm_sync();
}

/**
 * Reads back the accumulator and checks every touched element equals
 * `expected`. A single short element means at least one dispatch never reached
 * the GPU.
 *
 * @param accumulator   Chain target buffer
 * @param element_count Elements the arm's dispatches actually covered
 * @param expected      Analytically predicted increment total
 * @param observed_out  Receives the first mismatching value, or element 0
 */
static bool verify_accumulator(fm_buffer *accumulator, int element_count,
                               double expected, double *observed_out)
{
    const float *data = (const float *)fm_buffer_data(accumulator);
    bool matched = true;

    *observed_out = (double)data[0];
    for (int i = 0; i < element_count; i++) {
        if ((double)data[i] != expected) {
            matched = false;
            *observed_out = (double)data[i];
            break;
        }
    }
    return matched;
}

/**
 * Runs warmup + measured tokens for one fm_ configuration.
 *
 * @param buffers      Probe buffers
 * @param config       Configuration to measure
 * @param samples_out  Receives MEASURED_TOKEN_COUNT per-token times, or NULL
 */
static probe_result measure_fm_configuration(fm_probe_buffers *buffers,
                                             const dispatch_probe_config *config,
                                             double *samples_out)
{
    double samples[MEASURED_TOKEN_COUNT];
    probe_result result = {0};
    long global_index = 0;

    fm_set_auto_batch(config->strategy == BATCH_STRATEGY_AUTO_SYNC);
    reset_fm_accumulator(buffers);

    for (int token = 0; token < WARMUP_TOKEN_COUNT; token++)
        run_fm_simulated_token(buffers, config, &global_index);

    /* Warmup adds count toward the expected total: the chain is never reset
     * mid-configuration, so pipeline creation cost is excluded from timing but
     * not from the correctness ledger. */
    for (int token = 0; token < MEASURED_TOKEN_COUNT; token++) {
        double started_at = monotonic_milliseconds();
        run_fm_simulated_token(buffers, config, &global_index);
        samples[token] = monotonic_milliseconds() - started_at;
    }

    fm_set_auto_batch(false);
    fm_sync();

    if (samples_out)
        memcpy(samples_out, samples, sizeof(samples));
    result.timing = summarise_latency_samples(samples, MEASURED_TOKEN_COUNT);
    result.expected_accumulator =
        (double)expected_increments_per_token(config) *
        (double)(WARMUP_TOKEN_COUNT + MEASURED_TOKEN_COUNT);
    result.work_verified = verify_accumulator(buffers->accumulator,
                                              config->element_count,
                                              result.expected_accumulator,
                                              &result.observed_accumulator);
    return result;
}

/* ================================================================
 * Raw-Metal reference path
 *
 * Isolates encoder creation from dispatch. `fm_metal.m` opens one
 * MTLComputeCommandEncoder per dispatch (fm_metal.m:388, 417, 447); a real
 * forward would encode many dispatches into a single encoder. Two identical
 * kernels are alternated so pipeline switching is exercised for real.
 * ================================================================ */

/* The two kernels are byte-identical, so alternating them changes only the
 * MTLComputePipelineState object bound before each dispatch. Comparing the
 * alternating arm with the one-pipeline arm isolates pipeline-state switching
 * from every other per-dispatch cost. */
typedef enum {
    RAW_ENCODING_ENCODER_PER_DISPATCH,  /* mirrors today's fm_metal.m */
    RAW_ENCODING_SINGLE_ENCODER,        /* one encoder per token */
    RAW_ENCODING_SINGLE_PIPELINE        /* one encoder, no pipeline switching */
} raw_encoding_mode;

static NSString *const RAW_KERNEL_SOURCE = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void bench_accumulate_a(device float* accumulator [[buffer(0)]],\n"
    "                               device const float* ones  [[buffer(1)]],\n"
    "                               constant uint& count      [[buffer(2)]],\n"
    "                               uint idx [[thread_position_in_grid]]) {\n"
    "    if (idx >= count) return;\n"
    "    accumulator[idx] = accumulator[idx] + ones[idx];\n"
    "}\n"
    "kernel void bench_accumulate_b(device float* accumulator [[buffer(0)]],\n"
    "                               device const float* ones  [[buffer(1)]],\n"
    "                               constant uint& count      [[buffer(2)]],\n"
    "                               uint idx [[thread_position_in_grid]]) {\n"
    "    if (idx >= count) return;\n"
    "    accumulator[idx] = accumulator[idx] + ones[idx];\n"
    "}\n";

static id<MTLDevice>               raw_device;
static id<MTLCommandQueue>         raw_queue;
static id<MTLComputePipelineState> raw_pipelines[2];
static id<MTLBuffer>               raw_accumulator;
static id<MTLBuffer>               raw_ones;

static bool initialise_raw_metal_probe(void)
{
    NSError *error = nil;
    size_t bytes = (size_t)PROBE_ELEMENT_COUNT * sizeof(float);

    raw_device = MTLCreateSystemDefaultDevice();
    if (!raw_device)
        return false;
    raw_queue = [raw_device newCommandQueue];

    id<MTLLibrary> library = [raw_device newLibraryWithSource:RAW_KERNEL_SOURCE
                                                      options:nil
                                                        error:&error];
    if (!library) {
        fprintf(stderr, "raw probe: shader compile failed: %s\n",
                error.localizedDescription.UTF8String);
        return false;
    }

    NSArray<NSString *> *names = @[ @"bench_accumulate_a", @"bench_accumulate_b" ];
    for (int i = 0; i < 2; i++) {
        id<MTLFunction> function = [library newFunctionWithName:names[(NSUInteger)i]];
        raw_pipelines[i] = [raw_device newComputePipelineStateWithFunction:function
                                                                     error:&error];
        if (!raw_pipelines[i]) {
            fprintf(stderr, "raw probe: pipeline failed: %s\n",
                    error.localizedDescription.UTF8String);
            return false;
        }
    }

    raw_accumulator = [raw_device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    raw_ones = [raw_device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
    float *ones_data = (float *)raw_ones.contents;
    for (int i = 0; i < PROBE_ELEMENT_COUNT; i++)
        ones_data[i] = 1.0f;
    memset(raw_accumulator.contents, 0, bytes);
    return true;
}

/** Binds and dispatches one accumulate kernel on an existing encoder. */
static void encode_raw_dispatch(id<MTLComputeCommandEncoder> encoder,
                                long pipeline_index)
{
    id<MTLComputePipelineState> pipeline = raw_pipelines[pipeline_index & 1];
    uint32_t count = (uint32_t)PROBE_ELEMENT_COUNT;
    NSUInteger threads_per_group = MIN((NSUInteger)PROBE_ELEMENT_COUNT,
                                       pipeline.maxTotalThreadsPerThreadgroup);

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:raw_accumulator offset:0 atIndex:0];
    [encoder setBuffer:raw_ones offset:0 atIndex:1];
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(PROBE_ELEMENT_COUNT, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
}

/** One simulated token on the raw path: one command buffer, commit and wait. */
static void run_raw_simulated_token(int dispatches_per_layer,
                                    raw_encoding_mode mode, long *global_index)
{
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [raw_queue commandBuffer];
        id<MTLComputeCommandEncoder> shared_encoder = nil;

        if (mode != RAW_ENCODING_ENCODER_PER_DISPATCH)
            shared_encoder = [command_buffer computeCommandEncoder];

        for (int layer = 0; layer < TRANSFORMER_LAYER_COUNT; layer++) {
            for (int index = 0; index < dispatches_per_layer; index++) {
                long pipeline_index =
                    mode == RAW_ENCODING_SINGLE_PIPELINE ? 0 : *global_index;

                if (shared_encoder) {
                    encode_raw_dispatch(shared_encoder, pipeline_index);
                } else {
                    id<MTLComputeCommandEncoder> encoder =
                        [command_buffer computeCommandEncoder];
                    encode_raw_dispatch(encoder, pipeline_index);
                    [encoder endEncoding];
                }
                (*global_index)++;
            }
        }

        if (shared_encoder)
            [shared_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

/**
 * Runs warmup + measured tokens on the raw-Metal reference path.
 *
 * @param dispatches_per_layer  K
 * @param mode                  Encoder granularity under test
 * @param samples_out           Receives MEASURED_TOKEN_COUNT times, or NULL
 */
static probe_result measure_raw_configuration(int dispatches_per_layer,
                                              raw_encoding_mode mode,
                                              double *samples_out)
{
    double samples[MEASURED_TOKEN_COUNT];
    probe_result result = {0};
    long global_index = 0;

    memset(raw_accumulator.contents, 0,
           (size_t)PROBE_ELEMENT_COUNT * sizeof(float));

    for (int token = 0; token < WARMUP_TOKEN_COUNT; token++)
        run_raw_simulated_token(dispatches_per_layer, mode, &global_index);

    for (int token = 0; token < MEASURED_TOKEN_COUNT; token++) {
        double started_at = monotonic_milliseconds();
        run_raw_simulated_token(dispatches_per_layer, mode, &global_index);
        samples[token] = monotonic_milliseconds() - started_at;
    }

    if (samples_out)
        memcpy(samples_out, samples, sizeof(samples));
    result.timing = summarise_latency_samples(samples, MEASURED_TOKEN_COUNT);
    result.expected_accumulator =
        (double)dispatches_per_layer * TRANSFORMER_LAYER_COUNT *
        (double)(WARMUP_TOKEN_COUNT + MEASURED_TOKEN_COUNT);

    const float *data = (const float *)raw_accumulator.contents;
    result.work_verified = true;
    result.observed_accumulator = (double)data[0];
    for (int i = 0; i < PROBE_ELEMENT_COUNT; i++) {
        if ((double)data[i] != result.expected_accumulator) {
            result.work_verified = false;
            result.observed_accumulator = (double)data[i];
            break;
        }
    }
    return result;
}

/* ================================================================
 * Measurement arms
 *
 * Every configuration is measured once per round and the rounds are
 * interleaved, so no configuration is penalised for running immediately after
 * a heavy one (an earlier revision of this probe reported a bogus 3x penalty
 * on the arm that happened to follow the no-batching sweep).
 *
 * All per-token samples from all rounds are pooled. The reported ceiling comes
 * from the pool minimum — the cost when the GPU is momentarily uncontended,
 * which is what "dispatch overhead" means. Median and p95 of the same pool show
 * what a desktop with a browser open actually delivers, and the gap between
 * them is the contention band rather than something hidden inside a mean.
 * ================================================================ */

#define MEASUREMENT_ROUND_COUNT   6
#define MAX_POOLED_SAMPLE_COUNT   (MEASUREMENT_ROUND_COUNT * MEASURED_TOKEN_COUNT)

typedef struct {
    const char           *label;
    dispatch_probe_config config;    /* fm arms only */
    int                   raw_dispatches_per_layer;
    raw_encoding_mode     raw_mode;
    bool                  is_raw_arm;
    double                pooled_samples[MAX_POOLED_SAMPLE_COUNT];
    int                   pooled_count;
    bool                  work_verified;
} measurement_arm;

/* Stride used to rotate arm order between rounds. Coprime with any realistic
 * arm count, so every arm follows a different predecessor each round — the GPU
 * is left in a different state before each measurement instead of one arm
 * always inheriting the wake of the heaviest one. */
#define ARM_ORDER_ROTATION_STRIDE   7

/** Runs every arm once per round, rotating order, pooling all samples. */
static void run_measurement_arms(fm_probe_buffers *buffers,
                                 measurement_arm *arms, int count)
{
    for (int round = 0; round < MEASUREMENT_ROUND_COUNT; round++) {
        for (int slot = 0; slot < count; slot++) {
            int i = (slot + round * ARM_ORDER_ROTATION_STRIDE) % count;
            measurement_arm *arm = &arms[i];
            double *sample_slot = &arm->pooled_samples[arm->pooled_count];
            probe_result result;

            if (arm->is_raw_arm) {
                result = measure_raw_configuration(arm->raw_dispatches_per_layer,
                                                   arm->raw_mode, sample_slot);
            } else {
                result = measure_fm_configuration(buffers, &arm->config,
                                                  sample_slot);
            }
            arm->pooled_count += MEASURED_TOKEN_COUNT;
            arm->work_verified = round == 0 ? result.work_verified
                                            : (arm->work_verified &&
                                               result.work_verified);
        }
    }
}

/** Summary of an arm's pooled samples. Sorts the pool in place. */
static timing_statistics summarise_arm(measurement_arm *arm)
{
    return summarise_latency_samples(arm->pooled_samples, arm->pooled_count);
}

static int arm_dispatches_per_layer(const measurement_arm *arm)
{
    return arm->is_raw_arm ? arm->raw_dispatches_per_layer
                           : arm->config.dispatches_per_layer;
}

/* ================================================================
 * Reporting
 * ================================================================ */

static const char *batch_strategy_name(fm_batch_strategy strategy)
{
    switch (strategy) {
    case BATCH_STRATEGY_PER_TOKEN: return "per-token  (begin/end x1)";
    case BATCH_STRATEGY_PER_LAYER: return "per-layer  (begin/end x40)";
    case BATCH_STRATEGY_AUTO_SYNC: return "auto-batch (fm_sync x1)";
    case BATCH_STRATEGY_NONE:      return "none       (commit+wait each)";
    }
    return "unknown";
}

static void print_table_header(const char *title)
{
    printf("\n%s\n", title);
    printf("%-32s %4s %9s %9s %9s %9s %11s %11s\n",
           "strategy", "K", "dispatch", "min ms", "p50 ms", "p95 ms",
           "tok/s @min", "tok/s @p50");
    printf("%-32s %4s %9s %9s %9s %9s %11s %11s\n",
           "--------------------------------", "----", "---------", "---------",
           "---------", "---------", "-----------", "-----------");
}

static void format_ceiling(char *out, size_t out_len, double milliseconds)
{
    double ceiling = implied_tokens_per_second(milliseconds);

    if (ceiling == UNBOUNDED_TOKENS_PER_SECOND)
        snprintf(out, out_len, "%s", "n/a");
    else
        snprintf(out, out_len, "%.1f", ceiling);
}

static void print_arm_row(measurement_arm *arm)
{
    timing_statistics stats = summarise_arm(arm);
    int dispatches_per_layer = arm_dispatches_per_layer(arm);
    char ceiling_at_min[16];
    char ceiling_at_median[16];

    format_ceiling(ceiling_at_min, sizeof(ceiling_at_min), stats.min_ms);
    format_ceiling(ceiling_at_median, sizeof(ceiling_at_median), stats.median_ms);
    printf("%-32s %4d %9d %9.3f %9.3f %9.3f %11s %11s%s\n",
           arm->label, dispatches_per_layer,
           dispatches_per_layer * TRANSFORMER_LAYER_COUNT,
           stats.min_ms, stats.median_ms, stats.p95_ms,
           ceiling_at_min, ceiling_at_median,
           arm->work_verified ? "" : "   <-- WORK CHECK FAILED");
}

static void print_arm_table(const char *title, measurement_arm *arms, int count)
{
    print_table_header(title);
    for (int i = 0; i < count; i++)
        print_arm_row(&arms[i]);
}

/* ================================================================
 * Arm construction
 * ================================================================ */

/** Fills the fm_ sweep arms: every batching strategy x every K. */
static int build_fm_sweep_arms(measurement_arm *arms)
{
    const fm_batch_strategy strategies[] = {
        BATCH_STRATEGY_PER_TOKEN, BATCH_STRATEGY_PER_LAYER,
        BATCH_STRATEGY_AUTO_SYNC, BATCH_STRATEGY_NONE
    };
    int count = 0;

    for (int s = 0; s < 4; s++) {
        for (int k = 0; k < DISPATCHES_PER_LAYER_SWEEP_COUNT; k++) {
            arms[count].label = batch_strategy_name(strategies[s]);
            arms[count].config.dispatches_per_layer = DISPATCHES_PER_LAYER_SWEEP[k];
            arms[count].config.strategy = strategies[s];
            arms[count].config.op_mix = OP_MIX_ADD_CHAIN;
            arms[count].config.element_count = PROBE_ELEMENT_COUNT;
            arms[count].config.dropped_dispatch_period = 0;
            count++;
        }
    }
    return count;
}

/* Sweep indices replicated for the noise floor: per-token K=10 and K=40. */
static const int NOISE_FLOOR_REPLICATED_ARMS[] = { 2, 4 };
#define NOISE_FLOOR_REPLICA_COUNT \
    ((int)(sizeof(NOISE_FLOOR_REPLICATED_ARMS) / sizeof(NOISE_FLOOR_REPLICATED_ARMS[0])))

/**
 * Appends byte-identical copies of two per-token arms. Because a replica runs
 * through exactly the same interleaved rounds as its original, the gap between
 * the two is pure measurement noise on the statistic actually reported — a
 * better floor than repeating one configuration in a separate stage under
 * different machine conditions.
 *
 * @param arms   Arm array already holding the sweep
 * @param count  Arms currently present
 * @return       New arm count
 */
static int build_noise_floor_replica_arms(measurement_arm *arms, int count)
{
    for (int i = 0; i < NOISE_FLOOR_REPLICA_COUNT; i++) {
        arms[count] = arms[NOISE_FLOOR_REPLICATED_ARMS[i]];
        arms[count].label = "per-token REPLICA (noise floor)";
        arms[count].pooled_count = 0;
        count++;
    }
    return count;
}

/**
 * Kernel-mix cross-check arms: an add/mul/silu round-robin should cost the same
 * as the pure-add chain if cost tracks dispatch count, not kernel identity.
 */
static int build_kernel_mix_arms(measurement_arm *arms)
{
    int count = 0;

    for (int k = 1; k < DISPATCHES_PER_LAYER_SWEEP_COUNT; k++) {
        arms[count].label = "add/mul/silu round-robin";
        arms[count].config.dispatches_per_layer = DISPATCHES_PER_LAYER_SWEEP[k];
        arms[count].config.strategy = BATCH_STRATEGY_PER_TOKEN;
        arms[count].config.op_mix = OP_MIX_ROUND_ROBIN;
        arms[count].config.element_count = PROBE_ELEMENT_COUNT;
        arms[count].config.dropped_dispatch_period = 0;
        count++;
    }
    return count;
}

/**
 * Compute-sensitivity arms: the same 1600 dispatches over buffers from 1 KiB to
 * 256 KiB. If the probe is measuring dispatch overhead, these are flat; where
 * they start to rise, the kernels have become real work and any ceiling read off
 * that row is a compute number wearing a dispatch number's clothes.
 */
static int build_compute_sensitivity_arms(measurement_arm *arms)
{
    const int k_index = DISPATCHES_PER_LAYER_SWEEP_COUNT - 1;   /* K = 40 */
    static char labels[ELEMENT_COUNT_SWEEP_COUNT][32];
    int count = 0;

    for (int e = 0; e < ELEMENT_COUNT_SWEEP_COUNT; e++) {
        snprintf(labels[e], sizeof(labels[e]), "elements = %d (%d KiB)",
                 ELEMENT_COUNT_SWEEP[e],
                 (int)((size_t)ELEMENT_COUNT_SWEEP[e] * sizeof(float) / 1024));
        arms[count].label = labels[e];
        arms[count].config.dispatches_per_layer = DISPATCHES_PER_LAYER_SWEEP[k_index];
        arms[count].config.strategy = BATCH_STRATEGY_PER_TOKEN;
        arms[count].config.op_mix = OP_MIX_ADD_CHAIN;
        arms[count].config.element_count = ELEMENT_COUNT_SWEEP[e];
        arms[count].config.dropped_dispatch_period = 0;
        count++;
    }
    return count;
}

/** Raw-Metal arms isolating encoder creation from dispatch. */
static int build_raw_metal_arms(measurement_arm *arms)
{
    const raw_encoding_mode modes[] = { RAW_ENCODING_ENCODER_PER_DISPATCH,
                                        RAW_ENCODING_SINGLE_ENCODER,
                                        RAW_ENCODING_SINGLE_PIPELINE };
    const char *labels[] = { "raw: encoder/dispatch, 2 pipes",
                             "raw: 1 encoder, 2 pipelines",
                             "raw: 1 encoder, 1 pipeline" };
    int count = 0;

    for (int m = 0; m < 3; m++) {
        for (int k = 0; k < DISPATCHES_PER_LAYER_SWEEP_COUNT; k++) {
            arms[count].label = labels[m];
            arms[count].is_raw_arm = true;
            arms[count].raw_mode = modes[m];
            arms[count].raw_dispatches_per_layer = DISPATCHES_PER_LAYER_SWEEP[k];
            count++;
        }
    }
    return count;
}

/* ================================================================
 * Analysis
 * ================================================================ */

/**
 * Reports the noise floor as the gap between each per-token arm and its
 * byte-identical replica. Nothing smaller than this may be called a difference.
 *
 * @return  The largest replica gap on the min statistic, in milliseconds
 */
static double report_noise_floor(measurement_arm *arms, int replica_start)
{
    double largest_min_gap_ms = 0.0;

    printf("\n== Noise floor: identical arms measured twice, interleaved ==\n");
    for (int i = 0; i < NOISE_FLOOR_REPLICA_COUNT; i++) {
        measurement_arm *original = &arms[NOISE_FLOOR_REPLICATED_ARMS[i]];
        measurement_arm *replica = &arms[replica_start + i];
        timing_statistics original_stats = summarise_arm(original);
        timing_statistics replica_stats = summarise_arm(replica);
        double min_gap = fabs(original_stats.min_ms - replica_stats.min_ms);
        double median_gap = fabs(original_stats.median_ms - replica_stats.median_ms);

        if (min_gap > largest_min_gap_ms)
            largest_min_gap_ms = min_gap;
        printf("  per-token K=%d: min %.3f vs %.3f ms (gap %.3f, %.2f%%)   "
               "p50 %.3f vs %.3f ms (gap %.3f, %.2f%%)\n",
               original->config.dispatches_per_layer,
               original_stats.min_ms, replica_stats.min_ms, min_gap,
               100.0 * min_gap / original_stats.min_ms,
               original_stats.median_ms, replica_stats.median_ms, median_gap,
               100.0 * median_gap / original_stats.median_ms);
    }
    printf("  NOISE FLOOR (min statistic): %.3f ms per token. Differences "
           "below this are not reportable.\n", largest_min_gap_ms);
    return largest_min_gap_ms;
}

/**
 * Turns the sweep into the numbers the kernel design actually needs: marginal
 * cost per dispatch, fixed cost per command-buffer round trip, and the
 * dispatch budget a 25 tok/s decode target allows.
 *
 * @param sweep  The fm_ sweep arms, strategy-major then K-ascending
 */
static void report_derived_costs(measurement_arm *sweep)
{
    const int per_token_base = 0 * DISPATCHES_PER_LAYER_SWEEP_COUNT;
    const int no_batch_base = 3 * DISPATCHES_PER_LAYER_SWEEP_COUNT;
    const int low = 1;                                     /* K = 5  -> 200 */
    const int high = DISPATCHES_PER_LAYER_SWEEP_COUNT - 1; /* K = 40 -> 1600 */
    double dispatch_delta = (double)(DISPATCHES_PER_LAYER_SWEEP[high] -
                                     DISPATCHES_PER_LAYER_SWEEP[low]) *
                            TRANSFORMER_LAYER_COUNT;

    double per_token_high = summarise_arm(&sweep[per_token_base + high]).min_ms;
    double per_token_low = summarise_arm(&sweep[per_token_base + low]).min_ms;
    double per_token_zero = summarise_arm(&sweep[per_token_base]).min_ms;
    double no_batch_high = summarise_arm(&sweep[no_batch_base + high]).min_ms;
    double no_batch_low = summarise_arm(&sweep[no_batch_base + low]).min_ms;

    double per_dispatch_us = 1000.0 * (per_token_high - per_token_low) / dispatch_delta;
    double fixed_per_token_us = 1000.0 * per_token_zero;
    double per_commit_us = 1000.0 * (no_batch_high - no_batch_low) / dispatch_delta;
    double budget_us = 1000.0 * 1000.0 / 25.0;   /* 25 tok/s => 40 ms per token */
    double affordable = (budget_us - fixed_per_token_us) / per_dispatch_us;

    printf("\n== Derived costs (pool minimum, one command buffer per token) ==\n");
    printf("  marginal cost per dispatch           : %.3f us\n", per_dispatch_us);
    printf("  fixed cost per token (K=0, 1 commit) : %.1f us\n", fixed_per_token_us);
    printf("  cost per command-buffer round trip   : %.1f us "
           "(from the no-batching arm)\n", per_commit_us);
    printf("  25 tok/s budget                      : %.0f us per token\n", budget_us);
    printf("  dispatch budget at 25 tok/s          : %.0f total = %.0f per layer "
           "across %d layers\n", affordable, affordable / TRANSFORMER_LAYER_COUNT,
           TRANSFORMER_LAYER_COUNT);
    printf("  planned granularity (K=40, 1600 disp): %.2f ms per token = "
           "%.1f tok/s ceiling\n", per_token_high,
           implied_tokens_per_second(per_token_high));
    printf("  dispatch overhead as a share of a 40 ms token budget: %.1f%%\n",
           100.0 * per_token_high / 40.0);
}

/**
 * Checks whether `fm_sync()` commits work encoded inside a `fm_batch_begin()`
 * batch. The plan's Phase 5 specifies "one fm_batch_begin() per token, single
 * fm_sync() before reading logits", so whether that pairing actually flushes
 * decides whether the planned forward would ever produce logits.
 *
 * @return 0 if the pairing works, 1 if fm_sync leaves the batch uncommitted
 */
static int report_batch_begin_sync_pairing(fm_probe_buffers *buffers)
{
    const int dispatch_count = 8;
    const float *data = (const float *)fm_buffer_data(buffers->accumulator);
    double after_sync = 0.0;
    double after_end = 0.0;

    reset_fm_accumulator(buffers);
    fm_batch_begin();
    for (int i = 0; i < dispatch_count; i++)
        fm_add(buffers->accumulator, buffers->ones, buffers->accumulator,
               PROBE_ELEMENT_COUNT);
    fm_sync();
    after_sync = (double)data[0];
    fm_batch_end();
    after_end = (double)data[0];

    printf("\n== fm_batch_begin + fm_sync pairing (plan Phase 5) ==\n");
    printf("  %d dispatches encoded; after fm_sync() accumulator = %.0f, "
           "after fm_batch_end() = %.0f (expected %d)\n",
           dispatch_count, after_sync, after_end, dispatch_count);
    if (after_sync == (double)dispatch_count)
        return 0;
    printf("  DEFECT: fm_sync() does NOT commit a fm_batch_begin() batch — it "
           "only flushes the auto-batch buffer (fm_metal.m:667-677), and\n"
           "  fm_submit_if_not_batching() returns early while batching so "
           "_fm_pending_ops stays 0. Phase 5 must call fm_batch_end(),\n"
           "  or use fm_set_auto_batch(true) + fm_sync(), but not "
           "fm_batch_begin() + fm_sync().\n");
    return 1;
}

/**
 * Demonstrates the elision check can fail: drops every 7th dispatch and
 * confirms the accumulator verification reports a mismatch. Per the harness
 * rule, a green check that was never shown to fail is not a gate.
 */
static int report_can_fail_demonstration(fm_probe_buffers *buffers)
{
    dispatch_probe_config honest = { 10, PROBE_ELEMENT_COUNT,
                                     BATCH_STRATEGY_PER_TOKEN,
                                     OP_MIX_ADD_CHAIN, 0 };
    dispatch_probe_config faulty = { 10, PROBE_ELEMENT_COUNT,
                                     BATCH_STRATEGY_PER_TOKEN,
                                     OP_MIX_ADD_CHAIN, 7 };
    probe_result honest_result = measure_fm_configuration(buffers, &honest, NULL);
    probe_result faulty_result = measure_fm_configuration(buffers, &faulty, NULL);

    printf("\n== Can-fail demonstration (harness rule) ==\n");
    printf("  honest run: expected %.0f, observed %.0f -> %s\n",
           honest_result.expected_accumulator, honest_result.observed_accumulator,
           honest_result.work_verified ? "verified" : "MISMATCH");
    printf("  1-in-7 dispatches dropped: expected %.0f, observed %.0f -> %s\n",
           faulty_result.expected_accumulator, faulty_result.observed_accumulator,
           faulty_result.work_verified ? "verified" : "MISMATCH");

    if (honest_result.work_verified && !faulty_result.work_verified) {
        printf("  RESULT: the work check detects missing dispatches. "
               "Its green is trustworthy.\n");
        return 0;
    }
    printf("  RESULT: the work check is NOT a gate — it did not behave as "
           "required.\n");
    return 1;
}

/* ================================================================
 * Entry point
 * ================================================================ */

#define MAX_ARM_COUNT   32

static void run_full_probe(fm_probe_buffers *buffers)
{
    static measurement_arm sweep_arms[MAX_ARM_COUNT];
    static measurement_arm mix_arms[MAX_ARM_COUNT];
    static measurement_arm raw_arms[MAX_ARM_COUNT];
    int sweep_count = build_fm_sweep_arms(sweep_arms);
    int replica_start = sweep_count;

    sweep_count = build_noise_floor_replica_arms(sweep_arms, sweep_count);
    run_measurement_arms(buffers, sweep_arms, sweep_count);

    int mix_count = build_kernel_mix_arms(mix_arms);
    mix_count += build_compute_sensitivity_arms(&mix_arms[mix_count]);
    run_measurement_arms(buffers, mix_arms, mix_count);

    report_noise_floor(sweep_arms, replica_start);
    print_arm_table("== fm_ API sweep: 40 layers x K dispatches per simulated "
                    "token (fm_add chain) ==", sweep_arms, sweep_count);
    print_arm_table("== Kernel-mix and compute-sensitivity cross-checks "
                    "(per-token batching) ==", mix_arms, mix_count);

    if (initialise_raw_metal_probe()) {
        int raw_count = build_raw_metal_arms(raw_arms);
        run_measurement_arms(buffers, raw_arms, raw_count);
        print_arm_table("== Raw-Metal reference: one command buffer per token, "
                        "encoder count varied ==", raw_arms, raw_count);
    } else {
        fprintf(stderr, "raw-Metal reference unavailable, skipped.\n");
    }

    report_derived_costs(sweep_arms);
}

int main(int argc, char **argv)
{
    fm_probe_buffers buffers = {0};
    bool prove_can_fail_only = argc > 1 && strcmp(argv[1], "--prove-can-fail") == 0;
    int exit_code = 0;

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "Metal unavailable — cannot run the dispatch probe.\n");
        return 1;
    }
    if (!allocate_fm_probe_buffers(&buffers)) {
        fprintf(stderr, "Failed to allocate probe buffers.\n");
        fm_free();
        return 1;
    }

    printf("qwen35moe Metal dispatch-overhead probe\n");
    printf("  layers per simulated token : %d\n", TRANSFORMER_LAYER_COUNT);
    printf("  elements per buffer        : %d (%zu KiB)\n", PROBE_ELEMENT_COUNT,
           (size_t)PROBE_ELEMENT_COUNT * sizeof(float) / 1024);
    printf("  warmup / measured tokens   : %d / %d per round\n",
           WARMUP_TOKEN_COUNT, MEASURED_TOKEN_COUNT);
    printf("  interleaved rounds per arm : %d (%d pooled samples)\n",
           MEASUREMENT_ROUND_COUNT, MAX_POOLED_SAMPLE_COUNT);

    exit_code = report_can_fail_demonstration(&buffers);
    report_batch_begin_sync_pairing(&buffers);
    if (!prove_can_fail_only)
        run_full_probe(&buffers);

    free_fm_probe_buffers(&buffers);
    fm_free();
    return exit_code;
}
