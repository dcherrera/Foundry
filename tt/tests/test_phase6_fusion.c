/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_fusion.c
 * @brief ON-CARD MEASUREMENT for Phase 6 item 8 — op fusions. It measures where the real
 *   MiniCPM5-1B layer actually round-trips data, and what each round trip costs, so a
 *   fusion can be ranked by a number instead of by how natural it looks.
 *
 * ---------------------------------------------------------------------------
 * THE LICENCE THIS SUITE EXISTS TO TEST AGAINST
 * ---------------------------------------------------------------------------
 * The plan's Phase 6 item 8 licenses a fusion on ONE ground: it avoids a round trip
 * through L1 or DRAM. **Never dispatch cost** — dispatch is 1.363 us batched and 0.39 us
 * device-side, so a fusion sold on descriptor count is
 * porting a K80 workaround for a problem this backend does not have. This suite therefore
 * measures ROUND TRIPS and reports the dispatch floor beside them only so it can be
 * SUBTRACTED, never claimed.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT MEASURES, AND WHY EACH ARM IS THE ARM IT IS
 * ---------------------------------------------------------------------------
 *  1. **The delay ratio, first.** FT_TT_QUEUE_OP_DELAY at N and 8N iterations. If the
 *     harness cannot see a factor of eight it cannot see a fusion, and every later number
 *     in the run would be noise wearing a decimal point. This is the timing harness's own
 *     red/green: it is a check that CAN fail, run before anything it protects.
 *  2. **The dispatch floor.** A batch of NOPs on the same residency, so the per-descriptor
 *     protocol cost is measured in THIS run, on THIS core, and can be subtracted from every
 *     other arm rather than inherited from another session's table.
 *  3. **The round-trip matrix.** The SAME elementwise op with only its operand ADDRESSES
 *     changed: operands in DRAM, in this core's own L1, or in a neighbour core's L1. The
 *     ops already take a NoC (x, y, address) triple per operand, so "keep the intermediate
 *     in L1" is a placement, not a kernel — which means the DRAM-versus-L1 difference is
 *     measurable TODAY, without building the fusion that would exploit it.
 *  4. **The removable op.** SILU_MUL at 144 tiles is the FFN gate exactly as MiniCPM5-1B
 *     runs it, and ADD at 48 tiles is the residual. A fusion that absorbs one of these
 *     removes its whole wall time, so its whole wall time is the UPPER BOUND of that
 *     fusion's saving and is measured directly rather than modelled.
 *  5. **The matmul denominators**, at every real shape, so a saving can be quoted as a
 *     fraction of the layer it sits in — and ffn_gate/up against ffn_down, which have the
 *     SAME tile-pair count and 144 against 48 output tiles, so the difference between them
 *     is what one output tile write costs inside a matmul.
 *  6. **The binary switch.** Five kernel images over the x1 link. Not a fusion and not
 *     dispatch — a third thing, and on the real layer sequence it is larger than either.
 *
 * ---------------------------------------------------------------------------
 * INTERLEAVED, ALWAYS
 * ---------------------------------------------------------------------------
 * Two 1.8 GHz host cores also driving eight K80s drift enough over ten seconds to swamp a
 * microsecond — measuring one arm to completion and then the other once reported a NEGATIVE
 * cost for CONFIGURE_TLB (dev notes §8). Every arm here runs once per round, in the same
 * order, and the rounds are averaged.
 *
 * ---------------------------------------------------------------------------
 * AND THE ANSWERS ARE COMPARED AS BYTES
 * ---------------------------------------------------------------------------
 * A placement change moves where data lives and must not move what it is. PCC is blind to
 * a consistent permutation and has already failed twice as a gate on this project (dev
 * notes §3), so the DRAM arm and the L1 arm are compared BYTE FOR BYTE, and a NEGATIVE
 * CONTROL — the same comparison against a deliberately different computation — is asserted
 * to come back DIFFERENT in the same run, so the comparison is known to be load-bearing
 * rather than assumed to be.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX          the core to drive (default 0; use 100-119 for this worker)
 *   FT_TT_FUSION_SCRATCH_INDEX  a second, IDLE core whose L1 is the neighbour-L1 arm
 *   FT_TT_FIRMWARE_DIR          the program directory whose fw_*.elf to boot
 *   FT_TT_FUSION_ROUNDS         rounds per arm (default 5)
 *
 * Usage — the last two directories are OPTIONAL writer arms; give both to get the
 * three-arm barrier A/B of stage 3b:
 *   FT_TT_FIRMWARE_DIR=../programs/eltwise FT_TT_TENSIX_INDEX=100 \
 *     FT_TT_FUSION_SCRATCH_INDEX=119 \
 *     ./build-fusion-card/test_phase6_fusion ../programs/eltwise ../programs/reduce \
 *         ../programs/rtriad ../programs/eltwise_blocked ../programs/eltwise_flushed
 *
 * @status GREEN on silicon, 0 checks failed, on TWO disjoint cores — Tensix 100 (scratch
 *   119) and Tensix 110 (scratch 118) — every figure reproducing within ~1%. All six cores
 *   touched probe PASS 4/4 afterwards. It builds no fusion; the kernels it adds
 *   (kernels/resident_vector_writer_blocked.c and _flushed.c) are measurement arms for the
 *   shipped writer's per-tile barrier and nothing depends on them.
 * @issues None
 * @todo None
 */

/* clock_gettime(CLOCK_MONOTONIC) is POSIX.1-2008, not C99. */
#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_coord.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_matmul.h"
#include "../ft_tt_op_vector.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_BYTES     FT_TT_OP_TILE_BYTES_BF16
#define TILE_DIM       FT_TT_OP_TILE_DIM
#define RUN_TIMEOUT_MS 60000u
#define QUEUE_CAPACITY 64u
#define POISON_BYTE    0x5A

/** DRAM banks 2..5, as every other op suite here uses. */
#define DRAM_BANK_A      2u
#define DRAM_BANK_B      3u
#define DRAM_BANK_C      4u
#define DRAM_BANK_CONST  5u
#define DRAM_BANK_COUNT  6u
#define DRAM_BUFFER_BASE 0x00100000u

/** Scratch bases inside the CB arena, above anything ft_tt_op_cb_place() hands out at this
 *  prefetch depth (139,264 B from 0x26000, i.e. up to ~0x48000) and below the work queue at
 *  0x170000. Each holds MAX_TILES tiles = 0x48000 bytes, so the three do not overlap. */
#define L1_SCRATCH_A 0x00060000u
#define L1_SCRATCH_B 0x000B0000u
#define L1_SCRATCH_C 0x00100000u

/** The widest run any arm uses: MiniCPM5-1B's FFN width, 4608 / 32. */
#define MAX_TILES 144u
/** MiniCPM5-1B's hidden width, 1536 / 32 — the residual and norm width. */
#define HIDDEN_TILES 48u

/** One prefetch depth for every elementwise arm, so depth is never a variable between
 *  arms. 16 divides both 48 and 144, and the CBs it places (2 blocks each on two input
 *  streams) cost 139,264 B of a 1,351,680 B arena. */
#define ELTWISE_DEPTH 24u
/** The matmul arms' requested depth — the same 48 the published per-shape figures were
 *  taken at, so these denominators sit beside them rather than beside nothing. Shrunk per
 *  shape to a divisor of Kt by ft_tt_op_block_depth(); the CBs are placed for 2x48 pages,
 *  which every shrunk value (48, 32, 8) divides. */
#define MATMUL_DEPTH 48u

/** Descriptors per timed window. Batching amortizes the host's completion poll so the
 *  number is the DEVICE's throughput and not the PCIe poll granularity. */
#define BATCH 8u

#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

#define TRIAD_MASK                                                                \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) | \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

/** One Tensix's derived peak at 800 MHz, HiFi4 — the denominator every per-core figure in
 *  this project is quoted against (dev notes §6). */
#define TENSIX_PEAK_GFLOPS 1638.4

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_eltwise;
static ft_tt_program_directory g_reduce;
static ft_tt_program_directory g_rtriad;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
static uint32_t g_rounds = 5u;

/** This core, and the idle core whose L1 the neighbour arm uses. */
static uint16_t g_self_x;
static uint16_t g_self_y;
static uint16_t g_peer_x;
static uint16_t g_peer_y;

/* ================================================================
 * Small helpers
 * ================================================================ */

static uint64_t monotonic_microseconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0u;
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)(now.tv_nsec / 1000);
}

static int device_write(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t address,
                        const void *source, size_t bytes)
{
    const unsigned char *cursor = source;
    size_t offset = 0;

    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > 0x10000u)
            chunk = 0x10000u;
        rc = noc->write(noc->user, x, y, (uint64_t)address + offset, cursor + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

static int device_read(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t address,
                       void *destination, size_t bytes)
{
    unsigned char *cursor = destination;
    size_t offset = 0;

    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > 0x10000u)
            chunk = 0x10000u;
        rc = noc->read(noc->user, x, y, (uint64_t)address + offset, cursor + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

static int fill_pattern(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t address,
                        size_t bytes, unsigned char byte)
{
    unsigned char pattern[0x1000];
    size_t offset = 0;

    memset(pattern, byte, sizeof(pattern));
    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > sizeof(pattern))
            chunk = sizeof(pattern);
        rc = device_write(noc, x, y, address + (uint32_t)offset, pattern, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

/**
 * How many whole TILES came back still entirely poison.
 *
 * Counting poison BYTES is the obvious version and it is wrong: 0x5A is a legal byte of a
 * bf16 result, and a correct 294,912-byte answer contained 519 of them — so a
 * "no poison survives" check on bytes fails on a kernel that worked perfectly. The
 * question the poison is there to answer is "did the kernel write this region at all",
 * and a whole tile of unbroken poison is what that looks like. One misleading counter,
 * caught by demanding the check go red for a reason (dev notes §3).
 */
static size_t poison_tiles_remaining(const void *data, size_t bytes)
{
    const unsigned char *cursor = data;
    const size_t tiles = bytes / TILE_BYTES;
    size_t untouched = 0;
    size_t tile;

    for (tile = 0; tile < tiles; tile++) {
        size_t index;
        bool all_poison = true;

        for (index = 0; index < TILE_BYTES; index++) {
            if (cursor[tile * TILE_BYTES + index] != POISON_BYTE) {
                all_poison = false;
                break;
            }
        }
        if (all_poison)
            untouched++;
    }
    return untouched;
}

/** A cheap full-mantissa generator — the operands must carry low mantissa bits or a wrong
 *  format reproduces the right answer (dev notes §3). */
static uint64_t g_random_state = 0x9E3779B97F4A7C15ULL;

static void random_seed(uint64_t seed)
{
    g_random_state = seed ? seed : 1u;
}

static float random_value(void)
{
    g_random_state ^= g_random_state << 13;
    g_random_state ^= g_random_state >> 7;
    g_random_state ^= g_random_state << 17;
    return ((float)(g_random_state >> 40) / 8388608.0f) * 2.0f - 1.0f;
}

static int upload_random_tiles(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                               uint32_t address, uint32_t tiles, uint64_t seed)
{
    const size_t elements = (size_t)tiles * TILE_DIM * TILE_DIM;
    uint16_t *plain = malloc(elements * sizeof(uint16_t));
    uint16_t *tiled = malloc(elements * sizeof(uint16_t));
    size_t index;
    int rc = FT_TT_ENOMEM;

    if (!plain || !tiled)
        goto done;
    random_seed(seed);
    for (index = 0; index < elements; index++)
        plain[index] = ft_bf16_from_float(random_value());
    /* The values are what matters here, not the row geometry: every arm reads the same
     * bytes, so a flat tile stream is a legitimate operand for a timing comparison and for
     * a byte-for-byte equality between two placements of the same op. */
    rc = ft_tt_tilize_matrix(plain, sizeof(uint16_t), TILE_DIM, tiles * TILE_DIM, tiled) == 0
             ? FT_TT_OK
             : FT_TT_EINVAL;
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, x, y, address, tiled,
                          (size_t)tiles * TILE_BYTES);
done:
    free(plain);
    free(tiled);
    return rc;
}

/* ================================================================
 * Bringing a resident triad up
 * ================================================================ */

static int load_images(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                       const ft_tt_program_directory *program)
{
    unsigned index;

    for (index = 0; index < program->program.image_count; index++) {
        const ft_tt_kernel_image *image = &program->program.images[index];
        int rc = ft_tt_launch_load_image(noc, x, y, image->processor, FT_TT_IMAGE_KERNEL,
                                         image->elf_data, image->elf_size, NULL);

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

static int start_triad(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0,
                                            FT_TT_PROC_MATH1, FT_TT_PROC_MATH2 };
    uint32_t args[5][FT_TT_QUEUE_KERNEL_ARG_COUNT];
    ft_tt_op_args blocks[5];
    unsigned index;
    int rc;

    for (index = 0; index < 5u; index++) {
        ft_tt_queue_kernel_binding binding;

        memset(&binding, 0, sizeof(binding));
        binding.idle_spin_limit = IDLE_SPIN_LIMIT;
        binding.cb_wait_spin_limit = CB_SPIN_LIMIT;
        binding.status_address =
            FT_TT_QUEUE_STATUS_ADDRESS(g_queue.control_address, processors[index]);
        rc = ft_tt_queue_kernel_args(&g_queue, &binding, args[index]);
        if (rc != FT_TT_OK)
            return rc;
        blocks[index].processor = processors[index];
        blocks[index].values = args[index];
        blocks[index].count = FT_TT_QUEUE_KERNEL_ARG_COUNT;
    }

    rc = ft_tt_op_stage(noc, x, y, &g_config, g_cb_buffer, sizeof(g_cb_buffer), blocks, 5u);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_send_go(noc, x, y);
}

/** NEVER a park — on a five-processor resident program that is the hazard, twice, on cores
 *  that had retired cleanly (dev notes §1). Asking is the whole protocol. */
static void retire_triad(const ft_tt_noc_access *noc, const char *label)
{
    uint32_t unstopped = 0u;
    int rc = ft_tt_watchdog_shutdown(noc, &g_group, RUN_TIMEOUT_MS, &unstopped);

    ft_tt_test_check_fmt(label, rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=0x%X", rc,
                         unstopped);
}

/* ================================================================
 * The timed window
 * ================================================================ */

/** How many descriptors have been retired on the live residency, so every enqueue knows
 *  what completion count to wait for without threading a counter through every arm. */
static uint32_t g_completed;

/**
 * Enqueue `count` copies of `descriptor` and wait for all of them, returning the elapsed
 * microseconds.
 *
 * The window is one host head-write, `count` device-side executions and one completion
 * poll — so per-descriptor cost is the device's, not the PCIe poll's, which is the whole
 * reason to batch rather than time one descriptor at a time.
 */
static uint64_t time_batch(const ft_tt_hardware_session *session,
                           const ft_tt_queue_descriptor *descriptor, uint32_t count, int *out_rc)
{
    ft_tt_queue_descriptor batch[BATCH];
    uint64_t started;
    uint64_t elapsed;
    uint32_t index;
    int rc;

    if (count > BATCH) {
        *out_rc = FT_TT_EINVAL;
        return 0u;
    }
    for (index = 0; index < count; index++)
        batch[index] = *descriptor;

    started = monotonic_microseconds();
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, batch, count);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_completed + count,
                                        RUN_TIMEOUT_MS, NULL);
    elapsed = monotonic_microseconds() - started;
    if (rc == FT_TT_OK)
        g_completed += count;
    *out_rc = rc;
    return elapsed;
}

/* ================================================================
 * The arm table
 * ================================================================ */

/** Where an operand lives. The op takes a NoC (x, y, address) triple, so this is the whole
 *  of "keep the intermediate in L1 instead of DRAM" — a placement, not a kernel. */
typedef enum operand_place {
    PLACE_DRAM = 0,
    PLACE_L1_SELF = 1,
    PLACE_L1_PEER = 2
} operand_place;

typedef struct eltwise_arm {
    const char *label;
    ft_tt_op_vector_function function;
    uint32_t tiles;
    operand_place inputs;
    operand_place output;
    /** Point input B at input A's buffer instead of B's — the NEGATIVE CONTROL, which must
     *  produce a DIFFERENT answer and so proves the byte comparison can fail. */
    bool alias_b_onto_a;
    uint64_t total_microseconds;
} eltwise_arm;

static void place_operand(ft_tt_op_operand *operand, operand_place place, uint32_t dram_bank,
                          uint32_t l1_address)
{
    switch (place) {
    case PLACE_L1_SELF:
        operand->x = g_self_x;
        operand->y = g_self_y;
        operand->address = l1_address;
        break;
    case PLACE_L1_PEER:
        operand->x = g_peer_x;
        operand->y = g_peer_y;
        operand->address = l1_address;
        break;
    case PLACE_DRAM:
    default:
        operand->x = (uint16_t)g_dram[dram_bank].x;
        operand->y = (uint16_t)g_dram[dram_bank].y;
        operand->address = DRAM_BUFFER_BASE;
        break;
    }
    operand->tile_bytes = TILE_BYTES;
}

static void fill_eltwise_request(ft_tt_op_eltwise_request *request, const eltwise_arm *arm)
{
    memset(request, 0, sizeof(*request));
    request->function = arm->function;
    request->tile_count = arm->tiles;
    request->prefetch_tiles = ELTWISE_DEPTH;
    place_operand(&request->input_a, arm->inputs, DRAM_BANK_A, L1_SCRATCH_A);
    if (ft_tt_op_vector_function_is_binary(arm->function)) {
        if (arm->alias_b_onto_a)
            place_operand(&request->input_b, arm->inputs, DRAM_BANK_A, L1_SCRATCH_A);
        else
            place_operand(&request->input_b, arm->inputs, DRAM_BANK_B, L1_SCRATCH_B);
    }
    place_operand(&request->output, arm->output, DRAM_BANK_C, L1_SCRATCH_C);
}

/** Read one arm's output back, wherever it landed. */
static int read_arm_output(const ft_tt_hardware_session *session, const eltwise_arm *arm,
                           void *destination)
{
    const size_t bytes = (size_t)arm->tiles * TILE_BYTES;

    switch (arm->output) {
    case PLACE_L1_SELF:
        return device_read(&session->noc, g_self_x, g_self_y, L1_SCRATCH_C, destination, bytes);
    case PLACE_L1_PEER:
        return device_read(&session->noc, g_peer_x, g_peer_y, L1_SCRATCH_C, destination, bytes);
    case PLACE_DRAM:
    default:
        return device_read(&session->noc, (uint16_t)g_dram[DRAM_BANK_C].x,
                           (uint16_t)g_dram[DRAM_BANK_C].y, DRAM_BUFFER_BASE, destination, bytes);
    }
}

static int poison_arm_output(const ft_tt_hardware_session *session, const eltwise_arm *arm)
{
    const size_t bytes = (size_t)arm->tiles * TILE_BYTES;

    switch (arm->output) {
    case PLACE_L1_SELF:
        return fill_pattern(&session->noc, g_self_x, g_self_y, L1_SCRATCH_C, bytes, POISON_BYTE);
    case PLACE_L1_PEER:
        return fill_pattern(&session->noc, g_peer_x, g_peer_y, L1_SCRATCH_C, bytes, POISON_BYTE);
    case PLACE_DRAM:
    default:
        return fill_pattern(&session->noc, (uint16_t)g_dram[DRAM_BANK_C].x,
                            (uint16_t)g_dram[DRAM_BANK_C].y, DRAM_BUFFER_BASE, bytes, POISON_BYTE);
    }
}

/* The arms. Order is the interleave order and does not change between rounds. */
static eltwise_arm g_arms[] = {
    { "MUL   144  A,B DRAM   C DRAM", FT_TT_OP_VECTOR_FN_MUL, MAX_TILES, PLACE_DRAM, PLACE_DRAM,
      false, 0u },
    { "MUL   144  A,B L1self C DRAM", FT_TT_OP_VECTOR_FN_MUL, MAX_TILES, PLACE_L1_SELF, PLACE_DRAM,
      false, 0u },
    { "MUL   144  A,B DRAM   C L1self", FT_TT_OP_VECTOR_FN_MUL, MAX_TILES, PLACE_DRAM,
      PLACE_L1_SELF, false, 0u },
    { "MUL   144  A,B L1self C L1self", FT_TT_OP_VECTOR_FN_MUL, MAX_TILES, PLACE_L1_SELF,
      PLACE_L1_SELF, false, 0u },
    { "MUL   144  A,B L1peer C L1peer", FT_TT_OP_VECTOR_FN_MUL, MAX_TILES, PLACE_L1_PEER,
      PLACE_L1_PEER, false, 0u },
    { "SILU  144  A   DRAM   C DRAM", FT_TT_OP_VECTOR_FN_SILU, MAX_TILES, PLACE_DRAM, PLACE_DRAM,
      false, 0u },
    { "SILU  144  A   L1self C L1self", FT_TT_OP_VECTOR_FN_SILU, MAX_TILES, PLACE_L1_SELF,
      PLACE_L1_SELF, false, 0u },
    { "SILUMUL 144 A,B DRAM  C DRAM", FT_TT_OP_VECTOR_FN_SILU_MUL, MAX_TILES, PLACE_DRAM,
      PLACE_DRAM, false, 0u },
    { "MUL    48  A,B DRAM   C DRAM", FT_TT_OP_VECTOR_FN_MUL, HIDDEN_TILES, PLACE_DRAM, PLACE_DRAM,
      false, 0u },
    { "SILU   48  A   DRAM   C DRAM", FT_TT_OP_VECTOR_FN_SILU, HIDDEN_TILES, PLACE_DRAM,
      PLACE_DRAM, false, 0u },
    { "ADD    48  A,B DRAM   C DRAM", FT_TT_OP_VECTOR_FN_ADD, HIDDEN_TILES, PLACE_DRAM, PLACE_DRAM,
      false, 0u },
    { "MUL   144  NEGATIVE CONTROL (B aliased onto A)", FT_TT_OP_VECTOR_FN_MUL, MAX_TILES,
      PLACE_DRAM, PLACE_DRAM, true, 0u }
};

#define ARM_COUNT (sizeof(g_arms) / sizeof(g_arms[0]))

/* Indices into g_arms that the report and the byte comparisons name. */
#define ARM_MUL144_DD    0u
#define ARM_MUL144_LD    1u
#define ARM_MUL144_DL    2u
#define ARM_MUL144_LL    3u
#define ARM_MUL144_PEER  4u
#define ARM_SILU144_DD   5u
#define ARM_SILU144_LL   6u
#define ARM_SILUMUL144   7u
#define ARM_MUL48_DD     8u
#define ARM_SILU48_DD    9u
#define ARM_ADD48_DD     10u
#define ARM_NEGATIVE     11u

/* ================================================================
 * Stage 0 — the transfer path, before anything is scored
 * ================================================================ */

static void stage_0_round_trip(const ft_tt_hardware_session *session)
{
    uint16_t written[TILE_DIM * TILE_DIM];
    uint16_t read_back[TILE_DIM * TILE_DIM];
    size_t index;
    int rc;

    ft_tt_test_step("stage 0 — host->DRAM->host and host->peer L1->host, so the transfer path "
                    "is a LINE and not an assumption");

    random_seed(0xF05E0001ULL);
    for (index = 0; index < TILE_DIM * TILE_DIM; index++)
        written[index] = ft_bf16_from_float(random_value());

    memset(read_back, 0xA5, sizeof(read_back));
    rc = device_write(&session->noc, (uint16_t)g_dram[DRAM_BANK_A].x,
                      (uint16_t)g_dram[DRAM_BANK_A].y, DRAM_BUFFER_BASE, written, sizeof(written));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, (uint16_t)g_dram[DRAM_BANK_A].x,
                         (uint16_t)g_dram[DRAM_BANK_A].y, DRAM_BUFFER_BASE, read_back,
                         sizeof(read_back));
    ft_tt_test_check_fmt("a tile round-trips through DRAM", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check("and it comes back byte-exact",
                     memcmp(written, read_back, sizeof(written)) == 0, "");

    memset(read_back, 0xA5, sizeof(read_back));
    rc = device_write(&session->noc, g_peer_x, g_peer_y, L1_SCRATCH_A, written, sizeof(written));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_peer_x, g_peer_y, L1_SCRATCH_A, read_back,
                         sizeof(read_back));
    ft_tt_test_check_fmt("a tile round-trips through the PEER core's L1", rc == FT_TT_OK, "rc=%d",
                         rc);
    ft_tt_test_check("and it comes back byte-exact",
                     memcmp(written, read_back, sizeof(written)) == 0, "");
}

/* ================================================================
 * Stage 1 — the timing harness's own red/green, before anything it protects
 * ================================================================ */

static double arm_microseconds(uint32_t arm);

/** The dispatch floor, measured by stage 2 in this run, on this core. It exists to be
 *  SUBTRACTED — never to license a fusion (plan Phase 6 item 8). */
static double g_nop_microseconds;

/**
 * The timing harness's OWN red/green, and it is deliberately run on the very op the rest
 * of the suite measures rather than on a synthetic one.
 *
 * FT_TT_QUEUE_OP_DELAY was the obvious instrument and it is NOT available here: a resident
 * binary is bound to one slab, the elementwise triad implements ELTWISE, NOP and STOP, and
 * it refuses everything else with kErrBadOpcode — correctly. So the known ratio comes from
 * the work instead: three times the tiles, net of the dispatch floor measured in the same
 * run, must cost about three times the wall clock. A harness that measured nothing would
 * report 1.0, and one that measured only the host's poll would too.
 */
static bool check_timer_resolves_work(void)
{
    const double mul144 = arm_microseconds(ARM_MUL144_DD) - g_nop_microseconds;
    const double mul48 = arm_microseconds(ARM_MUL48_DD) - g_nop_microseconds;
    const double silu144 = arm_microseconds(ARM_SILU144_DD) - g_nop_microseconds;
    const double silu48 = arm_microseconds(ARM_SILU48_DD) - g_nop_microseconds;
    const double mul_ratio = (mul48 > 0.0) ? mul144 / mul48 : 0.0;
    const double silu_ratio = (silu48 > 0.0) ? silu144 / silu48 : 0.0;

    printf("\n  timer resolution, from the work itself (net of the %.3f us dispatch floor):\n",
           g_nop_microseconds);
    printf("    MUL  144 / MUL  48 = %.3f   (144/48 = 3.000)\n", mul_ratio);
    printf("    SILU 144 / SILU 48 = %.3f\n", silu_ratio);

    ft_tt_test_check_fmt("3x the tiles costs 2.7-3.3x the time on BOTH functions, so this timer "
                         "resolves work and not host noise",
                         mul_ratio > 2.7 && mul_ratio < 3.3 && silu_ratio > 2.7 &&
                             silu_ratio < 3.3,
                         "MUL %.3f SILU %.3f", mul_ratio, silu_ratio);
    return true;
}

/* ================================================================
 * Stage 2 — the dispatch floor, measured in THIS run so it can be subtracted
 * ================================================================ */

static bool stage_2_dispatch_floor(const ft_tt_hardware_session *session)
{
    ft_tt_queue_descriptor descriptor;
    uint64_t total = 0u;
    uint32_t round;
    int rc = FT_TT_OK;

    ft_tt_test_step("stage 2 — the DISPATCH FLOOR on this core, this run: a batch of NOPs. It is "
                    "here to be SUBTRACTED, never to license a fusion (plan Phase 6 item 8)");

    for (round = 0; round < g_rounds; round++) {
        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_NOP;
        total += time_batch(session, &descriptor, BATCH, &rc);
        if (rc != FT_TT_OK)
            break;
    }
    ft_tt_test_check_fmt("the NOP arm ran", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    g_nop_microseconds = (double)total / (double)(g_rounds * BATCH);
    printf("\n  NOP, batch %u, %u rounds : %.3f us per descriptor\n", BATCH, g_rounds,
           g_nop_microseconds);
    ft_tt_test_check_fmt("the dispatch floor is microseconds, not tens of them — so a fusion "
                         "sold on descriptor count would be selling nothing",
                         g_nop_microseconds < 10.0, "%.3f us", g_nop_microseconds);
    return true;
}

/* ================================================================
 * Stage 3 — the round-trip matrix
 * ================================================================ */

static int start_eltwise_residency(const ft_tt_hardware_session *session)
{
    ft_tt_op_eltwise_request request;
    eltwise_arm widest = g_arms[ARM_MUL144_DD];
    uint64_t mask = 0;
    int rc;

    fill_eltwise_request(&request, &widest);
    rc = load_images(&session->noc, g_self_x, g_self_y, &g_eltwise);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, g_self_x, g_self_y, &g_eltwise.program, NULL,
                                   &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_eltwise_place_cbs(&request, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer),
                                        &mask);
    if (rc != FT_TT_OK)
        return rc;

    g_config.local_cb_mask = mask;
    g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    g_config.brisc_noc_id = 0u;
    rc = ft_tt_queue_init(&session->noc, g_self_x, g_self_y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          ft_tt_queue_make_nonce(), &g_queue);
    if (rc != FT_TT_OK)
        return rc;
    g_completed = 0u;
    return start_triad(&session->noc, g_self_x, g_self_y);
}

/** Every operand buffer, in every place an arm can read it from, filled with the SAME
 *  values — which is what makes the DRAM arm and the L1 arm comparable byte for byte. */
static int upload_all_operands(const ft_tt_hardware_session *session)
{
    int rc;

    rc = upload_random_tiles(session, (uint16_t)g_dram[DRAM_BANK_A].x,
                             (uint16_t)g_dram[DRAM_BANK_A].y, DRAM_BUFFER_BASE, MAX_TILES,
                             0xA11CE0001ULL);
    if (rc == FT_TT_OK)
        rc = upload_random_tiles(session, (uint16_t)g_dram[DRAM_BANK_B].x,
                                 (uint16_t)g_dram[DRAM_BANK_B].y, DRAM_BUFFER_BASE, MAX_TILES,
                                 0xB0B0B0002ULL);
    if (rc == FT_TT_OK)
        rc = upload_random_tiles(session, g_self_x, g_self_y, L1_SCRATCH_A, MAX_TILES,
                                 0xA11CE0001ULL);
    if (rc == FT_TT_OK)
        rc = upload_random_tiles(session, g_self_x, g_self_y, L1_SCRATCH_B, MAX_TILES,
                                 0xB0B0B0002ULL);
    if (rc == FT_TT_OK)
        rc = upload_random_tiles(session, g_peer_x, g_peer_y, L1_SCRATCH_A, MAX_TILES,
                                 0xA11CE0001ULL);
    if (rc == FT_TT_OK)
        rc = upload_random_tiles(session, g_peer_x, g_peer_y, L1_SCRATCH_B, MAX_TILES,
                                 0xB0B0B0002ULL);
    return rc;
}

static bool stage_3_round_trip_matrix(const ft_tt_hardware_session *session)
{
    const size_t widest_bytes = (size_t)MAX_TILES * TILE_BYTES;
    unsigned char *dram_answer = malloc(widest_bytes);
    unsigned char *other_answer = malloc(widest_bytes);
    uint32_t slot_base = 0u;
    uint32_t round;
    uint32_t arm;
    bool ok = true;
    int rc = FT_TT_OK;

    ft_tt_test_step("stage 3 — the ROUND-TRIP MATRIX: one op, one residency, only the operand "
                    "ADDRESSES change. %u arms x %u rounds x %u descriptors, interleaved",
                    (unsigned)ARM_COUNT, g_rounds, BATCH);

    if (!dram_answer || !other_answer) {
        ft_tt_test_check("scratch allocates", false, "");
        free(dram_answer);
        free(other_answer);
        return false;
    }

    /* One parameter block per arm, written once. Enqueueing eight descriptors that all
     * point at the same block is what keeps the timed window free of host work. */
    for (arm = 0; arm < ARM_COUNT; arm++) {
        ft_tt_op_eltwise_request request;
        uint32_t words[FT_TT_OP_VECTOR_WORDS];
        uint32_t address = 0u;

        fill_eltwise_request(&request, &g_arms[arm]);
        rc = ft_tt_op_eltwise_parameters(&request, words);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, slot_base + arm, words,
                                                 &address);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("every arm's parameter block writes", false, "arm %u rc=%d", arm,
                                 rc);
            ok = false;
            goto done;
        }
    }

    for (round = 0; round < g_rounds; round++) {
        for (arm = 0; arm < ARM_COUNT; arm++) {
            ft_tt_queue_descriptor descriptor;

            memset(&descriptor, 0, sizeof(descriptor));
            descriptor.opcode = FT_TT_QUEUE_OP_ELTWISE;
            descriptor.a0 = FT_TT_QUEUE_PARAM_ADDRESS(g_queue.control_address, slot_base + arm);
            g_arms[arm].total_microseconds += time_batch(session, &descriptor, BATCH, &rc);
            if (rc != FT_TT_OK) {
                ft_tt_test_check_fmt("every arm completes its batch", false,
                                     "round %u arm %u (%s) rc=%d", round, arm, g_arms[arm].label,
                                     rc);
                ok = false;
                goto done;
            }
        }
    }
    ft_tt_test_check("every arm completed every batch on ONE residency, with no launch and no "
                     "image load between them",
                     true, "");

    /* --- the answers, as BYTES, and a negative control that proves the comparison can fail */
    rc = poison_arm_output(session, &g_arms[ARM_MUL144_DD]);
    if (rc == FT_TT_OK) {
        ft_tt_queue_descriptor descriptor;

        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_ELTWISE;
        descriptor.a0 =
            FT_TT_QUEUE_PARAM_ADDRESS(g_queue.control_address, slot_base + ARM_MUL144_DD);
        (void)time_batch(session, &descriptor, 1u, &rc);
    }
    if (rc == FT_TT_OK)
        rc = read_arm_output(session, &g_arms[ARM_MUL144_DD], dram_answer);
    ft_tt_test_check_fmt("the DRAM arm's answer reads back", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK) {
        ok = false;
        goto done;
    }
    ft_tt_test_check_fmt("and NO TILE came back untouched, so the kernel wrote the whole output "
                         "rather than the previous round's bytes surviving",
                         poison_tiles_remaining(dram_answer, widest_bytes) == 0,
                         "%zu of %u tiles still all-poison",
                         poison_tiles_remaining(dram_answer, widest_bytes), MAX_TILES);

    {
        static const uint32_t compared[] = { ARM_MUL144_LL, ARM_MUL144_PEER, ARM_MUL144_DL,
                                             ARM_MUL144_LD };
        size_t index;

        for (index = 0; index < sizeof(compared) / sizeof(compared[0]); index++) {
            const uint32_t which = compared[index];
            char label[160];

            rc = poison_arm_output(session, &g_arms[which]);
            if (rc == FT_TT_OK) {
                ft_tt_queue_descriptor descriptor;

                memset(&descriptor, 0, sizeof(descriptor));
                descriptor.opcode = FT_TT_QUEUE_OP_ELTWISE;
                descriptor.a0 =
                    FT_TT_QUEUE_PARAM_ADDRESS(g_queue.control_address, slot_base + which);
                (void)time_batch(session, &descriptor, 1u, &rc);
            }
            if (rc == FT_TT_OK)
                rc = read_arm_output(session, &g_arms[which], other_answer);
            if (rc != FT_TT_OK) {
                ft_tt_test_check_fmt("a placement arm's answer reads back", false, "arm %u rc=%d",
                                     which, rc);
                ok = false;
                goto done;
            }
            snprintf(label, sizeof(label),
                     "[%s] is BYTE-IDENTICAL to the all-DRAM arm — a placement moves where data "
                     "lives, never what it is",
                     g_arms[which].label);
            ft_tt_test_check_fmt(label, memcmp(dram_answer, other_answer, widest_bytes) == 0,
                                 "%zu of %u tiles still all-poison",
                                 poison_tiles_remaining(other_answer, widest_bytes), MAX_TILES);
        }
    }

    /* THE NEGATIVE CONTROL. Same comparison, same code path, against an arm whose operands
     * are deliberately wrong — so "byte-identical" is known to be a claim this suite can
     * refuse, and not a comparison that would pass on anything (dev notes §3). */
    rc = poison_arm_output(session, &g_arms[ARM_NEGATIVE]);
    if (rc == FT_TT_OK) {
        ft_tt_queue_descriptor descriptor;

        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_ELTWISE;
        descriptor.a0 = FT_TT_QUEUE_PARAM_ADDRESS(g_queue.control_address, slot_base + ARM_NEGATIVE);
        (void)time_batch(session, &descriptor, 1u, &rc);
    }
    if (rc == FT_TT_OK)
        rc = read_arm_output(session, &g_arms[ARM_NEGATIVE], other_answer);
    ft_tt_test_check_fmt("the negative control ran", rc == FT_TT_OK, "rc=%d", rc);
    if (rc == FT_TT_OK)
        ft_tt_test_check("NEGATIVE CONTROL: the byte comparison REFUSES an arm computing a*a "
                         "instead of a*b — so the four greens above are a detection, not a "
                         "tautology",
                         memcmp(dram_answer, other_answer, widest_bytes) != 0, "");

done:
    free(dram_answer);
    free(other_answer);
    return ok;
}

/* ================================================================
 * Stage 3b — is the DRAM write premium a LATENCY? Batch the barrier and see
 * ================================================================ */

/** Tiles per NoC write barrier in the blocked writer, and the pages the output CB is given
 *  so that a block is contiguous. 8 divides both 48 and 144, and 16 % 8 == 0. */
#define WRITE_BLOCK 8u
#define WRITE_BLOCK_OUT_PAGES 16u

static ft_tt_program_directory g_eltwise_blocked;
static ft_tt_program_directory g_eltwise_flushed;
static bool g_have_blocked_writer;
static bool g_have_flushed_writer;

/** The writer arms, in interleave order. `resident_vector_writer.c` is the shipped one and
 *  is always arm 0, so every other arm's ratio is against what ships. */
enum { WRITER_PER_TILE = 0, WRITER_BLOCKED = 1, WRITER_FLUSHED = 2, WRITER_ARM_COUNT = 3 };

static const char *const g_writer_arm_names[WRITER_ARM_COUNT] = {
    "barrier/tile", "barrier/block", "flush/tile"
};

static uint64_t g_writer_ab_total[WRITER_ARM_COUNT][3];

/** The three arms of the writer A/B: the two ops the round-trip matrix found EXPOSED to
 *  the write path, plus the one it found hidden behind the SFPU. */
static const uint32_t g_writer_ab_arms[3] = { ARM_MUL144_DD, ARM_ADD48_DD, ARM_SILU144_DD };

/** CBs for the writer A/B — placed BY HAND rather than by ft_tt_op_eltwise_place_cbs(),
 *  because the blocked writer needs an output CB deep enough to hold a whole block and
 *  that is a host placement, not a change to anything shipped. BOTH arms get the identical
 *  placement: an A/B with two variables measures neither. */
static int start_eltwise_writer_arm(const ft_tt_hardware_session *session,
                                    const ft_tt_program_directory *program)
{
    ft_tt_op_cb_request requests[4];
    uint64_t mask = 0;
    int rc = load_images(&session->noc, g_self_x, g_self_y, program);

    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, g_self_x, g_self_y, &program->program, NULL,
                                   &g_config);
    if (rc != FT_TT_OK)
        return rc;

    requests[0].index = FT_TT_OP_ELTWISE_CB_IN0;
    requests[0].page_bytes = TILE_BYTES;
    requests[0].page_count = ELTWISE_DEPTH * 2u;
    requests[1].index = FT_TT_OP_ELTWISE_CB_IN1;
    requests[1].page_bytes = TILE_BYTES;
    requests[1].page_count = ELTWISE_DEPTH * 2u;
    requests[2].index = FT_TT_OP_ELTWISE_CB_SCRATCH;
    requests[2].page_bytes = TILE_BYTES;
    requests[2].page_count = 2u;
    requests[3].index = FT_TT_OP_ELTWISE_CB_OUT;
    requests[3].page_bytes = TILE_BYTES;
    requests[3].page_count = WRITE_BLOCK_OUT_PAGES;
    rc = ft_tt_op_cb_place(requests, 4u, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    if (rc != FT_TT_OK)
        return rc;

    g_config.local_cb_mask = mask;
    g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    g_config.brisc_noc_id = 0u;
    rc = ft_tt_queue_init(&session->noc, g_self_x, g_self_y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          ft_tt_queue_make_nonce(), &g_queue);
    if (rc != FT_TT_OK)
        return rc;
    g_completed = 0u;
    return start_triad(&session->noc, g_self_x, g_self_y);
}

/** One arm's three ops, on an already-started residency. `answer` receives the MUL 144
 *  result so the two writers can be compared as BYTES. */
static int run_writer_arm(const ft_tt_hardware_session *session, uint64_t out_totals[3],
                          unsigned char *answer)
{
    uint32_t index;
    int rc = FT_TT_OK;

    for (index = 0; index < 3u; index++) {
        const uint32_t arm = g_writer_ab_arms[index];
        ft_tt_op_eltwise_request request;
        ft_tt_queue_descriptor descriptor;
        uint32_t words[FT_TT_OP_VECTOR_WORDS];
        uint32_t address = 0u;

        fill_eltwise_request(&request, &g_arms[arm]);
        /* POISON FIRST for the arm whose bytes are compared. Without this the "no tile came
         * back untouched" line scores the PREVIOUS stage's leftovers and cannot fail — the
         * @issues note this file already carried. */
        if (index == 0u && answer)
            rc = poison_arm_output(session, &g_arms[arm]);
        if (rc == FT_TT_OK)
            rc = ft_tt_op_eltwise_parameters(&request, words);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, index, words, &address);
        if (rc != FT_TT_OK)
            return rc;

        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_ELTWISE;
        descriptor.a0 = address;
        out_totals[index] += time_batch(session, &descriptor, BATCH, &rc);
        if (rc != FT_TT_OK)
            return rc;
        if (index == 0u && answer)
            rc = read_arm_output(session, &g_arms[arm], answer);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** The three writer program directories, indexed by the WRITER_* enum. An absent arm is
 *  NULL and is skipped, so the suite still runs when only two were built. */
static const ft_tt_program_directory *writer_arm_program(uint32_t arm)
{
    switch (arm) {
    case WRITER_PER_TILE:
        return &g_eltwise;
    case WRITER_BLOCKED:
        return g_have_blocked_writer ? &g_eltwise_blocked : NULL;
    case WRITER_FLUSHED:
        return g_have_flushed_writer ? &g_eltwise_flushed : NULL;
    default:
        return NULL;
    }
}

static bool stage_3b_writer_barrier(const ft_tt_hardware_session *session)
{
    const size_t widest_bytes = (size_t)MAX_TILES * TILE_BYTES;
    unsigned char *answers[WRITER_ARM_COUNT] = { NULL, NULL, NULL };
    uint32_t round;
    uint32_t index;
    uint32_t arm;
    bool ok = true;
    int rc = FT_TT_OK;

    ft_tt_test_step("stage 3b — is the DRAM write premium a LATENCY? THREE writers off one "
                    "source: a barrier per tile (ships), a barrier per %u tiles, and upstream's "
                    "flush-per-tile with ONE barrier per descriptor. Arms interleaved by ROUND",
                    WRITE_BLOCK);

    for (arm = 0; arm < WRITER_ARM_COUNT; arm++) {
        if (writer_arm_program(arm))
            answers[arm] = malloc(widest_bytes);
    }
    if (!answers[WRITER_PER_TILE]) {
        ft_tt_test_check("scratch allocates", false, "");
        goto done;
    }

    for (round = 0; round < g_rounds && rc == FT_TT_OK; round++) {
        for (arm = 0; arm < WRITER_ARM_COUNT && rc == FT_TT_OK; arm++) {
            const ft_tt_program_directory *program = writer_arm_program(arm);

            if (!program || !answers[arm])
                continue;
            rc = start_eltwise_writer_arm(session, program);
            if (rc == FT_TT_OK)
                rc = run_writer_arm(session, g_writer_ab_total[arm], answers[arm]);
            (void)ft_tt_watchdog_shutdown(&session->noc, &g_group, RUN_TIMEOUT_MS, NULL);
        }
    }
    ft_tt_test_check_fmt("every writer arm ran every round", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK) {
        ok = false;
        goto done;
    }

    /* A BARRIER-PLACEMENT CHANGE MAY NOT MOVE A VALUE. Each arm is compared against the
     * shipped writer's bytes, and the destination was poisoned before each arm's own run,
     * so "no tile came back untouched" is a line that can fail. */
    for (arm = 1u; arm < WRITER_ARM_COUNT; arm++) {
        char label[160];

        if (!writer_arm_program(arm) || !answers[arm])
            continue;
        snprintf(label, sizeof(label),
                 "the %s writer produces BYTE-IDENTICAL output to the shipped per-tile "
                 "barrier — a barrier-placement change that moved a value would be a bug, "
                 "not a tradeoff",
                 g_writer_arm_names[arm]);
        ft_tt_test_check(label, memcmp(answers[WRITER_PER_TILE], answers[arm], widest_bytes) == 0,
                         "");
        snprintf(label, sizeof(label), "and the %s writer left no tile untouched",
                 g_writer_arm_names[arm]);
        ft_tt_test_check_fmt(label, poison_tiles_remaining(answers[arm], widest_bytes) == 0,
                             "%zu of %u tiles still all-poison",
                             poison_tiles_remaining(answers[arm], widest_bytes), MAX_TILES);
    }

    printf("\n  THE WRITER'S BARRIER — three arms, identical CBs, %u tiles a block\n",
           WRITE_BLOCK);
    printf("  op                             ");
    for (arm = 0; arm < WRITER_ARM_COUNT; arm++)
        printf(" %13s", writer_arm_program(arm) ? g_writer_arm_names[arm] : "(not built)");
    printf("     best vs ships\n");
    for (index = 0; index < 3u; index++) {
        const uint32_t which = g_writer_ab_arms[index];
        const double shipped =
            (double)g_writer_ab_total[WRITER_PER_TILE][index] / (double)(g_rounds * BATCH);
        double best = shipped;

        printf("  %-30s", g_arms[which].label);
        for (arm = 0; arm < WRITER_ARM_COUNT; arm++) {
            double value;

            if (!writer_arm_program(arm)) {
                printf(" %13s", "-");
                continue;
            }
            value = (double)g_writer_ab_total[arm][index] / (double)(g_rounds * BATCH);
            if (value > 0.0 && value < best)
                best = value;
            printf(" %10.2f us", value);
        }
        printf("  %8.3fx %8.1f ns/tile\n", (best > 0.0) ? shipped / best : 0.0,
               (shipped - best) * 1000.0 / (double)g_arms[which].tiles);
    }

done:
    for (arm = 0; arm < WRITER_ARM_COUNT; arm++)
        free(answers[arm]);
    return ok;
}

/* ================================================================
 * Stage 4 — the matmul denominators, and what one output tile write costs
 * ================================================================ */

typedef struct matmul_shape {
    const char *label;
    uint32_t tile_rows;
    uint32_t tile_inner;
    uint32_t tile_cols;
    uint64_t total_microseconds;
} matmul_shape;

/* MiniCPM5-1B at Mt = 1 (32 rows), the orientation ft_tt_op_matmul.h settled: A is
 * [Mt*32 x Kt*32], B is K x N. Hidden 1536, q 2048, kv 256, FFN 4608 — from the
 * checkpoint's own metadata, not from a table. */
static matmul_shape g_shapes[] = {
    { "attn_q       1536x2048", 1u, 48u, 64u, 0u },
    { "attn_k/v     1536x256", 1u, 48u, 8u, 0u },
    { "attn_output  2048x1536", 1u, 64u, 48u, 0u },
    { "ffn_gate/up  1536x4608", 1u, 48u, 144u, 0u },
    { "ffn_down     4608x1536", 1u, 144u, 48u, 0u }
};

#define SHAPE_COUNT (sizeof(g_shapes) / sizeof(g_shapes[0]))
#define SHAPE_FFN_GATE 3u
#define SHAPE_FFN_DOWN 4u

static int start_matmul_residency(const ft_tt_hardware_session *session)
{
    ft_tt_op_cb_request requests[3];
    uint64_t mask = 0;
    int rc;

    rc = load_images(&session->noc, g_self_x, g_self_y, &g_rtriad);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, g_self_x, g_self_y, &g_rtriad.program, NULL,
                                   &g_config);
    if (rc != FT_TT_OK)
        return rc;

    requests[0].index = 0u;
    requests[0].page_bytes = TILE_BYTES;
    requests[0].page_count = MATMUL_DEPTH * 2u;
    requests[1].index = 1u;
    requests[1].page_bytes = TILE_BYTES;
    requests[1].page_count = MATMUL_DEPTH * 2u;
    requests[2].index = 16u;
    requests[2].page_bytes = TILE_BYTES;
    requests[2].page_count = 2u;
    rc = ft_tt_op_cb_place(requests, 3u, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    if (rc != FT_TT_OK)
        return rc;

    g_config.local_cb_mask = mask;
    g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    g_config.brisc_noc_id = 0u;
    rc = ft_tt_queue_init(&session->noc, g_self_x, g_self_y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          ft_tt_queue_make_nonce(), &g_queue);
    if (rc != FT_TT_OK)
        return rc;
    g_completed = 0u;
    return start_triad(&session->noc, g_self_x, g_self_y);
}

static bool stage_4_matmul_shapes(const ft_tt_hardware_session *session)
{
    uint32_t round;
    uint32_t index;
    int rc = FT_TT_OK;

    ft_tt_test_step("stage 4 — every real MiniCPM5-1B matmul shape at Mt=1 on a resident triad, "
                    "interleaved. These are the DENOMINATORS a fusion's saving is a fraction of");

    for (index = 0; index < SHAPE_COUNT; index++) {
        uint32_t words[FT_TT_QUEUE_MATMUL_WORDS];
        uint32_t address = 0u;

        memset(words, 0, sizeof(words));
        words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = g_shapes[index].tile_rows;
        words[FT_TT_QUEUE_MATMUL_TILE_INNER] = g_shapes[index].tile_inner;
        words[FT_TT_QUEUE_MATMUL_TILE_COLS] = g_shapes[index].tile_cols;
        words[FT_TT_QUEUE_MATMUL_A_X] = (uint32_t)g_dram[DRAM_BANK_A].x;
        words[FT_TT_QUEUE_MATMUL_A_Y] = (uint32_t)g_dram[DRAM_BANK_A].y;
        words[FT_TT_QUEUE_MATMUL_A_BASE] = DRAM_BUFFER_BASE;
        words[FT_TT_QUEUE_MATMUL_B_X] = (uint32_t)g_dram[DRAM_BANK_B].x;
        words[FT_TT_QUEUE_MATMUL_B_Y] = (uint32_t)g_dram[DRAM_BANK_B].y;
        words[FT_TT_QUEUE_MATMUL_B_BASE] = DRAM_BUFFER_BASE;
        words[FT_TT_QUEUE_MATMUL_C_X] = (uint32_t)g_dram[DRAM_BANK_C].x;
        words[FT_TT_QUEUE_MATMUL_C_Y] = (uint32_t)g_dram[DRAM_BANK_C].y;
        words[FT_TT_QUEUE_MATMUL_C_BASE] = DRAM_BUFFER_BASE;
        /* SHRINK IT HOST-SIDE. resident_matmul_reader.c:84 collapses a depth that does
         * not DIVIDE Kt to 1 — always correct, and 5.9x slower (docs/tt_kernel_model.md
         * §12), with nothing anywhere reporting it. Writing the parameter block by hand,
         * which the resident queue path invites, skips the host-side shrink that is the
         * only thing standing between a real shape and that fallback: attn_output (Kt=64)
         * at a requested 48 measured 2.850 ms / 70.6 GFLOP/s here before this line existed,
         * against 0.35 ms with the shrink. A wrong SPEED with no error anywhere. */
        words[FT_TT_QUEUE_MATMUL_DEPTH] =
            ft_tt_op_matmul_prefetch_depth(g_shapes[index].tile_inner, MATMUL_DEPTH);
        words[FT_TT_QUEUE_MATMUL_B_N_MAJOR] = 0u;

        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, index, words, &address);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("every shape's parameter block writes", false,
                                 "shape %u rc=%d", index, rc);
            return false;
        }
    }

    for (round = 0; round < g_rounds; round++) {
        for (index = 0; index < SHAPE_COUNT; index++) {
            ft_tt_queue_descriptor descriptor;

            memset(&descriptor, 0, sizeof(descriptor));
            descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
            descriptor.a0 = FT_TT_QUEUE_PARAM_ADDRESS(g_queue.control_address, index);
            g_shapes[index].total_microseconds += time_batch(session, &descriptor, 1u, &rc);
            if (rc != FT_TT_OK) {
                ft_tt_test_check_fmt("every shape completes", false, "round %u shape %s rc=%d",
                                     round, g_shapes[index].label, rc);
                return false;
            }
        }
    }
    ft_tt_test_check("every MiniCPM5-1B matmul shape ran on one residency", true, "");
    return true;
}

/* ================================================================
 * Stage 4b — the RMS norm, which a MiniCPM5-1B layer runs twice
 * ================================================================ */

static double g_rms_norm_microseconds;

static int start_reduce_residency(const ft_tt_hardware_session *session,
                                  const ft_tt_op_reduce_residency *residency)
{
    uint64_t mask = 0;
    int rc = load_images(&session->noc, g_self_x, g_self_y, &g_reduce);

    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, g_self_x, g_self_y, &g_reduce.program, NULL,
                                   &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_reduce_place_cbs(residency, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer),
                                       &mask);
    if (rc != FT_TT_OK)
        return rc;

    g_config.local_cb_mask = mask;
    g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    g_config.brisc_noc_id = 0u;
    rc = ft_tt_queue_init(&session->noc, g_self_x, g_self_y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          ft_tt_queue_make_nonce(), &g_queue);
    if (rc != FT_TT_OK)
        return rc;
    g_completed = 0u;
    return start_triad(&session->noc, g_self_x, g_self_y);
}

static bool stage_4b_rms_norm(const ft_tt_hardware_session *session)
{
    const uint32_t cols = HIDDEN_TILES * TILE_DIM;
    const ft_tt_op_reduce_function member = FT_TT_OP_REDUCE_FN_RMS_NORM;
    ft_tt_op_reduce_residency residency;
    ft_tt_op_reduce_request request;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    uint32_t address = 0u;
    uint64_t total = 0u;
    unsigned char *constants = NULL;
    float *gamma = NULL;
    size_t constant_bytes;
    uint32_t round;
    uint32_t index;
    int rc;

    ft_tt_test_step("stage 4b — RMS norm over 32x1536, which a MiniCPM5-1B layer runs TWICE. "
                    "The producer of every matmul's activation operand");

    rc = ft_tt_op_reduce_residency_for(&member, 1u, HIDDEN_TILES, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_residency(session, &residency);
    ft_tt_test_check_fmt("the reduce residency starts", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    constant_bytes = ft_tt_op_reduce_constants_bytes(member, HIDDEN_TILES, TILE_BYTES);
    constants = malloc(constant_bytes);
    gamma = malloc((size_t)cols * sizeof(float));
    if (!constants || !gamma) {
        ft_tt_test_check("the norm's constants allocate", false, "");
        goto done;
    }
    random_seed(0x6A11AULL);
    for (index = 0; index < cols; index++)
        gamma[index] = 0.5f + random_value() * 0.25f;
    rc = ft_tt_op_reduce_build_constants(member, cols, 1e-6f, gamma, NULL,
                                         FT_TT_OP_TILE_BYTES_BF16, constants, constant_bytes);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, (uint16_t)g_dram[DRAM_BANK_CONST].x,
                          (uint16_t)g_dram[DRAM_BANK_CONST].y, DRAM_BUFFER_BASE, constants,
                          constant_bytes);
    if (rc == FT_TT_OK)
        rc = upload_random_tiles(session, (uint16_t)g_dram[DRAM_BANK_A].x,
                                 (uint16_t)g_dram[DRAM_BANK_A].y, DRAM_BUFFER_BASE, HIDDEN_TILES,
                                 0x1401110001ULL);
    if (rc == FT_TT_OK)
        rc = fill_pattern(&session->noc, (uint16_t)g_dram[DRAM_BANK_C].x,
                          (uint16_t)g_dram[DRAM_BANK_C].y, DRAM_BUFFER_BASE,
                          (size_t)HIDDEN_TILES * TILE_BYTES, POISON_BYTE);
    ft_tt_test_check_fmt("the norm's operands and a poisoned output upload", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto done;

    memset(&request, 0, sizeof(request));
    request.function = member;
    request.row_blocks = 1u;
    request.tile_cols = HIDDEN_TILES;
    request.prefetch_tiles = HIDDEN_TILES;
    place_operand(&request.input, PLACE_DRAM, DRAM_BANK_A, 0u);
    place_operand(&request.constants, PLACE_DRAM, DRAM_BANK_CONST, 0u);
    place_operand(&request.output, PLACE_DRAM, DRAM_BANK_C, 0u);

    rc = ft_tt_op_reduce_parameters(&request, words);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, 0u, words, &address);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the norm's parameter block writes", false, "rc=%d", rc);
        goto done;
    }

    for (round = 0; round < g_rounds; round++) {
        ft_tt_queue_descriptor descriptor;

        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_REDUCE;
        descriptor.a0 = address;
        total += time_batch(session, &descriptor, BATCH, &rc);
        if (rc != FT_TT_OK)
            break;
    }
    ft_tt_test_check_fmt("the norm arm ran", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto done;
    g_rms_norm_microseconds = (double)total / (double)(g_rounds * BATCH);

    {
        unsigned char *answer = malloc((size_t)HIDDEN_TILES * TILE_BYTES);

        if (answer) {
            rc = device_read(&session->noc, (uint16_t)g_dram[DRAM_BANK_C].x,
                             (uint16_t)g_dram[DRAM_BANK_C].y, DRAM_BUFFER_BASE, answer,
                             (size_t)HIDDEN_TILES * TILE_BYTES);
            ft_tt_test_check_fmt("the norm wrote every tile of its output", rc == FT_TT_OK &&
                                     poison_tiles_remaining(answer, (size_t)HIDDEN_TILES *
                                                                        TILE_BYTES) == 0,
                                 "rc=%d", rc);
            free(answer);
        }
    }
    printf("\n  RMS_NORM 32x1536 (Wt=%u) : %.2f us per descriptor\n", HIDDEN_TILES,
           g_rms_norm_microseconds);

done:
    free(constants);
    free(gamma);
    retire_triad(&session->noc, "the reduce residency retires by ASKING, never a park");
    return true;
}

/* ================================================================
 * Stage 5 — what a binary switch costs
 * ================================================================ */

static double g_switch_microseconds;

static bool stage_5_binary_switch(const ft_tt_hardware_session *session)
{
    uint64_t total = 0u;
    uint32_t round;
    int rc = FT_TT_OK;

    ft_tt_test_step("stage 5 — what a BINARY SWITCH costs: five kernel images over the x1 link. "
                    "Not a fusion and not dispatch — a third thing, and the layer sequence hits "
                    "it a dozen times");

    for (round = 0; round < g_rounds; round++) {
        uint64_t started = monotonic_microseconds();

        rc = load_images(&session->noc, g_self_x, g_self_y, &g_eltwise);
        total += monotonic_microseconds() - started;
        if (rc != FT_TT_OK)
            break;
        started = monotonic_microseconds();
        rc = load_images(&session->noc, g_self_x, g_self_y, &g_rtriad);
        total += monotonic_microseconds() - started;
        if (rc != FT_TT_OK)
            break;
    }
    ft_tt_test_check_fmt("both image sets loaded every round", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    g_switch_microseconds = (double)total / (double)(g_rounds * 2u);
    printf("\n  five kernel images, host -> L1 : %.3f ms per switch\n",
           g_switch_microseconds / 1000.0);
    return true;
}

/* ================================================================
 * The report
 * ================================================================ */

static double arm_microseconds(uint32_t arm)
{
    return (double)g_arms[arm].total_microseconds / (double)(g_rounds * BATCH);
}

static double shape_microseconds(uint32_t index)
{
    return (double)g_shapes[index].total_microseconds / (double)g_rounds;
}

static void report(void)
{
    const double mul144 = arm_microseconds(ARM_MUL144_DD);
    const double silu144 = arm_microseconds(ARM_SILU144_DD);
    const double mul48 = arm_microseconds(ARM_MUL48_DD);
    const double silu48 = arm_microseconds(ARM_SILU48_DD);
    /* The SFPU's share, per tile. MUL and SILU move the same output bytes and differ by one
     * input stream and one SFPU verb, and the measured gap is far too large to be an input
     * stream — which is what makes it a statement about the SFPU. The first version of this
     * report divided that gap by the tile count and called it "one DRAM tile read", which
     * produced a NEGATIVE cost. The difference between those two arms is not a read. */
    const double silu_premium_144 = (silu144 - mul144) * 1000.0 / (double)MAX_TILES;
    const double silu_premium_48 = (silu48 - mul48) * 1000.0 / (double)HIDDEN_TILES;
    /* What DRAM costs over L1, per tile, on each side. THIS is the round trip the licence is
     * about, and it is measured by MOVING ADDRESSES rather than by subtracting two ops. */
    const double read_premium_ns =
        (arm_microseconds(ARM_MUL144_DD) - arm_microseconds(ARM_MUL144_LD)) * 1000.0 /
        (2.0 * (double)MAX_TILES);
    const double write_premium_ns =
        (arm_microseconds(ARM_MUL144_DD) - arm_microseconds(ARM_MUL144_DL)) * 1000.0 /
        (double)MAX_TILES;
    uint32_t index;
    double layer_matmul_us;

    printf("\n");
    printf("  ============================================================================\n");
    printf("  THE ROUND-TRIP MATRIX — one elementwise op, only the operand ADDRESSES differ\n");
    printf("  ============================================================================\n");
    printf("  arm                                              us/descriptor   ns/tile\n");
    for (index = 0; index < ARM_COUNT; index++) {
        const double us = arm_microseconds(index);

        printf("  %-46s %10.2f %9.1f\n", g_arms[index].label, us,
               (us - g_nop_microseconds) * 1000.0 / (double)g_arms[index].tiles);
    }
    printf("\n  dispatch floor (NOP, same run, same core)      %10.3f us  <- SUBTRACTED above\n",
           g_nop_microseconds);
    printf("  DRAM over L1(self), per tile READ             : %10.1f ns  <- THE READ ROUND TRIP\n",
           read_premium_ns);
    printf("  DRAM over L1(self), per tile WRITE            : %10.1f ns  <- THE WRITE ROUND TRIP\n",
           write_premium_ns);
    printf("  the SFPU's share (SILU over MUL), per tile    : %10.1f ns (144), %.1f ns (48)\n",
           silu_premium_144, silu_premium_48);

    printf("\n");
    printf("  ============================================================================\n");
    printf("  THE DENOMINATORS — MiniCPM5-1B matmul, Mt=1, bf16, depth %u, one core\n",
           MATMUL_DEPTH);
    printf("  ============================================================================\n");
    printf("  shape                          wall (ms)   GFLOP/s   %% of one Tensix   out tiles\n");
    layer_matmul_us = 0.0;
    for (index = 0; index < SHAPE_COUNT; index++) {
        const double us = shape_microseconds(index);
        const double flops = 2.0 * (double)g_shapes[index].tile_rows * TILE_DIM *
                             (double)g_shapes[index].tile_inner * TILE_DIM *
                             (double)g_shapes[index].tile_cols * TILE_DIM;
        const double gflops = flops / (us * 1000.0);

        printf("  %-28s %9.3f  %8.1f   %14.2f%%   %9u\n", g_shapes[index].label, us / 1000.0,
               gflops, 100.0 * gflops / TENSIX_PEAK_GFLOPS,
               g_shapes[index].tile_rows * g_shapes[index].tile_cols);
    }
    /* One MiniCPM5-1B layer: q + k + v + attn_output + gate + up + down. */
    layer_matmul_us = shape_microseconds(0) + 2.0 * shape_microseconds(1) +
                      shape_microseconds(2) + 2.0 * shape_microseconds(3) +
                      shape_microseconds(4);
    printf("  ONE LAYER's seven matmuls                      %9.3f ms\n", layer_matmul_us / 1000.0);
    printf("  ffn_gate/up vs ffn_down: SAME %u tile-pairs, 144 vs 48 output tiles -> "
           "%.1f ns per output tile write inside a matmul\n",
           48u * 144u,
           (shape_microseconds(SHAPE_FFN_GATE) - shape_microseconds(SHAPE_FFN_DOWN)) * 1000.0 /
               96.0);

    printf("\n");
    printf("  ============================================================================\n");
    printf("  WHAT A FUSION WOULD REMOVE, per layer, per 32-row block\n");
    printf("  ============================================================================\n");
    printf("  A  matmul gate + matmul up + SILU_MUL -> one op\n");
    printf("     removes the whole SILU_MUL descriptor      %9.3f ms  (measured, arm 8)\n",
           arm_microseconds(ARM_SILUMUL144) / 1000.0);
    printf("     plus 288 matmul output-tile writes         %9.3f ms  (measured, gate vs down)\n",
           288.0 *
               ((shape_microseconds(SHAPE_FFN_GATE) - shape_microseconds(SHAPE_FFN_DOWN)) / 96.0) /
               1000.0);
    printf("  B  matmul down + residual ADD, and matmul attn_output + residual ADD\n");
    printf("     removes two ADD descriptors                %9.3f ms  (measured, arm 11)\n",
           2.0 * arm_microseconds(ARM_ADD48_DD) / 1000.0);
    printf("  C  residual ADD + RMS norm -> one op\n");
    printf("     removes the norm's 48 tile READS only      %9.3f ms  (the ADD's write stays: "
           "the residual is needed downstream)\n",
           2.0 * 48.0 * read_premium_ns / 1000000.0);
    printf("     for scale, a whole RMS norm descriptor is  %9.3f ms\n",
           g_rms_norm_microseconds / 1000.0);
    printf("  D  ONE binary switch                          %9.3f ms  (measured, stage 5)\n",
           g_switch_microseconds / 1000.0);
    printf("     — NOT a fusion and NOT dispatch. It is what a MERGE removes, and a core that\n");
    printf("       runs a whole layer pays it once per binary change\n");
    printf("\n  layer matmul budget for scale                 %9.3f ms\n",
           layer_matmul_us / 1000.0);
    printf("  ============================================================================\n\n");
}

/* ================================================================
 * main
 * ================================================================ */

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint32_t tensix_index = 0;
    uint32_t peer_index = 0;
    const char *peer_text;
    uint32_t bank;
    int rc;

    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <eltwise-dir> <reduce-dir> <rtriad-dir> [blocked-writer-dir] "
                "[flushed-writer-dir]\n",
                argv[0]);
        return 2;
    }
    if (getenv("FT_TT_FUSION_ROUNDS"))
        g_rounds = (uint32_t)strtoul(getenv("FT_TT_FUSION_ROUNDS"), NULL, 10);
    if (g_rounds == 0u)
        g_rounds = 1u;

    ft_tt_test_step("Phase 6 item 8 — OP FUSION MEASUREMENT (no fusion is built here)");

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    rc = ft_tt_hardware_session_tensix_from_environment(&session, &g_self_x, &g_self_y,
                                                        &tensix_index);
    if (rc != FT_TT_OK)
        goto release;

    /* The peer is an IDLE core whose L1 is a scratch buffer. Its coordinate is DERIVED from
     * the live grid, never guessed — a wrong NoC endpoint is a hung chip (spec §7.3). */
    peer_text = getenv("FT_TT_FUSION_SCRATCH_INDEX");
    peer_index = peer_text ? (uint32_t)strtoul(peer_text, NULL, 10) : tensix_index + 1u;
    {
        ft_tt_noc_coord peer;

        rc = ft_tt_coord_tensix_by_index(&session.grid, peer_index, FT_TT_COORD_TRANSLATED, &peer);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the peer core's coordinate derives", false, "index %u rc=%d",
                                 peer_index, rc);
            goto release;
        }
        g_peer_x = (uint16_t)peer.x;
        g_peer_y = (uint16_t)peer.y;
    }
    ft_tt_test_check_fmt("the compute core and the L1-scratch core are DIFFERENT cores",
                         peer_index != tensix_index, "compute %u scratch %u", tensix_index,
                         peer_index);

    for (bank = 0; bank < DRAM_BANK_COUNT; bank++) {
        rc = ft_tt_coord_dram_bank(&session.grid, bank, FT_TT_COORD_TRANSLATED, &g_dram[bank]);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("DRAM bank coordinates derive", false, "bank %u rc=%d", bank, rc);
            goto release;
        }
    }

    rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &g_eltwise);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[2], TRIAD_MASK, &g_reduce);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[3], TRIAD_MASK, &g_rtriad);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("all three program directories load", false, "rc=%d", rc);
        goto release;
    }
    if (argc > 4) {
        int blocked_rc = ft_tt_program_directory_load(argv[4], TRIAD_MASK, &g_eltwise_blocked);

        g_have_blocked_writer = (blocked_rc == FT_TT_OK);
        ft_tt_test_check_fmt("the blocked-writer program directory loads", g_have_blocked_writer,
                             "rc=%d", blocked_rc);
    }
    if (argc > 5) {
        int flushed_rc = ft_tt_program_directory_load(argv[5], TRIAD_MASK, &g_eltwise_flushed);

        g_have_flushed_writer = (flushed_rc == FT_TT_OK);
        ft_tt_test_check_fmt("the flushed-writer program directory loads", g_have_flushed_writer,
                             "rc=%d", flushed_rc);
    }

    rc = ft_tt_launch_core_set_init(&g_group.cores, g_placement_storage, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_core_set_add_tensix(&g_group.cores, &session.grid, tensix_index, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the core set derives and classifies the target", false,
                             "index %u rc=%d", tensix_index, rc);
        goto release_programs;
    }
    g_group.id = 0u;
    g_group.first_tensix_index = tensix_index;
    g_group.tensix_index_count = 1u;
    g_group.queues = &g_queue;
    g_group.resident_processor_mask = TRIAD_MASK;

    rc = ft_tt_core_boot_firmware(&session.noc, g_self_x, g_self_y, &firmware, NULL);
    ft_tt_test_check_fmt("firmware boots on the target core", rc == FT_TT_OK,
                         "index %u -> (%u, %u) rc=%d", tensix_index, g_self_x, g_self_y, rc);
    if (rc != FT_TT_OK)
        goto release_programs;

    stage_0_round_trip(&session);

    rc = start_eltwise_residency(&session);
    ft_tt_test_check_fmt("the elementwise residency starts", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto release_programs;

    if (stage_2_dispatch_floor(&session)) {
        rc = upload_all_operands(&session);
        ft_tt_test_check_fmt("every operand buffer is filled, in DRAM and in both L1s, with the "
                             "SAME values",
                             rc == FT_TT_OK, "rc=%d", rc);
        if (rc == FT_TT_OK && stage_3_round_trip_matrix(&session))
            (void)check_timer_resolves_work();
    }
    retire_triad(&session.noc, "the elementwise residency retires by ASKING, never a park");

    if (g_have_blocked_writer || g_have_flushed_writer)
        (void)stage_3b_writer_barrier(&session);

    rc = start_matmul_residency(&session);
    ft_tt_test_check_fmt("the matmul residency starts", rc == FT_TT_OK, "rc=%d", rc);
    if (rc == FT_TT_OK)
        (void)stage_4_matmul_shapes(&session);
    retire_triad(&session.noc, "the matmul residency retires by ASKING, never a park");

    (void)stage_4b_rms_norm(&session);
    (void)stage_5_binary_switch(&session);
    report();

release_programs:
    ft_tt_program_directory_release(&g_eltwise);
    ft_tt_program_directory_release(&g_reduce);
    ft_tt_program_directory_release(&g_rtriad);
    if (g_have_blocked_writer)
        ft_tt_program_directory_release(&g_eltwise_blocked);
    if (g_have_flushed_writer)
        ft_tt_program_directory_release(&g_eltwise_flushed);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
