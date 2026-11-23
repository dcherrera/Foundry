/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_queue.c
 * @brief Phase 5 off-device tests: the work-queue wire protocol, the completion-evidence
 *   rules, the shutdown protocol, and the core-group partitioner — including a SYNTHETIC
 *   12-column grid that must partition into 120 cores with no code change.
 *
 * NO DEVICE IS TOUCHED. Cores are host memory behind the same `ft_tt_noc_access` vtable
 * the real TLB path implements, and the resident kernel is a SOFTWARE MODEL stepped by
 * this file.
 *
 * **What a model can and cannot prove.** The model executes the same loop the real
 * kernel does, from the same header constants, so it pins the HOST half: ring wrap,
 * batch writes, head-last ordering, token derivation, the four completion checks, the
 * stop protocol and the watchdog's alive-vs-progressing distinction. It cannot tell you
 * that `kernels/resident_queue.c` compiles to that loop — that is
 * `tests/test_phase5_resident.c` on silicon, and nothing here substitutes for it. Two
 * halves written from one header agreeing with each other is exactly the shape of
 * self-consistency this project has been caught by before (dev notes §3).
 *
 * The negative cases are the point of the file. Each of them asserts that a check
 * REFUSES something, so a check that was quietly weakened goes red here rather than on
 * the card:
 *   - a queue whose kernel never ran must fail completion, not pass on a zero;
 *   - a completed_count that is right with a stale token must fail;
 *   - two groups holding one core must fail the disjointness check;
 *   - a set with no storage must refuse an add rather than write through NULL.
 *
 * @status Green, 92/92, on macOS/arm64 and on the Linux card host's gcc under the
 *   tests/Makefile.queue flags. Seven mutations recorded red
 *   (docs/tt_bringup_log.md 17.7).
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_arch.h"
#include "ft_tt_coord.h"
#include "ft_tt_group.h"
#include "ft_tt_launch.h"
#include "ft_tt_noc.h"
#include "ft_tt_queue.h"
#include "ft_tt_watchdog.h"

/* ------------------------------------------------------------------------- */
/* Tiny harness                                                              */
/* ------------------------------------------------------------------------- */

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got == want) {
        printf("  ok    %s = 0x%" PRIx64 "\n", what, got);
    } else {
        checks_failed++;
        printf("  FAIL  %s: got 0x%" PRIx64 ", want 0x%" PRIx64 "\n", what, got, want);
    }
}

/* ------------------------------------------------------------------------- */
/* A simulated card: several cores, each with its own L1                     */
/* ------------------------------------------------------------------------- */

#define SIM_MAX_CORES 8u
#define SIM_L1_BYTES  FT_TT_MEM_L1_SIZE

typedef struct simulated_core {
    uint16_t x;
    uint16_t y;
    bool present;
    unsigned char *l1;
    /* The model kernel's private registers — the things a real kernel keeps in its own
     * local memory rather than in L1. */
    bool kernel_running;
    uint32_t kernel_tail;
    uint32_t kernel_completed;
    uint32_t kernel_idle_spins;
    uint32_t kernel_heartbeat;
    uint32_t kernel_error;
    uint32_t kernel_control_address;
    uint32_t kernel_capacity_mask;
    uint32_t kernel_ring_address;
    uint32_t kernel_idle_limit;
} simulated_core;

static simulated_core sim_cores[SIM_MAX_CORES];
static unsigned sim_core_count;
static uint32_t sim_read_calls;
static uint32_t sim_write_calls;

static simulated_core *sim_find(uint16_t x, uint16_t y)
{
    unsigned index;

    for (index = 0; index < sim_core_count; index++) {
        if (sim_cores[index].present && sim_cores[index].x == x && sim_cores[index].y == y)
            return &sim_cores[index];
    }
    return NULL;
}

static int simulated_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                           size_t bytes)
{
    simulated_core *core = sim_find(x, y);

    (void)user;
    sim_write_calls++;
    if (!core)
        return FT_TT_EINVAL;
    if (addr + bytes > SIM_L1_BYTES)
        return FT_TT_EINVAL;
    memcpy(core->l1 + addr, src, bytes);
    return FT_TT_OK;
}

static int simulated_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst,
                          size_t bytes)
{
    simulated_core *core = sim_find(x, y);

    (void)user;
    sim_read_calls++;
    if (!core)
        return FT_TT_EINVAL;
    if (addr + bytes > SIM_L1_BYTES)
        return FT_TT_EINVAL;
    memcpy(dst, core->l1 + addr, bytes);
    return FT_TT_OK;
}

static ft_tt_noc_access simulated_access(void)
{
    ft_tt_noc_access access;

    memset(&access, 0, sizeof(access));
    access.write = simulated_write;
    access.read = simulated_read;
    access.user = NULL;
    return access;
}

/** Bring up `count` simulated cores at the coordinates a real grid would derive. */
static void sim_reset(const uint16_t *xs, const uint16_t *ys, unsigned count)
{
    unsigned index;

    for (index = 0; index < SIM_MAX_CORES; index++) {
        if (sim_cores[index].l1)
            memset(sim_cores[index].l1, 0, SIM_L1_BYTES);
        sim_cores[index].present = false;
    }
    for (index = 0; index < count && index < SIM_MAX_CORES; index++) {
        if (!sim_cores[index].l1)
            sim_cores[index].l1 = calloc(1, SIM_L1_BYTES);
        sim_cores[index].x = xs[index];
        sim_cores[index].y = ys[index];
        sim_cores[index].present = true;
        sim_cores[index].kernel_running = false;
        sim_cores[index].kernel_tail = 0u;
        sim_cores[index].kernel_completed = 0u;
        sim_cores[index].kernel_idle_spins = 0u;
        sim_cores[index].kernel_heartbeat = 0u;
        sim_cores[index].kernel_error = FT_TT_QUEUE_KERR_NONE;
    }
    sim_core_count = (count < SIM_MAX_CORES) ? count : SIM_MAX_CORES;
    sim_read_calls = 0u;
    sim_write_calls = 0u;
}

static uint32_t sim_l1_u32(const simulated_core *core, uint32_t address)
{
    uint32_t value;

    memcpy(&value, core->l1 + address, sizeof(value));
    return value;
}

static void sim_l1_store_u32(simulated_core *core, uint32_t address, uint32_t value)
{
    memcpy(core->l1 + address, &value, sizeof(value));
}

static uint32_t sim_control(const simulated_core *core, ft_tt_queue_control_field field)
{
    return sim_l1_u32(core, core->kernel_control_address + (uint32_t)field * 4u);
}

static void sim_set_control(simulated_core *core, ft_tt_queue_control_field field, uint32_t value)
{
    sim_l1_store_u32(core, core->kernel_control_address + (uint32_t)field * 4u, value);
}

/* ------------------------------------------------------------------------- */
/* The model resident kernel                                                 */
/* ------------------------------------------------------------------------- */

/** Start the model kernel on a core, as a launch would. */
static void model_kernel_start(simulated_core *core, const uint32_t args[4])
{
    core->kernel_control_address = args[0];
    core->kernel_capacity_mask = args[1];
    core->kernel_ring_address = args[2];
    core->kernel_idle_limit = args[3];
    core->kernel_tail = sim_control(core, FT_TT_QUEUE_FIELD_TAIL);
    core->kernel_completed = sim_control(core, FT_TT_QUEUE_FIELD_COMPLETED_COUNT);
    core->kernel_heartbeat = sim_control(core, FT_TT_QUEUE_FIELD_HEARTBEAT);
    core->kernel_idle_spins = 0u;
    core->kernel_error = FT_TT_QUEUE_KERR_NONE;
    core->kernel_running = true;
    sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_BOOT);
    if (sim_control(core, FT_TT_QUEUE_FIELD_MAGIC) != FT_TT_QUEUE_MAGIC) {
        core->kernel_error = FT_TT_QUEUE_KERR_BAD_MAGIC;
        core->kernel_running = false;
        sim_set_control(core, FT_TT_QUEUE_FIELD_ERROR, core->kernel_error);
        sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_EXIT);
    }
}

/**
 * One turn of the model kernel's loop. Mirrors kernels/resident_queue.c.
 *
 * The order of the two completion stores is deliberate and is the thing this model is
 * here to pin: the TOKEN is written before the COUNT, so a host that has seen the count
 * is entitled to trust the token.
 */
static void model_kernel_step(simulated_core *core)
{
    uint32_t head;
    uint32_t slot;
    uint32_t base;
    uint32_t opcode;
    uint32_t token;
    bool item_completed = true;

    if (!core->kernel_running)
        return;

    sim_set_control(core, FT_TT_QUEUE_FIELD_HEARTBEAT, ++core->kernel_heartbeat);

    if (sim_control(core, FT_TT_QUEUE_FIELD_STOP_REQUEST) != 0u) {
        core->kernel_running = false;
        sim_set_control(core, FT_TT_QUEUE_FIELD_ERROR, core->kernel_error);
        sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_EXIT);
        return;
    }

    head = sim_control(core, FT_TT_QUEUE_FIELD_HEAD);
    if (head == core->kernel_tail) {
        sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_IDLE);
        core->kernel_idle_spins++;
        sim_set_control(core, FT_TT_QUEUE_FIELD_IDLE_SPINS, core->kernel_idle_spins);
        if (core->kernel_idle_spins >= core->kernel_idle_limit) {
            core->kernel_error = FT_TT_QUEUE_KERR_IDLE_TIMEOUT;
            core->kernel_running = false;
            sim_set_control(core, FT_TT_QUEUE_FIELD_ERROR, core->kernel_error);
            sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_EXIT);
        }
        return;
    }

    core->kernel_idle_spins = 0u;
    sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_EXEC);

    slot = core->kernel_tail & core->kernel_capacity_mask;
    base = core->kernel_ring_address + slot * FT_TT_QUEUE_DESCRIPTOR_BYTES;
    opcode = sim_l1_u32(core, base + 0u);
    token = sim_l1_u32(core, base + 4u);
    sim_set_control(core, FT_TT_QUEUE_FIELD_LAST_OPCODE, opcode);

    switch (opcode) {
    case FT_TT_QUEUE_OP_NOP:
        break;
    case FT_TT_QUEUE_OP_WRITE_WORD:
        sim_l1_store_u32(core, sim_l1_u32(core, base + 8u), sim_l1_u32(core, base + 12u));
        break;
    case FT_TT_QUEUE_OP_SEM_WAIT: {
        uint32_t semaphore_address = sim_l1_u32(core, base + 8u);
        uint32_t target = sim_l1_u32(core, base + 12u);

        /* The model blocks the same way the kernel does: it stays on this descriptor
         * until the semaphore reaches its target or the host asks it to stop, and an
         * abandoned wait is NOT completed. */
        sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_SEM_WAIT);
        if (sim_l1_u32(core, semaphore_address) < target)
            return; /* still waiting; tail does not advance */
        break;
    }
    case FT_TT_QUEUE_OP_STOP:
        core->kernel_running = false;
        break;
    default:
        core->kernel_error = FT_TT_QUEUE_KERR_BAD_OPCODE;
        item_completed = false;
        core->kernel_running = false;
        break;
    }

    /* AN ABANDONED DESCRIPTOR ADVANCES NOTHING — not `tail` and not `completed_count`. It
     * was handed to nobody, so the next launch re-attempts it. This is what makes `tail`
     * mean one thing in both kernels: with the resident TRIAD it is the handoff pointer the
     * WRITER's cursor follows, and advancing it for an item the reader abandoned would let
     * the writer retire work that never ran. */
    if (item_completed) {
        core->kernel_tail++;
        sim_set_control(core, FT_TT_QUEUE_FIELD_TAIL, core->kernel_tail);
        sim_set_control(core, FT_TT_QUEUE_FIELD_COMPLETED_TOKEN, token);
        sim_set_control(core, FT_TT_QUEUE_FIELD_COMPLETED_COUNT, ++core->kernel_completed);
    }
    if (!core->kernel_running) {
        sim_set_control(core, FT_TT_QUEUE_FIELD_ERROR, core->kernel_error);
        sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_EXIT);
    }
}

static void model_kernel_run(simulated_core *core, unsigned max_steps)
{
    unsigned step;

    for (step = 0; step < max_steps && core->kernel_running; step++)
        model_kernel_step(core);
}

/* ------------------------------------------------------------------------- */
/* 1. The control block the host writes                                      */
/* ------------------------------------------------------------------------- */

static const uint16_t sim_xs[SIM_MAX_CORES] = { 1, 2, 3, 4, 5, 6, 7, 8 };
static const uint16_t sim_ys[SIM_MAX_CORES] = { 2, 2, 2, 2, 3, 3, 3, 3 };

#define TEST_CAPACITY 16u
#define TEST_IDLE_LIMIT 1000u

/**
 * The kernel binding these off-hardware cases use: both spin bounds set to `spins`, no
 * status slot and no group block.
 *
 * A helper rather than a struct per call site so that a new REQUIRED field in the binding
 * shows up in one place. The two bounds are set together on purpose: `cb_wait_spin_limit`
 * is refused at 0, and a test that left it there would be exercising the refusal instead of
 * the case it names.
 */
static const ft_tt_queue_kernel_binding *sim_binding(uint32_t spins)
{
    static ft_tt_queue_kernel_binding binding;

    memset(&binding, 0, sizeof(binding));
    binding.idle_spin_limit = spins;
    binding.cb_wait_spin_limit = spins;
    return &binding;
}

static void test_queue_init_writes_a_recognisable_block(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];

    printf("\nqueue: init writes a block the kernel can recognise, and poisons the rest\n");
    sim_reset(sim_xs, sim_ys, 1u);

    check(ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x1234u, &queue) ==
              FT_TT_OK,
          "init");
    check(ft_tt_queue_read_control(&noc, &queue, fields) == FT_TT_OK, "the block reads back");
    check_u64(fields[FT_TT_QUEUE_FIELD_MAGIC], FT_TT_QUEUE_MAGIC, "magic");
    check_u64(fields[FT_TT_QUEUE_FIELD_CAPACITY], TEST_CAPACITY, "capacity");
    check_u64(fields[FT_TT_QUEUE_FIELD_HEAD], 0u, "head");
    check_u64(fields[FT_TT_QUEUE_FIELD_TAIL], 0u, "tail");
    check_u64(fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT], 0u, "completed count");

    /* The two fields that must NOT start at zero. A zero here is what makes "never ran"
     * and "finished" the same bit pattern, which is the trap this whole protocol is
     * shaped around (dev notes §3). */
    check_u64(fields[FT_TT_QUEUE_FIELD_STATE], (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON,
              "state starts poisoned, not zero");
    check_u64(fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN], (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON,
              "and so does the completed token");

    check(queue.ring_address == FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_RING_OFFSET,
          "the ring follows the control block");
    check(ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, 15u, 0x1234u, &queue) == FT_TT_EINVAL,
          "a capacity that is not a power of two is refused");
    check(ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE + 1u, TEST_CAPACITY, 0x1234u,
                           &queue) == FT_TT_EINVAL,
          "and so is a misaligned control address");
    check(ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0u, &queue) ==
              FT_TT_EINVAL,
          "a zero nonce is refused: it would make every token derivable without the run");
    check(ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY,
                           (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON, &queue) == FT_TT_EINVAL,
          "and so is a nonce that collides with the poison");
}

/* ------------------------------------------------------------------------- */
/* 2. Enqueue: one device write per batch, head last                         */
/* ------------------------------------------------------------------------- */

static void fill_nops(ft_tt_queue_descriptor *descriptors, uint32_t count)
{
    uint32_t index;

    memset(descriptors, 0, (size_t)count * sizeof(*descriptors));
    for (index = 0; index < count; index++)
        descriptors[index].opcode = FT_TT_QUEUE_OP_NOP;
}

static void test_enqueue_is_one_write_plus_a_head_bump(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    ft_tt_queue_descriptor descriptors[TEST_CAPACITY];
    simulated_core *core;
    uint32_t before;

    printf("\nqueue: a batch is one descriptor write and one head bump — no round trip\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(1u, 2u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0xABCDu, &queue);

    fill_nops(descriptors, 8u);
    before = sim_write_calls;
    check(ft_tt_queue_enqueue(&noc, &queue, descriptors, 8u) == FT_TT_OK, "eight enqueued");
    check_u64(sim_write_calls - before, 2u,
              "eight descriptors cost TWO device accesses in total, not eight");
    check_u64(sim_read_calls, 0u, "and enqueue never reads the card: there is no round trip");
    check_u64(queue.head, 8u, "the host's head advanced by the batch");

    /* Tokens are stamped by the enqueue, not by the caller. */
    check_u64(descriptors[0].token, ft_tt_queue_token_at(&queue, 0u), "descriptor 0 got its token");
    check_u64(descriptors[7].token, ft_tt_queue_token_at(&queue, 7u), "and descriptor 7 got its");
    check(ft_tt_queue_token_at(&queue, 0u) != ft_tt_queue_token_at(&queue, 1u),
          "consecutive tokens differ");
    check(ft_tt_queue_token_at(&queue, 0u) != 0u &&
              ft_tt_queue_token_at(&queue, 0u) != (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON,
          "and a token is never zero and never the poison");

    check_u64(sim_l1_u32(core, queue.ring_address), FT_TT_QUEUE_OP_NOP, "slot 0 holds the opcode");
    check_u64(sim_l1_u32(core, queue.ring_address + 4u), descriptors[0].token,
              "and slot 0 holds its token");
    check_u64(sim_l1_u32(core, queue.control_address + FT_TT_QUEUE_FIELD_HEAD * 4u), 8u,
              "the device head is the last thing written");

    check(ft_tt_queue_enqueue(&noc, &queue, descriptors, TEST_CAPACITY + 1u) == FT_TT_ENOMEM,
          "a batch bigger than the ring is refused");
    check(ft_tt_queue_enqueue(&noc, &queue, descriptors, 0u) == FT_TT_EINVAL,
          "and so is an empty one");
}

static void test_enqueue_wraps_the_ring(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    ft_tt_queue_descriptor descriptors[TEST_CAPACITY];
    simulated_core *core;

    printf("\nqueue: a batch that crosses the end of the ring lands in both halves\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(1u, 2u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x55AAu, &queue);

    /* Push the head to two before the wrap, then enqueue four. */
    fill_nops(descriptors, TEST_CAPACITY - 2u);
    (void)ft_tt_queue_enqueue(&noc, &queue, descriptors, TEST_CAPACITY - 2u);

    fill_nops(descriptors, 4u);
    descriptors[0].a0 = 0xF00Du;
    descriptors[3].a0 = 0xBEEFu;
    check(ft_tt_queue_enqueue(&noc, &queue, descriptors, 4u) == FT_TT_OK, "four more enqueued");
    check_u64(queue.head, TEST_CAPACITY + 2u, "the head is monotonic, not wrapped");
    check_u64(sim_l1_u32(core, queue.ring_address +
                                   (TEST_CAPACITY - 2u) * FT_TT_QUEUE_DESCRIPTOR_BYTES + 8u),
              0xF00Du, "the first of the four is in the last-but-one slot");
    check_u64(sim_l1_u32(core, queue.ring_address + 1u * FT_TT_QUEUE_DESCRIPTOR_BYTES + 8u),
              0xBEEFu, "and the fourth wrapped round to slot 1");
}

/* ------------------------------------------------------------------------- */
/* 3. Completion is produced by the work item                                */
/* ------------------------------------------------------------------------- */

static void test_completion_cannot_report_work_that_never_ran(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    ft_tt_queue_descriptor descriptors[TEST_CAPACITY];
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    const char *reason = NULL;
    simulated_core *core;

    printf("\nqueue: a core that never ran fails the completion check\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(1u, 2u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x0F0Fu, &queue);

    fill_nops(descriptors, 4u);
    (void)ft_tt_queue_enqueue(&noc, &queue, descriptors, 4u);

    /* The kernel is never started. This is the exact case ft_tt_launch_wait_done_all()
     * reported as complete on 2026-08-12. */
    (void)ft_tt_queue_read_control(&noc, &queue, fields);
    check(ft_tt_queue_check_completion(&queue, fields, 4u, &reason) == FT_TT_ERROR,
          "four descriptors enqueued and nothing started is NOT complete");
    check(reason != NULL && strstr(reason, "never ran") != NULL,
          "and the reason says so rather than being a bare failure");

    /* Now run it and the same four descriptors do pass. */
    {
        uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];

        check(ft_tt_queue_kernel_args(&queue, sim_binding(TEST_IDLE_LIMIT), args) == FT_TT_OK, "kernel args");
        check_u64(args[1], TEST_CAPACITY - 1u, "the capacity arrives as a MASK");
        check(ft_tt_queue_kernel_args(&queue, sim_binding(0u), args) == FT_TT_EINVAL,
              "an idle limit of 0 is refused: residency is bounded, always");
        (void)ft_tt_queue_kernel_args(&queue, sim_binding(TEST_IDLE_LIMIT), args);
        model_kernel_start(core, args);
        model_kernel_run(core, 64u);
    }
    (void)ft_tt_queue_read_control(&noc, &queue, fields);
    check(ft_tt_queue_check_completion(&queue, fields, 4u, &reason) == FT_TT_OK,
          "after the kernel runs them, the same four ARE complete");
    check_u64(fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN], ft_tt_queue_token_at(&queue, 3u),
              "and the token is this run's fourth");
}

static void test_completion_rejects_a_stale_token(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    const char *reason = NULL;

    printf("\nqueue: the right COUNT with the wrong TOKEN is still a failure\n");
    sim_reset(sim_xs, sim_ys, 1u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x77u, &queue);

    memset(fields, 0, sizeof(fields));
    fields[FT_TT_QUEUE_FIELD_STATE] = FT_TT_QUEUE_STATE_EXIT;
    fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] = 4u;
    fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] = ft_tt_queue_token_at(&queue, 3u);
    check(ft_tt_queue_check_completion(&queue, fields, 4u, &reason) == FT_TT_OK,
          "the honest case passes");

    /* A previous run's residue: a plausible non-zero token that this run never issued. */
    fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] = ft_tt_queue_token_at(&queue, 3u) ^ 1u;
    check(ft_tt_queue_check_completion(&queue, fields, 4u, &reason) == FT_TT_ERROR,
          "a token this run never issued fails even with the count right");

    fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] = ft_tt_queue_token_at(&queue, 3u);
    fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] = 3u;
    check(ft_tt_queue_check_completion(&queue, fields, 4u, &reason) == FT_TT_ERROR,
          "a short count fails even with a token that is a real one");

    fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] = 4u;
    fields[FT_TT_QUEUE_FIELD_ERROR] = FT_TT_QUEUE_KERR_BAD_OPCODE;
    check(ft_tt_queue_check_completion(&queue, fields, 4u, &reason) == FT_TT_ERROR,
          "and a kernel that reported an error fails whatever its counters say");
}

static void test_two_runs_cannot_share_a_token(void)
{
    ft_tt_queue first;
    ft_tt_queue second;
    ft_tt_noc_access noc = simulated_access();
    uint32_t index;
    bool any_collision = false;

    printf("\nqueue: two runs' token sets are disjoint — residue cannot be mistaken for work\n");
    sim_reset(sim_xs, sim_ys, 1u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x1111u, &first);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x1112u, &second);

    for (index = 0; index < 256u && !any_collision; index++) {
        uint32_t j;

        for (j = 0; j < 256u; j++) {
            if (ft_tt_queue_token_at(&first, index) == ft_tt_queue_token_at(&second, j)) {
                any_collision = true;
                break;
            }
        }
    }
    check(!any_collision, "no token of run A equals any of the first 256 of run B");
}

/* ------------------------------------------------------------------------- */
/* 4. Shutdown by asking, and the deadlock it has to survive                 */
/* ------------------------------------------------------------------------- */

static void test_stop_request_retires_a_blocked_kernel(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    ft_tt_queue_descriptor descriptors[4];
    uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];
    simulated_core *core;
    const uint32_t semaphore_address = FT_TT_OP_CB_ARENA_BASE;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];

    printf("\nqueue: a kernel blocked on a semaphore nobody will post is retired by ASKING\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(1u, 2u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x2222u, &queue);

    memset(descriptors, 0, sizeof(descriptors));
    descriptors[0].opcode = FT_TT_QUEUE_OP_NOP;
    descriptors[1].opcode = FT_TT_QUEUE_OP_SEM_WAIT;
    descriptors[1].a0 = semaphore_address;
    descriptors[1].a1 = 1u; /* nobody ever posts it */
    (void)ft_tt_queue_enqueue(&noc, &queue, descriptors, 2u);

    (void)ft_tt_queue_kernel_args(&queue, sim_binding(TEST_IDLE_LIMIT), args);
    model_kernel_start(core, args);
    model_kernel_run(core, 64u);

    (void)ft_tt_queue_read_control(&noc, &queue, fields);
    check_u64(fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT], 1u,
              "the NOP completed and the blocked wait did NOT");
    check_u64(fields[FT_TT_QUEUE_FIELD_STATE], FT_TT_QUEUE_STATE_SEM_WAIT, "the kernel is blocked");
    check(core->kernel_running, "and it is still running: a stall, not a crash");
    {
        uint32_t heartbeat_before = fields[FT_TT_QUEUE_FIELD_HEARTBEAT];

        model_kernel_run(core, 4u);
        (void)ft_tt_queue_read_control(&noc, &queue, fields);
        check(fields[FT_TT_QUEUE_FIELD_HEARTBEAT] > heartbeat_before,
              "the heartbeat still moves — ALIVE and blocked is distinguishable from dead");
    }

    check(ft_tt_queue_request_stop(&noc, &queue) == FT_TT_OK, "the host asks it to stop");
    model_kernel_run(core, 8u);
    (void)ft_tt_queue_read_control(&noc, &queue, fields);
    check(!core->kernel_running, "the kernel left its loop");
    check_u64(fields[FT_TT_QUEUE_FIELD_STATE], FT_TT_QUEUE_STATE_EXIT, "and says EXIT");
    check_u64(fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT], 1u,
              "the ABANDONED descriptor was not retired — stopping is not finishing");
}

static void test_stop_sentinel_is_ordered_with_the_work(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    ft_tt_queue_descriptor descriptors[4];
    uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    simulated_core *core;

    printf("\nqueue: a STOP descriptor runs after the work queued before it\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(1u, 2u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x3333u, &queue);

    memset(descriptors, 0, sizeof(descriptors));
    descriptors[0].opcode = FT_TT_QUEUE_OP_NOP;
    descriptors[1].opcode = FT_TT_QUEUE_OP_NOP;
    descriptors[2].opcode = FT_TT_QUEUE_OP_STOP;
    (void)ft_tt_queue_enqueue(&noc, &queue, descriptors, 3u);

    (void)ft_tt_queue_kernel_args(&queue, sim_binding(TEST_IDLE_LIMIT), args);
    model_kernel_start(core, args);
    model_kernel_run(core, 64u);

    (void)ft_tt_queue_read_control(&noc, &queue, fields);
    check_u64(fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT], 3u,
              "all three retired, the STOP among them");
    check_u64(fields[FT_TT_QUEUE_FIELD_STATE], FT_TT_QUEUE_STATE_EXIT, "and the kernel exited");
}

static void test_bounded_residency(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    simulated_core *core;

    printf("\nqueue: a kernel whose host stopped feeding it returns on its own\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(1u, 2u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x4444u, &queue);

    (void)ft_tt_queue_kernel_args(&queue, sim_binding(5u), args);
    model_kernel_start(core, args);
    model_kernel_run(core, 100u);

    (void)ft_tt_queue_read_control(&noc, &queue, fields);
    check(!core->kernel_running, "it is not spinning forever");
    check_u64(fields[FT_TT_QUEUE_FIELD_STATE], FT_TT_QUEUE_STATE_EXIT, "it reached EXIT");
    check_u64(fields[FT_TT_QUEUE_FIELD_ERROR], FT_TT_QUEUE_KERR_IDLE_TIMEOUT,
              "and it says WHY — the idle bound, not a mystery");
}

/* ------------------------------------------------------------------------- */
/* 5. The partitioner — derived, never constant                              */
/* ------------------------------------------------------------------------- */

static ft_tt_grid_info p150a_grid(void)
{
    ft_tt_grid_info grid;

    memset(&grid, 0, sizeof(grid));
    grid.valid = true;
    grid.arch = ft_tt_arch_blackhole();
    grid.noc_translation_enabled = 1u;
    grid.tensix_enabled_col = 16383u;
    grid.tensix_column_count = 14u;
    grid.tensix_core_count = 140u;
    grid.enabled_gddr = 255u;
    grid.dram_bank_count = 8u;
    return grid;
}

/**
 * A SYNTHETIC 12-column part.
 *
 * This is the acceptance criterion, not a nicety: TT's 140→120 reduction is a
 * compatibility measure rather than yield binning, so 12-column p150s are a shipping
 * configuration running this same software. The partitioner has to produce 120 cores here
 * with no code change, and a literal 140 or 14 anywhere in it makes this case red and
 * nothing else.
 */
static const uint8_t twelve_columns[12] = { 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14 };
static const uint8_t twelve_bits[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
static ft_tt_arch twelve_column_arch;

static ft_tt_grid_info twelve_column_grid(void)
{
    ft_tt_grid_info grid;

    twelve_column_arch = *ft_tt_arch_blackhole();
    twelve_column_arch.tensix.column_count = 12u;
    twelve_column_arch.tensix.noc_x_by_column = twelve_columns;
    twelve_column_arch.tensix.telemetry_bit_by_column = twelve_bits;

    memset(&grid, 0, sizeof(grid));
    grid.valid = true;
    grid.arch = &twelve_column_arch;
    grid.noc_translation_enabled = 1u;
    grid.tensix_enabled_col = 0xFFFu;
    grid.enabled_gddr = 255u;
    grid.dram_bank_count = 8u;
    return grid;
}

#define PARTITION_STORAGE 160u
static ft_tt_launch_core_placement partition_placements[PARTITION_STORAGE];
static ft_tt_queue partition_queues[PARTITION_STORAGE];
static ft_tt_core_group partition_groups[8];

static void test_partition_is_derived_from_the_live_grid(void)
{
    const ft_tt_grid_info grid = p150a_grid();
    uint32_t live = ft_tt_coord_tensix_core_count(&grid);
    const char *reason = NULL;

    printf("\ngroups: the partition comes from the LIVE core count\n");
    check_u64(live, 140u, "this fixture is a 14-column part");

    check(ft_tt_group_partition(&grid, 0u, 0u, 4u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE) == FT_TT_OK,
          "140 cores partition into four groups");
    check_u64(ft_tt_launch_core_set_count(&partition_groups[0].cores), 35u, "group 0 size");
    check_u64(ft_tt_launch_core_set_count(&partition_groups[3].cores), 35u, "group 3 size");
    check_u64(partition_groups[3].first_tensix_index, 105u, "group 3 starts where group 2 ended");
    check(ft_tt_group_check_disjoint(partition_groups, 4u, &reason) == FT_TT_OK,
          "and no core is in two groups");

    /* A remainder must be spread, not dropped — a partitioner that leaves cores
     * unassigned is one that silently runs a smaller machine than it was given. */
    check(ft_tt_group_partition(&grid, 0u, 0u, 3u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE) == FT_TT_OK,
          "140 cores partition into three groups");
    check_u64(ft_tt_launch_core_set_count(&partition_groups[0].cores), 47u, "group 0 takes one extra");
    check_u64(ft_tt_launch_core_set_count(&partition_groups[1].cores), 47u, "so does group 1");
    check_u64(ft_tt_launch_core_set_count(&partition_groups[2].cores), 46u, "group 2 takes the base");
    check_u64(ft_tt_launch_core_set_count(&partition_groups[0].cores) +
                  ft_tt_launch_core_set_count(&partition_groups[1].cores) +
                  ft_tt_launch_core_set_count(&partition_groups[2].cores),
              live, "and every live core is assigned exactly once");
}

static void test_partition_on_a_synthetic_twelve_column_part(void)
{
    const ft_tt_grid_info grid = twelve_column_grid();
    uint32_t live = ft_tt_coord_tensix_core_count(&grid);
    const char *reason = NULL;

    printf("\ngroups: a synthetic 12-column part partitions into 120 with NO code change\n");
    check_u64(live, 120u, "the live core count is 12 x 10, not 140");

    check(ft_tt_group_partition(&grid, 0u, 0u, 4u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE) == FT_TT_OK,
          "and it partitions");
    check_u64(ft_tt_launch_core_set_count(&partition_groups[0].cores), 30u,
              "into groups of 120/4, not 140/4");
    check_u64(partition_groups[3].first_tensix_index + partition_groups[3].tensix_index_count, 120u,
              "the last group ends exactly at the live count");
    check(ft_tt_group_check_disjoint(partition_groups, 4u, &reason) == FT_TT_OK, "disjoint");

    /* And an index the 14-column part has but this one does not must be REFUSED rather
     * than addressed — that is a NoC transaction at a coordinate nobody derived. */
    check(ft_tt_group_partition(&grid, 130u, 4u, 1u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE) != FT_TT_OK,
          "an index range past this part's live count is refused, not clamped");
}

static void test_partition_refusals_and_disjointness(void)
{
    const ft_tt_grid_info grid = p150a_grid();
    const char *reason = NULL;

    printf("\ngroups: refusals, sub-ranges, and the pairwise check that catches a shared core\n");

    check(ft_tt_group_partition(&grid, 1u, 8u, 2u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE) == FT_TT_OK,
          "a sub-range partitions: indices 1..8 into two groups");
    check_u64(partition_groups[0].first_tensix_index, 1u, "group 0 starts at the first index given");
    check_u64(partition_groups[1].first_tensix_index, 5u, "group 1 follows it");
    check(ft_tt_group_check_disjoint(partition_groups, 2u, &reason) == FT_TT_OK, "disjoint");

    /* DRAM banks are handed out by stride, so two groups never share one while the part
     * has at least as many banks as groups. */
    {
        uint32_t bank_a = 0u;
        uint32_t bank_b = 0u;

        check(ft_tt_group_dram_bank(&partition_groups[0], 0u, &bank_a) == FT_TT_OK, "group 0 bank 0");
        check(ft_tt_group_dram_bank(&partition_groups[1], 0u, &bank_b) == FT_TT_OK, "group 1 bank 0");
        check(bank_a != bank_b, "and the two groups' first DRAM banks differ");
        check(ft_tt_group_dram_bank(&partition_groups[0], 99u, &bank_a) == FT_TT_EINVAL,
              "a bank index past what the group owns is refused");
    }

    check(ft_tt_group_partition(&grid, 0u, 4u, 5u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE) == FT_TT_EINVAL,
          "more groups than cores is refused");
    check(ft_tt_group_partition(&grid, 0u, 0u, 4u, partition_groups, partition_placements,
                                partition_queues, 8u) == FT_TT_ENOMEM,
          "storage smaller than the range is refused before anything is placed");
    check(ft_tt_group_partition(&grid, 0u, 0u, 0u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE) == FT_TT_EINVAL,
          "and so is zero groups");

    /* The negative case for the disjointness check itself: hand it two groups holding the
     * same core and it must say so. A checker nobody has seen refuse is not a checker. */
    (void)ft_tt_group_partition(&grid, 0u, 8u, 2u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE);
    partition_placements[4] = partition_placements[0]; /* group 1 slot 0 := group 0 slot 0 */
    check(ft_tt_group_check_disjoint(partition_groups, 2u, &reason) == FT_TT_ERROR,
          "two groups holding one core is caught");
    check(reason != NULL, "with a reason");
}

/* ------------------------------------------------------------------------- */
/* 6. The watchdog                                                           */
/* ------------------------------------------------------------------------- */

static void test_watchdog_reports_alive_but_blocked(void)
{
    const ft_tt_grid_info grid = p150a_grid();
    ft_tt_noc_access noc = simulated_access();
    ft_tt_watchdog_config config;
    ft_tt_watchdog_report report;
    uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];
    ft_tt_queue_descriptor descriptors[4];
    unsigned slot;

    printf("\nwatchdog: a blocked group times out, and the report says ALIVE\n");

    (void)ft_tt_group_partition(&grid, 0u, 2u, 1u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE);
    {
        uint16_t xs[2];
        uint16_t ys[2];

        for (slot = 0; slot < 2u; slot++) {
            const ft_tt_launch_core_placement *placement =
                ft_tt_launch_core_set_at(&partition_groups[0].cores, slot);
            xs[slot] = placement->x;
            ys[slot] = placement->y;
        }
        sim_reset(xs, ys, 2u);
    }

    check(ft_tt_group_queues_init(&noc, &partition_groups[0], FT_TT_OP_QUEUE_BASE, TEST_CAPACITY,
                                  0x9999u) == FT_TT_OK,
          "both cores' queues initialise");

    /* Core 0 gets a NOP it can finish; core 1 gets a wait nobody will satisfy. */
    memset(descriptors, 0, sizeof(descriptors));
    descriptors[0].opcode = FT_TT_QUEUE_OP_NOP;
    (void)ft_tt_queue_enqueue(&noc, &partition_groups[0].queues[0], descriptors, 1u);
    memset(descriptors, 0, sizeof(descriptors));
    descriptors[0].opcode = FT_TT_QUEUE_OP_SEM_WAIT;
    descriptors[0].a0 = FT_TT_OP_CB_ARENA_BASE;
    descriptors[0].a1 = 1u;
    (void)ft_tt_queue_enqueue(&noc, &partition_groups[0].queues[1], descriptors, 1u);

    for (slot = 0; slot < 2u; slot++) {
        const ft_tt_launch_core_placement *placement =
            ft_tt_launch_core_set_at(&partition_groups[0].cores, slot);
        simulated_core *core = sim_find(placement->x, placement->y);

        (void)ft_tt_queue_kernel_args(&partition_groups[0].queues[slot], sim_binding(1000000u),
                                      args);
        model_kernel_start(core, args);
        model_kernel_run(core, 16u);
    }

    config.stall_timeout_ms = 20u;
    config.overall_timeout_ms = 200u;
    check(ft_tt_watchdog_wait(&noc, &partition_groups[0], 1u, &config, &report) == FT_TT_EIO,
          "the wait times out rather than hanging");
    check_u64(report.stalled_slot, 1u, "and names the core that is short");
    check_u64(report.completed_total, 1u, "with the one that finished counted");
    check_u64(report.stalled_state, FT_TT_QUEUE_STATE_SEM_WAIT, "the stalled core is in SEM_WAIT");

    config.overall_timeout_ms = 0u;
    check(ft_tt_watchdog_wait(&noc, &partition_groups[0], 1u, &config, &report) == FT_TT_EINVAL,
          "a zero overall timeout is refused: there is no untimed path");
}

/* ------------------------------------------------------------------------- */


/* ------------------------------------------------------------------------- */
/* 6. Group-level completion — "the group finished item N"                   */
/* ------------------------------------------------------------------------- */

/**
 * The hole this closes: completion was per CORE, and the only group-level answer available
 * was a SUM of `completed_count` across the group. A sum is satisfiable by the wrong thing
 * — one member doing 2N items and another doing none sums to the same 2N as every member
 * doing N — which is the "RUN_MSG_DONE is 0" family of lie one level up again.
 *
 * These cases are written against the SLOT ARRAY the kernels write, and each one names the
 * REASON the check refuses rather than only that it refused. A check with several
 * independent grounds for failing tests whichever is cheapest, and the one you care about
 * may never run (dev notes §3, and Phase 5's own H1).
 */
static void write_group_slot(simulated_core *core, uint32_t address, uint32_t member,
                             uint32_t sequence, uint32_t token)
{
    uint32_t base = address + (FT_TT_QUEUE_GROUP_HEADER_WORDS + member * FT_TT_QUEUE_GROUP_SLOT_WORDS) * 4u;

    sim_l1_store_u32(core, base, sequence);
    sim_l1_store_u32(core, base + 4u, token);
}

/* ------------------------------------------------------------------------- */
/* The HOT-PATH slot read — the one that has to respect the 16-byte L1 unit  */
/* ------------------------------------------------------------------------- */

/** Last length ft_tt_queue_group_read_slots() asked the accessor for. */
static size_t sim_last_read_bytes;

/**
 * The simulated accessor with the L1 rule the REAL one carries: 16 bytes in address AND
 * length. Without this the off-hardware suite cannot see the defect at all — a plain
 * memcpy model accepts any length, and the refusal only appears on silicon, where it cost
 * a residency three retire/re-arm cycles and a core.
 */
static int l1_strict_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                          size_t bytes)
{
    sim_last_read_bytes = bytes;
    if ((address % 16u) != 0u || (bytes % 16u) != 0u)
        return FT_TT_EINVAL;
    return simulated_read(user, x, y, address, dst, bytes);
}

static void test_group_slots_read_fits_the_l1_unit(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_noc_access strict = simulated_access();
    const uint32_t address = FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_GROUP_OFFSET;
    ft_tt_queue_group_slot wide[FT_TT_QUEUE_GROUP_MAX_MEMBERS];
    ft_tt_queue_group_slot narrow[FT_TT_QUEUE_GROUP_MAX_MEMBERS];
    simulated_core *leader;
    unsigned members;
    bool every_odd_count_fits = true;
    bool wide_agrees_with_the_header_read = true;

    printf("\ngroup: the hot-path slot read is WIDE, so its length must be a multiple of 16\n");
    sim_reset(sim_xs, sim_ys, 1u);
    leader = sim_find(sim_xs[0], sim_ys[0]);
    strict.read = l1_strict_read;

    check(ft_tt_queue_group_init(&noc, sim_xs[0], sim_ys[0], address, 8u) == FT_TT_OK,
          "a block for eight members writes");
    for (members = 0; members < 8u; members++)
        write_group_slot(leader, address, members, members + 1u, 0x1000u + members);

    /* EVERY count 1..8, not just the even ones. A slot is 8 bytes, so an ODD count is 8
     * short of the unit and is exactly what the divisibility item made reachable: N=3, 5
     * and 7 were unusable before and are the first callers this ever had. */
    for (members = 1u; members <= 8u; members++) {
        if (ft_tt_queue_group_read_slots(&strict, sim_xs[0], sim_ys[0], address, members,
                                         narrow) != FT_TT_OK ||
            (sim_last_read_bytes % 16u) != 0u)
            every_odd_count_fits = false;
        if (ft_tt_queue_group_read(&noc, sim_xs[0], sim_ys[0], address, members, wide) !=
                FT_TT_OK ||
            narrow[members - 1u].sequence != wide[members - 1u].sequence ||
            narrow[members - 1u].token != wide[members - 1u].token)
            wide_agrees_with_the_header_read = false;
    }
    check(every_odd_count_fits,
          "every member count 1..8 reads through an accessor that ENFORCES the 16-byte unit");
    check(wide_agrees_with_the_header_read,
          "and the slots-only read says the same as the read that checks the magic");

    /* The rounding must not be a licence to read past the block. */
    check_u64((uint64_t)(((size_t)FT_TT_QUEUE_GROUP_MAX_MEMBERS *
                              FT_TT_QUEUE_GROUP_SLOT_WORDS * 4u + 15u) & ~(size_t)15u) +
                  FT_TT_QUEUE_GROUP_HEADER_WORDS * 4u,
              (uint64_t)(FT_TT_QUEUE_GROUP_HEADER_WORDS * 4u +
                         FT_TT_QUEUE_GROUP_MAX_MEMBERS * FT_TT_QUEUE_GROUP_SLOT_WORDS * 4u),
              "at the widest group the rounding adds nothing, so nothing runs off the block");

    /* And the magic is what turns a wrong ADDRESS into a refusal rather than a number. */
    check(ft_tt_queue_group_read(&noc, sim_xs[0], sim_ys[0], address + 0x40u, 2u, wide) ==
              FT_TT_ERROR,
          "a block read at the wrong address is REFUSED on the magic, not believed");
}

/**
 * The COMPLETION LINE: one transaction, three assertions, and the fourth PROVABLY not
 * checked.
 *
 * Three separate things have to hold and each has its own red arm:
 *  1. the read is ONE 16-byte-aligned, 16-byte-long access — proved through the accessor
 *     that REFUSES anything else, the same instrument the group-slot read uses;
 *  2. it agrees with the wide read on the three fields it carries;
 *  3. it does NOT carry `error`, and the check beside it does not pretend to — a queue
 *     whose kernel set an error word AFTER publishing completion passes the line check and
 *     fails the wide one. That asymmetry is the entire cost of the transaction saved, and
 *     it is measured here rather than described in a comment.
 */
static void test_completion_line_is_one_transaction(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_noc_access strict = simulated_access();
    ft_tt_queue queue;
    ft_tt_queue_descriptor descriptors[TEST_CAPACITY];
    uint32_t line[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t wide[FT_TT_QUEUE_FIELD_COUNT];
    const char *reason = NULL;
    simulated_core *core;
    uint32_t reads_before;

    printf("\nqueue: the completion LINE is one transaction and drops exactly one assertion\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(sim_xs[0], sim_ys[0]);
    strict.read = l1_strict_read;
    (void)ft_tt_queue_init(&noc, sim_xs[0], sim_ys[0], FT_TT_OP_QUEUE_BASE, TEST_CAPACITY,
                           0x7331u, &queue);

    fill_nops(descriptors, 4u);
    (void)ft_tt_queue_enqueue(&noc, &queue, descriptors, 4u);
    {
        uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];

        (void)ft_tt_queue_kernel_args(&queue, sim_binding(TEST_IDLE_LIMIT), args);
        model_kernel_start(core, args);
        model_kernel_run(core, 64u);
    }

    /* (1) ONE access, through the accessor that enforces the L1 unit. */
    reads_before = sim_read_calls;
    check(ft_tt_queue_read_completion_line(&strict, &queue, line) == FT_TT_OK,
          "the line reads through an accessor that REFUSES a non-16-byte access");
    check_u64(sim_read_calls - reads_before, 1u, "and it is ONE accessor call, not two");
    check_u64((uint64_t)sim_last_read_bytes, 16u, "asking for exactly 16 bytes");

    check(ft_tt_queue_read_completion_wide(&strict, &queue, wide) == FT_TT_OK,
          "the wide read is the same answer for twice the transactions");
    check_u64((uint64_t)sim_last_read_bytes, 32u, "it asks for 32 bytes — two of them");

    /* (2) They agree on the three fields the line carries. */
    check(line[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] == wide[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] &&
              line[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] ==
                  wide[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] &&
              line[FT_TT_QUEUE_FIELD_STATE] == wide[FT_TT_QUEUE_FIELD_STATE],
          "count, token and state read the same out of one transaction as out of two");
    check(ft_tt_queue_check_completion_line(&queue, line, 4u, &reason) == FT_TT_OK,
          "a finished run passes the line check");

    /* RED: one more descriptor than the kernel ran. */
    check(ft_tt_queue_check_completion_line(&queue, line, 5u, &reason) == FT_TT_ERROR &&
              reason != NULL && strstr(reason, "completed_count") != NULL,
          "RED: a count short of `expected` is REFUSED by the line check");

    /* RED: the host poison. */
    {
        uint32_t poisoned[FT_TT_QUEUE_FIELD_COUNT];

        memcpy(poisoned, line, sizeof(poisoned));
        poisoned[FT_TT_QUEUE_FIELD_STATE] = (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON;
        check(ft_tt_queue_check_completion_line(&queue, poisoned, 4u, &reason) == FT_TT_ERROR &&
                  reason != NULL && strstr(reason, "never ran") != NULL,
              "RED: the host poison is REFUSED by the line check");
    }

    /* RED: a token that is not this run's last. */
    {
        uint32_t stale[FT_TT_QUEUE_FIELD_COUNT];

        memcpy(stale, line, sizeof(stale));
        stale[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] ^= 1u;
        check(ft_tt_queue_check_completion_line(&queue, stale, 4u, &reason) == FT_TT_ERROR &&
                  reason != NULL && strstr(reason, "token") != NULL,
              "RED: a stale token is REFUSED by the line check");
    }

    /* (3) THE ASYMMETRY. An error word set AFTER completion was published — the group
     *     post's own failure mode — is invisible to the line and visible to the wide read.
     *     A queue that arms the post must therefore keep the wide read, and this is the
     *     check that says so in a form that can fail. */
    sim_l1_store_u32(core, queue.control_address + FT_TT_QUEUE_FIELD_ERROR * 4u,
                     (uint32_t)FT_TT_QUEUE_KERR_BAD_PARAMETERS);
    check(ft_tt_queue_read_completion_line(&strict, &queue, line) == FT_TT_OK &&
              ft_tt_queue_check_completion_line(&queue, line, 4u, &reason) == FT_TT_OK,
          "an error word set after completion is INVISIBLE to the line check");
    check(ft_tt_queue_read_completion_wide(&strict, &queue, wide) == FT_TT_OK &&
              ft_tt_queue_check_completion(&queue, wide, 4u, &reason) == FT_TT_ERROR,
          "and VISIBLE to the wide one — the saved transaction IS that assertion");
    check_u64(line[FT_TT_QUEUE_FIELD_ERROR], (uint64_t)FT_TT_QUEUE_KERR_NONE,
              "the line buffer's ERROR word is ZEROED, never read");
}

/* ------------------------------------------------------------------------- */
/* The RESIDENT path's per-descriptor transaction shape                      */
/* ------------------------------------------------------------------------- */

/** PCIe accesses the WORD32 accessor has issued, and the BLOCK16 one. Counted as
 *  `bytes / unit` where the units are issued, which is what this link is priced in — a CALL
 *  count cannot tell a 4-byte read from a 64-byte one, and a BYTE count cannot tell sixteen
 *  words from four blocks. */
static uint64_t sim_word32_reads, sim_word32_writes, sim_block16_reads, sim_block16_writes;

static int word32_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                       size_t bytes)
{
    if ((address % 4u) != 0u || (bytes % 4u) != 0u)
        return FT_TT_EINVAL;
    sim_word32_reads += bytes / 4u;
    return simulated_read(user, x, y, address, dst, bytes);
}

static int word32_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                        size_t bytes)
{
    if ((address % 4u) != 0u || (bytes % 4u) != 0u)
        return FT_TT_EINVAL;
    sim_word32_writes += bytes / 4u;
    return simulated_write(user, x, y, address, src, bytes);
}

static int block16_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                        size_t bytes)
{
    if ((address % 16u) != 0u || (bytes % 16u) != 0u)
        return FT_TT_EINVAL;
    sim_block16_reads += bytes / 16u;
    return simulated_read(user, x, y, address, dst, bytes);
}

static int block16_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                         size_t bytes)
{
    /* THE SAME REFUSAL THE REAL ACCESSOR MAKES, and it is the whole reason the head bump
     * cannot come through here: `ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, ...)` demands
     * 16 bytes of alignment in ADDRESS and in LENGTH, and a head bump is four bytes at an
     * arbitrary word. So `wide` issue is structurally two accessors, never one. */
    if ((address % 16u) != 0u || (bytes % 16u) != 0u)
        return FT_TT_EINVAL;
    sim_block16_writes += bytes / 16u;
    return simulated_write(user, x, y, address, src, bytes);
}

/**
 * WHAT ONE RESIDENT DESCRIPTOR COSTS THE HOST, IN PCIe ACCESSES, ON THREE SHAPES.
 *
 * `pn_tt_residency_enqueue()` and `pn_tt_residency_await()` drive exactly this sequence
 * through the UC / WORD32 accessor, which moves FOUR BYTES PER ACCESS. That is why the
 * eltwise split's per-member await measured flat at ~58 µs whatever the width: it is
 * SEVENTEEN non-posted reads and it never touched the card's work. The recorded matmul
 * figures — 2.08 µs for a 64-byte parameter block, 4.84 µs for a descriptor plus a head
 * bump — are the ISSUE side, which is posted stores, and were never a measurement of this.
 *
 *   A  `read_field(COUNT)` + `read_control()`      1 + 16 = 17   what ships
 *   B  `read_field(COUNT)` + `read_completion()`   1 +  5 =  6   the SAME four agreements
 *   C  `read_completion_line()` on BLOCK16              1        three of the four
 *
 * B gives up nothing: the completion WINDOW is words 5..9 and all four of rule 1's
 * agreements are inside it. This test is what makes that a fact about the code rather than
 * a claim in a header.
 */
static void test_resident_await_transaction_shape(void)
{
    ft_tt_noc_access word32 = simulated_access();
    ft_tt_noc_access block16 = simulated_access();
    ft_tt_queue_descriptor descriptors[TEST_CAPACITY];
    uint32_t parameters[FT_TT_QUEUE_PARAM_WORDS];
    uint32_t control[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t window[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t line[FT_TT_QUEUE_FIELD_COUNT];
    const char *reason = NULL;
    ft_tt_queue queue;
    simulated_core *core;
    uint32_t address = 0u;
    uint32_t value = 0u;
    unsigned index;

    printf("\nresident: one descriptor is 25 posted writes and SEVENTEEN non-posted reads\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(sim_xs[0], sim_ys[0]);
    word32.read = word32_read;
    word32.write = word32_write;
    block16.read = block16_read;
    (void)ft_tt_queue_init(&word32, sim_xs[0], sim_ys[0], FT_TT_OP_QUEUE_BASE, TEST_CAPACITY,
                           0x51D3u, &queue);

    /* THE ISSUE SIDE, in the order and the calls pn_tt_residency_enqueue() makes. */
    for (index = 0; index < FT_TT_QUEUE_PARAM_WORDS; index++)
        parameters[index] = 0xA5000000u + index;
    sim_word32_writes = 0u;
    check(ft_tt_queue_write_op_parameters(&word32, &queue, 0u, parameters, &address) == FT_TT_OK,
          "the 64-byte parameter block writes");
    check_u64(sim_word32_writes, 16u, "and it is SIXTEEN word accesses, not one");
    sim_word32_writes = 0u;
    fill_nops(descriptors, 1u);
    descriptors[0].a0 = address;
    check(ft_tt_queue_enqueue(&word32, &queue, descriptors, 1u) == FT_TT_OK,
          "the 32-byte descriptor and the 4-byte head bump write");
    check_u64(sim_word32_writes, 9u, "eight words of ring and one of head");

    {
        uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];

        (void)ft_tt_queue_kernel_args(&queue, sim_binding(TEST_IDLE_LIMIT), args);
        model_kernel_start(core, args);
        model_kernel_run(core, 64u);
    }

    /* SHAPE A — what ships. The poll, then the whole 64-byte control block. */
    sim_word32_reads = 0u;
    check(ft_tt_queue_read_field(&word32, &queue, FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &value) ==
                  FT_TT_OK &&
              value >= 1u,
          "the poll sees the descriptor complete");
    check(ft_tt_queue_read_control(&word32, &queue, control) == FT_TT_OK,
          "and the control block reads");
    check_u64(sim_word32_reads, 17u,
              "SHAPE A: the shipped await is SEVENTEEN non-posted reads a descriptor");
    check(ft_tt_queue_check_completion(&queue, control, 1u, &reason) == FT_TT_OK,
          "and all four agreements hold");

    /* SHAPE B — the completion WINDOW. Same accessor, same verdict, six accesses. */
    sim_word32_reads = 0u;
    check(ft_tt_queue_read_field(&word32, &queue, FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &value) ==
              FT_TT_OK,
          "the same poll");
    check(ft_tt_queue_read_completion(&word32, &queue, window) == FT_TT_OK,
          "and the completion window reads");
    check_u64(sim_word32_reads, 6u, "SHAPE B: the window is SIX — eleven accesses removed");
    check(ft_tt_queue_check_completion(&queue, window, 1u, &reason) == FT_TT_OK,
          "the FOUR-agreement check still runs, because ERROR is word 9 and inside it");
    check(window[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] ==
                  control[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] &&
              window[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] ==
                  control[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] &&
              window[FT_TT_QUEUE_FIELD_STATE] == control[FT_TT_QUEUE_FIELD_STATE] &&
              window[FT_TT_QUEUE_FIELD_ERROR] == control[FT_TT_QUEUE_FIELD_ERROR],
          "and it is the SAME four words the control block carried");

    /* SHAPE C — the line, on the accessor that enforces the 16-byte L1 unit. */
    sim_block16_reads = 0u;
    check(ft_tt_queue_completion_line_is_addressable(&queue),
          "the queue's line is addressable through a BLOCK16 accessor");
    check(ft_tt_queue_read_completion_line(&block16, &queue, line) == FT_TT_OK,
          "the line reads");
    check_u64(sim_block16_reads, 1u, "SHAPE C: ONE access — the poll IS the check");
    check(ft_tt_queue_check_completion_line(&queue, line, 1u, &reason) == FT_TT_OK,
          "and the three agreements it carries hold");

    /* RED, on every shape: a stale token must be refused, or none of the above is a check. */
    {
        uint32_t stale[FT_TT_QUEUE_FIELD_COUNT];

        memcpy(stale, control, sizeof(stale));
        stale[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] ^= 1u;
        check(ft_tt_queue_check_completion(&queue, stale, 1u, &reason) == FT_TT_ERROR,
              "RED: shape A refuses a stale token");
        memcpy(stale, window, sizeof(stale));
        stale[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] ^= 1u;
        check(ft_tt_queue_check_completion(&queue, stale, 1u, &reason) == FT_TT_ERROR,
              "RED: shape B refuses a stale token");
        memcpy(stale, line, sizeof(stale));
        stale[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] ^= 1u;
        check(ft_tt_queue_check_completion_line(&queue, stale, 1u, &reason) == FT_TT_ERROR,
              "RED: shape C refuses a stale token");
    }

    /* RED: shape B is not vacuous about ERROR. An error word set AFTER completion — the one
     * state the line cannot see — must still be refused by the window, because keeping that
     * fourth agreement is the whole argument for making B unconditional. */
    sim_l1_store_u32(core, queue.control_address + FT_TT_QUEUE_FIELD_ERROR * 4u,
                     (uint32_t)FT_TT_QUEUE_KERR_BAD_PARAMETERS);
    check(ft_tt_queue_read_completion(&word32, &queue, window) == FT_TT_OK &&
              ft_tt_queue_check_completion(&queue, window, 1u, &reason) == FT_TT_ERROR,
          "RED: the completion WINDOW still sees an error word the line cannot");
    sim_l1_store_u32(core, queue.control_address + FT_TT_QUEUE_FIELD_ERROR * 4u,
                     (uint32_t)FT_TT_QUEUE_KERR_NONE);

    /* RED: a queue the fast path CANNOT reach has to say so before it is used. A word-aligned
     * control address is legal to `ft_tt_queue_init()` and fatal to a BLOCK16 read, and on
     * the resident path that failure arrives as a completion that never completes. */
    {
        ft_tt_queue misplaced = queue;

        misplaced.control_address = FT_TT_OP_QUEUE_BASE + 4u;
        check(!ft_tt_queue_completion_line_is_addressable(&misplaced),
              "RED: a word-aligned control block is REFUSED as line-addressable");
        check(ft_tt_queue_read_completion_line(&block16, &misplaced, line) != FT_TT_OK,
              "RED: and the BLOCK16 accessor really does refuse it");
        misplaced.control_address = FT_TT_OP_QUEUE_BASE + 16u;
        check(ft_tt_queue_completion_line_is_addressable(&misplaced),
              "while a 16-byte-aligned one is accepted — the guard is not a constant false");
    }
}

/**
 * WHAT ONE RESIDENT DESCRIPTOR COSTS THE HOST TO *ISSUE*, IN PCIe ACCESSES, ON TWO SHAPES —
 * AND WHETHER THE ORDERING THE SECOND SHAPE PUTS AT RISK IS CHECKED.
 *
 * The await ladder above took the READ half from seventeen accesses to one. The write half is
 * the same arithmetic on the same link: `pn_tt_residency_enqueue()` writes a 64-byte parameter
 * block, a 32-byte descriptor and a 4-byte head bump, and through the UC / WORD32 accessor
 * that is **25 posted accesses**. The parameter block and the ring slot are both 16-byte
 * aligned in address AND length, so both fit the BLOCK16 L1 accessor; the head bump is four
 * bytes at an arbitrary word and does not. So:
 *
 *   word   16 + 8 + 1 = 25 accesses, ONE accessor
 *   wide    4 + 2      =  6 BLOCK16 accesses, PLUS 1 WORD32 head bump = **7**
 *
 * **AND THE SPLIT IS THE WHOLE RISK.** `wide` puts the descriptor on the write-combined
 * window and the head bump on the uncached one, and "descriptors before head" is one of the
 * two orderings this protocol rests on (see ft_tt_queue.h). If the head lands first the kernel
 * executes whatever the ring slot held — a *previous* descriptor, which is well-formed and
 * carries a STALE TOKEN. That is not a hang and not a refusal on the card; it is a wrong
 * answer, and the only thing standing between it and the model is the four-agreement check.
 *
 * So the last part of this test does not check arithmetic at all. It publishes a head for a
 * descriptor that was never written — the exact inversion — and requires the check to REFUSE
 * and to NAME the token. A `wide` arm that runs clean on silicon is evidence only because
 * this is what happens when it does not.
 */
static void test_resident_issue_transaction_shape(void)
{
    ft_tt_noc_access word32 = simulated_access();
    ft_tt_noc_access block16 = simulated_access();
    ft_tt_queue_descriptor descriptors[TEST_CAPACITY];
    uint32_t parameters[FT_TT_QUEUE_PARAM_WORDS];
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    const char *reason = NULL;
    ft_tt_queue queue;
    simulated_core *core;
    uint32_t address = 0u;
    unsigned index;

    printf("\nresident: the ISSUE side is 25 posted accesses, and SEVEN on the wide shape\n");
    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(sim_xs[0], sim_ys[0]);
    word32.read = word32_read;
    word32.write = word32_write;
    block16.read = block16_read;
    block16.write = block16_write;
    (void)ft_tt_queue_init(&word32, sim_xs[0], sim_ys[0], FT_TT_OP_QUEUE_BASE, TEST_CAPACITY,
                           0x1550u, &queue);
    for (index = 0; index < FT_TT_QUEUE_PARAM_WORDS; index++)
        parameters[index] = 0x5E000000u + index;
    {
        uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];

        (void)ft_tt_queue_kernel_args(&queue, sim_binding(TEST_IDLE_LIMIT), args);
        model_kernel_start(core, args);
    }

    /* SHAPE `word` — every write through the UC accessor, which is what ships. */
    sim_word32_writes = 0u;
    sim_block16_writes = 0u;
    check(ft_tt_queue_write_op_parameters(&word32, &queue, 0u, parameters, &address) == FT_TT_OK,
          "word: the 64-byte parameter block writes");
    fill_nops(descriptors, 1u);
    descriptors[0].a0 = address;
    check(ft_tt_queue_stage(&word32, &queue, descriptors, 1u) == FT_TT_OK,
          "word: the 32-byte descriptor stages");
    check(ft_tt_queue_publish(&word32, &queue) == FT_TT_OK, "word: the head bump publishes it");
    check_u64(sim_word32_writes, 25u, "word: 16 params + 8 ring + 1 head = TWENTY-FIVE accesses");
    check_u64(sim_block16_writes, 0u, "word: and nothing went wide");
    model_kernel_run(core, 64u);
    check(ft_tt_queue_read_completion(&word32, &queue, fields) == FT_TT_OK &&
              ft_tt_queue_check_completion(&queue, fields, 1u, &reason) == FT_TT_OK,
          "word: the descriptor completes and all four agreements hold");

    /* SHAPE `wide` — the two 16-byte-aligned writes through BLOCK16, the head bump through
     * WORD32 because nothing else can carry it. Same bytes, same order, same L1. */
    sim_word32_writes = 0u;
    sim_block16_writes = 0u;
    check(ft_tt_queue_write_op_parameters(&block16, &queue, 0u, parameters, &address) == FT_TT_OK,
          "wide: the parameter block writes through the BLOCK16 accessor");
    check_u64(sim_block16_writes, 4u, "wide: 64 bytes is FOUR accesses, not sixteen");
    fill_nops(descriptors, 1u);
    descriptors[0].a0 = address;
    check(ft_tt_queue_stage(&block16, &queue, descriptors, 1u) == FT_TT_OK,
          "wide: the descriptor stages through the same accessor");
    check_u64(sim_block16_writes, 6u, "wide: 32 bytes is TWO more");
    check(ft_tt_queue_publish(&word32, &queue) == FT_TT_OK,
          "wide: the head bump still goes through WORD32");
    check_u64(sim_word32_writes, 1u, "wide: ONE word access, and it is the head");
    check_u64(sim_block16_writes + sim_word32_writes, 7u, "wide: SEVEN accesses against 25");
    model_kernel_run(core, 64u);
    check(ft_tt_queue_read_completion(&word32, &queue, fields) == FT_TT_OK &&
              ft_tt_queue_check_completion(&queue, fields, 2u, &reason) == FT_TT_OK,
          "wide: the SAME four agreements hold — the bytes that landed are the same bytes");

    /* RED: the head bump CANNOT go wide. A four-byte write at a word address is refused by
     * the 16-byte L1 rule, so the two-accessor split is structural and not a convention —
     * a caller that tried to put the whole issue on one wide accessor gets an error, not a
     * silently different transaction. */
    check(ft_tt_queue_write_head(&block16, &queue, queue.head) != FT_TT_OK,
          "RED: the BLOCK16 accessor REFUSES a 4-byte head bump");
    /* AND THE STAND-IN IS TIED TO THE SHIPPED RULE. The refusal above comes from this file's
     * simulated accessor; this asks the function the real one calls, so the two cannot drift
     * and the structural claim is not a property of the test. */
    check(ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false,
                                   queue.control_address + FT_TT_QUEUE_FIELD_HEAD * 4u,
                                   4u) != FT_TT_OK,
          "RED: and ft_tt_noc_check_transfer() refuses it for the same reason on the card");
    check(ft_tt_queue_issue_is_wide_addressable(&queue),
          "while the parameter blocks and the ring slot BOTH fit it — the guard is not a "
          "constant false");
    {
        /* RED: a queue the wide shape cannot reach has to SAY SO before it is used. A
         * word-aligned control block is legal to ft_tt_queue_init() and is refused by every
         * 16-byte write, so a caller that armed `wide` on one would fail on its first
         * descriptor instead of falling back. */
        ft_tt_queue misplaced = queue;

        misplaced.control_address = FT_TT_OP_QUEUE_BASE + 4u;
        misplaced.ring_address = misplaced.control_address + FT_TT_QUEUE_RING_OFFSET;
        check(!ft_tt_queue_issue_is_wide_addressable(&misplaced),
              "RED: a word-aligned control block is REFUSED as wide-issue addressable");
    }

    /* RED — THE ORDERING, WHICH IS THE ONLY THING `wide` PUTS AT RISK.
     *
     * The realistic inversion is not "the slot is empty" — it is "the slot still holds the
     * descriptor that occupied it one lap of the ring ago", which is WELL FORMED and executes
     * cleanly. So run the ring all the way round first, and only then publish a head for a
     * descriptor that was never written. The card then completes a legal descriptor, the
     * COUNT is exactly what the host expected, and the ONLY thing that separates that from a
     * correct token is the per-index nonce. If this reads green, a clean `wide` arm on silicon
     * means nothing. */
    while (queue.issued < TEST_CAPACITY) {
        fill_nops(descriptors, 1u);
        descriptors[0].a0 = address;
        check(ft_tt_queue_enqueue(&word32, &queue, descriptors, 1u) == FT_TT_OK,
              "the ring runs one full lap, so slot 0 holds a well-formed old descriptor");
        model_kernel_run(core, 64u);
    }
    {
        const uint32_t expected = queue.issued + 1u;

        check(ft_tt_queue_write_head(&word32, &queue, queue.head + 1u) == FT_TT_OK,
              "RED: a head is published for a descriptor that was never written");
        model_kernel_run(core, 64u);
        check(ft_tt_queue_read_completion(&word32, &queue, fields) == FT_TT_OK,
              "RED: and the card completed something");
        check_u64(fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT], expected,
                  "RED: the COUNT is exactly what the host expected — it catches nothing");
        check_u64(fields[FT_TT_QUEUE_FIELD_ERROR], (uint64_t)FT_TT_QUEUE_KERR_NONE,
                  "RED: and the kernel reports NO error — a stale slot is a legal descriptor");
        check(ft_tt_queue_check_completion(&queue, fields, expected, &reason) == FT_TT_ERROR &&
                  reason != NULL && strstr(reason, "token") != NULL,
              "RED: the four-agreement check REFUSES it and the TOKEN is what caught it");
    }
}

static void test_group_completion_needs_every_member(void)
{
    const ft_tt_grid_info grid = p150a_grid();
    ft_tt_noc_access noc = simulated_access();
    ft_tt_group_member_completion members[2];
    const uint32_t address = FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_GROUP_OFFSET;
    const uint32_t token = 0xA1B2C3D4u;
    const char *reason = NULL;
    uint32_t short_slot = 0xFFFFFFFFu;
    simulated_core *leader;
    unsigned slot;

    printf("\ngroup: completion is per GROUP now, and one member behind refuses it\n");

    (void)ft_tt_group_partition(&grid, 0u, 2u, 1u, partition_groups, partition_placements,
                                partition_queues, PARTITION_STORAGE);
    {
        uint16_t xs[2];
        uint16_t ys[2];

        for (slot = 0; slot < 2u; slot++) {
            const ft_tt_launch_core_placement *placement =
                ft_tt_launch_core_set_at(&partition_groups[0].cores, slot);
            xs[slot] = placement->x;
            ys[slot] = placement->y;
        }
        sim_reset(xs, ys, 2u);
        leader = sim_find(xs[0], ys[0]);
    }

    check(ft_tt_group_completion_init(&noc, &partition_groups[0], address) == FT_TT_OK,
          "the completion block writes on the group's FIRST core");
    check_u64(partition_groups[0].completion_address, address,
              "and the group remembers where it put it");
    check_u64(sim_l1_u32(leader, address + FT_TT_QUEUE_GROUP_MAGIC * 4u),
              FT_TT_QUEUE_GROUP_MAGIC_VALUE, "the block carries its magic");
    check_u64(sim_l1_u32(leader, address + FT_TT_QUEUE_GROUP_MEMBERS * 4u), 2u,
              "and the live member count, not a constant");

    /* Nothing posted yet. */
    check(ft_tt_group_completion_read(&noc, &partition_groups[0], members, 2u) == FT_TT_OK,
          "the block reads back in one access");
    check(ft_tt_group_check_completion(&partition_groups[0], members, 2u, 1u, token, &reason,
                                       &short_slot) == FT_TT_ERROR,
          "with nothing posted, sequence 1 is REFUSED");
    check(reason != NULL && strstr(reason, "has not reached") != NULL,
          "and the reason is that a member has not reached it");

    /* ONE member posts. This is the case a SUM would pass and a slot array must not. */
    write_group_slot(leader, address, 0u, 1u, token);
    (void)ft_tt_group_completion_read(&noc, &partition_groups[0], members, 2u);
    check_u64(members[0].sequence, 1u, "member 0's slot carries the sequence it posted");
    check_u64(members[1].sequence, 0u, "and member 1's is untouched");
    check(ft_tt_group_check_completion(&partition_groups[0], members, 2u, 1u, token, &reason,
                                       &short_slot) == FT_TT_ERROR,
          "ONE member at sequence 1 is still REFUSED — a sum would have passed here");
    check_u64(short_slot, 1u, "and it names WHICH member is behind");

    /* Both post. */
    write_group_slot(leader, address, 1u, 1u, token);
    (void)ft_tt_group_completion_read(&noc, &partition_groups[0], members, 2u);
    check(ft_tt_group_check_completion(&partition_groups[0], members, 2u, 1u, token, &reason,
                                       &short_slot) == FT_TT_OK,
          "with BOTH members at sequence 1, it passes");

    /* A stale block from a previous run: a large enough sequence, the wrong token. This is
     * the case the sequence alone cannot refuse. */
    write_group_slot(leader, address, 0u, 99u, 0xDEADBEEFu);
    write_group_slot(leader, address, 1u, 99u, 0xDEADBEEFu);
    (void)ft_tt_group_completion_read(&noc, &partition_groups[0], members, 2u);
    check(ft_tt_group_check_completion(&partition_groups[0], members, 2u, 1u, token, &reason,
                                       &short_slot) == FT_TT_ERROR,
          "a residue with a HIGHER sequence and the wrong token is refused");
    check(reason != NULL && strstr(reason, "token") != NULL,
          "and it refuses BECAUSE of the token, not the sequence");

    /* A block at the wrong address has no magic, and that is a refusal rather than a
     * plausible number. Zeroed storage answers the same as absent storage (dev notes §3),
     * so the magic is what makes "you read the wrong place" visible. */
    sim_l1_store_u32(leader, address + FT_TT_QUEUE_GROUP_MAGIC * 4u, 0u);
    check(ft_tt_group_completion_read(&noc, &partition_groups[0], members, 2u) == FT_TT_ERROR,
          "a block without the magic is REFUSED, not read as zeros");
}

/* ------------------------------------------------------------------------- */
/* 7. Per-processor status, and the retirement it makes honest               */
/* ------------------------------------------------------------------------- */

/**
 * With five processors on one queue, the control block's `state` word has ONE writer, so a
 * reader that has exited while the compute thread is still spinning out a bound reads EXIT
 * — and a shutdown that believed it would relaunch a second kernel onto a core that still
 * has one running. The status array is what makes "retired" mean all five.
 */
static void test_status_says_which_processor(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    const uint32_t triad = (1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) |
                           (1u << FT_TT_PROC_MATH0) | (1u << FT_TT_PROC_MATH1) |
                           (1u << FT_TT_PROC_MATH2);
    uint32_t status[FT_TT_QUEUE_STATUS_WORDS];
    uint32_t pending = 0xFFFFFFFFu;
    simulated_core *core;
    unsigned processor;

    printf("\nstatus: 'the core exited' is not the same as 'all five exited'\n");

    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(1u, 2u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x7777u, &queue);

    /* The control block says EXIT — which is exactly the lie a triad can tell. */
    sim_set_control(core, FT_TT_QUEUE_FIELD_STATE, FT_TT_QUEUE_STATE_EXIT);
    check(ft_tt_queue_all_processors_exited(&noc, &queue, 0u, &pending) == FT_TT_OK,
          "with NO mask, the control block is the whole answer (the BRISC-only kernel)");
    check(ft_tt_queue_all_processors_exited(&noc, &queue, triad, &pending) == FT_TT_EIO,
          "with the TRIAD mask, that same EXIT is NOT enough");
    check_u64(pending, triad, "and every processor is still pending");

    /* Retire four of the five. */
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (processor == FT_TT_PROC_MATH0)
            continue;
        sim_l1_store_u32(core,
                         FT_TT_QUEUE_STATUS_ADDRESS(queue.control_address, processor) +
                             FT_TT_QUEUE_STATUS_STATE * 4u,
                         FT_TT_QUEUE_STATE_EXIT);
    }
    check(ft_tt_queue_all_processors_exited(&noc, &queue, triad, &pending) == FT_TT_EIO,
          "four of five is still not retired");
    check_u64(pending, 1u << FT_TT_PROC_MATH0, "and the pending mask names the ONE that is not");

    sim_l1_store_u32(core,
                     FT_TT_QUEUE_STATUS_ADDRESS(queue.control_address, FT_TT_PROC_MATH0) +
                         FT_TT_QUEUE_STATUS_STATE * 4u,
                     FT_TT_QUEUE_STATE_EXIT);
    check(ft_tt_queue_all_processors_exited(&noc, &queue, triad, &pending) == FT_TT_OK,
          "all five at EXIT is retired");

    /* The status slot is readable as a block, and it is per PROCESSOR — two processors must
     * not share one, or a heartbeat from one would look like liveness from the other. */
    sim_l1_store_u32(core,
                     FT_TT_QUEUE_STATUS_ADDRESS(queue.control_address, FT_TT_PROC_MATH2) +
                         FT_TT_QUEUE_STATUS_CURSOR * 4u,
                     42u);
    (void)ft_tt_queue_read_status(&noc, &queue, FT_TT_PROC_MATH2, status);
    check_u64(status[FT_TT_QUEUE_STATUS_CURSOR], 42u, "a status slot reads back per processor");
    (void)ft_tt_queue_read_status(&noc, &queue, FT_TT_PROC_MATH1, status);
    check_u64(status[FT_TT_QUEUE_STATUS_CURSOR], 0u, "and its neighbour is unaffected");
    check(ft_tt_queue_read_status(&noc, &queue, FT_TT_PROC_COUNT, status) == FT_TT_EINVAL,
          "a processor index off the end is refused");
}

/* ------------------------------------------------------------------------- */
/* 8. The matmul parameter block                                             */
/* ------------------------------------------------------------------------- */

static void test_matmul_parameter_blocks(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    uint32_t words[FT_TT_QUEUE_MATMUL_WORDS];
    uint32_t first = 0u;
    uint32_t second = 0u;
    simulated_core *core;
    unsigned index;

    printf("\nmatmul params: a shape that does not fit six descriptor words gets a pointer\n");

    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(1u, 2u);
    (void)ft_tt_queue_init(&noc, 1u, 2u, FT_TT_OP_QUEUE_BASE, TEST_CAPACITY, 0x8888u, &queue);

    for (index = 0; index < FT_TT_QUEUE_MATMUL_WORDS; index++)
        words[index] = 0x1000u + index;

    check(ft_tt_queue_write_matmul_parameters(&noc, &queue, 0u, words, &first) == FT_TT_OK,
          "a parameter block writes");
    check(ft_tt_queue_write_matmul_parameters(&noc, &queue, 1u, words, &second) == FT_TT_OK,
          "and a second one writes");
    check_u64(second - first, FT_TT_QUEUE_MATMUL_PARAM_BYTES,
              "consecutive slots are one block apart, so a batch can be pipelined");
    check(first >= queue.control_address + FT_TT_QUEUE_MATMUL_PARAM_OFFSET,
          "and they land in the parameter region, clear of the ring");
    check_u64(sim_l1_u32(core, first + FT_TT_QUEUE_MATMUL_TILE_INNER * 4u), 0x1001u,
              "the words arrive in the order the kernel indexes them");
    check(ft_tt_queue_write_matmul_parameters(&noc, &queue, FT_TT_QUEUE_MATMUL_PARAM_SLOTS, words,
                                              &first) == FT_TT_EINVAL,
          "a slot past the end is refused rather than written over the group block");
}

/**
 * The bit-30 value of a matmul's flag word (`rtriad_mixw`), read off the kernel header's own
 * text: `enum { kMatmulWeightBfp4 = 1u << N };`. -1 if the header or the line is missing,
 * which the caller reports as a FAILURE: an unreadable contract is not a passing one.
 */
static long kernel_weight_bfp4_flag(void)
{
    const char *directory = getenv("FT_TT_KERNEL_DIR");
    char path[512];
    char line[256];
    long value = -1;
    FILE *header;

    snprintf(path, sizeof(path), "%s/resident_queue_protocol.h",
             directory && directory[0] != '\0' ? directory : "../kernels");
    header = fopen(path, "r");
    if (!header)
        return -1;
    while (fgets(line, sizeof(line), header)) {
        unsigned shift;

        if (sscanf(line, " enum { kMatmulWeightBfp4 = 1u << %u }", &shift) == 1 && shift < 32u)
            value = (long)(1ul << shift);
    }
    fclose(header);
    return value;
}

/**
 * The weight-format flag of the mixed-weight resident matmul: the encoder maps the two tile
 * sizes and refuses every other, and the pre-send check refuses a flag word that disagrees
 * with the weight — the only guard there is, because DRAM tiles carry no header and a wrong
 * flag is a wrong answer with a clean run.
 */
static void test_matmul_weight_flag(void)
{
    const uint32_t bfp4 = FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4;
    const uint32_t others = 1u | (1u << 8) | (1u << 18); /* N-major, pipelined, A-reuse */
    uint32_t flag = 0xDEADBEEFu;

    printf("\nmatmul weight flag: 1,088 B -> 0, 576 B -> bit 30, anything else refused\n");

    check_u64((uint64_t)kernel_weight_bfp4_flag(), bfp4,
              "the host mirror equals the kernel header's kMatmulWeightBfp4");
    check(ft_tt_queue_matmul_weight_flag(1088u, &flag) == FT_TT_OK && flag == 0u,
          "a 1,088 B (Bfp8_b) weight encodes as 0");
    flag = 0xDEADBEEFu;
    check(ft_tt_queue_matmul_weight_flag(576u, &flag) == FT_TT_OK && flag == 0x40000000u,
          "a 576 B (Bfp4_b) weight encodes as bit 30");
    flag = 0xDEADBEEFu;
    check(ft_tt_queue_matmul_weight_flag(2048u, &flag) == FT_TT_EINVAL &&
              ft_tt_queue_matmul_weight_flag(832u, &flag) == FT_TT_EINVAL &&
              ft_tt_queue_matmul_weight_flag(0u, &flag) == FT_TT_EINVAL && flag == 0xDEADBEEFu,
          "bf16, Bfp6_b and zero tiles are refused, and the out word is left alone");
    check(ft_tt_queue_matmul_weight_flag(576u, NULL) == FT_TT_EINVAL, "a NULL out is refused");

    check(ft_tt_queue_matmul_check_weight(1088u, others, false) == FT_TT_OK &&
              ft_tt_queue_matmul_check_weight(1088u, others, true) == FT_TT_OK &&
              ft_tt_queue_matmul_check_weight(576u, others | bfp4, true) == FT_TT_OK,
          "Bfp8 anywhere and flagged Bfp4 on a mixw residency pass, other bits ignored");
    check(ft_tt_queue_matmul_check_weight(576u, others, true) == FT_TT_EINVAL,
          "REFUSED: a 576 B weight WITHOUT the flag");
    check(ft_tt_queue_matmul_check_weight(1088u, others | bfp4, true) == FT_TT_EINVAL,
          "REFUSED: a 1,088 B weight WITH the flag");
    check(ft_tt_queue_matmul_check_weight(576u, others | bfp4, false) == FT_TT_EINVAL &&
              ft_tt_queue_matmul_check_weight(1088u, others | bfp4, false) == FT_TT_EINVAL,
          "REFUSED: a 576 B weight or bit 30 on a non-mixw residency (it ignores the flag)");
    check(ft_tt_queue_matmul_check_weight(2048u, others, true) == FT_TT_EINVAL,
          "REFUSED: a tile size that is neither format");
    check(ft_tt_queue_matmul_check_weight(576u, others | bfp4 | 0x80000000u, true) ==
              FT_TT_EINVAL,
          "REFUSED: bit 31 on a mixw residency (its reader refuses it)");
}

/**
 * The parameter CACHE: a second region whose blocks are pointed at, not rewritten.
 *
 * What has to hold is geometric and it is the whole safety argument — the cache must not
 * overlap the rotating slots (which ARE rewritten, every few descriptors) and must not
 * overlap the group block. Both ends are checked against a real write here as well as by
 * the static asserts, because an assert on a macro cannot catch a function that computes
 * the address a second way.
 */
static void test_parameter_cache_is_a_disjoint_region(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_queue queue;
    uint32_t words[FT_TT_QUEUE_PARAM_WORDS];
    uint32_t rotating = 0u;
    uint32_t cached = 0u;
    uint32_t last = 0u;
    simulated_core *core;
    unsigned index;

    printf("\nmatmul params: the CACHE region is disjoint from the slots and the group block\n");

    sim_reset(sim_xs, sim_ys, 1u);
    core = sim_find(sim_xs[0], sim_ys[0]);
    (void)ft_tt_queue_init(&noc, sim_xs[0], sim_ys[0], FT_TT_OP_QUEUE_BASE, TEST_CAPACITY,
                           0x9999u, &queue);

    for (index = 0; index < FT_TT_QUEUE_PARAM_WORDS; index++)
        words[index] = 0x2000u + index;

    check(ft_tt_queue_write_matmul_parameters(
              &noc, &queue, FT_TT_QUEUE_MATMUL_PARAM_SLOTS - 1u, words, &rotating) == FT_TT_OK,
          "the LAST rotating slot writes");
    check(ft_tt_queue_write_param_cache(&noc, &queue, 0u, words, &cached) == FT_TT_OK,
          "and cache slot 0 writes");
    check(cached >= rotating + FT_TT_QUEUE_MATMUL_PARAM_BYTES,
          "the cache begins AFTER the last rotating slot ends — a cached block cannot be "
          "rewritten by the slot rotation");

    check(ft_tt_queue_write_param_cache(&noc, &queue, FT_TT_QUEUE_PARAM_CACHE_PRIMARY_SLOTS - 1u,
                                        words, &last) == FT_TT_OK,
          "the LAST primary cache slot writes");
    check(last + FT_TT_QUEUE_MATMUL_PARAM_BYTES <=
              queue.control_address + FT_TT_QUEUE_GROUP_OFFSET,
          "and it ends at or before the group completion block");
    {
        uint32_t first_overflow = 0u, last_overflow = 0u, overlaps = 0u, slot, other;

        check(ft_tt_queue_write_param_cache(&noc, &queue, FT_TT_QUEUE_PARAM_CACHE_PRIMARY_SLOTS,
                                            words, &first_overflow) == FT_TT_OK &&
                  first_overflow == queue.control_address + FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_OFFSET,
              "the first OVERFLOW slot (plan S2) writes at 0x2000 from the control block");
        check(first_overflow >= queue.control_address + FT_TT_QUEUE_RING_OFFSET +
                                    64u * FT_TT_QUEUE_DESCRIPTOR_BYTES &&
                  first_overflow >= queue.control_address + 0x1040u,
              "past the largest ring (64 slots) and the MoE mailbox (0x1000..0x1040)");
        check(ft_tt_queue_write_param_cache(&noc, &queue, FT_TT_QUEUE_PARAM_CACHE_SLOTS - 1u, words,
                                            &last_overflow) == FT_TT_OK &&
                  last_overflow + FT_TT_QUEUE_MATMUL_PARAM_BYTES <= rotating -
                      (FT_TT_QUEUE_MATMUL_PARAM_SLOTS - 1u) * FT_TT_QUEUE_MATMUL_PARAM_BYTES,
              "the LAST overflow slot ends at or before rotating slot 0");
        for (slot = 0u; slot < FT_TT_QUEUE_PARAM_CACHE_SLOTS; slot++)
            for (other = slot + 1u; other < FT_TT_QUEUE_PARAM_CACHE_SLOTS; other++) {
                const uint32_t a = FT_TT_QUEUE_PARAM_CACHE_ADDRESS(0u, slot);
                const uint32_t b = FT_TT_QUEUE_PARAM_CACHE_ADDRESS(0u, other);

                overlaps += (a < b + FT_TT_QUEUE_MATMUL_PARAM_BYTES &&
                             b < a + FT_TT_QUEUE_MATMUL_PARAM_BYTES);
            }
        check_u64(overlaps, 0u, "no two of the 768 cache blocks overlap");
    }
    check(ft_tt_queue_write_param_cache(&noc, &queue, FT_TT_QUEUE_PARAM_CACHE_SLOTS, words,
                                        &last) == FT_TT_EINVAL,
          "a cache slot past the end is REFUSED, not written over the group block");

    check_u64(sim_l1_u32(core, cached + FT_TT_QUEUE_MATMUL_TILE_INNER * 4u), 0x2001u,
              "the cached words arrive in the order the kernel indexes them");
    /* The magic of the group block is what a run-off would destroy, so prove it survives
     * the widest cache write this region admits. */
    check(ft_tt_queue_group_init(&noc, sim_xs[0], sim_ys[0],
                                 FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_GROUP_OFFSET, 8u) == FT_TT_OK,
          "a group block writes above the cache");
    (void)ft_tt_queue_write_param_cache(&noc, &queue, FT_TT_QUEUE_PARAM_CACHE_SLOTS - 1u, words,
                                        &last);
    check_u64(sim_l1_u32(core, FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_GROUP_OFFSET),
              FT_TT_QUEUE_GROUP_MAGIC_VALUE,
              "and the last cache write leaves the group magic intact");
}

int main(void)
{
    printf("Phase 5 work queue and core groups — off-device\n");

    test_queue_init_writes_a_recognisable_block();
    test_enqueue_is_one_write_plus_a_head_bump();
    test_enqueue_wraps_the_ring();
    test_completion_cannot_report_work_that_never_ran();
    test_completion_rejects_a_stale_token();
    test_two_runs_cannot_share_a_token();
    test_stop_request_retires_a_blocked_kernel();
    test_stop_sentinel_is_ordered_with_the_work();
    test_bounded_residency();
    test_partition_is_derived_from_the_live_grid();
    test_partition_on_a_synthetic_twelve_column_part();
    test_partition_refusals_and_disjointness();
    test_watchdog_reports_alive_but_blocked();
    test_group_completion_needs_every_member();
    test_group_slots_read_fits_the_l1_unit();
    test_completion_line_is_one_transaction();
    test_resident_await_transaction_shape();
    test_resident_issue_transaction_shape();
    test_status_says_which_processor();
    test_matmul_parameter_blocks();
    test_parameter_cache_is_a_disjoint_region();
    test_matmul_weight_flag();

    printf("\n%d/%d checks failed\n", checks_failed, checks_run);
    return checks_failed == 0 ? 0 : 1;
}
