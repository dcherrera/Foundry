/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase5_resident.c
 * @brief Phase 5 on the card: a RESIDENT kernel polling an L1 work queue, the dispatch
 *   A/B against the Phase 4 slow-dispatch arm, two independent workloads on disjoint core
 *   groups, a cross-group semaphore, and a DELIBERATE DEADLOCK the watchdog catches with
 *   the card still usable afterwards.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES THE A/B HONEST
 * ---------------------------------------------------------------------------
 * Both arms run `kernels/resident_queue.c`. The only difference is compile-time arg 0:
 * 0 is RESIDENT (the loop), 1 is ONE-SHOT (the loop stopped after one descriptor). So the
 * per-descriptor code is the same source, the same opcode and the same stores, and the
 * measured difference is the DISPATCH — not two implementations of the work.
 *
 *   arm A, slow dispatch: per descriptor, plant it, write the launch message, send GO,
 *                         poll for RUN_MSG_DONE. This is the Phase 4 shape, and Phase 2
 *                         measured it at 16 us over 1000 launches.
 *   arm B, queue dispatch: enqueue a batch, and poll ONCE at the end. The host never
 *                         round-trips the card between descriptors.
 *
 * **The arms are interleaved.** Two 1.8 GHz cores also driving eight K80s drift enough
 * over ten seconds to swamp a microsecond, and measuring one arm to completion and then
 * the other once reported a NEGATIVE cost for CONFIGURE_TLB (dev notes §8).
 *
 * A third number is reported and is the one comparable to upstream's 2.08 us op-to-op
 * golden: DEVICE-SIDE op-to-op, measured by filling the ring, bumping the head ONCE, and
 * timing until the completion count arrives. No host work is inside that window at all.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES THE COMPLETION CHECKS HONEST
 * ---------------------------------------------------------------------------
 * Nothing here retires work on `ft_tt_launch_wait_done_all()`. That call reported eight
 * cores complete with one never sent a go message, because `RUN_MSG_DONE` is 0 and is
 * what a core's own boot leaves in the mailbox (dev notes §3). Completion is
 * `completed_count` plus `completed_token`, both written by the resident kernel AFTER the
 * descriptor's work, with the token a per-run nonce — and stage 1 runs the negative case,
 * checking that an unlaunched core FAILS.
 *
 * Per-group data is checked PAIRWISE DISTINCT as well as against a reference, because the
 * multicore seed once collided mod 16 and four of eight cores passed against their own
 * equally-wrong reference (dev notes §3).
 *
 * ---------------------------------------------------------------------------
 * SAFETY
 * ---------------------------------------------------------------------------
 * Every coordinate comes from `ft_tt_coord` against live telemetry and is classified
 * before it can enter a launch (inside `ft_tt_launch_core_set_add_tensix()`), because a
 * wrong NoC endpoint hangs the chip and costs a physical power cycle (spec §7.3).
 *
 * NOTHING IN THIS FILE ASSERTS RESET ON A BLOCKED KERNEL. The deadlock stage recovers by
 * writing `stop_request` and waiting for the kernel to say EXIT. `assert_reset` on a
 * kernel blocked in a wait is what wedged Tensix 0 (dev notes §1), and in this phase every
 * resident kernel is blocked in a wait by design. Cores are parked only after their
 * kernels have reported EXIT.
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=../programs/resident \
 *   FT_TT_TENSIX_INDICES=71,72,73,74 \
 *   ./build-phase5/test_phase5_resident ../programs/resident ../programs/oneshot
 *
 * FT_TT_TENSIX_INDICES exists so the SECOND, DISJOINT core set is the same binary rather
 * than an edited source: a result believed on one set of cores does not generalise, and
 * recompiling to move cores makes the two runs differ by more than the cores.
 *
 * @status Green on the card, 0 failures, on TWO disjoint core sets: 71-74 and 76-79,
 *   2026-08-12. Four mutations recorded red (docs/tt_bringup_log.md 17.7), two of them
 *   rebuilt KERNELS driven through this same binary.
 * @issues None
 * @todo None
 */

/* clock_gettime and CLOCK_MONOTONIC are POSIX.1-2008; -std=c99 turns off glibc's
 * default feature set, so ask for them by name or the Linux card host's gcc hides them. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_group.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "ft_tt_watchdog.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/** Cores this test drives, and the two groups they split into. Both are DERIVED from the
 *  index list rather than fixed: the partitioner is handed a count and a group count. */
#define CORE_COUNT  4u
#define GROUP_COUNT 2u

#define QUEUE_CAPACITY   1024u
#define IDLE_SPIN_LIMIT  400000000u /* the host-died safety net, not flow control */
#define RUN_TIMEOUT_MS   5000u

/** Where a stage's verifiable output lands. Inside the CB arena, which no CB is using
 *  here, and comfortably clear of the queue region at the top of L1. */
#define SCRATCH_BASE     FT_TT_OP_CB_ARENA_BASE
#define SEMAPHORE_BASE   (FT_TT_OP_CB_ARENA_BASE + 0x1000u)

#define CORE_INDICES_ENVIRONMENT "FT_TT_TENSIX_INDICES"
#define AB_ITERATIONS_ENVIRONMENT "FT_TT_QUEUE_AB_ITERATIONS"

static uint32_t g_core_indices[CORE_COUNT] = { 71u, 72u, 73u, 74u };
static uint32_t g_ab_iterations = 256u;

static ft_tt_launch_core_placement g_placements[CORE_COUNT];
static ft_tt_queue g_queues[CORE_COUNT];
static ft_tt_core_group g_groups[GROUP_COUNT];

/** The two builds of kernels/resident_queue.c. */
static ft_tt_program_directory g_resident_program;
static ft_tt_program_directory g_oneshot_program;
static ft_tt_launch_config g_resident_config[CORE_COUNT];
static ft_tt_launch_config g_oneshot_config[CORE_COUNT];

/** A BRISC-only program: nothing but the data-movement processor is enabled, so nothing
 *  in this test can block in a circular-buffer wait — which is the one wait the host has
 *  no way to break. */
#define BRISC_ONLY_MASK (1u << FT_TT_PROC_DM0)

/** No circular buffers. The descriptor array still has to be written, because the launch
 *  message points at it and firmware reads it. */
static uint32_t g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * 4u];

/** The flat slot of a (group, slot) pair, which is the index into g_resident_config and
 *  g_core_indices. Derived from the group sizes rather than written out, so it stays right
 *  if the group count changes. */
static unsigned flat_index(unsigned group, unsigned slot)
{
    unsigned flat = 0;
    unsigned index;

    for (index = 0; index < group; index++)
        flat += ft_tt_launch_core_set_count(&g_groups[index].cores);
    return flat + slot;
}

/** Group sizes, derived the same way ft_tt_group_partition() derives them: the base share
 *  plus one extra to the first `total % groups` of them, so nothing is left unassigned. */
static uint32_t derived_group_size(unsigned group, unsigned total, unsigned groups)
{
    return total / groups + (group < total % groups ? 1u : 0u);
}

static uint64_t monotonic_microseconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0u;
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)(now.tv_nsec / 1000);
}

/* ------------------------------------------------------------------------- */
/* Environment                                                               */
/* ------------------------------------------------------------------------- */

static bool load_environment(void)
{
    const char *indices = getenv(CORE_INDICES_ENVIRONMENT);
    const char *iterations = getenv(AB_ITERATIONS_ENVIRONMENT);

    if (indices && indices[0] != '\0') {
        unsigned count = 0;
        const char *cursor = indices;

        while (*cursor != '\0') {
            char *end = NULL;
            unsigned long value = strtoul(cursor, &end, 10);

            if (end == cursor || count >= CORE_COUNT) {
                ft_tt_test_check_fmt(CORE_INDICES_ENVIRONMENT " parses", false, "at \"%s\"",
                                     cursor);
                return false;
            }
            g_core_indices[count++] = (uint32_t)value;
            cursor = (*end == ',') ? end + 1 : end;
        }
        if (count != CORE_COUNT) {
            ft_tt_test_check_fmt(CORE_INDICES_ENVIRONMENT " names every core", false,
                                 "got %u, need %u", count, CORE_COUNT);
            return false;
        }
    }
    if (iterations && iterations[0] != '\0')
        g_ab_iterations = (uint32_t)strtoul(iterations, NULL, 10);
    if (g_ab_iterations == 0u || g_ab_iterations > QUEUE_CAPACITY)
        g_ab_iterations = 256u;
    return true;
}

/* ------------------------------------------------------------------------- */
/* Setup                                                                     */
/* ------------------------------------------------------------------------- */

/**
 * Partition the four chosen indices into two groups.
 *
 * The indices are a LIST, not a range, so the partitioner is applied to a contiguous run
 * and then each placement is replaced by the one the list actually names. Doing it this
 * way keeps ft_tt_group_partition() the only thing that decides SIZES — the derived part
 * — while the test still gets to pick discontiguous cores, which is what stops a bug that
 * walks a contiguous run from passing.
 */
static bool build_groups(const ft_tt_hardware_session *session)
{
    unsigned group;
    unsigned assigned = 0;
    const char *reason = NULL;

    ft_tt_test_step("groups: partition %u cores into %u disjoint groups, sizes derived",
                    CORE_COUNT, GROUP_COUNT);

    for (group = 0; group < GROUP_COUNT; group++) {
        uint32_t size = derived_group_size(group, CORE_COUNT, GROUP_COUNT);
        unsigned slot;
        int rc;

        memset(&g_groups[group], 0, sizeof(g_groups[group]));
        g_groups[group].id = group;
        g_groups[group].first_tensix_index = g_core_indices[assigned];
        g_groups[group].tensix_index_count = size;
        g_groups[group].queues = g_queues + assigned;
        g_groups[group].dram_bank_first = group % session->grid.dram_bank_count;
        g_groups[group].dram_bank_stride = GROUP_COUNT;
        g_groups[group].dram_bank_count =
            (session->grid.dram_bank_count - g_groups[group].dram_bank_first + GROUP_COUNT - 1u) /
            GROUP_COUNT;

        rc = ft_tt_launch_core_set_init(&g_groups[group].cores, g_placements + assigned, size);
        if (rc != FT_TT_OK)
            return false;

        for (slot = 0; slot < size; slot++) {
            const ft_tt_launch_core_placement *placement;

            rc = ft_tt_launch_core_set_add_tensix(&g_groups[group].cores, &session->grid,
                                                  g_core_indices[assigned + slot], NULL);
            placement = ft_tt_launch_core_set_at(&g_groups[group].cores, slot);
            ft_tt_test_check_fmt("coordinate derives and classifies as a Tensix",
                                 rc == FT_TT_OK && placement != NULL,
                                 "group %u slot %u: index %u -> translated (%u, %u), rc=%d", group,
                                 slot, g_core_indices[assigned + slot],
                                 placement ? placement->x : 0u, placement ? placement->y : 0u, rc);
            if (rc != FT_TT_OK)
                return false;
        }
        assigned += size;
    }

    {
        bool disjoint = ft_tt_group_check_disjoint(g_groups, GROUP_COUNT, &reason) == FT_TT_OK;

        ft_tt_test_check("the two groups share no core", disjoint, reason);
    }
    return ft_tt_test_failure_count() == 0;
}

/** Boot Tenstorrent's firmware on every core in the group, and load both kernel builds. */
static bool boot_and_load(const ft_tt_hardware_session *session,
                          const ft_tt_firmware_set *firmware)
{
    unsigned group;
    unsigned flat = 0;

    ft_tt_test_step("boot firmware on all %u cores and load both kernel builds", CORE_COUNT);

    for (group = 0; group < GROUP_COUNT; group++) {
        unsigned count = ft_tt_launch_core_set_count(&g_groups[group].cores);
        unsigned slot;

        for (slot = 0; slot < count; slot++, flat++) {
            const ft_tt_launch_core_placement *placement =
                ft_tt_launch_core_set_at(&g_groups[group].cores, slot);
            int rc = ft_tt_core_boot_firmware(&session->noc, placement->x, placement->y, firmware,
                                              NULL);

            ft_tt_test_check_fmt("firmware booted", rc == FT_TT_OK, "(%u, %u) rc=%d",
                                 placement->x, placement->y, rc);
            if (rc != FT_TT_OK)
                return false;

            rc = ft_tt_op_program_load(&session->noc, placement->x, placement->y,
                                       &g_resident_program.program, NULL, &g_resident_config[flat]);
            if (rc == FT_TT_OK) {
                g_resident_config[flat].local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
                g_resident_config[flat].local_cb_mask = 0u;
                g_resident_config[flat].brisc_noc_id = 0u;
            }
            ft_tt_test_check_fmt("resident kernel image loaded", rc == FT_TT_OK, "(%u, %u) rc=%d",
                                 placement->x, placement->y, rc);
            if (rc != FT_TT_OK)
                return false;

            /* The one-shot build is loaded into the SAME slab, so only one of the two can
             * be resident on a core at a time. Staging the other one is a reload, which
             * is exactly what the A/B's arm A is paying for. */
            g_oneshot_config[flat] = g_resident_config[flat];
        }
    }
    return true;
}

/** Put the one-shot image into a core's BRISC slab, replacing whatever is there.
 *
 *  Straight to ft_tt_launch_load_image(): there is no separate "relocated kernel" kind,
 *  because the ELF's own text address says which of the two legal regions it was linked
 *  into and the ELF is the only party that cannot be wrong about that. */
static int load_oneshot_image(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    return ft_tt_launch_load_image(noc, x, y, FT_TT_PROC_DM0, FT_TT_IMAGE_KERNEL,
                                   g_oneshot_program.program.images[0].elf_data,
                                   g_oneshot_program.program.images[0].elf_size, NULL);
}

static int load_resident_image(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    return ft_tt_launch_load_image(noc, x, y, FT_TT_PROC_DM0, FT_TT_IMAGE_KERNEL,
                                   g_resident_program.program.images[0].elf_data,
                                   g_resident_program.program.images[0].elf_size, NULL);
}

/** Stage and KICK a resident kernel, without waiting for it. */
/** The binding every core in this test uses: bounded residency, its own BRISC status slot,
 *  and no group block (stage 3 checks groups through per-core completion, which is what
 *  this BRISC-only kernel publishes; group-level completion is tested against the triad in
 *  test_phase6_triad.c, where a group has more than one processor to disagree about). */
static void fill_binding(const ft_tt_queue *queue, ft_tt_queue_kernel_binding *binding)
{
    memset(binding, 0, sizeof(*binding));
    binding->idle_spin_limit = IDLE_SPIN_LIMIT;
    binding->cb_wait_spin_limit = IDLE_SPIN_LIMIT;
    binding->status_address =
        FT_TT_QUEUE_STATUS_ADDRESS(queue->control_address, FT_TT_PROC_DM0);
}

static int start_resident(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                          const ft_tt_launch_config *config, const ft_tt_queue *queue)
{
    uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];
    ft_tt_queue_kernel_binding binding;
    ft_tt_op_args block;
    int rc;

    fill_binding(queue, &binding);
    rc = ft_tt_queue_kernel_args(queue, &binding, args);
    if (rc != FT_TT_OK)
        return rc;
    block.processor = FT_TT_PROC_DM0;
    block.values = args;
    block.count = FT_TT_QUEUE_KERNEL_ARG_COUNT;

    rc = ft_tt_op_stage(noc, x, y, config, g_cb_buffer, sizeof(g_cb_buffer), &block, 1u);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_send_go(noc, x, y);
}

/* ------------------------------------------------------------------------- */
/* Stage 1 — the resident kernel runs, and completion is produced by the work */
/* ------------------------------------------------------------------------- */

/** Poison a scratch word so a "result" that is really untouched memory is visible. */
static int poison_scratch(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t address,
                          uint32_t count)
{
    uint32_t pattern[8];
    uint32_t index;

    for (index = 0; index < 8u; index++)
        pattern[index] = 0xDEAD0000u + index;
    for (index = 0; index < count; index += 8u) {
        uint32_t words = (count - index < 8u) ? count - index : 8u;
        int rc = noc->write(noc->user, x, y, address + index * 4u, pattern, words * 4u);

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

static uint32_t stage1_expected_value(uint32_t core_seed, uint32_t index)
{
    /* The per-core seed is an OFFSET, never inside a modulus: `(element + index * 7) % 16`
     * made any two indices congruent mod 16 byte-identical and four of eight cores passed
     * against their own wrong reference (dev notes §3). */
    return 0x51000000u + core_seed * 0x10000u + index;
}

static bool stage_1_resident_runs(const ft_tt_hardware_session *session)
{
    const ft_tt_launch_core_placement *placement =
        ft_tt_launch_core_set_at(&g_groups[0].cores, 0u);
    ft_tt_queue *queue = &g_groups[0].queues[0];
    ft_tt_queue_descriptor descriptors[8];
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t readback[8];
    const uint32_t items = 8u;
    const char *reason = NULL;
    uint32_t index;
    int rc;

    ft_tt_test_step("stage 1 — a resident kernel executes descriptors, and says so itself");

    rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                          QUEUE_CAPACITY, ft_tt_queue_make_nonce(), queue);
    ft_tt_test_check_fmt("queue control block written", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    (void)poison_scratch(&session->noc, placement->x, placement->y, SCRATCH_BASE, items);

    memset(descriptors, 0, sizeof(descriptors));
    for (index = 0; index < items; index++) {
        descriptors[index].opcode = FT_TT_QUEUE_OP_WRITE_WORD;
        descriptors[index].a0 = SCRATCH_BASE + index * 4u;
        descriptors[index].a1 = stage1_expected_value(0u, index);
    }
    rc = ft_tt_queue_enqueue(&session->noc, queue, descriptors, items);
    ft_tt_test_check_fmt("eight descriptors enqueued", rc == FT_TT_OK, "rc=%d", rc);

    /* THE NEGATIVE CASE, before the kernel is started. This is exactly the situation
     * ft_tt_launch_wait_done_all() reported as complete on 2026-08-12. */
    (void)ft_tt_queue_read_control(&session->noc, queue, fields);
    {
        /* The condition is evaluated into a local FIRST. C leaves the order of a call's
         * arguments unspecified, and gcc evaluates the format arguments before the
         * condition — which prints the PREVIOUS value of `reason` next to a check that
         * passed, and reads a stale `readback`. The check was right and the report was a
         * lie, which is the one combination this project cannot afford. */
        bool refused =
            ft_tt_queue_check_completion(queue, fields, items, &reason) == FT_TT_ERROR;

        /* The REASON is asserted, not just the refusal. Without this the case is passed by
         * the count check alone (0 != 8) and the state poison is never exercised — a
         * mutation that stopped writing the poison produced ZERO failures here until the
         * assertion was tightened to name the reason. A check that would pass for a
         * different reason than the one it is testing is a vacuous check. */
        ft_tt_test_check_fmt("with nothing launched, the completion check REFUSES", refused, "%s",
                             reason ? reason : "(it passed, which is the bug)");
        ft_tt_test_check_fmt("and it refuses BECAUSE the kernel never ran, not merely on the count",
                             reason != NULL && strstr(reason, "never ran") != NULL, "%s",
                             reason ? reason : "(no reason given)");
    }
    {
        bool still_poison =
            session->noc.read(session->noc.user, placement->x, placement->y, SCRATCH_BASE,
                              readback, sizeof(uint32_t)) == FT_TT_OK &&
            readback[0] == 0xDEAD0000u;

        ft_tt_test_check_fmt("and the destination is still the poison", still_poison, "0x%08x",
                             readback[0]);
    }

    rc = start_resident(&session->noc, placement->x, placement->y, &g_resident_config[0], queue);
    ft_tt_test_check_fmt("the resident kernel is staged and kicked", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    {
        uint32_t completed = 0u;

        rc = ft_tt_queue_wait_completed(&session->noc, queue, items, RUN_TIMEOUT_MS, &completed);
        ft_tt_test_check_fmt("all eight descriptors completed", rc == FT_TT_OK, "completed=%u/%u",
                             completed, items);
    }

    (void)ft_tt_queue_read_control(&session->noc, queue, fields);
    {
        bool passed = ft_tt_queue_check_completion(queue, fields, items, &reason) == FT_TT_OK;

        ft_tt_test_check_fmt("the completion check now passes", passed, "%s",
                             reason ? reason : "ok");
    }
    ft_tt_test_check_fmt("and the state is no longer the host poison",
                         fields[FT_TT_QUEUE_FIELD_STATE] !=
                             (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON,
                         "state=%s", ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]));
    ft_tt_test_check_fmt("the token is THIS run's eighth",
                         fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] ==
                             ft_tt_queue_token_at(queue, items - 1u),
                         "0x%08x vs 0x%08x", fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN],
                         ft_tt_queue_token_at(queue, items - 1u));

    /* And the WORK, independently of the completion counters. Two checks that can fail
     * separately is the point: a protocol that says "done" and a destination that says
     * nothing happened is the failure this phase has to be able to see. */
    rc = session->noc.read(session->noc.user, placement->x, placement->y, SCRATCH_BASE, readback,
                           sizeof(readback));
    ft_tt_test_check_fmt("the destination reads back", rc == FT_TT_OK, "rc=%d", rc);
    for (index = 0; index < items; index++) {
        ft_tt_test_check_fmt("word matches, and is not the poison",
                             readback[index] == stage1_expected_value(0u, index),
                             "[%u] 0x%08x want 0x%08x", index, readback[index],
                             stage1_expected_value(0u, index));
    }

    /* The kernel is STILL RUNNING — that is what resident means. Prove it by enqueueing
     * more work without launching anything. */
    memset(descriptors, 0, sizeof(descriptors));
    for (index = 0; index < 4u; index++) {
        descriptors[index].opcode = FT_TT_QUEUE_OP_WRITE_WORD;
        descriptors[index].a0 = SCRATCH_BASE + index * 4u;
        descriptors[index].a1 = 0xC0FFEE00u + index;
    }
    rc = ft_tt_queue_enqueue(&session->noc, queue, descriptors, 4u);
    ft_tt_test_check_fmt("four more enqueued with NO launch of any kind", rc == FT_TT_OK, "rc=%d",
                         rc);
    {
        uint32_t completed = 0u;

        rc = ft_tt_queue_wait_completed(&session->noc, queue, items + 4u, RUN_TIMEOUT_MS,
                                        &completed);
        ft_tt_test_check_fmt("and the still-resident kernel executed them", rc == FT_TT_OK,
                             "completed=%u/%u", completed, items + 4u);
    }
    (void)session->noc.read(session->noc.user, placement->x, placement->y, SCRATCH_BASE, readback,
                            4u * sizeof(uint32_t));
    ft_tt_test_check_fmt("the second batch's result is present too", readback[3] == 0xC0FFEE03u,
                         "0x%08x", readback[3]);

    /* And it retires by ASKING. */
    {
        uint32_t unstopped = 0u;
        ft_tt_core_group solo = g_groups[0];

        solo.cores.count = 1u;
        rc = ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
        ft_tt_test_check_fmt("the kernel stops when asked, with no reset anywhere",
                             rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=%u", rc,
                             unstopped);
    }
    return ft_tt_test_failure_count() == 0;
}

/* ------------------------------------------------------------------------- */
/* Stage 2 — the dispatch A/B                                                */
/* ------------------------------------------------------------------------- */


/** The full Phase 4 op-dispatch shape for ONE descriptor: plant it, write the CB
 *  descriptor array, write every enabled processor's runtime-arg block, write the launch
 *  message, GO, poll for RUN_MSG_DONE. This is exactly what ft_tt_op_launch() does. */
static int slow_dispatch_full(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                              const ft_tt_launch_config *config, ft_tt_queue *queue)
{
    ft_tt_queue_descriptor descriptor;
    int rc;

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_NOP;
    rc = ft_tt_queue_enqueue(&session->noc, queue, &descriptor, 1u);
    if (rc != FT_TT_OK)
        return rc;
    rc = start_resident(&session->noc, x, y, config, queue);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_wait_done(&session->noc, x, y, RUN_TIMEOUT_MS, NULL);
}

/** The LEANEST slow dispatch there is: the launch message and the descriptors are already
 *  in L1, so a descriptor costs one head bump, one GO and the completion poll. This is the
 *  floor of the go/done model, and it is the number Phase 2's 16 us over 1000 launches is
 *  comparable to — quoting only the full shape would flatter this phase. */
static int slow_dispatch_go_only(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                                 ft_tt_queue *queue)
{
    int rc = ft_tt_queue_publish(&session->noc, queue);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go(&session->noc, x, y);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_wait_done(&session->noc, x, y, RUN_TIMEOUT_MS, NULL);
}

/** Fill a batch with NOPs. NOP is the right op for a dispatch measurement: its wall time
 *  IS the protocol's, with no work to hide behind. */
static void fill_nop_batch(ft_tt_queue_descriptor *batch, uint32_t count)
{
    uint32_t index;

    memset(batch, 0, (size_t)count * sizeof(batch[0]));
    for (index = 0; index < count; index++)
        batch[index].opcode = FT_TT_QUEUE_OP_NOP;
}

static ft_tt_queue_descriptor g_batch[QUEUE_CAPACITY];

/** Arm B at one batch size: enqueue and poll once. Returns microseconds per descriptor. */
static double measure_queue_dispatch(const ft_tt_hardware_session *session, uint16_t x,
                                     uint16_t y, const ft_tt_launch_config *config,
                                     uint32_t batch, uint32_t batches, int *out_rc)
{
    ft_tt_queue queue;
    uint64_t elapsed = 0u;
    uint32_t round;
    int rc;

    rc = load_resident_image(&session->noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&session->noc, x, y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                              ft_tt_queue_make_nonce(), &queue);
    if (rc == FT_TT_OK)
        rc = start_resident(&session->noc, x, y, config, &queue);
    if (rc != FT_TT_OK) {
        *out_rc = rc;
        return 0.0;
    }

    for (round = 0; round < batches; round++) {
        uint64_t started;

        fill_nop_batch(g_batch, batch);
        started = monotonic_microseconds();
        rc = ft_tt_queue_enqueue(&session->noc, &queue, g_batch, batch);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &queue, (round + 1u) * batch,
                                            RUN_TIMEOUT_MS, NULL);
        elapsed += monotonic_microseconds() - started;
        if (rc != FT_TT_OK)
            break;
    }

    {
        uint32_t unstopped = 0u;
        ft_tt_core_group solo = g_groups[0];

        solo.cores.count = 1u;
        solo.queues = &queue;
        (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
    }
    *out_rc = rc;
    return (double)elapsed / (double)(batch * batches);
}

static bool stage_2_dispatch_ab(const ft_tt_hardware_session *session)
{
    const ft_tt_launch_core_placement *placement =
        ft_tt_launch_core_set_at(&g_groups[0].cores, 0u);
    ft_tt_queue queue;
    uint64_t full_total = 0u;
    uint64_t go_only_total = 0u;
    uint64_t queue_total = 0u;
    uint64_t device_total = 0u;
    const uint32_t rounds = 8u;
    const uint32_t per_round = (g_ab_iterations / rounds) ? (g_ab_iterations / rounds) : 1u;
    uint32_t round;
    int rc = FT_TT_OK;

    ft_tt_test_step("stage 2 - dispatch A/B/C/D, interleaved, %u descriptors per arm",
                    rounds * per_round);

    for (round = 0; round < rounds; round++) {
        uint64_t started;
        uint32_t index;

        /* ---- arm A1: the full Phase 4 op dispatch, one host round trip per op ---- */
        rc = load_oneshot_image(&session->noc, placement->x, placement->y);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                                  QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &queue);
        if (rc != FT_TT_OK)
            return false;

        started = monotonic_microseconds();
        for (index = 0; index < per_round; index++) {
            rc = slow_dispatch_full(session, placement->x, placement->y, &g_oneshot_config[0],
                                    &queue);
            if (rc != FT_TT_OK)
                break;
        }
        full_total += monotonic_microseconds() - started;
        ft_tt_test_check_fmt("arm A1 completed its round", rc == FT_TT_OK, "round %u rc=%d", round,
                             rc);
        if (rc != FT_TT_OK)
            return false;

        /* ---- arm A2: go/done only. Launch message and descriptors pre-written ---- */
        rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                              QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &queue);
        if (rc == FT_TT_OK) {
            fill_nop_batch(g_batch, per_round);
            rc = ft_tt_queue_stage(&session->noc, &queue, g_batch, per_round);
        }
        if (rc == FT_TT_OK) {
            /* Stage the launch ONCE; from here a descriptor is a head bump and a GO. */
            uint32_t args[FT_TT_QUEUE_KERNEL_ARG_COUNT];
            ft_tt_queue_kernel_binding binding;
            ft_tt_op_args block;

            fill_binding(&queue, &binding);
            (void)ft_tt_queue_kernel_args(&queue, &binding, args);
            block.processor = FT_TT_PROC_DM0;
            block.values = args;
            block.count = FT_TT_QUEUE_KERNEL_ARG_COUNT;
            rc = ft_tt_op_stage(&session->noc, placement->x, placement->y, &g_oneshot_config[0],
                                g_cb_buffer, sizeof(g_cb_buffer), &block, 1u);
        }
        if (rc != FT_TT_OK)
            return false;

        started = monotonic_microseconds();
        for (index = 0; index < per_round; index++) {
            /* One descriptor becomes visible per iteration: publish covers everything
             * staged, so the head is bumped to the full batch on the first pass and the
             * one-shot kernel takes exactly one item per GO. */
            rc = slow_dispatch_go_only(session, placement->x, placement->y, &queue);
            if (rc != FT_TT_OK)
                break;
        }
        go_only_total += monotonic_microseconds() - started;
        ft_tt_test_check_fmt("arm A2 completed its round", rc == FT_TT_OK, "round %u rc=%d", round,
                             rc);
        if (rc != FT_TT_OK)
            return false;

        /* ---- arm B: queue dispatch, one enqueue and one completion poll ---- */
        rc = load_resident_image(&session->noc, placement->x, placement->y);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                                  QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &queue);
        if (rc == FT_TT_OK)
            rc = start_resident(&session->noc, placement->x, placement->y, &g_resident_config[0],
                                &queue);
        if (rc != FT_TT_OK)
            return false;

        fill_nop_batch(g_batch, per_round);
        started = monotonic_microseconds();
        rc = ft_tt_queue_enqueue(&session->noc, &queue, g_batch, per_round);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &queue, per_round, RUN_TIMEOUT_MS, NULL);
        queue_total += monotonic_microseconds() - started;
        ft_tt_test_check_fmt("arm B completed its round", rc == FT_TT_OK, "round %u rc=%d", round,
                             rc);
        if (rc != FT_TT_OK)
            return false;

        /* ---- arm C: DEVICE-SIDE op-to-op. The descriptors are STAGED first, so the only
         * host action inside the window is one 4-byte head write; everything after that is
         * the resident kernel executing. This is the number comparable to upstream's
         * 2.08 us per-core op-to-op golden. ---- */
        fill_nop_batch(g_batch, per_round);
        rc = ft_tt_queue_stage(&session->noc, &queue, g_batch, per_round);
        if (rc == FT_TT_OK) {
            started = monotonic_microseconds();
            rc = ft_tt_queue_publish(&session->noc, &queue);
            if (rc == FT_TT_OK)
                rc = ft_tt_queue_wait_completed(&session->noc, &queue, 2u * per_round,
                                                RUN_TIMEOUT_MS, NULL);
            device_total += monotonic_microseconds() - started;
        }
        ft_tt_test_check_fmt("arm C completed its round", rc == FT_TT_OK, "round %u rc=%d", round,
                             rc);
        if (rc != FT_TT_OK)
            return false;

        {
            uint32_t unstopped = 0u;
            ft_tt_core_group solo = g_groups[0];

            solo.cores.count = 1u;
            solo.queues = &queue;
            (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
        }
    }

    {
        double total = (double)(rounds * per_round);
        double full_us = (double)full_total / total;
        double go_only_us = (double)go_only_total / total;
        double queue_us = (double)queue_total / total;
        double device_us = (double)device_total / total;

        printf("\n  DISPATCH COST PER DESCRIPTOR (NOP), %u per arm, arms interleaved\n",
               rounds * per_round);
        printf("    A1  Phase 4 op dispatch: CB config + args + launch msg + GO + poll : %8.3f us\n",
               full_us);
        printf("    A2  go/done only: launch msg pre-written, head bump + GO + poll    : %8.3f us\n",
               go_only_us);
        printf("    B   queue dispatch: batch of %-4u enqueued, ONE completion poll     : %8.3f us\n",
               per_round, queue_us);
        printf("    C   device-side: descriptors pre-staged, one head bump, no polling : %8.3f us\n",
               device_us);
        printf("    --- reference points ---\n");
        printf("    Phase 2 slow-dispatch baseline (1000 launches, noop compute)       :   16.000 us\n");
        printf("    upstream bh_p150 op-to-op golden, per core                         :    2.080 us\n");
        printf("    speedup A1 -> B  %6.2fx      A2 -> B  %6.2fx\n", queue_us > 0.0 ? full_us / queue_us : 0.0,
               queue_us > 0.0 ? go_only_us / queue_us : 0.0);

        ft_tt_test_check_fmt("queue dispatch beats the full Phase 4 op dispatch",
                             queue_us < full_us, "%.3f us vs %.3f us", queue_us, full_us);
        ft_tt_test_check_fmt("queue dispatch beats the leanest go/done dispatch",
                             queue_us < go_only_us, "%.3f us vs %.3f us", queue_us, go_only_us);
        ft_tt_test_check_fmt("and it beats upstream's 2.08 us per-core op-to-op golden",
                             queue_us < 2.08, "%.3f us", queue_us);
        ft_tt_test_check_fmt("the device-side op-to-op number is smaller still",
                             device_us < queue_us, "%.3f us vs %.3f us", device_us, queue_us);
    }

    /* The batch-size sweep. Measured separately rather than interleaved, and labelled as
     * such: what it shows is the SHAPE of the model's cost — a fixed per-batch cost of two
     * device accesses plus one completion poll, divided by the batch, over a per-descriptor
     * device cost that does not depend on the batch at all. */
    {
        static const uint32_t sweep[] = { 1u, 4u, 16u, 64u, 256u };
        unsigned index;

        printf("\n  ARM B BY BATCH SIZE (measured separately, not interleaved)\n");
        for (index = 0; index < sizeof(sweep) / sizeof(sweep[0]); index++) {
            uint32_t batch = sweep[index];
            uint32_t batches = (256u / batch) ? (256u / batch) : 1u;
            int sweep_rc = FT_TT_OK;
            double us = measure_queue_dispatch(session, placement->x, placement->y,
                                               &g_resident_config[0], batch, batches, &sweep_rc);

            printf("    batch %4u x %-4u : %8.3f us per descriptor%s\n", batch, batches, us,
                   sweep_rc == FT_TT_OK ? "" : "  (FAILED)");
            ft_tt_test_check_fmt("the sweep point ran", sweep_rc == FT_TT_OK, "batch %u rc=%d",
                                 batch, sweep_rc);
        }
    }
    return ft_tt_test_failure_count() == 0;
}


/* ------------------------------------------------------------------------- */

static bool stage_3_two_groups(const ft_tt_hardware_session *session)
{
    static ft_tt_queue_descriptor batch[64];
    const uint32_t items = 16u;
    uint32_t observed[CORE_COUNT][16];
    uint32_t nonce = ft_tt_queue_make_nonce();
    unsigned group;
    unsigned flat = 0;
    bool all_started = true;

    ft_tt_test_step("stage 3 — two workloads on disjoint groups, neither stalling the other");

    for (group = 0; group < GROUP_COUNT; group++) {
        unsigned count = ft_tt_launch_core_set_count(&g_groups[group].cores);
        unsigned slot;
        int rc = ft_tt_group_queues_init(&session->noc, &g_groups[group], FT_TT_OP_QUEUE_BASE,
                                         QUEUE_CAPACITY, nonce + group);

        ft_tt_test_check_fmt("group queues initialised", rc == FT_TT_OK, "group %u rc=%d", group,
                             rc);
        if (rc != FT_TT_OK)
            return false;

        for (slot = 0; slot < count; slot++, flat++) {
            const ft_tt_launch_core_placement *placement =
                ft_tt_launch_core_set_at(&g_groups[group].cores, slot);
            uint32_t index;

            rc = load_resident_image(&session->noc, placement->x, placement->y);
            if (rc == FT_TT_OK)
                rc = poison_scratch(&session->noc, placement->x, placement->y, SCRATCH_BASE, items);
            if (rc == FT_TT_OK)
                rc = start_resident(&session->noc, placement->x, placement->y,
                                    &g_resident_config[flat], &g_groups[group].queues[slot]);
            if (rc != FT_TT_OK) {
                all_started = false;
                break;
            }

            memset(batch, 0, sizeof(batch));
            for (index = 0; index < items; index++) {
                batch[index].opcode = FT_TT_QUEUE_OP_WRITE_WORD;
                batch[index].a0 = SCRATCH_BASE + index * 4u;
                /* The seed is the CORE INDEX as an offset, so a core reading another
                 * core's L1 produces a wrong answer rather than a coincidentally right
                 * one, and no two cores can collide however the indices are chosen. */
                batch[index].a1 = stage1_expected_value(g_core_indices[flat], index);
            }
            rc = ft_tt_queue_enqueue(&session->noc, &g_groups[group].queues[slot], batch, items);
            if (rc != FT_TT_OK)
                all_started = false;
        }
    }
    ft_tt_test_check("every core in both groups is resident and fed", all_started, NULL);
    if (!all_started)
        return false;

    for (group = 0; group < GROUP_COUNT; group++) {
        ft_tt_watchdog_config config;
        ft_tt_watchdog_report report;
        int rc;

        config.stall_timeout_ms = 1000u;
        config.overall_timeout_ms = RUN_TIMEOUT_MS;
        rc = ft_tt_watchdog_wait(&session->noc, &g_groups[group], items, &config, &report);
        ft_tt_test_check_fmt("group finished its own work", rc == FT_TT_OK,
                             "group %u: %u/%u in %u ms", group, report.completed_total,
                             report.completed_target, report.elapsed_ms);
    }

    flat = 0;
    for (group = 0; group < GROUP_COUNT; group++) {
        unsigned count = ft_tt_launch_core_set_count(&g_groups[group].cores);
        unsigned slot;

        for (slot = 0; slot < count; slot++, flat++) {
            const ft_tt_launch_core_placement *placement =
                ft_tt_launch_core_set_at(&g_groups[group].cores, slot);
            uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
            const char *reason = NULL;
            uint32_t index;
            bool matched = true;

            bool evidenced;

            (void)ft_tt_queue_read_control(&session->noc, &g_groups[group].queues[slot], fields);
            evidenced = ft_tt_queue_check_completion(&g_groups[group].queues[slot], fields, items,
                                                     &reason) == FT_TT_OK;
            ft_tt_test_check_fmt("completion is evidenced by the kernel", evidenced,
                                 "group %u slot %u: %s", group, slot, reason ? reason : "ok");

            (void)session->noc.read(session->noc.user, placement->x, placement->y, SCRATCH_BASE,
                                    observed[flat], items * sizeof(uint32_t));
            for (index = 0; index < items; index++) {
                if (observed[flat][index] != stage1_expected_value(g_core_indices[flat], index))
                    matched = false;
            }
            ft_tt_test_check_fmt("and every word matches this core's own reference", matched,
                                 "core index %u, first word 0x%08x", g_core_indices[flat],
                                 observed[flat][0]);
        }
    }

    /* PAIRWISE DISTINCT, checked against each OTHER and not only against the model. This
     * is the check that caught the mod-16 seed collision, and it does not depend on the
     * reference being right (dev notes §3). */
    {
        unsigned a;
        bool all_distinct = true;

        for (a = 0; a < CORE_COUNT; a++) {
            unsigned b;

            for (b = a + 1u; b < CORE_COUNT; b++) {
                if (memcmp(observed[a], observed[b], items * sizeof(uint32_t)) == 0)
                    all_distinct = false;
            }
        }
        ft_tt_test_check("the four cores' results are pairwise distinct", all_distinct, NULL);
    }

    for (group = 0; group < GROUP_COUNT; group++) {
        uint32_t unstopped = 0u;
        int rc = ft_tt_watchdog_shutdown(&session->noc, &g_groups[group], RUN_TIMEOUT_MS,
                                         &unstopped);

        ft_tt_test_check_fmt("group retired cleanly", rc == FT_TT_OK && unstopped == 0u,
                             "group %u rc=%d unstopped=%u", group, rc, unstopped);
    }
    return ft_tt_test_failure_count() == 0;
}

/* ------------------------------------------------------------------------- */
/* Stage 4 — a cross-group semaphore                                         */
/* ------------------------------------------------------------------------- */

static bool stage_4_cross_group_semaphore(const ft_tt_hardware_session *session)
{
    const ft_tt_launch_core_placement *waiter =
        ft_tt_launch_core_set_at(&g_groups[1].cores, 0u);
    const ft_tt_launch_core_placement *poster =
        ft_tt_launch_core_set_at(&g_groups[0].cores, 0u);
    ft_tt_queue waiter_queue;
    ft_tt_queue poster_queue;
    ft_tt_queue_descriptor descriptors[4];
    uint32_t zero = 0u;
    uint32_t completed = 0u;
    uint32_t readback = 0xFFFFFFFFu;
    uint32_t nonce = ft_tt_queue_make_nonce();
    int rc;

    ft_tt_test_step("stage 4 — group 1 waits on a semaphore that group 0 posts");

    (void)session->noc.write(session->noc.user, waiter->x, waiter->y, SEMAPHORE_BASE, &zero,
                             sizeof(zero));
    (void)poison_scratch(&session->noc, waiter->x, waiter->y, SCRATCH_BASE, 1u);

    rc = ft_tt_queue_init(&session->noc, waiter->x, waiter->y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          nonce, &waiter_queue);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&session->noc, poster->x, poster->y, FT_TT_OP_QUEUE_BASE,
                              QUEUE_CAPACITY, nonce + 1u, &poster_queue);
    if (rc == FT_TT_OK)
        rc = load_resident_image(&session->noc, waiter->x, waiter->y);
    if (rc == FT_TT_OK)
        rc = load_resident_image(&session->noc, poster->x, poster->y);
    if (rc == FT_TT_OK)
        rc = start_resident(&session->noc, waiter->x, waiter->y, &g_resident_config[flat_index(1u, 0u)],
                            &waiter_queue);
    if (rc == FT_TT_OK)
        rc = start_resident(&session->noc, poster->x, poster->y, &g_resident_config[flat_index(0u, 0u)],
                            &poster_queue);
    ft_tt_test_check_fmt("both cores are resident", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    /* The waiter goes first and BLOCKS: it cannot proceed until the other group posts. */
    memset(descriptors, 0, sizeof(descriptors));
    descriptors[0].opcode = FT_TT_QUEUE_OP_SEM_WAIT;
    descriptors[0].a0 = SEMAPHORE_BASE;
    descriptors[0].a1 = 1u;
    descriptors[0].a2 = 200000000u;
    descriptors[1].opcode = FT_TT_QUEUE_OP_WRITE_WORD;
    descriptors[1].a0 = SCRATCH_BASE;
    descriptors[1].a1 = 0x5E4A0001u;
    (void)ft_tt_queue_enqueue(&session->noc, &waiter_queue, descriptors, 2u);

    (void)ft_tt_queue_read_field(&session->noc, &waiter_queue, FT_TT_QUEUE_FIELD_COMPLETED_COUNT,
                                 &completed);
    ft_tt_test_check_fmt("the waiter has not run past the semaphore", completed == 0u,
                         "completed=%u", completed);
    (void)session->noc.read(session->noc.user, waiter->x, waiter->y, SCRATCH_BASE, &readback,
                            sizeof(readback));
    ft_tt_test_check_fmt("and its destination is still the poison", readback == 0xDEAD0000u,
                         "0x%08x", readback);

    /* Now group 0 posts to group 1's semaphore. */
    memset(descriptors, 0, sizeof(descriptors));
    descriptors[0].opcode = FT_TT_QUEUE_OP_SEM_POST;
    descriptors[0].a0 = waiter->x;
    descriptors[0].a1 = waiter->y;
    descriptors[0].a2 = SEMAPHORE_BASE;
    descriptors[0].a3 = 1u;
    (void)ft_tt_queue_enqueue(&session->noc, &poster_queue, descriptors, 1u);

    rc = ft_tt_queue_wait_completed(&session->noc, &poster_queue, 1u, RUN_TIMEOUT_MS, &completed);
    ft_tt_test_check_fmt("the poster completed its post", rc == FT_TT_OK, "completed=%u", completed);

    rc = ft_tt_queue_wait_completed(&session->noc, &waiter_queue, 2u, RUN_TIMEOUT_MS, &completed);
    ft_tt_test_check_fmt("and the waiter then finished BOTH its descriptors", rc == FT_TT_OK,
                         "completed=%u/2", completed);
    (void)session->noc.read(session->noc.user, waiter->x, waiter->y, SCRATCH_BASE, &readback,
                            sizeof(readback));
    ft_tt_test_check_fmt("the work behind the semaphore ran", readback == 0x5E4A0001u, "0x%08x",
                         readback);

    {
        uint32_t unstopped = 0u;
        ft_tt_core_group solo = g_groups[0];
        ft_tt_core_group other = g_groups[1];

        solo.cores.count = 1u;
        solo.queues = &poster_queue;
        other.cores.count = 1u;
        other.queues = &waiter_queue;
        (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
        (void)ft_tt_watchdog_shutdown(&session->noc, &other, RUN_TIMEOUT_MS, &unstopped);
    }
    return ft_tt_test_failure_count() == 0;
}

/* ------------------------------------------------------------------------- */
/* Stage 5 — a deliberate deadlock, and the card surviving it                */
/* ------------------------------------------------------------------------- */

static bool stage_5_deadlock_and_recovery(const ft_tt_hardware_session *session)
{
    const ft_tt_launch_core_placement *placement =
        ft_tt_launch_core_set_at(&g_groups[1].cores, 0u);
    ft_tt_core_group solo = g_groups[1];
    ft_tt_queue queue;
    ft_tt_queue_descriptor descriptors[2];
    ft_tt_watchdog_config config;
    ft_tt_watchdog_report report;
    uint32_t zero = 0u;
    uint32_t unstopped = 0u;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    int rc;

    ft_tt_test_step("stage 5 — a deliberate deadlock: the watchdog catches it and the card lives");

    solo.cores.count = 1u;
    solo.queues = &queue;

    (void)session->noc.write(session->noc.user, placement->x, placement->y, SEMAPHORE_BASE, &zero,
                             sizeof(zero));
    rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                          QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &queue);
    if (rc == FT_TT_OK)
        rc = load_resident_image(&session->noc, placement->x, placement->y);
    if (rc == FT_TT_OK)
        rc = start_resident(&session->noc, placement->x, placement->y,
                            &g_resident_config[flat_index(1u, 0u)], &queue);
    if (rc != FT_TT_OK)
        return false;

    /* A wait nobody will ever satisfy. This is the hang the plan asks for, and on this
     * host an unrecoverable one is a walk to the machine. */
    memset(descriptors, 0, sizeof(descriptors));
    descriptors[0].opcode = FT_TT_QUEUE_OP_SEM_WAIT;
    descriptors[0].a0 = SEMAPHORE_BASE;
    descriptors[0].a1 = 0xFFFFFFFFu;
    descriptors[0].a2 = 0u; /* fall back to the idle bound, i.e. effectively forever */
    (void)ft_tt_queue_enqueue(&session->noc, &queue, descriptors, 1u);

    config.stall_timeout_ms = 500u;
    config.overall_timeout_ms = 3000u;
    rc = ft_tt_watchdog_wait(&session->noc, &solo, 1u, &config, &report);
    ft_tt_test_check_fmt("the watchdog returns instead of hanging the session", rc == FT_TT_EIO,
                         "rc=%d after %u ms", rc, report.elapsed_ms);
    ft_tt_test_check_fmt("it says the stalled core is ALIVE, not dead", report.stalled_core_is_alive,
                         "state=%s", ft_tt_queue_state_string(report.stalled_state));
    ft_tt_test_check_fmt("and names the state it is blocked in",
                         report.stalled_state == (uint32_t)FT_TT_QUEUE_STATE_SEM_WAIT, "%s",
                         ft_tt_queue_state_string(report.stalled_state));

    /* Recovery is a WRITE. No reset is asserted anywhere in this path — that is what
     * wedged Tensix 0, and every resident kernel is blocked in a wait by design. */
    rc = ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
    ft_tt_test_check_fmt("the deadlocked kernel is retired by ASKING, not by reset",
                         rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=%u", rc, unstopped);

    (void)ft_tt_queue_read_control(&session->noc, &queue, fields);
    ft_tt_test_check_fmt("it reached EXIT",
                         fields[FT_TT_QUEUE_FIELD_STATE] == (uint32_t)FT_TT_QUEUE_STATE_EXIT, "%s",
                         ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]));
    ft_tt_test_check_fmt("and the abandoned descriptor was NOT retired",
                         fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] == 0u, "completed=%u",
                         fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT]);

    /* THE ACCEPTANCE: the same core runs a fresh resident kernel afterwards. If this
     * needed a power cycle, the watchdog does not work. */
    {
        uint32_t readback = 0u;

        (void)poison_scratch(&session->noc, placement->x, placement->y, SCRATCH_BASE, 1u);
        rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                              QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &queue);
        if (rc == FT_TT_OK) {
            memset(descriptors, 0, sizeof(descriptors));
            descriptors[0].opcode = FT_TT_QUEUE_OP_WRITE_WORD;
            descriptors[0].a0 = SCRATCH_BASE;
            descriptors[0].a1 = 0xA11E0001u;
            rc = ft_tt_queue_enqueue(&session->noc, &queue, descriptors, 1u);
        }
        if (rc == FT_TT_OK)
            rc = start_resident(&session->noc, placement->x, placement->y,
                                &g_resident_config[flat_index(1u, 0u)], &queue);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &queue, 1u, RUN_TIMEOUT_MS, NULL);
        (void)session->noc.read(session->noc.user, placement->x, placement->y, SCRATCH_BASE,
                                &readback, sizeof(readback));
        ft_tt_test_check_fmt("THE CARD SURVIVED: the same core runs a fresh kernel correctly",
                             rc == FT_TT_OK && readback == 0xA11E0001u, "rc=%d readback=0x%08x", rc,
                             readback);
        (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
    }
    return ft_tt_test_failure_count() == 0;
}

/* ------------------------------------------------------------------------- */
/* Stage 6 — the two opcodes that were implemented and never run              */
/* ------------------------------------------------------------------------- */

/**
 * `FT_TT_QUEUE_OP_L1_COPY` and `FT_TT_QUEUE_OP_DELAY` shipped with Phase 5 and neither had
 * ever executed on the card: only NOP, WRITE_WORD, SEM_POST, SEM_WAIT and STOP had. An
 * opcode nobody has run is a claim, not a feature.
 *
 * L1_COPY is checked the way dev notes §3 says a transfer has to be: the destination is
 * POISONED first, and the payload is VARIED across three patterns including an ALL-ZERO
 * one. Zero earns its place because it is the pattern that catches "the copy did nothing"
 * against a non-zero poison — and it is also the one a WRONG SOURCE passes for free, since
 * uninitialised L1 is usually zero too, so a set that is only zero proves the opposite of
 * what it looks like. The copy is also CROSS-CORE, from group 0's core into group 1's,
 * because a same-core copy cannot tell a NoC write from a local store.
 *
 * DELAY is checked as a RATIO, not against a wall-clock constant: two descriptors whose
 * iteration counts differ 8x must differ in elapsed time by roughly 8x. An absolute
 * threshold on a 2-core host also driving eight K80s measures the host's mood.
 */
static bool stage_6_unexercised_opcodes(const ft_tt_hardware_session *session)
{
    const ft_tt_launch_core_placement *source =
        ft_tt_launch_core_set_at(&g_groups[0].cores, 0u);
    const ft_tt_launch_core_placement *destination =
        ft_tt_launch_core_set_at(&g_groups[1].cores, 0u);
    static const uint32_t patterns[3][4] = {
        { 0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u },
        { 0u, 0u, 0u, 0u },
        { 0xFEEDFACEu, 0x00000001u, 0xFFFFFFFFu, 0x5A5A5A5Au }
    };
    const uint32_t copy_source = SCRATCH_BASE + 0x800u;
    const uint32_t copy_destination = SCRATCH_BASE + 0xC00u;
    ft_tt_queue queue;
    ft_tt_queue_descriptor descriptors[2];
    uint32_t completed = 0u;
    unsigned pattern;
    int rc;

    ft_tt_test_step("stage 6 — L1_COPY and DELAY, the two opcodes that had never run");

    rc = ft_tt_queue_init(&session->noc, source->x, source->y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          ft_tt_queue_make_nonce(), &queue);
    if (rc == FT_TT_OK)
        rc = load_resident_image(&session->noc, source->x, source->y);
    if (rc == FT_TT_OK)
        rc = start_resident(&session->noc, source->x, source->y,
                            &g_resident_config[flat_index(0u, 0u)], &queue);
    ft_tt_test_check_fmt("the copying core is resident", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    for (pattern = 0; pattern < 3u; pattern++) {
        uint32_t readback[4] = { 0u, 0u, 0u, 0u };

        /* Poison the DESTINATION on the other core, and plant the payload on this one. */
        (void)poison_scratch(&session->noc, destination->x, destination->y, copy_destination, 4u);
        (void)session->noc.write(session->noc.user, source->x, source->y, copy_source,
                                 patterns[pattern], sizeof(patterns[pattern]));

        memset(descriptors, 0, sizeof(descriptors));
        descriptors[0].opcode = FT_TT_QUEUE_OP_L1_COPY;
        descriptors[0].a0 = destination->x;
        descriptors[0].a1 = destination->y;
        descriptors[0].a2 = copy_destination;
        descriptors[0].a3 = copy_source;
        descriptors[0].a4 = sizeof(patterns[pattern]);
        rc = ft_tt_queue_enqueue(&session->noc, &queue, descriptors, 1u);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &queue, pattern + 1u, RUN_TIMEOUT_MS,
                                            &completed);
        ft_tt_test_check_fmt("L1_COPY completed", rc == FT_TT_OK, "pattern %u completed=%u", pattern,
                             completed);

        (void)session->noc.read(session->noc.user, destination->x, destination->y, copy_destination,
                                readback, sizeof(readback));
        ft_tt_test_check_fmt("and the OTHER core's L1 now holds this pattern, not the poison",
                             memcmp(readback, patterns[pattern], sizeof(readback)) == 0,
                             "pattern %u: [0]=0x%08x want 0x%08x, [3]=0x%08x want 0x%08x", pattern,
                             readback[0], patterns[pattern][0], readback[3], patterns[pattern][3]);
    }

    /* DELAY. Timed on the DEVICE-SIDE path — the descriptors are staged first and the
     * window contains one head write — so the number is the kernel's spin and not the
     * host's enqueue. */
    {
        static const uint32_t iterations[2] = { 250000u, 2000000u };
        uint64_t elapsed[2] = { 0u, 0u };
        unsigned index;

        for (index = 0; index < 2u; index++) {
            uint64_t started;

            memset(descriptors, 0, sizeof(descriptors));
            descriptors[0].opcode = FT_TT_QUEUE_OP_DELAY;
            descriptors[0].a0 = iterations[index];
            rc = ft_tt_queue_stage(&session->noc, &queue, descriptors, 1u);
            started = monotonic_microseconds();
            if (rc == FT_TT_OK)
                rc = ft_tt_queue_publish(&session->noc, &queue);
            if (rc == FT_TT_OK)
                rc = ft_tt_queue_wait_completed(&session->noc, &queue, 3u + index + 1u,
                                                RUN_TIMEOUT_MS, &completed);
            elapsed[index] = monotonic_microseconds() - started;
            ft_tt_test_check_fmt("DELAY completed", rc == FT_TT_OK, "%u iterations, %llu us",
                                 iterations[index], (unsigned long long)elapsed[index]);
        }

        printf("\n  DELAY: %u iterations in %llu us, %u iterations in %llu us\n", iterations[0],
               (unsigned long long)elapsed[0], iterations[1], (unsigned long long)elapsed[1]);

        /* The 8x request must show up as a time ratio in a generous band. A DELAY that did
         * nothing would land at 1.0, and one that ignored its argument would too — which is
         * exactly what an unexercised opcode most plausibly does. */
        ft_tt_test_check_fmt("DELAY takes measurably longer than a NOP", elapsed[0] > 100u,
                             "%llu us for %u iterations", (unsigned long long)elapsed[0],
                             iterations[0]);
        ft_tt_test_check_fmt("and 8x the iterations takes 4x-16x the time, so a0 is read",
                             elapsed[0] > 0u && elapsed[1] >= elapsed[0] * 4u &&
                                 elapsed[1] <= elapsed[0] * 16u,
                             "ratio %.2f", elapsed[0] ? (double)elapsed[1] / (double)elapsed[0] : 0.0);
    }

    {
        uint32_t unstopped = 0u;
        ft_tt_core_group solo = g_groups[0];

        solo.cores.count = 1u;
        solo.queues = &queue;
        rc = ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
        ft_tt_test_check_fmt("the copying core retires by asking", rc == FT_TT_OK && unstopped == 0u,
                             "rc=%d unstopped=%u", rc, unstopped);
    }
    return ft_tt_test_failure_count() == 0;
}

/* ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* Stage 7 — PHASE D's gate: an ON-CARD dispatcher against the host round trip  */
/* ------------------------------------------------------------------------- */

/** The dispatcher program (kernels/dispatch_gate.c), loaded on the SECOND core. */
static ft_tt_program_directory g_dispatch_program;
static bool g_dispatch_loaded;

#define GATE_TABLE_ADDRESS   (FT_TT_OP_CB_ARENA_BASE + 0x4000u) /* 64-aligned, clear of scratch */
#define GATE_MAGIC           0x47505344u
#define GATE_ENTRY_WORDS     16u
#define GATE_FIRST_ENTRY     32u
#define GATE_MAX_ENTRIES     1024u
#define GATE_POLL_BOUND      2000000u

static uint32_t g_gate_table[GATE_FIRST_ENTRY + GATE_MAX_ENTRIES * GATE_ENTRY_WORDS];

/**
 * Write the table for `entries` NOP descriptors into the dispatcher core's L1. Tokens are the
 * TARGET queue's own, stamped from the host's copy of its run nonce and issued counter — the
 * host owns the nonce, which is the PHASE D design. `corrupt` (>= 0) stamps a wrong token on
 * that entry's DESCRIPTOR while its expected token stays right: the red arm.
 */
static int gate_write_table(const ft_tt_noc_access *noc, uint16_t dx, uint16_t dy,
                            const ft_tt_queue *target, uint32_t entries, int corrupt)
{
    uint32_t i;

    memset(g_gate_table, 0, sizeof(g_gate_table));
    g_gate_table[0] = GATE_MAGIC;
    g_gate_table[1] = entries;
    g_gate_table[2] = target->x;
    g_gate_table[3] = target->y;
    g_gate_table[4] = target->control_address;
    g_gate_table[5] = target->ring_address;
    g_gate_table[6] = target->capacity - 1u;
    g_gate_table[7] = target->issued;
    g_gate_table[8] = GATE_POLL_BOUND;
    for (i = 0; i < entries; i++) {
        uint32_t *record = g_gate_table + GATE_FIRST_ENTRY + i * GATE_ENTRY_WORDS;
        const uint32_t token = ft_tt_queue_token_at(target, target->issued + i);

        record[0] = FT_TT_QUEUE_OP_NOP;
        record[1] = ((int)i == corrupt) ? (token ^ 0x00F0F0F0u) : token;
        record[8] = token;
    }
    return noc->write(noc->user, dx, dy, GATE_TABLE_ADDRESS, g_gate_table,
                      (size_t)(GATE_FIRST_ENTRY + entries * GATE_ENTRY_WORDS) * 4u);
}

/** Load, stage and kick the dispatcher, then poll its state word. Returns elapsed us. */
static double gate_run_dispatcher(const ft_tt_noc_access *noc, uint16_t dx, uint16_t dy,
                                  const ft_tt_launch_config *config, uint32_t *out_status,
                                  int *out_rc)
{
    uint32_t args[1] = { GATE_TABLE_ADDRESS };
    ft_tt_op_args block;
    uint64_t started, deadline;
    int rc;

    rc = ft_tt_launch_load_image(noc, dx, dy, FT_TT_PROC_DM0, FT_TT_IMAGE_KERNEL,
                                 g_dispatch_program.program.images[0].elf_data,
                                 g_dispatch_program.program.images[0].elf_size, NULL);
    block.processor = FT_TT_PROC_DM0;
    block.values = args;
    block.count = 1u;
    if (rc == FT_TT_OK)
        rc = ft_tt_op_stage(noc, dx, dy, config, g_cb_buffer, sizeof(g_cb_buffer), &block, 1u);
    if (rc != FT_TT_OK) {
        *out_rc = rc;
        return 0.0;
    }
    started = monotonic_microseconds();
    rc = ft_tt_launch_send_go(noc, dx, dy);
    deadline = started + (uint64_t)RUN_TIMEOUT_MS * 1000u;
    while (rc == FT_TT_OK) {
        rc = noc->read(noc->user, dx, dy, GATE_TABLE_ADDRESS + 24u * 4u, out_status, 24u);
        if (rc != FT_TT_OK || out_status[0] == 2u || out_status[0] == 3u)
            break;
        if (monotonic_microseconds() > deadline) {
            rc = FT_TT_EIO;
            break;
        }
    }
    *out_rc = rc;
    return (double)(monotonic_microseconds() - started);
}

static bool stage_7_dispatch_gate(const ft_tt_hardware_session *session)
{
    const unsigned target_flat = 0u, dispatcher_flat = 1u;
    const ft_tt_launch_core_placement *tp = ft_tt_launch_core_set_at(&g_groups[0].cores, 0u);
    const ft_tt_launch_core_placement *dp = ft_tt_launch_core_set_at(&g_groups[0].cores, 1u);
    const uint32_t sizes[3] = { 64u, 256u, 1024u };
    ft_tt_queue queue;
    uint32_t status[6];
    unsigned round, size;
    int rc;

    ft_tt_test_step("PHASE D gate: core (%u, %u) dispatches NOPs into core (%u, %u)'s queue "
                    "on-card, against the host issuing and awaiting each one",
                    dp->x, dp->y, tp->x, tp->y);
    rc = load_resident_image(&session->noc, tp->x, tp->y);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&session->noc, tp->x, tp->y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                              ft_tt_queue_make_nonce(), &queue);
    if (rc == FT_TT_OK)
        rc = start_resident(&session->noc, tp->x, tp->y, &g_resident_config[target_flat], &queue);
    ft_tt_test_check_fmt("the target residency is up", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    for (size = 0; size < 3u; size++) {
        for (round = 0; round < 2u; round++) {
            const uint32_t n = sizes[size];
            uint64_t started = monotonic_microseconds();
            uint32_t i;
            double host_us, card_us;

            /* ARM H — the host's own shape today: one descriptor, one await, n times. */
            for (i = 0; i < n && rc == FT_TT_OK; i++) {
                fill_nop_batch(g_batch, 1u);
                rc = ft_tt_queue_enqueue(&session->noc, &queue, g_batch, 1u);
                if (rc == FT_TT_OK)
                    rc = ft_tt_queue_wait_completed(&session->noc, &queue, queue.issued,
                                                    RUN_TIMEOUT_MS, NULL);
            }
            host_us = (double)(monotonic_microseconds() - started) / (double)n;
            ft_tt_test_check_fmt("host arm completed", rc == FT_TT_OK, "n=%u rc=%d", n, rc);
            if (rc != FT_TT_OK)
                return false;

            /* ARM D — the dispatcher issues and awaits the same n on-card. */
            rc = gate_write_table(&session->noc, dp->x, dp->y, &queue, n, -1);
            card_us = (rc == FT_TT_OK)
                          ? gate_run_dispatcher(&session->noc, dp->x, dp->y,
                                                &g_resident_config[dispatcher_flat], status, &rc)
                          : 0.0;
            ft_tt_test_check_fmt("dispatcher completed every entry",
                                 rc == FT_TT_OK && status[0] == 2u && status[1] == n,
                                 "rc=%d state %u done %u error %u count %u", rc, status[0],
                                 status[1], status[2], status[3]);
            if (rc != FT_TT_OK || status[0] != 2u)
                return false;
            /* THE HOST TAKES ITS COUNTERS BACK, advanced by what the card issued. */
            queue.issued += n;
            queue.head += n;
            rc = ft_tt_queue_wait_completed(&session->noc, &queue, queue.issued, RUN_TIMEOUT_MS,
                                            NULL);
            ft_tt_test_check_fmt("the target's completion agrees with the host's resync",
                                 rc == FT_TT_OK, "issued %u rc=%d", queue.issued, rc);
            printf("GATE n=%u round %u: host %.2f us/descriptor, on-card %.2f us/descriptor "
                   "(doorbell to done %.0f us)\n", n, round, host_us, card_us / (double)n,
                   card_us);
            if (rc != FT_TT_OK)
                return false;
        }
    }

    /* RED: one entry's descriptor carries a wrong token. The target executes it and echoes
     * that token, so the dispatcher must stop AT that entry with error 3 (token mismatch). */
    rc = gate_write_table(&session->noc, dp->x, dp->y, &queue, 64u, 17);
    if (rc == FT_TT_OK)
        (void)gate_run_dispatcher(&session->noc, dp->x, dp->y,
                                  &g_resident_config[dispatcher_flat], status, &rc);
    ft_tt_test_check_fmt("RED: a wrong token stops the dispatcher at that entry, error 3",
                         rc == FT_TT_OK && status[0] == 3u && status[1] == 17u &&
                             status[2] == 3u,
                         "state %u done %u error %u", status[0], status[1], status[2]);
    queue.issued += 18u;
    queue.head += 18u;

    {
        uint32_t unstopped = 0u;
        ft_tt_core_group solo = g_groups[0];

        solo.cores.count = 1u;
        solo.queues = &queue;
        rc = ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
        ft_tt_test_check_fmt("the target retires by asking", rc == FT_TT_OK && unstopped == 0u,
                             "rc=%d unstopped %u", rc, unstopped);
    }
    return ft_tt_test_failure_count() == 0;
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    const char *resident_directory = (argc > 1) ? argv[1] : ft_tt_firmware_directory();
    const char *oneshot_directory = (argc > 2) ? argv[2] : NULL;
    unsigned group;
    int rc;

    if (!oneshot_directory) {
        fprintf(stderr, "usage: %s <resident-program-dir> <oneshot-program-dir>\n", argv[0]);
        return 2;
    }
    if (!load_environment())
        return ft_tt_test_finish();

    printf("Phase 5 resident dispatch — cores %u, %u, %u, %u\n", g_core_indices[0],
           g_core_indices[1], g_core_indices[2], g_core_indices[3]);

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;

    /* Own the four cores for the whole run, per-core rather than card-wide, and
     * refuse with the holder's name if any of them is taken. `retire` is NULL:
     * this suite boots firmware from scratch on every core it claims, which is
     * the one case where accepting whatever state a core is in is defensible.
     * Anything RESUMING a residency must pass a real retirement instead — a
     * released claim is not a retired kernel. */
    rc = ft_tt_hardware_session_claim_cores(&session, g_core_indices, CORE_COUNT,
                                            "phase5 resident dispatch", NULL, NULL);
    ft_tt_test_check_fmt("all four cores are claimed for this run", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return ft_tt_test_finish();
    }

    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    rc = ft_tt_program_directory_load(resident_directory, BRISC_ONLY_MASK, &g_resident_program);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(oneshot_directory, BRISC_ONLY_MASK, &g_oneshot_program);
    if (rc != FT_TT_OK) {
        ft_tt_firmware_set_release(&firmware);
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    if (argc > 3 && ft_tt_program_directory_load(argv[3], BRISC_ONLY_MASK,
                                                 &g_dispatch_program) == FT_TT_OK)
        g_dispatch_loaded = true;
    if (build_groups(&session) && boot_and_load(&session, &firmware)) {
        /* `FT_TT_PHASE5_GATE_ONLY=1` runs PHASE D's gate alone. */
        const char *gate_only = getenv("FT_TT_PHASE5_GATE_ONLY");

        if (gate_only && gate_only[0] == '1') {
            if (g_dispatch_loaded)
                (void)stage_7_dispatch_gate(&session);
            else
                ft_tt_test_check("the dispatcher program loaded (argv[3])", false, "missing");
        } else if (stage_1_resident_runs(&session) && stage_2_dispatch_ab(&session) &&
                   stage_3_two_groups(&session)) {
            (void)(stage_4_cross_group_semaphore(&session) &&
                   stage_5_deadlock_and_recovery(&session) &&
                   stage_6_unexercised_opcodes(&session) &&
                   (!g_dispatch_loaded || stage_7_dispatch_gate(&session)));
        }
    }

    /* Park on EVERY exit path. By here every kernel has reported EXIT, so this is a park
     * of an idle core rather than a reset of a blocked one. */
    for (group = 0; group < GROUP_COUNT; group++)
        (void)ft_tt_launch_park_all(&session.noc, &g_groups[group].cores);

    if (g_dispatch_loaded)
        ft_tt_program_directory_release(&g_dispatch_program);
    ft_tt_program_directory_release(&g_oneshot_program);
    ft_tt_program_directory_release(&g_resident_program);
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
