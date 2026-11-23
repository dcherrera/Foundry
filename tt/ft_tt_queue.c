/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_queue.c
 * @brief Host half of the Phase 5 work-queue protocol — see ft_tt_queue.h for the
 *   contract and the three rules it is shaped by.
 *
 * @status Green off-hardware in tests/test_queue.c and on the card in
 *   tests/test_phase5_resident.c, on two disjoint core sets, 2026-08-12.
 * @issues None
 * @todo None
 */

/* clock_gettime and CLOCK_MONOTONIC are POSIX.1-2008; -std=c99 turns off glibc's
 * default feature set, so ask for them by name or the Linux card host's gcc hides them. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_queue.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ft_tt_device.h"

/** Descriptors written in one device access. A batch is contiguous in the ring, so this
 *  bounds a stack buffer rather than the protocol. */
#define QUEUE_MAX_BATCH 1024u

/* LAYOUT PINS. Deliberately literal, not derived: a derived pin re-derives the mistake and
 * passes, which is exactly how eleven self-consistent asserts on ft_tt_mailbox.h missed a
 * 16-byte insertion (dev notes §3). These are the offsets the kernels have compiled
 * against on silicon. Moving one is a rebuild of every resident kernel, not a header edit. */
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_CONTROL_BYTES == 64u, control_block_is_64_bytes);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_STATUS_OFFSET == 0x40u, status_at_0x40);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_STATUS_BYTES == 32u, status_slot_is_32_bytes);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_STATUS_OFFSET + FT_TT_PROC_COUNT * FT_TT_QUEUE_STATUS_BYTES <=
                        FT_TT_QUEUE_RING_OFFSET,
                    status_array_clears_the_ring);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_RING_OFFSET == 0x100u, ring_at_0x100);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_DESCRIPTOR_BYTES == 32u, descriptor_is_32_bytes);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_MATMUL_PARAM_OFFSET == 0x9000u, matmul_params_at_0x9000);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_GROUP_OFFSET == 0xF000u, group_block_at_0xF000);
/* The ring at its largest must not reach the parameter blocks, and the parameter blocks
 * must not reach the group block. */
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_RING_OFFSET + QUEUE_MAX_BATCH * FT_TT_QUEUE_DESCRIPTOR_BYTES <=
                        FT_TT_QUEUE_MATMUL_PARAM_OFFSET,
                    max_ring_clears_the_parameter_blocks);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_MATMUL_PARAM_OFFSET +
                            FT_TT_QUEUE_MATMUL_PARAM_SLOTS * FT_TT_QUEUE_MATMUL_PARAM_BYTES <=
                        FT_TT_QUEUE_GROUP_OFFSET,
                    parameter_blocks_clear_the_group_block);
/* And the whole thing must fit the region ft_tt_op_l1_layout.h reserved for it. */
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_GROUP_OFFSET +
                            (FT_TT_QUEUE_GROUP_HEADER_WORDS +
                             FT_TT_QUEUE_GROUP_MAX_MEMBERS * FT_TT_QUEUE_GROUP_SLOT_WORDS) * 4u <=
                        FT_TT_OP_QUEUE_BYTES,
                    group_block_fits_the_queue_region);
/* The weight-format bit: kMatmulWeightBfp4 in kernels/resident_queue_protocol.h, literal for
 * the same reason as the pins above. Bits 0, 8..19 (that header) and 20..29 (the MoE build)
 * are taken; the bit must be none of them. */
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4 == 0x40000000u,
                    weight_bfp4_flag_is_word13_bit30);
FT_TT_STATIC_ASSERT((FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4 & 0x3FFFFF01u) == 0u,
                    weight_bfp4_flag_collides_with_no_taken_bit);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_MATMUL_WEIGHT_TILE_BYTES_BFP8 == FT_TT_OP_TILE_BYTES_BFP8_B &&
                        FT_TT_QUEUE_MATMUL_WEIGHT_TILE_BYTES_BFP4 == FT_TT_OP_TILE_BYTES_BFP4_B,
                    weight_tile_sizes_match_the_l1_layout);

/* ------------------------------------------------------------------------- */
/* Small helpers                                                             */
/* ------------------------------------------------------------------------- */

static bool is_power_of_two(uint32_t value)
{
    return value != 0u && (value & (value - 1u)) == 0u;
}

static uint32_t control_field_address(const ft_tt_queue *queue, ft_tt_queue_control_field field)
{
    return queue->control_address + (uint32_t)field * 4u;
}

static int write_control_word(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                              ft_tt_queue_control_field field, uint32_t value)
{
    return noc->write(noc->user, queue->x, queue->y, control_field_address(queue, field), &value,
                      sizeof(value));
}

static uint64_t monotonic_microseconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0u;
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)(now.tv_nsec / 1000);
}

/* ------------------------------------------------------------------------- */
/* Setup                                                                     */
/* ------------------------------------------------------------------------- */

uint32_t ft_tt_queue_make_nonce(void)
{
    uint64_t mixed = monotonic_microseconds();
    uint32_t nonce;

    mixed ^= (uint64_t)getpid() << 32;
    mixed ^= mixed >> 29;
    mixed *= 0xBF58476D1CE4E5B9u;
    mixed ^= mixed >> 32;

    nonce = (uint32_t)mixed;
    /* Never 0 and never the poison: both are values a completion check must be able to
     * tell a real token APART from. */
    if (nonce == 0u || nonce == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
        nonce = 0x5150C0DEu;
    return nonce;
}

uint32_t ft_tt_queue_token_at(const ft_tt_queue *queue, uint32_t index)
{
    uint32_t token;

    if (!queue)
        return 0u;
    /* Index is mixed in as an XOR of a rotated counter rather than added, so two runs
     * whose nonces differ by a small amount cannot produce overlapping token sets. */
    token = queue->run_nonce ^ ((index + 1u) * 0x9E3779B9u);
    if (token == 0u || token == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
        token = queue->run_nonce ^ 0xA5A5A5A5u;
    return token;
}

int ft_tt_queue_init(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                     uint32_t control_address, uint32_t capacity, uint32_t run_nonce,
                     ft_tt_queue *out_queue)
{
    uint32_t block[FT_TT_QUEUE_FIELD_COUNT];
    ft_tt_queue queue = {0};
    int rc;

    if (!noc || !noc->write || !out_queue)
        return FT_TT_EINVAL;
    if ((control_address & 3u) != 0u)
        return FT_TT_EINVAL;
    if (!is_power_of_two(capacity) || capacity < 2u)
        return FT_TT_EINVAL;
    if (run_nonce == 0u || run_nonce == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
        return FT_TT_EINVAL;

    queue.x = x;
    queue.y = y;
    queue.control_address = control_address;
    queue.ring_address = control_address + FT_TT_QUEUE_RING_OFFSET;
    queue.capacity = capacity;
    queue.head = 0u;
    queue.run_nonce = run_nonce;
    queue.issued = 0u;

    memset(block, 0, sizeof(block));
    block[FT_TT_QUEUE_FIELD_MAGIC] = FT_TT_QUEUE_MAGIC;
    block[FT_TT_QUEUE_FIELD_CAPACITY] = capacity;
    block[FT_TT_QUEUE_FIELD_STATE] = (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON;
    /* The token slot is poisoned too. A zero there would be indistinguishable from a
     * kernel that completed an item whose token happened to be zero — which is why
     * ft_tt_queue_token_at() never issues one. */
    block[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] = (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON;

    rc = noc->write(noc->user, x, y, control_address, block, sizeof(block));
    if (rc != FT_TT_OK) {
        ft_tt_log_error("queue: writing the control block at 0x%08x failed (%d)",
                        control_address, rc);
        return rc;
    }

    *out_queue = queue;
    return FT_TT_OK;
}

int ft_tt_queue_kernel_args(const ft_tt_queue *queue, const ft_tt_queue_kernel_binding *binding,
                            uint32_t out_args[FT_TT_QUEUE_KERNEL_ARG_COUNT])
{
    ft_tt_queue_kernel_binding defaults = {0};

    if (!queue || !out_args)
        return FT_TT_EINVAL;
    if (!is_power_of_two(queue->capacity))
        return FT_TT_EINVAL;
    if (!binding) {
        defaults.idle_spin_limit = FT_TT_QUEUE_DEFAULT_IDLE_SPINS;
        defaults.cb_wait_spin_limit = FT_TT_QUEUE_DEFAULT_IDLE_SPINS;
        binding = &defaults;
    }
    /* 0 would have to mean "spin forever" on either bound, and a resident kernel with an
     * unbounded wait is the one thing this phase may not ship (ft_tt_queue.h rule 3, and
     * kernels/resident_queue_protocol.h rule 4). */
    if (binding->idle_spin_limit == 0u || binding->cb_wait_spin_limit == 0u)
        return FT_TT_EINVAL;
    if (binding->group_address != 0u && binding->group_member >= FT_TT_QUEUE_GROUP_MAX_MEMBERS)
        return FT_TT_EINVAL;

    out_args[0] = queue->control_address;
    out_args[1] = queue->capacity - 1u;
    out_args[2] = queue->ring_address;
    out_args[3] = binding->idle_spin_limit;
    out_args[4] = binding->status_address;
    out_args[5] = binding->cb_wait_spin_limit;
    out_args[6] = binding->group_x;
    out_args[7] = binding->group_y;
    out_args[8] = binding->group_address;
    out_args[9] = binding->group_member;
    return FT_TT_OK;
}

int ft_tt_queue_read_status(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                            unsigned processor, uint32_t out_fields[FT_TT_QUEUE_STATUS_WORDS])
{
    if (!noc || !noc->read || !queue || !out_fields)
        return FT_TT_EINVAL;
    if (processor >= FT_TT_PROC_COUNT)
        return FT_TT_EINVAL;
    return noc->read(noc->user, queue->x, queue->y,
                     FT_TT_QUEUE_STATUS_ADDRESS(queue->control_address, processor), out_fields,
                     FT_TT_QUEUE_STATUS_BYTES);
}

/* ------------------------------------------------------------------------- */
/* The group completion block, addressed by COORDINATE                       */
/* ------------------------------------------------------------------------- */

/** L1 address of slot 0 — where the hot-path read starts, past the header. */
static uint32_t group_slots_address(uint32_t group_address)
{
    return group_address + FT_TT_QUEUE_GROUP_HEADER_WORDS * 4u;
}

uint32_t ft_tt_queue_group_block_words(uint32_t members)
{
    return FT_TT_QUEUE_GROUP_HEADER_WORDS + members * FT_TT_QUEUE_GROUP_SLOT_WORDS;
}

int ft_tt_queue_group_init(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           uint32_t group_address, uint32_t members)
{
    uint32_t block[FT_TT_QUEUE_GROUP_HEADER_WORDS +
                   FT_TT_QUEUE_GROUP_MAX_MEMBERS * FT_TT_QUEUE_GROUP_SLOT_WORDS];

    if (!noc || !noc->write || group_address == 0u || (group_address & 3u) != 0u)
        return FT_TT_EINVAL;
    if (members == 0u)
        return FT_TT_EINVAL;
    if (members > FT_TT_QUEUE_GROUP_MAX_MEMBERS)
        return FT_TT_ENOMEM;

    memset(block, 0, sizeof(block));
    block[FT_TT_QUEUE_GROUP_MAGIC] = FT_TT_QUEUE_GROUP_MAGIC_VALUE;
    block[FT_TT_QUEUE_GROUP_MEMBERS] = members;
    return noc->write(noc->user, x, y, group_address, block,
                      (size_t)ft_tt_queue_group_block_words(members) * 4u);
}

int ft_tt_queue_group_read(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           uint32_t group_address, uint32_t members,
                           ft_tt_queue_group_slot *out_slots)
{
    uint32_t block[FT_TT_QUEUE_GROUP_HEADER_WORDS +
                   FT_TT_QUEUE_GROUP_MAX_MEMBERS * FT_TT_QUEUE_GROUP_SLOT_WORDS];
    uint32_t slot;
    int rc;

    if (!noc || !noc->read || !out_slots || group_address == 0u || (group_address & 3u) != 0u)
        return FT_TT_EINVAL;
    if (members == 0u || members > FT_TT_QUEUE_GROUP_MAX_MEMBERS)
        return FT_TT_EINVAL;

    rc = noc->read(noc->user, x, y, group_address, block,
                   (size_t)ft_tt_queue_group_block_words(members) * 4u);
    if (rc != FT_TT_OK)
        return rc;
    if (block[FT_TT_QUEUE_GROUP_MAGIC] != FT_TT_QUEUE_GROUP_MAGIC_VALUE)
        return FT_TT_ERROR;

    for (slot = 0u; slot < members; slot++) {
        const uint32_t base =
            FT_TT_QUEUE_GROUP_HEADER_WORDS + slot * FT_TT_QUEUE_GROUP_SLOT_WORDS;

        out_slots[slot].sequence = block[base];
        out_slots[slot].token = block[base + 1u];
    }
    return FT_TT_OK;
}

int ft_tt_queue_group_read_slots(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                 uint32_t group_address, uint32_t members,
                                 ft_tt_queue_group_slot *out_slots)
{
    /* One slot of slack: an ODD member count is rounded UP to the 16-byte unit below. */
    uint32_t words[(FT_TT_QUEUE_GROUP_MAX_MEMBERS + 1u) * FT_TT_QUEUE_GROUP_SLOT_WORDS];
    size_t bytes;
    uint32_t slot;
    int rc;

    if (!noc || !noc->read || !out_slots || group_address == 0u || (group_address & 3u) != 0u)
        return FT_TT_EINVAL;
    if (members == 0u || members > FT_TT_QUEUE_GROUP_MAX_MEMBERS)
        return FT_TT_EINVAL;

    /* THE 16-BYTE UNIT IS THE POINT OF THIS CALL, so the length has to respect it: a slot is
     * 8 bytes and an ODD member count is not a multiple of 16. Reading one slot further is
     * free — the block is 4 KiB and the header keeps the slots 16-byte aligned — and the
     * extra slot is simply not looked at. Getting this wrong is not a wrong answer: the L1
     * accessor REFUSES the length, and here that refusal reached the forward as a failed
     * projection, which retired and re-armed the residency three times in a row. */
    bytes = ((size_t)members * FT_TT_QUEUE_GROUP_SLOT_WORDS * 4u + 15u) & ~(size_t)15u;
    rc = noc->read(noc->user, x, y, group_slots_address(group_address), words, bytes);
    if (rc != FT_TT_OK)
        return rc;
    for (slot = 0u; slot < members; slot++) {
        out_slots[slot].sequence = words[slot * FT_TT_QUEUE_GROUP_SLOT_WORDS];
        out_slots[slot].token = words[slot * FT_TT_QUEUE_GROUP_SLOT_WORDS + 1u];
    }
    return FT_TT_OK;
}

int ft_tt_queue_all_processors_exited(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                      uint32_t processor_mask, uint32_t *out_pending)
{
    uint32_t pending = 0u;
    unsigned processor;

    if (!noc || !noc->read || !queue)
        return FT_TT_EINVAL;

    if (processor_mask == 0u) {
        /* The BRISC-only kernel: the control block's own state IS the whole answer, because
         * there is exactly one processor and it is the writer of that word. */
        uint32_t state = 0u;
        int rc = ft_tt_queue_read_field(noc, queue, FT_TT_QUEUE_FIELD_STATE, &state);

        if (rc != FT_TT_OK)
            return rc;
        if (out_pending)
            *out_pending = (state == (uint32_t)FT_TT_QUEUE_STATE_EXIT) ? 0u : 1u;
        return (state == (uint32_t)FT_TT_QUEUE_STATE_EXIT) ? FT_TT_OK : FT_TT_EIO;
    }

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        uint32_t fields[FT_TT_QUEUE_STATUS_WORDS];
        int rc;

        if ((processor_mask & (1u << processor)) == 0u)
            continue;
        rc = ft_tt_queue_read_status(noc, queue, processor, fields);
        if (rc != FT_TT_OK)
            return rc;
        if (fields[FT_TT_QUEUE_STATUS_STATE] != (uint32_t)FT_TT_QUEUE_STATE_EXIT)
            pending |= 1u << processor;
    }

    if (out_pending)
        *out_pending = pending;
    return pending ? FT_TT_EIO : FT_TT_OK;
}

int ft_tt_queue_matmul_weight_flag(uint32_t tile_bytes, uint32_t *out_flag)
{
    if (!out_flag)
        return FT_TT_EINVAL;
    if (tile_bytes == FT_TT_QUEUE_MATMUL_WEIGHT_TILE_BYTES_BFP8) {
        *out_flag = 0u;
        return FT_TT_OK;
    }
    if (tile_bytes == FT_TT_QUEUE_MATMUL_WEIGHT_TILE_BYTES_BFP4) {
        *out_flag = FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4;
        return FT_TT_OK;
    }
    return FT_TT_EINVAL;
}

int ft_tt_queue_matmul_check_weight(uint32_t tile_bytes, uint32_t flag_word, bool mixw_residency)
{
    uint32_t expected;

    if (ft_tt_queue_matmul_weight_flag(tile_bytes, &expected) != FT_TT_OK)
        return FT_TT_EINVAL;
    if ((flag_word & FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4) != expected)
        return FT_TT_EINVAL;
    /* A non-mixw build ignores bit 30, so a 576 B weight sent to one is read at 1,088 B. */
    if (!mixw_residency && expected != 0u)
        return FT_TT_EINVAL;
    /* The mixw reader refuses bit 31 (free); refused here so no triad is stranded on it. */
    if (mixw_residency && (flag_word & 0x80000000u) != 0u)
        return FT_TT_EINVAL;
    return FT_TT_OK;
}

int ft_tt_queue_write_op_parameters(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                    uint32_t slot, const uint32_t words[FT_TT_QUEUE_PARAM_WORDS],
                                    uint32_t *out_addr)
{
    return ft_tt_queue_write_matmul_parameters(noc, queue, slot, words, out_addr);
}

int ft_tt_queue_write_matmul_parameters(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                        uint32_t slot,
                                        const uint32_t words[FT_TT_QUEUE_MATMUL_WORDS],
                                        uint32_t *out_addr)
{
    uint32_t address;
    int rc;

    if (!noc || !noc->write || !queue || !words)
        return FT_TT_EINVAL;
    if (slot >= FT_TT_QUEUE_MATMUL_PARAM_SLOTS)
        return FT_TT_EINVAL;

    address = FT_TT_QUEUE_MATMUL_PARAM_ADDRESS(queue->control_address, slot);
    rc = noc->write(noc->user, queue->x, queue->y, address, words, FT_TT_QUEUE_MATMUL_PARAM_BYTES);
    if (rc != FT_TT_OK)
        return rc;
    if (out_addr)
        *out_addr = address;
    return FT_TT_OK;
}

int ft_tt_queue_write_param_cache(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                  uint32_t slot, const uint32_t words[FT_TT_QUEUE_PARAM_WORDS],
                                  uint32_t *out_addr)
{
    uint32_t address;
    int rc;

    if (!noc || !noc->write || !queue || !words)
        return FT_TT_EINVAL;
    if (slot >= FT_TT_QUEUE_PARAM_CACHE_SLOTS)
        return FT_TT_EINVAL;

    address = FT_TT_QUEUE_PARAM_CACHE_ADDRESS(queue->control_address, slot);
    rc = noc->write(noc->user, queue->x, queue->y, address, words,
                    FT_TT_QUEUE_MATMUL_PARAM_BYTES);
    if (rc != FT_TT_OK)
        return rc;
    if (out_addr)
        *out_addr = address;
    return FT_TT_OK;
}

/* The cache must start where the rotating slots END and end where the group block BEGINS.
 * The point of the region is that it is the GAP, so a pin that re-derived the gap could not
 * notice the gap closing — these compare the three offsets against each other directly. */
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_MATMUL_PARAM_OFFSET +
                            FT_TT_QUEUE_MATMUL_PARAM_SLOTS * FT_TT_QUEUE_MATMUL_PARAM_BYTES <=
                        FT_TT_QUEUE_PARAM_CACHE_OFFSET,
                    param_cache_clears_the_rotating_slots);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_PARAM_CACHE_OFFSET +
                            FT_TT_QUEUE_PARAM_CACHE_PRIMARY_SLOTS * FT_TT_QUEUE_MATMUL_PARAM_BYTES <=
                        FT_TT_QUEUE_GROUP_OFFSET,
                    param_cache_clears_the_group_block);
/* The overflow region sits past the largest ring (64 slots) and the MoE mailbox (0x1000..0x1040)
 * and ends where the rotating slots begin. */
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_OFFSET >= 0x1040u &&
                        FT_TT_QUEUE_RING_OFFSET + 64u * FT_TT_QUEUE_DESCRIPTOR_BYTES <= 0x1000u,
                    param_cache_overflow_clears_the_ring_and_mailbox);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_OFFSET +
                            FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_SLOTS * FT_TT_QUEUE_MATMUL_PARAM_BYTES <=
                        FT_TT_QUEUE_MATMUL_PARAM_OFFSET,
                    param_cache_overflow_clears_the_rotating_slots);
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_PARAM_CACHE_OFFSET % 16u == 0u, param_cache_is_l1_aligned);

/* ------------------------------------------------------------------------- */
/* Enqueue — the whole per-op cost of the model                              */
/* ------------------------------------------------------------------------- */

/** Write a run of descriptors that does not wrap. */
static int write_ring_run(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                          uint32_t first_slot, const ft_tt_queue_descriptor *descriptors,
                          uint32_t count)
{
    uint32_t address = queue->ring_address + first_slot * FT_TT_QUEUE_DESCRIPTOR_BYTES;

    return noc->write(noc->user, queue->x, queue->y, address, descriptors,
                      (size_t)count * FT_TT_QUEUE_DESCRIPTOR_BYTES);
}

int ft_tt_queue_stage(const ft_tt_noc_access *noc, ft_tt_queue *queue,
                      ft_tt_queue_descriptor *descriptors, uint32_t count)
{
    uint32_t first_slot;
    uint32_t contiguous;
    uint32_t index;
    int rc;

    if (!noc || !noc->write || !queue || !descriptors || count == 0u)
        return FT_TT_EINVAL;
    if (count > queue->capacity || count > QUEUE_MAX_BATCH)
        return FT_TT_ENOMEM;

    /* Stamp the tokens here rather than trusting the caller. A caller that supplies its
     * own token can supply one a previous run also used, and then the completion check
     * passes on residue. */
    for (index = 0u; index < count; index++)
        descriptors[index].token = ft_tt_queue_token_at(queue, queue->issued + index);

    first_slot = (queue->head + queue->staged) & (queue->capacity - 1u);
    contiguous = queue->capacity - first_slot;
    if (contiguous > count)
        contiguous = count;

    rc = write_ring_run(noc, queue, first_slot, descriptors, contiguous);
    if (rc != FT_TT_OK)
        return rc;
    if (contiguous < count) {
        rc = write_ring_run(noc, queue, 0u, descriptors + contiguous, count - contiguous);
        if (rc != FT_TT_OK)
            return rc;
    }

    queue->staged += count;
    queue->issued += count;
    return FT_TT_OK;
}

int ft_tt_queue_publish(const ft_tt_noc_access *noc, ft_tt_queue *queue)
{
    uint32_t published;
    int rc;

    if (!noc || !noc->write || !queue)
        return FT_TT_EINVAL;
    if (queue->staged == 0u)
        return FT_TT_OK;

    published = queue->head + queue->staged;
    rc = write_control_word(noc, queue, FT_TT_QUEUE_FIELD_HEAD, published);
    if (rc != FT_TT_OK)
        return rc;
    queue->head = published;
    queue->staged = 0u;
    return FT_TT_OK;
}

bool ft_tt_queue_issue_is_wide_addressable(const ft_tt_queue *queue)
{
    if (!queue)
        return false;
    /* ASKED OF THE SHIPPED RULE, not re-derived from it. `ft_tt_noc_check_transfer()` is the
     * function the BLOCK16 L1 accessor itself calls, so this predicate and the refusal it
     * predicts cannot drift apart. Slot 0 answers for every slot: both parameter regions
     * stride by 64 bytes and the ring by 32, and every one of those is a multiple of 16. */
    return ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false,
                                    FT_TT_QUEUE_PARAM_ADDRESS(queue->control_address, 0u),
                                    FT_TT_QUEUE_MATMUL_PARAM_BYTES) == FT_TT_OK &&
           ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false,
                                    FT_TT_QUEUE_PARAM_CACHE_ADDRESS(queue->control_address, 0u),
                                    FT_TT_QUEUE_MATMUL_PARAM_BYTES) == FT_TT_OK &&
           ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_L1, false, queue->ring_address,
                                    FT_TT_QUEUE_DESCRIPTOR_BYTES) == FT_TT_OK;
}

int ft_tt_queue_write_head(const ft_tt_noc_access *noc, const ft_tt_queue *queue, uint32_t head)
{
    if (!noc || !noc->write || !queue)
        return FT_TT_EINVAL;
    return write_control_word(noc, queue, FT_TT_QUEUE_FIELD_HEAD, head);
}

int ft_tt_queue_enqueue(const ft_tt_noc_access *noc, ft_tt_queue *queue,
                        ft_tt_queue_descriptor *descriptors, uint32_t count)
{
    int rc = ft_tt_queue_stage(noc, queue, descriptors, count);

    if (rc != FT_TT_OK)
        return rc;
    /* The head bump is LAST and is one word. */
    return ft_tt_queue_publish(noc, queue);
}

/* ------------------------------------------------------------------------- */
/* Observation                                                               */
/* ------------------------------------------------------------------------- */

int ft_tt_queue_read_control(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                             uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT])
{
    if (!noc || !noc->read || !queue || !out_fields)
        return FT_TT_EINVAL;
    return noc->read(noc->user, queue->x, queue->y, queue->control_address, out_fields,
                     FT_TT_QUEUE_CONTROL_BYTES);
}

int ft_tt_queue_read_completion(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT])
{
    if (!noc || !noc->read || !queue || !out_fields)
        return FT_TT_EINVAL;
    /* The unread fields are ZEROED rather than left indeterminate. A caller that reaches for
     * HEAD out of this buffer then reads a 0 it can see, instead of whatever the stack had —
     * and the four the completion check names are all inside the window. */
    memset(out_fields, 0, FT_TT_QUEUE_FIELD_COUNT * sizeof(out_fields[0]));
    return noc->read(noc->user, queue->x, queue->y,
                     control_field_address(queue, FT_TT_QUEUE_COMPLETION_FIRST_FIELD),
                     &out_fields[FT_TT_QUEUE_COMPLETION_FIRST_FIELD],
                     FT_TT_QUEUE_COMPLETION_FIELD_COUNT * 4u);
}

/** First field of the 16-byte-aligned span that CONTAINS the completion window, and how
 *  many fields that span holds. Word 5 does not start on a 16-byte boundary; word 4 does,
 *  and eight fields from there is exactly two BLOCK16 transactions. */
#define QUEUE_WIDE_FIRST_FIELD 4u
#define QUEUE_WIDE_FIELD_COUNT 8u

int ft_tt_queue_read_completion_wide(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                     uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT])
{
    if (!noc || !noc->read || !queue || !out_fields)
        return FT_TT_EINVAL;
    memset(out_fields, 0, FT_TT_QUEUE_FIELD_COUNT * sizeof(out_fields[0]));
    return noc->read(noc->user, queue->x, queue->y,
                     queue->control_address + QUEUE_WIDE_FIRST_FIELD * 4u,
                     &out_fields[QUEUE_WIDE_FIRST_FIELD], QUEUE_WIDE_FIELD_COUNT * 4u);
}

/** The ONE 16-byte line that holds count, token and state. Word 4 starts it; four fields is
 *  exactly one BLOCK16 transaction. `error` is word 9 and is deliberately outside it. */
#define QUEUE_LINE_FIRST_FIELD 4u
#define QUEUE_LINE_FIELD_COUNT 4u

int ft_tt_queue_read_completion_line(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                     uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT])
{
    if (!noc || !noc->read || !queue || !out_fields)
        return FT_TT_EINVAL;
    memset(out_fields, 0, FT_TT_QUEUE_FIELD_COUNT * sizeof(out_fields[0]));
    return noc->read(noc->user, queue->x, queue->y,
                     queue->control_address + QUEUE_LINE_FIRST_FIELD * 4u,
                     &out_fields[QUEUE_LINE_FIRST_FIELD], QUEUE_LINE_FIELD_COUNT * 4u);
}

int ft_tt_queue_check_completion_line(const ft_tt_queue *queue,
                                      const uint32_t fields[FT_TT_QUEUE_FIELD_COUNT],
                                      uint32_t expected, const char **out_reason)
{
    const char *reason = NULL;

    if (!queue || !fields || expected == 0u) {
        if (out_reason)
            *out_reason = "bad argument";
        return FT_TT_EINVAL;
    }

    /* THE SAME THREE TESTS IN THE SAME ORDER as ft_tt_queue_check_completion(), with the
     * `error` clause removed rather than reordered — so a divergence between the two is a
     * missing clause and not a different verdict. */
    if (fields[FT_TT_QUEUE_FIELD_STATE] == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
        reason = "the kernel never ran: state is still the host poison";
    else if (fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] != expected)
        reason = "completed_count does not match the number enqueued";
    else if (fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] !=
             ft_tt_queue_token_at(queue, expected - 1u))
        reason = "completed_token is not this run's last token";

    if (out_reason)
        *out_reason = reason;
    return reason ? FT_TT_ERROR : FT_TT_OK;
}

bool ft_tt_queue_completion_line_is_addressable(const ft_tt_queue *queue)
{
    if (!queue)
        return false;
    /* The SPAN, not just the base: the length is a compile-time 16 and is pinned by the
     * assert below, so the only runtime variable is where the control block was placed. */
    return ((queue->control_address + QUEUE_LINE_FIRST_FIELD * 4u) % 16u) == 0u;
}

FT_TT_STATIC_ASSERT(QUEUE_LINE_FIRST_FIELD * 4u % 16u == 0u, line_span_is_16_byte_aligned);
FT_TT_STATIC_ASSERT(QUEUE_LINE_FIELD_COUNT * 4u == 16u, line_span_is_exactly_one_transaction);
FT_TT_STATIC_ASSERT(QUEUE_LINE_FIRST_FIELD <= FT_TT_QUEUE_FIELD_COMPLETED_COUNT &&
                        QUEUE_LINE_FIRST_FIELD + QUEUE_LINE_FIELD_COUNT >
                            FT_TT_QUEUE_FIELD_COMPLETED_COUNT,
                    line_span_holds_completed_count);
FT_TT_STATIC_ASSERT(QUEUE_LINE_FIRST_FIELD + QUEUE_LINE_FIELD_COUNT >
                        FT_TT_QUEUE_FIELD_COMPLETED_TOKEN,
                    line_span_holds_completed_token);
FT_TT_STATIC_ASSERT(QUEUE_LINE_FIRST_FIELD + QUEUE_LINE_FIELD_COUNT > FT_TT_QUEUE_FIELD_STATE,
                    line_span_holds_state);
/* AND THE ONE THAT IS THE POINT: `error` is NOT in the line. If a later renumbering brings
 * it in, this assert fires and whoever moved it must delete the line/wide split rather than
 * leave a check whose documentation is a lie. */
FT_TT_STATIC_ASSERT(FT_TT_QUEUE_FIELD_ERROR >= QUEUE_LINE_FIRST_FIELD + QUEUE_LINE_FIELD_COUNT,
                    line_span_deliberately_excludes_error);

FT_TT_STATIC_ASSERT(QUEUE_WIDE_FIRST_FIELD * 4u % 16u == 0u, wide_span_is_16_byte_aligned);
FT_TT_STATIC_ASSERT(QUEUE_WIDE_FIELD_COUNT * 4u % 16u == 0u, wide_span_is_a_16_byte_multiple);
FT_TT_STATIC_ASSERT(QUEUE_WIDE_FIRST_FIELD <= FT_TT_QUEUE_COMPLETION_FIRST_FIELD &&
                        QUEUE_WIDE_FIRST_FIELD + QUEUE_WIDE_FIELD_COUNT >=
                            FT_TT_QUEUE_COMPLETION_FIRST_FIELD +
                                FT_TT_QUEUE_COMPLETION_FIELD_COUNT,
                    wide_span_contains_the_completion_window);

int ft_tt_queue_read_field(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                           ft_tt_queue_control_field field, uint32_t *out_value)
{
    if (!noc || !noc->read || !queue || !out_value)
        return FT_TT_EINVAL;
    if ((unsigned)field >= FT_TT_QUEUE_FIELD_COUNT)
        return FT_TT_EINVAL;
    return noc->read(noc->user, queue->x, queue->y, control_field_address(queue, field), out_value,
                     sizeof(*out_value));
}

int ft_tt_queue_request_stop(const ft_tt_noc_access *noc, const ft_tt_queue *queue)
{
    if (!noc || !noc->write || !queue)
        return FT_TT_EINVAL;
    return write_control_word(noc, queue, FT_TT_QUEUE_FIELD_STOP_REQUEST, FT_TT_QUEUE_STOP_REQUESTED);
}

uint32_t ft_tt_queue_stop_request_address(const ft_tt_queue *queue)
{
    /* Deliberately the SAME derivation ft_tt_queue_request_stop() uses, so the address a
     * DYING process hands the driver and the address a LIVING one writes cannot drift
     * apart — two spellings of one address is how a cleanup silently stops firing. */
    return queue ? control_field_address(queue, FT_TT_QUEUE_FIELD_STOP_REQUEST) : 0u;
}

int ft_tt_queue_wait_completed(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                               uint32_t target, uint32_t timeout_ms, uint32_t *out_completed)
{
    uint64_t deadline;
    uint32_t completed = 0u;

    if (!noc || !noc->read || !queue)
        return FT_TT_EINVAL;
    /* Same rule as ft_tt_launch_wait_done(): there is no untimed path on a board where a
     * hang costs a physical power cycle. */
    if (timeout_ms == 0u)
        return FT_TT_EINVAL;

    deadline = monotonic_microseconds() + (uint64_t)timeout_ms * 1000u;
    for (;;) {
        int rc = ft_tt_queue_read_field(noc, queue, FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &completed);
        if (rc != FT_TT_OK)
            return rc;
        if (completed >= target) {
            if (out_completed)
                *out_completed = completed;
            return FT_TT_OK;
        }
        if (monotonic_microseconds() >= deadline) {
            if (out_completed)
                *out_completed = completed;
            return FT_TT_EIO;
        }
    }
}

int ft_tt_queue_check_completion(const ft_tt_queue *queue,
                                 const uint32_t fields[FT_TT_QUEUE_FIELD_COUNT],
                                 uint32_t expected, const char **out_reason)
{
    const char *reason = NULL;

    if (!queue || !fields || expected == 0u) {
        if (out_reason)
            *out_reason = "bad argument";
        return FT_TT_EINVAL;
    }

    if (fields[FT_TT_QUEUE_FIELD_STATE] == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
        reason = "the kernel never ran: state is still the host poison";
    else if (fields[FT_TT_QUEUE_FIELD_ERROR] != (uint32_t)FT_TT_QUEUE_KERR_NONE)
        reason = ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]);
    else if (fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] != expected)
        reason = "completed_count does not match the number enqueued";
    else if (fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] !=
             ft_tt_queue_token_at(queue, expected - 1u))
        reason = "completed_token is not this run's last token";

    if (out_reason)
        *out_reason = reason;
    return reason ? FT_TT_ERROR : FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Strings                                                                   */
/* ------------------------------------------------------------------------- */

const char *ft_tt_queue_state_string(uint32_t state)
{
    switch (state) {
    case FT_TT_QUEUE_STATE_HOST_POISON:
        return "HOST_POISON (the kernel never ran)";
    case FT_TT_QUEUE_STATE_BOOT:
        return "BOOT";
    case FT_TT_QUEUE_STATE_IDLE:
        return "IDLE";
    case FT_TT_QUEUE_STATE_EXEC:
        return "EXEC";
    case FT_TT_QUEUE_STATE_SEM_WAIT:
        return "SEM_WAIT";
    case FT_TT_QUEUE_STATE_EXIT:
        return "EXIT";
    case FT_TT_QUEUE_STATE_CB_WAIT_FRONT:
        return "CB_WAIT_FRONT";
    case FT_TT_QUEUE_STATE_CB_RESERVE_BACK:
        return "CB_RESERVE_BACK";
    default:
        return "unknown";
    }
}

const char *ft_tt_queue_kernel_error_string(uint32_t error)
{
    switch (error) {
    case FT_TT_QUEUE_KERR_NONE:
        return "none";
    case FT_TT_QUEUE_KERR_BAD_MAGIC:
        return "the control block did not carry the queue magic";
    case FT_TT_QUEUE_KERR_BAD_OPCODE:
        return "a descriptor named an opcode the kernel does not implement";
    case FT_TT_QUEUE_KERR_IDLE_TIMEOUT:
        return "the idle spin limit expired: the host stopped feeding the queue";
    case FT_TT_QUEUE_KERR_SEM_WAIT_TIMEOUT:
        return "a semaphore wait exhausted its spin bound";
    case FT_TT_QUEUE_KERR_BAD_CAPACITY:
        return "the capacity mask is not one less than a power of two";
    case FT_TT_QUEUE_KERR_CB_WAIT_TIMEOUT:
        return "a circular-buffer wait exhausted its spin bound";
    case FT_TT_QUEUE_KERR_BAD_PARAMETERS:
        return "a runtime argument was refused: an unbounded wait, or a queue whose tail "
               "and completed_count disagree";
    case FT_TT_QUEUE_KERR_COOP_ROLE_MISMATCH:
        return "a descriptor's cooperate word disagreed with this member's role: "
               "cooperation is all-or-nothing for a residency";
    case FT_TT_QUEUE_KERR_COOP_WRITE_POINTER:
        return "CB_IN0's write pointer is not where this member's own block count says: "
               "the group's circular buffers have drifted apart";
    case FT_TT_QUEUE_KERR_COOP_RENDEZVOUS:
        return "a multicast rendezvous wait was abandoned: the group is not in lockstep, "
               "or the host asked it to stop";
    case FT_TT_QUEUE_KERR_COOP_ISSUE:
        return "a multicast was refused at the point of issue: a zero length, zero "
               "destinations, a misalignment or an inverted rectangle";
    case FT_TT_QUEUE_KERR_COOP_BAD_PARTITION:
        return "the cooperative runtime args are illegal: a piece over four destinations, "
               "an inverted rectangle, or a destination total that is not the receiver count";
    default:
        return "unknown";
    }
}

const char *ft_tt_queue_opcode_string(uint32_t opcode)
{
    switch (opcode) {
    case FT_TT_QUEUE_OP_NOP:
        return "NOP";
    case FT_TT_QUEUE_OP_WRITE_WORD:
        return "WRITE_WORD";
    case FT_TT_QUEUE_OP_L1_COPY:
        return "L1_COPY";
    case FT_TT_QUEUE_OP_SEM_POST:
        return "SEM_POST";
    case FT_TT_QUEUE_OP_SEM_WAIT:
        return "SEM_WAIT";
    case FT_TT_QUEUE_OP_DELAY:
        return "DELAY";
    case FT_TT_QUEUE_OP_STOP:
        return "STOP";
    case FT_TT_QUEUE_OP_MATMUL:
        return "MATMUL";
    case FT_TT_QUEUE_OP_ELTWISE:
        return "ELTWISE";
    case FT_TT_QUEUE_OP_SOFTMAX:
        return "SOFTMAX";
    case FT_TT_QUEUE_OP_REDUCE:
        return "REDUCE";
    case FT_TT_QUEUE_OP_EMBEDDING:
        return "EMBEDDING";
    case FT_TT_QUEUE_OP_KV_APPEND:
        return "KV_APPEND";
    case FT_TT_QUEUE_OP_ROPE:
        return "ROPE";
    case FT_TT_QUEUE_OP_ATTENTION:
        return "ATTENTION";
    case FT_TT_QUEUE_OP_GDN:
        return "GDN";
    case FT_TT_QUEUE_OP_GDN_CHUNK:
        return "GDN_CHUNK";
    case FT_TT_QUEUE_OP_GDN_GATE:
        return "GDN_GATE";
    default:
        return "unknown";
    }
}
