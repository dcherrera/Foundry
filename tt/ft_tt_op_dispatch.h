/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_dispatch.h
 * @brief How a Phase 4 op gets onto one Tensix and back: describe its circular buffers,
 *   plant its runtime args, kick it, wait.
 *
 * Slow dispatch, one host round trip per op. Phase 5's persistent kernel is what removes
 * that, and these are the numbers it has to beat.
 *
 * **Loading is not here any more.** Kernel text is linked into a slab of the
 * KERNEL_CONFIG region rather than the tail of its own processor's firmware hole, because
 * the tails are 1296-1952 bytes and only `vector_add` fits (ft_tt_op_l1_layout.h has the
 * measured table). That used to need a second loader in this file carrying the slab rule
 * while ft_tt_launch.c kept the in-hole rule. They are ONE rule — "inside its own
 * firmware hole, OR entirely inside its kernel-config slab, never straddling" — and since
 * the consolidation round they live in one function, `validate_placement()` in
 * ft_tt_launch.c, chosen by where the ELF put its text rather than by a caller's flag.
 *
 * @status Loads and runs all five images of the relocated matmul and rms_norm programs on
 *   the card.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_DISPATCH_H
#define FT_TT_OP_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_kernel_blob.h"
#include "ft_tt_launch.h"

/**
 * ft_tt_launch_load_image() under this file's older name, kept because Phase 5's resident
 * test calls it. It is exactly
 * `ft_tt_launch_load_image(..., FT_TT_IMAGE_KERNEL, ...)` — the merged placement rule
 * accepts a relocated image and an in-hole one alike, and refuses everything else, so the
 * relocation no longer needs an entry point of its own.
 *
 * @param noc         device access
 * @param x, y        TRANSLATED coordinates of the target Tensix, from ft_tt_coord —
 *                    never a guess (spec §7.3)
 * @param processor   FT_TT_PROC_* index this image belongs to
 * @param elf_data    the ELF blob
 * @param elf_size    its length
 * @param out_place   may be NULL
 * @return            FT_TT_OK, FT_TT_EINVAL, or whatever the accessor returned
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_op_image_load(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, unsigned processor,
                        const void *elf_data, size_t elf_size, ft_tt_image_placement *out_place);

/**
 * ft_tt_launch_load_image() for every image in `program`, plus the launch config those
 * placements imply: per-processor text address and size, runtime-arg addresses from
 * ft_tt_op_l1_layout.h, and `enables` from the program.
 *
 * The caller still fills in `local_cb_config_address`, `local_cb_mask` and
 * `brisc_noc_id`, because those are the op's business rather than the loader's.
 *
 * @param placements  indexed by FT_TT_PROC_*, may be NULL
 * @param out_config  zeroed and filled on success
 * @return            FT_TT_OK, or the first failing image's code (logged)
 */
int ft_tt_op_program_load(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                          const ft_tt_kernel_program *program,
                          ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                          ft_tt_launch_config *out_config);

/* ================================================================
 * Operands and block depth — shared by every op driver
 * ================================================================ */

/** Where one operand lives and how big its tiles are. The coordinate is TRANSLATED and
 *  comes from ft_tt_coord — a guessed endpoint hangs the chip (spec §7.3).
 *
 *  This lives here rather than in one op's header because every op driver needs it and
 *  none of them owns it. It was ft_tt_op_matmul.h's until the vector op family needed the
 *  same three fields. */
typedef struct ft_tt_op_operand {
    uint16_t x;
    uint16_t y;
    uint32_t address;    /**< base of tile 0 */
    uint32_t tile_bytes; /**< 2048 Float16_b, 1088 Bfp8_b, 576 Bfp4_b, 4096 Float32 */
} ft_tt_op_operand;

/**
 * The block-prefetch depth a reader may actually use over a run of `run` tiles.
 *
 * A reader reserves a contiguous block of `depth` pages and fills it with ONE NoC read,
 * which is what turned a per-tile barrier into a per-block one and was worth 5.9x.
 * A CB block is only contiguous when the FIFO write
 * pointer never straddles the wrap — which holds exactly when every push is the same size
 * and the CB holds a whole multiple of it.
 *
 * So a requested depth is SHRUNK TO THE LARGEST DIVISOR of `run` that is no larger, never
 * clamped: a ragged final block would move the write pointer off the block grid and
 * corrupt every block after it.
 *
 * `run` is the dimension a block may not span — Kt for the matmul (the K loop restarts per
 * output tile), Wt for softmax (the compute thread consumes a whole row at once), and the
 * whole tile count for a flat elementwise stream.
 *
 * @param run        tiles in the run a block must divide; 0 yields 1
 * @param requested  what the caller asked for; 0 means 1
 * @return           the effective depth, always >= 1 and always dividing `run`
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_block_depth(uint32_t run, uint32_t requested);

/* ================================================================
 * Circular buffers
 * ================================================================ */

/** Most CBs one op describes. Was 12 — three inputs, three intermediates, an output and
 *  slack — which flash-decode attention exceeds: it carries a running max (ping-pong), a
 *  running row sum, an output accumulator and this block's contribution to each, which is
 *  SIXTEEN buffers (ft_tt_op_attention.h). Raised rather than worked around, because the
 *  hardware bound is Blackhole's 64 circular buffers (`FT_TT_NUM_CIRCULAR_BUFFERS`) and 12
 *  was a guess at what an op would want. Every user of this constant sizes an array with
 *  it, so raising it is additive. */
#define FT_TT_OP_MAX_CB 20u

/** One CB to place: an index, a page size and a page count. Its L1 address is assigned by
 *  ft_tt_op_cb_place() out of the arena, so an op never hardcodes one. */
typedef struct ft_tt_op_cb_request {
    unsigned index;
    uint32_t page_bytes;
    uint32_t page_count;
} ft_tt_op_cb_request;

/**
 * Assign arena addresses to `count` CBs, bottom-up from FT_TT_OP_CB_ARENA_BASE, and
 * encode the descriptor array the firmware reads.
 *
 * Page sizes differ per CB and that is the normal case here, not an edge one: a Bfp8_b
 * weight page is 1088 bytes beside a 2048-byte Float16_b activation page. Every address
 * is left 64-byte aligned, which is the NoC's DRAM alignment and more than L1 needs.
 *
 * @param requests      what to place
 * @param count         how many, at most FT_TT_OP_MAX_CB
 * @param out_entries   receives the placed entries, same order, with `address` filled in
 * @param buffer        receives the encoded descriptor array
 * @param buffer_bytes  its size; ft_tt_launch_encode_cb_config() states the minimum
 * @param out_mask      receives the launch message's local_cb_mask
 * @return              FT_TT_OK; FT_TT_EINVAL on a bad argument; FT_TT_ENOMEM if the
 *                      requests do not fit the arena
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_op_cb_place(const ft_tt_op_cb_request *requests, unsigned count,
                      ft_tt_launch_cb_entry *out_entries, void *buffer, size_t buffer_bytes,
                      uint64_t *out_mask);

/** The placed entry for CB `index`, or NULL. */
const ft_tt_launch_cb_entry *ft_tt_op_cb_find(const ft_tt_launch_cb_entry *entries,
                                              unsigned count, unsigned index);

/* ================================================================
 * Runtime args and the kick
 * ================================================================ */

/** Runtime args for one processor, as the op wants them planted. */
typedef struct ft_tt_op_args {
    unsigned processor;
    const uint32_t *values;
    unsigned count;
} ft_tt_op_args;

/**
 * Everything before the kick: the CB descriptor array, every enabled processor's
 * runtime-arg block, and the launch message. No GO, no wait.
 *
 * Split out for Phase 5. A RESIDENT kernel is staged exactly like a one-shot op and then
 * kicked and left alone — the host does not wait for it, because waiting for it is the
 * dispatch cost this phase exists to remove. `ft_tt_op_launch()` is this call plus
 * `ft_tt_launch_send_go()` plus `ft_tt_launch_wait_done()`.
 *
 * @return FT_TT_OK, or the first failing step's code
 */
int ft_tt_op_stage(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                   const ft_tt_launch_config *config, const void *cb_buffer, size_t cb_bytes,
                   const ft_tt_op_args *args, unsigned arg_count);

/**
 * Write the CB descriptor array and every processor's runtime-arg block, write the launch
 * message, send GO, and wait for completion.
 *
 * EVERY enabled processor gets a block. One that reads no args still gets a written,
 * zero-filled one, because a stale block or the 0xFFFF sentinel resolves to an address a
 * kernel may dereference — which stalled a core once already
 * (docs/tt_launch_notes.md §3, "the trap that cost a run").
 *
 * @param config        from ft_tt_op_program_load(), with the CB fields filled in
 * @param cb_buffer     the encoded descriptor array from ft_tt_op_cb_place()
 * @param cb_bytes      its size
 * @param args          per-processor arg blocks; processors not named get a zero block
 * @param arg_count     how many entries `args` has
 * @param timeout_ms    bounds the wait. There is no untimed path here (spec §7.3)
 * @param out_state     may be NULL; filled on timeout
 * @return              FT_TT_OK; FT_TT_EIO on timeout, with the core dumped
 */
int ft_tt_op_launch(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                    const ft_tt_launch_config *config, const void *cb_buffer, size_t cb_bytes,
                    const ft_tt_op_args *args, unsigned arg_count, uint32_t timeout_ms,
                    ft_tt_launch_state *out_state);

#endif /* FT_TT_OP_DISPATCH_H */
