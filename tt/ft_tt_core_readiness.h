/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_readiness.h
 * @brief ONE question, asked of the card before anything boots firmware onto a Tensix:
 *        **may I boot on this core right now?**
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS — A SPINNING COMPUTE THREAD OUTLIVES ITS PROCESS
 * ---------------------------------------------------------------------------
 * A refused descriptor makes a resident reader abort and leaves the compute thread to spin
 * out its whole bound — that is the family's design, not a fault. `pn_tt_residency_stop()`
 * waits it out and `pn_tt_residency_retired()` reports whether it did, so the process that
 * CAUSED the spin will not re-arm onto it.
 *
 * **But the thread outlives the process.** A `serve` restart or a model reload is a NEW
 * process with an empty residency table, and its ordinary first arm boots firmware onto a
 * core that is still running. MEASURED ON SILICON 2026-08-21/22: that is a WEDGE — core 90
 * came back `instrn_buf_status = 0x66`, twice, and only a whole-card `ft_tt_core recover
 * --rung 3` cleared it. Nothing in a fresh process's own memory can see this. Only the card
 * can say, so the card is asked.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT ADDS TO ft_tt_core_status.h, AND WHY THAT VERDICT ALONE IS NOT ENOUGH
 * ---------------------------------------------------------------------------
 * The verdict is the backbone: HELD and FIRMWARE_IDLE are the two states firmware may be
 * booted onto, and every other one is a refusal. But the verdict has ONE blind spot for
 * exactly this bug, and it is in the heartbeat:
 *
 *   `ft_tt_core_status` samples the CONTROL block's heartbeat, which has ONE writer — the
 *   reader on BRISC. In this failure the reader is the processor that ALREADY EXITED. The
 *   control heartbeat is therefore frozen while TRISC0 spins, and a check resting on it
 *   would call the core quiet.
 *
 * So this module also samples the FIVE PER-PROCESSOR status slots — the same words
 * `ft_tt_queue_all_processors_exited()` polls, at the same fixed L1 addresses, which a
 * process that did not create the queue can still read. Every wait in
 * `resident_queue_protocol.h` bumps its own slot's heartbeat, so a processor that is
 * executing anything at all moves a word between two back-to-back reads.
 *
 * ---------------------------------------------------------------------------
 * A TICKING HEARTBEAT IS EVIDENCE. A STALE STATE WORD IS NOT.
 * ---------------------------------------------------------------------------
 * The slots are read WITHOUT the queue handle that wrote them, so their `state` words may
 * be anything a previous program left — and one family leaves non-EXIT words behind on a
 * perfectly clean exit: the memory family enables DM0|DM1 only, so its three TRISCs never
 * run the program and never write EXIT (`pn_tt_residency.c` documents the false alarm this
 * already caused once). Refusing on a stale word would refuse healthy cores, and a refusal
 * here drops the whole model to the ~26 s/position CPU forward.
 *
 * **So only a heartbeat that MOVES is ever treated as busy.** It is a positive statement
 * that a processor is executing right now; a state word is a statement about the past. That
 * is the same reasoning `ft_tt_core_status.c` already applies to the control heartbeat, one
 * level down.
 *
 * ---------------------------------------------------------------------------
 * NOT READY SPLITS IN TWO, AND THE SPLIT IS THE WHOLE POINT
 * ---------------------------------------------------------------------------
 * A core spinning out a bound WILL finish; a core at `instrn_buf_status = 0x66` never will.
 * Waiting is right for the first and is dead time before an inevitable refusal for the
 * second, so `waiting_can_help` separates them and the caller does not have to guess.
 *
 * **This module never waits, never writes and never recovers.** It issues reads. What to do
 * about a busy core is a policy the caller owns.
 *
 * @status New 2026-08-24. The classifier is covered off-hardware by
 *         tt/tests/test_core_readiness.c; the device read is the same two calls
 *         `ft_tt_core status` has run on the p150a since 2026-08-12.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_CORE_READINESS_H
#define FT_TT_CORE_READINESS_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_core_status.h"
#include "ft_tt_coord.h"
#include "ft_tt_noc.h"

/** Why a core is not ready. Ordered so a caller can print the name and be done. */
typedef enum ft_tt_core_readiness_state {
    /** Firmware may be booted onto this core. */
    FT_TT_CORE_READY = 0,
    /** The core did not answer a mailbox read. Past what a host write can fix. */
    FT_TT_CORE_NOT_READY_NO_ANSWER,
    /** The Tensix instruction FIFO is not ready — the 0x66 wedge. BRISC firmware's
     *  `device_setup()` pushes into it before it releases the subordinates, so a boot here
     *  stalls BRISC inside one store. **Only a card reset clears this.** */
    FT_TT_CORE_NOT_READY_INSTRUCTION_FIFO,
    /** A processor's heartbeat MOVED between two reads: something is executing. This is the
     *  leftover spinning compute thread, and it clears itself. */
    FT_TT_CORE_NOT_READY_STILL_EXECUTING,
    /** The go signal says GO and nothing is ticking: a kernel was launched and no processor
     *  is making progress. Waiting has nothing to wait for. */
    FT_TT_CORE_NOT_READY_KERNEL_STALLED,
    /** BRISC is out of reset and never wrote RUN_MSG_DONE, or is blocked in `tensix_sync`
     *  on subordinates held in reset. Both are stuck signatures a read can name. */
    FT_TT_CORE_NOT_READY_BOOT_STUCK,
    /** Some RISC-Vs running, some held, and nothing above matches. Report, do not guess. */
    FT_TT_CORE_NOT_READY_MIXED
} ft_tt_core_readiness_state;

/** Two samples of one processor's queue status slot. */
typedef struct ft_tt_core_processor_liveness {
    bool slot_read;      /**< both samples came back */
    bool ticking;        /**< the heartbeat MOVED between them — this processor is executing */
    uint32_t state;      /**< ft_tt_queue_state from the first sample; context only, never a verdict */
    uint32_t heartbeat;  /**< the first sample's heartbeat; for logs */
} ft_tt_core_processor_liveness;

/** Everything the readiness question read, and its answer. */
typedef struct ft_tt_core_readiness {
    ft_tt_core_status status;   /**< the whole underlying read, verdict included */
    ft_tt_core_readiness_state state;
    bool ready;                 /**< `state == FT_TT_CORE_READY` */
    /** Is this a wait or a refusal? True only when something is demonstrably executing and
     *  will therefore stop. A 0x66 core, a stalled kernel and a stuck boot are all false. */
    bool waiting_can_help;
    /** Bit FT_TT_PROC_* per processor whose heartbeat moved. 0 when nothing is ticking. */
    uint32_t ticking_mask;
    /** Per-processor detail, indexed by FT_TT_PROC_*. Only filled when a queue control
     *  block with our magic was found; `slot_read` says which entries mean anything. */
    ft_tt_core_processor_liveness processor[FT_TT_PROC_COUNT];
} ft_tt_core_readiness;

/** "ready", "the Tensix instruction FIFO is not ready (0x66 — needs a card reset)", ...
 *  Never NULL, and phrased to be printed straight into a refusal. */
const char *ft_tt_core_readiness_string(ft_tt_core_readiness_state state);

/**
 * Decide readiness from an already-taken reading. **PURE — no device access.**
 *
 * Split out so every branch can be driven from hand-built readings on a machine with no
 * card, which is the only way any of them gets a red arm. The device read below is a thin
 * wrapper that fills the inputs and calls this.
 *
 * @param status         a filled `ft_tt_core_status`, verdict included
 * @param processor      FT_TT_PROC_COUNT liveness samples; may be all-zero when no queue
 *                       control block was found, which is the ordinary cold-core case
 * @param out            zeroed except for `status`, which the caller owns and this copies
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void ft_tt_core_readiness_classify(const ft_tt_core_status *status,
                                   const ft_tt_core_processor_liveness *processor,
                                   ft_tt_core_readiness *out);

/**
 * Read one core and answer the question. **Issues only NoC reads.**
 *
 * Cost is one `ft_tt_core_status_read()` (a mailbox snapshot plus two control-block reads)
 * plus two reads of the five queue status slots — around 400 us of PIO, once per arm. That
 * is charged at model-load time against a failure whose cheapest repair is a whole-card
 * reset, so it is not a cost worth optimising.
 *
 * @param noc    device access
 * @param grid   live grid — the coordinate is resolved here, NEVER passed in: a guessed
 *               endpoint does not fail a read, it hangs the chip (spec §7.3)
 * @param index  Tensix index, as `ft_tt_coord_tensix_by_index` numbers them
 * @param out    zeroed and filled
 * @return       FT_TT_OK once the coordinate resolved, or the coordinate error. A core that
 *               does not answer is a RESULT (`FT_TT_CORE_NOT_READY_NO_ANSWER`), not an error
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_core_readiness_read(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                              uint32_t index, ft_tt_core_readiness *out);

#endif /* FT_TT_CORE_READINESS_H */
