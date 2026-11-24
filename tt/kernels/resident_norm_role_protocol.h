/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_norm_role_protocol.h
 * @brief PERSISTENT NORM ROLE, the wire half: the role skeleton (schedule table, role words,
 *        bridge descriptor) and the norm role's exchange layout. Pure C over plain words, so
 *        the host (ft_tt_op_norm_role.c, Pantheon pn_tt_norm_role.c, the host model in
 *        tests/test_norm_role.c) and the three kernels share one definition.
 *
 * WHAT THE ROLE IS (design spec §4, §11). One program,
 * `norm_role`, loaded ONCE per decode session on M cores (8 on Qwen3.8). It owns every
 * residual add and RMS norm of a token: step k of the token's schedule is
 *
 *     LOAD:  h  = input          (layer 0: the embedding row)
 *     ADD:   h  = h + input      (input = the out-projection / FFN-down output)
 *     then   normed = h * gamma_k * rsqrt(mean(h^2) + eps)
 *
 * with h RESIDENT in L1 across steps (no DRAM round trip of the residual stream). Member m
 * owns tile columns [m*S, (m+1)*S) of the 160-tile row (S = T / M): it adds and squares its
 * slice, sends its x^2 slice to every member, and each member then reduces ALL T tiles in
 * today's order (the P3 rule: every member's rstd is the same bits) and scales its own slice.
 *
 * BRIDGE MODE (P0). The role is driven by one descriptor per step per member on its own
 * resident queue: the descriptor is the step's clock and carries TODAY'S precondition (the
 * S2 sidecar, resident_precondition.h) and gets TODAY'S completion (completed count + token,
 * the S1 group pull on member 0), so the per-op producers and consumers do not change. The
 * descriptor carries the step index and the trip counts the compute threads need (MATH has
 * no CB records, so it cannot read the table) — identical every token, so the puller's replay of a
 * recorded token is unchanged); everything else is in the SCHEDULE TABLE, written into each
 * member's L1 once at arm. The role counts its own sequence n (steps run since arm, the same
 * on every member). P4 replaces the descriptor with credit rings.
 *
 * THE SCHEDULE TABLE (CB kNrCbTable, one page): a 16-word header, then one 8-word entry per
 * step of a token. The reader validates the header checksum on the first descriptor; a table
 * the host did not write (or wrote for another member) is refused, never executed.
 *
 * THE EXCHANGE. Member m's writer sends its S x^2 pages WHOLE into pages m*S .. m*S + S-1 of
 * every member's CB kNrCbSqAll, one write per rectangle per contiguous run (ftnr_exchange_runs,
 * spec §11.11 levers 1-2), one ack barrier, then sets flag word m of every member's kNrCbFlags
 * page to the role sequence n (1, 2, ...). Faces 2-3 of each page are the zeros the half-tile
 * map packs from a cleared DST, which is what today's cb_mid holds there; SqAll is also zeroed
 * at arm. The table's exchange mode (§11.13) may send faces 0-1 only (one write a page; faces
 * 2-3 keep the boot zeros) and stagger the target order by member. The reader pushes slice j's
 * S pages once flag j == n, in j order, so the reduce starts on slice 0 while later slices are
 * still in flight; it also times each flag's first sighting (kNrFlagArrival + k).
 *
 * WHY THE EXCHANGE NEEDS NO CREDIT BACK. Member m writes step n+1's slice only after step
 * n+1's precondition holds. That precondition is an out-projection or FFN-down (or, at a token
 * start, the embedding) which consumed step n's normed output, which only completed once
 * member 0's group pull saw EVERY member's writer finish step n — and a writer finishes a step
 * only after its compute popped step n's SqAll pages. tests/test_norm_role.c models it.
 *
 * ROLE WORDS. The exit word is the queue's stop word (kFieldStopRequest): the host's ordinary
 * retire stops a role like any residency. The watchdog word (61) is bumped by the reader once
 * per step and per 256 idle polls; 62 counts steps published; 63 accumulates the busy clock
 * (precondition met -> completion published, cycles >> 4) on the writer.
 *
 * WIRE CONTRACT FORM: one-line anonymous enums, as resident_queue_protocol.h.
 *
 * @status P0 (P0.1/P0.2). Host test tests/test_norm_role.c.
 *         Card gate pending.
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_NORM_ROLE_PROTOCOL_H
#define RESIDENT_NORM_ROLE_PROTOCOL_H

#include <stdbool.h>
#include <stdint.h>

#ifndef FTNR_INLINE
#define FTNR_INLINE static inline
#endif

/* ---- Parameter block (16 words at descriptor a0) ---- */
enum { kNrParamMagic = 0 };
enum { kNrParamStep = 1 };          /* step index within the token, 0 .. steps-1 */
enum { kNrParamKind = 2 };          /* the entry's kind, repeated: MATH has no CB records */
enum { kNrParamTileCols = 3 };      /* T, repeated (the compute threads' trip counts) */
enum { kNrParamSliceTiles = 4 };    /* S, repeated */
enum { kNrParamWords = 16 };
enum { kNrMagic = 0x4E524C31 };     /* "NRL1" */

/* ---- Schedule table header (words from the table base) ---- */
enum { kNrHeadMagic = 0 };
enum { kNrHeadSteps = 1 };
enum { kNrHeadTileCols = 2 };
enum { kNrHeadMembers = 3 };        /* members | member << 8 | exchange mode << 16 */
enum { kNrHeadSliceTiles = 4 };
enum { kNrHeadTargets = 5 };        /* exchange targets, 1 .. kNrMaxTargets */
enum { kNrHeadTarget0 = 6 };        /* 2 words a target, kNrMaxTargets of them */
enum { kNrHeadSelfXy = 14 };        /* this member's own NoC x | y << 8 */
enum { kNrHeadChecksum = 15 };
enum { kNrHeadWords = 16 };
enum { kNrTableMagic = 0x4E525431 }; /* "NRT1" */
enum { kNrMaxTargets = 4 };
enum { kNrMaxSteps = 160 };
enum { kNrMaxMembers = 8 };

/* ---- Target words: rect = x0 | y0 << 8 | x1 << 16 | y1 << 24; info = receivers (the
 *      sender excluded) | includes_sender << 8. A one-core rectangle is a unicast. ---- */
enum { kNrTargetIncludesSender = 0x100 };

/* ---- Schedule entry (8 words) ---- */
enum { kNrEntryKind = 0 };
enum { kNrEntryInputXy = 1 };       /* x | y << 8 */
enum { kNrEntryInputBase = 2 };
enum { kNrEntryConstXy = 3 };
enum { kNrEntryConstBase = 4 };     /* scaler tile, eps tile, then T gamma tiles */
enum { kNrEntryOutXy = 5 };
enum { kNrEntryOutBase = 6 };
enum { kNrEntryWords = 8 };
enum { kNrKindLoad = 1 };
enum { kNrKindAdd = 2 };

/* ---- Circular buffers (compile-time indices; norm_role_float16b.cbtable) ---- */
enum { kNrCbIn = 0 };               /* S pages: the input slice */
enum { kNrCbScaler = 1 };           /* 2 pages: this step's and the prefetched next */
enum { kNrCbCol = 2 };              /* 1 page: rsqrt(mean + eps), column 0 */
enum { kNrCbH = 3 };                /* 2S pages: resident h (front S) and the new h */
enum { kNrCbSq = 4 };               /* S pages: this member's x^2 slice */
enum { kNrCbEps = 5 };              /* 2 pages */
enum { kNrCbGamma = 6 };            /* 2S pages */
enum { kNrCbSqAll = 7 };            /* T pages: every member's x^2 slice, in tile order */
enum { kNrCbStage = 8 };            /* K pages: gamma-step scratch */
enum { kNrCbTable = 9 };            /* 1 page: the schedule table */
enum { kNrCbFlags = 10 };           /* 1 page: exchange flags, one word a member */
enum { kNrCbOut = 16 };             /* S pages */

/* ---- Exchange ---- */
enum { kNrExchangeBytes = 1024 };   /* faces 0-1 of a bf16 32x32 tile (exchange and output) */
enum { kNrFlagStage = 8 };          /* first of the writer's stage words (8 + member) */
enum { kNrFlagArrival = 16 };       /* reader's per-sender arrival clocks (16 + member) */
enum { kNrFlagAllSeen = 24 };       /* reader: every flag seen (clock, summed over steps) */
enum { kNrFlagSqAllWait = 25 };     /* reader: time in SqAll reserve (summed) */
enum { kNrFlagOutFirst = 28 };      /* writer: first output page in (clock, summed) */
enum { kNrFlagOutDone = 29 };       /* writer: output acked, before the publish (clock) */
enum { kNrFlagsBytes = 128 };

/* ---- Exchange modes (header word kNrHeadMembers bits 16-23; spec §11.13). 0 = whole pages,
 *      targets in table order (§11.12). ---- */
enum { kNrExchangeFaces = 1 };      /* faces 0-1 of each page to every rectangle */
enum { kNrExchangeStagger = 2 };    /* member m starts at target m mod targets */
enum { kNrExchangeModes = 3 };      /* every defined mode bit */

/* ---- Role words (control block word indices) ---- */
enum { kNrWordWatchdog = 61 };
enum { kNrWordSteps = 62 };
enum { kNrWordBusy = 63 };
/* Phase clocks (cycles >> 4 from the precondition stamp, summed over steps): input and
 * constants read (reader), own flag sent (writer), every flag in (reader). */
enum { kNrWordInputRead = 13 };
enum { kNrWordFlagSent = 14 };
enum { kNrWordFlagsIn = 15 };

/* ---- Refusals the reader records (control word kFieldError carries kErrBadParameters;
 *      word 12 the reason, for the host's report) ---- */
enum { kNrWordReason = 12 };
enum { kNrOk = 0 };
enum { kNrBadMagic = 1 };
enum { kNrBadTable = 2 };
enum { kNrBadStep = 3 };
enum { kNrBadKind = 4 };
enum { kNrBadShape = 5 };

/** x of an `x | y << 8` word. */
FTNR_INLINE uint32_t ftnr_x(uint32_t xy)
{
    return xy & 0xFFu;
}

/** y of an `x | y << 8` word. */
FTNR_INLINE uint32_t ftnr_y(uint32_t xy)
{
    return (xy >> 8) & 0xFFu;
}

/** The flags-page word member `member` stages its flag in before multicasting it to word
 *  `member` of every receiver. A NoC write lands only when source and destination share
 *  their offset within 16 bytes: word 8 + member is 32 bytes past word member, so the two
 *  always agree (one shared stage word reached only members 0 and 4). */
FTNR_INLINE uint32_t ftnr_flag_stage_word(uint32_t member)
{
#if defined(FTNR_HOST_RED_SHARED_STAGE)
    (void)member; /* RED (host tests only): the one shared stage word the card refused */
    return (uint32_t)kNrFlagStage;
#else
    return (uint32_t)kNrFlagStage + member;
#endif
}

/** The checksum the header carries: every header word but the checksum, then every entry
 *  word, folded. Never 0 for a written table (the magic is folded in). */
FTNR_INLINE uint32_t ftnr_checksum(const volatile uint32_t *table, uint32_t steps)
{
    uint32_t sum = 0x9E3779B9u;
    uint32_t words = (uint32_t)kNrHeadWords + steps * (uint32_t)kNrEntryWords;

    for (uint32_t w = 0; w < words; w++) {
        if (w == (uint32_t)kNrHeadChecksum)
            continue;
        sum = (sum ^ table[w]) * 0x01000193u + w;
    }
    return sum == 0u ? 1u : sum;
}

/** Is `table` a schedule this member can run? Checks the magic, the shape (T = M * S, every
 *  count in range), the member index and the checksum. */
FTNR_INLINE uint32_t ftnr_table_check(const volatile uint32_t *table, uint32_t member)
{
    const uint32_t steps = table[kNrHeadSteps];
    const uint32_t tile_cols = table[kNrHeadTileCols];
    const uint32_t members = table[kNrHeadMembers] & 0xFFu;
    const uint32_t slice = table[kNrHeadSliceTiles];
    const uint32_t targets = table[kNrHeadTargets];

    if (table[kNrHeadMagic] != (uint32_t)kNrTableMagic)
        return kNrBadTable;
    if (steps == 0u || steps > (uint32_t)kNrMaxSteps || members == 0u ||
        members > (uint32_t)kNrMaxMembers || slice == 0u || tile_cols != members * slice ||
        targets == 0u || targets > (uint32_t)kNrMaxTargets)
        return kNrBadShape;
    if (((table[kNrHeadMembers] >> 8) & 0xFFu) != member || member >= members)
        return kNrBadTable;
    if ((table[kNrHeadMembers] >> 16) & ~(uint32_t)kNrExchangeModes)
        return kNrBadShape;
    if (table[kNrHeadChecksum] != ftnr_checksum(table, steps))
        return kNrBadTable;
    return kNrOk;
}

/** Entry `step` of a checked table. */
FTNR_INLINE const volatile uint32_t *ftnr_entry(const volatile uint32_t *table, uint32_t step)
{
    return table + kNrHeadWords + step * (uint32_t)kNrEntryWords;
}

/** The step after `step`: the schedule wraps to step 0 at the next token. */
FTNR_INLINE uint32_t ftnr_next_step(uint32_t step, uint32_t steps)
{
    return step + 1u >= steps ? 0u : step + 1u;
}

/** The step a role that has run `sequence - 1` steps must see next. The host arms the role
 *  at a token boundary, so sequence 1 is step 0. */
FTNR_INLINE uint32_t ftnr_expected_step(uint32_t sequence, uint32_t steps)
{
    return (sequence - 1u) % steps;
}

/**
 * Is this descriptor the step the role expects? A role out of step with the host would add
 * the wrong residual — refused (kNrBad*), never run.
 */
FTNR_INLINE uint32_t ftnr_descriptor_check(const volatile uint32_t *parameters,
                                           const volatile uint32_t *table, uint32_t sequence)
{
    const uint32_t steps = table[kNrHeadSteps];
    const uint32_t step = parameters[kNrParamStep];
    const uint32_t kind = ftnr_entry(table, step < steps ? step : 0u)[kNrEntryKind];

    if (parameters[kNrParamMagic] != (uint32_t)kNrMagic)
        return kNrBadMagic;
    if (step >= steps || step != ftnr_expected_step(sequence, steps))
        return kNrBadStep;
    if ((kind != (uint32_t)kNrKindLoad && kind != (uint32_t)kNrKindAdd) ||
        parameters[kNrParamKind] != kind)
        return kNrBadKind;
    if (parameters[kNrParamTileCols] != table[kNrHeadTileCols] ||
        parameters[kNrParamSliceTiles] != table[kNrHeadSliceTiles])
        return kNrBadShape;
    /* A token's first step has no residual to add to; every other step needs one. */
    if ((step == 0u) != (kind == (uint32_t)kNrKindLoad))
        return kNrBadKind;
    return kNrOk;
}

/** Byte offset, from SqAll's base, of the page member `member`'s tile `tile` lands in. */
FTNR_INLINE uint32_t ftnr_sqall_offset(uint32_t member, uint32_t tile, uint32_t slice,
                                       uint32_t tile_bytes)
{
    return (member * slice + tile) * tile_bytes;
}

/** Ring page `index` pages past ring page `first` of a CB of `ring_pages` pages. */
FTNR_INLINE uint32_t ftnr_ring_page(uint32_t first, uint32_t index, uint32_t ring_pages)
{
    return (first + index) % ring_pages;
}

enum { kNrMaxExchangeRuns = 2 };    /* a slice read from a ring splits at most once */

/**
 * The exchange's writes (spec §11.11 levers 1-2): member m's S x^2 pages, read from the Sq ring
 * at page `read_page` of `ring_pages`, as at most two CONTIGUOUS runs of WHOLE pages (one where
 * the ring wraps). Run r is `pages[r]` pages from ring page `first[r]`, landing at SqAll page
 * m*S + `tile[r]`. The writer sends each run as ONE write per rectangle (plus one to itself),
 * then one ack barrier for the whole slice. Whole pages carry Sq's faces 2-3: the zeros the
 * half-tile map packs from a cleared DST, the same bits today's cb_mid holds (and rows 16-31
 * never reach a decode row-0 sum). Returns the run count (0 for a zero slice or ring).
 */
FTNR_INLINE uint32_t ftnr_exchange_runs(uint32_t read_page, uint32_t ring_pages, uint32_t slice,
                                        uint32_t first[kNrMaxExchangeRuns],
                                        uint32_t tile[kNrMaxExchangeRuns],
                                        uint32_t pages[kNrMaxExchangeRuns])
{
    uint32_t head;

    if (slice == 0u || ring_pages == 0u || slice > ring_pages || read_page >= ring_pages)
        return 0u;
    head = ring_pages - read_page < slice ? ring_pages - read_page : slice;
    first[0] = read_page;
    tile[0] = 0u;
    pages[0] = head;
    if (head == slice)
        return 1u;
    first[1] = 0u;
    tile[1] = head;
    pages[1] = slice - head;
    return 2u;
}

/** The exchange mode bits of a table (kNrExchange*). */
FTNR_INLINE uint32_t ftnr_exchange_mode(const volatile uint32_t *table)
{
    return (table[kNrHeadMembers] >> 16) & (uint32_t)kNrExchangeModes;
}

/**
 * The target member `member` sends to `k`-th (spec §11.13 lever 2). Plain: table order, so all
 * M senders hit target 0 first and its receivers take M slices while the others idle. Stagger:
 * member m starts at target m mod `targets`, so the first bursts split across the rectangles.
 * A permutation of 0 .. targets-1 for every member.
 */
FTNR_INLINE uint32_t ftnr_exchange_target(uint32_t k, uint32_t member, uint32_t targets,
                                          uint32_t mode)
{
    if (targets == 0u)
        return 0u;
#if defined(FTNR_HOST_RED_NO_STAGGER)
    (void)member; /* RED (host tests only): the stagger bit ignored */
    mode &= ~(uint32_t)kNrExchangeStagger;
#endif
    return (mode & (uint32_t)kNrExchangeStagger) ? (k + member) % targets : k % targets;
}

/** Writes of a run of `pages` pages to one rectangle (spec §11.13 lever 1): one whole-page
 *  write, or under kNrExchangeFaces one faces-0-1 write a page. */
FTNR_INLINE uint32_t ftnr_exchange_writes(uint32_t pages, uint32_t mode)
{
    return (mode & (uint32_t)kNrExchangeFaces) ? pages : (pages != 0u ? 1u : 0u);
}

/** Bytes of each of those writes. Write w starts w pages into the run, at source and landing
 *  alike. Faces 0-1 is half the bytes a receiver takes: SqAll faces 2-3 stay the zeros the
 *  reader wrote at boot, which are today's cb_mid bits. */
FTNR_INLINE uint32_t ftnr_exchange_write_bytes(uint32_t pages, uint32_t tile_bytes, uint32_t mode)
{
#if defined(FTNR_HOST_RED_FACE0_ONLY)
    /* RED (host tests only): face 0 alone, so face 1 of every remote page never lands */
    return (mode & (uint32_t)kNrExchangeFaces) ? (uint32_t)kNrExchangeBytes / 2u : pages * tile_bytes;
#else
    return (mode & (uint32_t)kNrExchangeFaces) ? (uint32_t)kNrExchangeBytes : pages * tile_bytes;
#endif
}

/** Has slice `member` of step `sequence` landed? Equality, not >=: a flag one step behind is
 *  the previous step's slice. */
FTNR_INLINE bool ftnr_flag_ready(uint32_t flag, uint32_t sequence)
{
    return flag == sequence;
}

/* ---- What one poll of the exchange wait does next ---- */
enum { kNrWaitReady = 0 };          /* the flag reached the sequence: push the slice */
enum { kNrWaitSpin = 1 };           /* poll again */
enum { kNrWaitStop = 2 };           /* the exit word is set: abandon the step, leave */
enum { kNrWaitTimeout = 3 };        /* the spin bound ran out */

/**
 * The exchange wait's next action after `spins` unsatisfied polls (pure, so the host model
 * drives it). The exit word is honoured every 1024 polls, which bounds how long a role
 * blocked on a member that will never send outlives a teardown.
 * FTNR_HOST_RED_IGNORE_EXIT (host test red only, never a kernel build) ignores the exit word.
 */
FTNR_INLINE uint32_t ftnr_exchange_wait_action(uint32_t flag, uint32_t sequence, uint32_t spins,
                                               bool exit_requested, uint32_t spin_limit)
{
    if (ftnr_flag_ready(flag, sequence))
        return kNrWaitReady;
    if ((spins & 1023u) != 0u)
        return kNrWaitSpin;
#if !defined(FTNR_HOST_RED_IGNORE_EXIT)
    if (exit_requested)
        return kNrWaitStop;
#else
    (void)exit_requested;
#endif
    return spins >= spin_limit ? kNrWaitTimeout : kNrWaitSpin;
}

#endif /* RESIDENT_NORM_ROLE_PROTOCOL_H */
