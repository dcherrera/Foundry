/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_puller.c
 * @brief STEP 3 (A) — the ON-CARD PULLER: a resident BRISC kernel that replays a recorded
 *        decode token's queue traffic (parameter blocks, ring slots, head bumps, awaits) into
 *        the other residencies' queues over the NoC, one SEGMENT per descriptor.
 *
 * @status A2: dry-run (log every word it would write) and live mode.
 * @issues None
 * @todo B: identical member ring words + multicast fan-out.
 *
 * THE PROGRAM (in this core's L1 at descriptor a0; built by Pantheon pn_tt_dispatch.c):
 *   header, 16 words: 0 magic 'DPLR'  1 version  2 queues  3 queue table offset (bytes)
 *     4 segments  5 segment table offset  6 log address (L1)  7 log capacity (bytes)
 *     8 OUT error  9 OUT failing entry offset  10 OUT log bytes  11 OUT last await count
 *     12 OUT cycles >> 4 inside PULL descriptors (cumulative)  13 OUT of those, in AWAIT polls
 *     14 multicast groups  15 group table offset (bytes)
 *   queue table, 4 words a queue: x | y << 16, nonce, mask (capacity-1), 0
 *   group table (STEP 3 B), 8 words a group: representative queue (its nonce/mask stand for
 *     all), x_lo, y_lo, x_hi, y_hi, destinations (<= 4: the recorded wedge rule), 0, 0.
 *     A RING or HEAD entry whose queue word has bit 31 set names GROUP (queue & 0xFFFF): the
 *     slot is MULTICAST to the rectangle (every member's ring words are identical: shared
 *     nonce, same index, same parameter-cache slot), the head by a 4-byte multicast write.
 *   segment table: one byte offset a segment
 *   entries, word 0 = type | count << 8 | patches << 20:
 *     1 WRITE  xy, addr, words[count], patches x {word, stride}
 *     2 RING   queue, index0, step, words[8*count] (word 1 of each slot = the token, computed
 *              here), patches x {word, stride}
 *     3 HEAD   queue, base, stride
 *     4 AWAIT  queue, count0, step (done once the queue's completed count is >= it; the
 *              token checked is the completed count's; a torn line is polled again:
 *              resident_puller_abi.h ft_tt_puller_await_step)
 *     5 SEGMENT END     6 PROGRAM END
 *     7 ROWCOPY  the 9 ROWCOPY parameter words, run inline (S2d streamed segments)
 *     8 WRITE DIRECT  xy, addr, `pad` words (head bits 20-21), words[count]: the payload is
 *              16 B-aligned like its destination, so the NoC writes it straight from the
 *              program (no copy through scratch). Lands by the next barrier like any write.
 *     9 FLUSH  write every queued head now (L10): the host's builder places it before the
 *              ring slots and parameter blocks it HOISTED above an await, so those writes go
 *              out while the awaited core works instead of after it finishes. No-op with no
 *              head queued.
 *    10 PRECOND  target, index0, step, producer, count0, count_step (S2): the WAIT SIDECAR
 *              entry of descriptor index0 + step*k on the target queue / group, naming the
 *              producer queue's coordinate and nonce and count0 + count_step*k
 *              (resident_puller_abi.h, resident_precondition_protocol.h). A plain write, like a
 *              ring slot: the next head flush's barrier lands it before the head.
 *   Cumulative cycles >> 4 also in control words 62 (inside PULL) and 63 (in AWAIT polls),
 *   and PER QUEUE at PULL_AWAIT_TABLE (cycles >> 4, awaits) — the AWAIT census (PM 8).
 *   Every per-token word is `base + stride * k` (k = a1); the ring index is `index0 + step*k`.
 *
 * THE DESCRIPTOR: opcode 0x14 PULL; a0 points at its parameter block (the host's ordinary
 * residency enqueue): 0 = program, 1 = k, 2 = first segment, 3 = last segment (inclusive),
 * 4 bit 0 = DRY RUN (log every word, write nothing to any other core); REDS: bit 1 PARAMETER-
 * block patches (KV row, positions) are frozen at the model token (k = 0) — ring words,
 * indices and tokens stay right (k+1 was argmax-lucky IDENTICAL: one zero KV row), bit 2 the first ring slot's token is flipped, bit 3
 * every AWAIT is skipped, bit 5 the first MULTICAST ring slot skips its last member; bit 6 (L10)
 * LAZY heads: no write barrier after a head flush inside a segment (one at the segment end).
 *
 * ORDERING. A target's ring slot and parameter block must LAND before its head says so: heads
 * are queued, and every queued head is written only after a NoC write barrier (one barrier for
 * up to 32 heads). An AWAIT, a segment end or a full head list flushes first. Awaits are a
 * 16-byte read of the target's words 4..7: count >= wanted, then the TOKEN of the last
 * completed descriptor and the state are checked, exactly as the host checks them.
 */

#include <stdbool.h>
#include <stdint.h>

#include "../fw/ft_bh_regs.h"
#include "ft_tensix_dataflow.h"
#include "ft_tensix_noc.h"
#include "ft_tensix_noc_multicast.h"
#include "resident_precondition_protocol.h"
#include "resident_puller_abi.h"
#include "resident_queue_protocol.h"

enum { kOpPull = 0x14 };
/* STEP 4 S2: data movers for the speculative VERIFY (host param block at a0):
 *   ROWCOPY 0x15 {src_x, src_y, src_addr, src_row, dst_x, dst_y, dst_addr, dst_row, tiles}:
 *     row `src_row` of every bf16 tile of a DRAM strip -> row `dst_row` of another (two 32 B
 *     face rows a tile; the source read is the 64 B-aligned pair, the DRAM read rule). */
enum { kOpRowCopy = 0x15 };
/* FILL 0x16 {dst_x, dst_y, dst_addr, bytes, value}: every 32-bit word of DRAM
 * [dst_addr, dst_addr + bytes) becomes `value` — the carried GDN state's zero at a sequence
 * start, on card instead of ~120 MB over the link. dst_addr and bytes 32 B multiples (the
 * NoC's write rule is 16 B; the host asks 64). Source: 32 KB of scratch written once. */
enum { kOpFill = 0x16 };
/* COPY 0x17 {src_x, src_y, src_addr, dst_x, dst_y, dst_addr, bytes}: DRAM [src_addr, +bytes)
 * -> [dst_addr, +bytes), card side, through the scratch in two halves (the read of the next
 * half overlaps the write of this one). Addresses and bytes 64 B multiples (the DRAM read rule).
 * The multi-turn checkpoint of the carried GDN state: ~120 MB that the link reads at 5 MB/s. */
enum { kOpCopy = 0x17 };

enum {
    kPlrMagic = 0x524C5044u, /* "DPLR" */
    kPlrEntryWrite = 1u,
    kPlrEntryRing = 2u,
    kPlrEntryHead = 3u,
    kPlrEntryAwait = 4u,
    kPlrEntrySegment = 5u,
    kPlrEntryEnd = 6u,
    kPlrEntryRowCopy = 7u, /* S2d: an inline ROWCOPY, its 9 parameter words follow */
    kPlrEntryWriteDirect = 8u, /* S2e: WRITE from the program itself (pad words to 16 B) */
    kPlrEntryFlush = FT_TT_PULLER_ENTRY_FLUSH, /* L10: write the queued heads now */
    kPlrEntryPrecond = FT_TT_PULLER_ENTRY_PRECOND /* S2: a descriptor's wait sidecar entry */
};

enum {
    kPlrErrNone = 0u,
    kPlrErrMagic = 1u,
    kPlrErrEntry = 2u,
    kPlrErrAwaitTimeout = 3u,
    kPlrErrAwaitToken = 4u,
    kPlrErrAwaitPoison = 5u,
    kPlrErrLogFull = 6u,
    kPlrErrInline = 7u,
    kPlrErrSegment = 8u
};

#define PULL_QUEUE_CONTROL   0x170000u
#define PULL_QUEUE_RING      0x170100u
#define PULL_HEAD_OFFSET     8u
#define PULL_WINDOW_OFFSET   16u /* words 4..7: stop, count, token, state */
#define PULL_MAX_HEADS       32u
#define PULL_SCRATCH_BYTES   0xC000u /* 48 KB of write sources */
#define PULL_AWAIT_BOUND     200000000u

/* Scratch: write sources, the poll buffer and the pending heads. Fixed L1, below the queue
 * region and above the program arena's end (the host keeps the program + log under it). */
#define PULL_SCRATCH_BASE    0x160000u
#define PULL_POLL_BUFFER     (PULL_SCRATCH_BASE + PULL_SCRATCH_BYTES)
/* PM 8: the AWAIT census — per queue-table index, two words: cumulative cycles >> 4 spent
 * polling that queue, and the number of AWAITs on it. Never reset here (the host diffs two
 * reads), so it needs no initialisation. 256 queues x 8 B, clear of the poll buffer. */
#define PULL_AWAIT_TABLE     (PULL_SCRATCH_BASE + PULL_SCRATCH_BYTES + 0x1000u)
#define PULL_AWAIT_QUEUES    256u
FTQ_STATIC_ASSERT(FT_TT_PULLER_AWAIT_RETRIES == PULL_AWAIT_TABLE + 8u * PULL_AWAIT_QUEUES,
                  "the retry words sit just past the AWAIT census");

typedef struct {
    volatile uint32_t *header;
    const volatile uint32_t *queues;
    uint32_t k;
    uint32_t patch_k;
    bool dry;
    bool red_bad_token;
    bool red_no_wait;
    bool red_group_token;
    bool lazy_heads; /* args[4] bit 6: no barrier after a head flush inside a segment */
    uint32_t scratch_used;
    bool direct_pending;
    uint32_t heads;
    uint32_t head_xy[PULL_MAX_HEADS];
    uint32_t head_value[PULL_MAX_HEADS];
    const volatile uint32_t *groups;
    volatile uint32_t *log;
    uint32_t log_words, log_capacity_words;
} pull_state;

static uint32_t pull_token(uint32_t nonce, uint32_t index)
{
    return ft_tt_puller_token(nonce, index);
}

static bool pull_log(pull_state *s, uint32_t type, uint32_t xy, uint32_t addr,
                     const volatile uint32_t *words, uint32_t count)
{
    uint32_t i;

    if (s->log_words + 4u + count > s->log_capacity_words)
        return false;
    s->log[s->log_words++] = type;
    s->log[s->log_words++] = xy;
    s->log[s->log_words++] = addr;
    s->log[s->log_words++] = count;
    for (i = 0; i < count; i++)
        s->log[s->log_words++] = words[i];
    return true;
}

static volatile uint32_t *pull_scratch(pull_state *s, uint32_t bytes);

/** Write every queued head, after one barrier that lands every ring slot and block before. */
static bool pull_flush_heads(pull_state *s)
{
    uint32_t i;

    if (s->heads == 0u)
        return true;
    if (!s->dry)
        ft_tensix_noc_write_barrier();
    for (i = 0; i < s->heads; i++) {
        const uint32_t xy = s->head_xy[i];

        if (s->dry) {
            volatile uint32_t one[1];

            one[0] = s->head_value[i];
            if (!pull_log(s, kPlrEntryHead, xy, PULL_QUEUE_CONTROL + PULL_HEAD_OFFSET, one, 1u))
                return false;
            continue;
        }
        if ((xy & 0x80000000u) != 0u) {
            const volatile uint32_t *g = s->groups + 8u * (xy & 0xFFFFu);
            volatile uint32_t *line = pull_scratch(s, 16u);
            uint64_t target;

            /* The source word sits at the destination's offset within its 16-byte line, as
             * for the unicast head: an L1 write lands lane for lane. */
            line[PULL_HEAD_OFFSET / 4u] = s->head_value[i];
            if (ft_tensix_multicast_address(g[1], g[2], g[3], g[4],
                                            PULL_QUEUE_CONTROL + PULL_HEAD_OFFSET,
                                            &target) != FT_TENSIX_MULTICAST_OK ||
                ft_tensix_noc_semaphore_set_multicast((uint32_t)(uintptr_t)line + PULL_HEAD_OFFSET,
                                                      target, g[5]) != FT_TENSIX_MULTICAST_OK)
                return false;
            continue;
        }
        {
            /* A PLAIN 4-byte NoC write, never the inline dword write: thousands of inline
             * writes from this BRISC wedged its NEXT firmware boot every time (instrn_buf
             * 0x66, cores 89 and 70; run_d_wedge4.sh: 3/3 clean with plain writes, the inline
             * control wedged). The source word sits at the destination's offset in a line. */
            volatile uint32_t *line = pull_scratch(s, 16u);

            line[PULL_HEAD_OFFSET / 4u] = s->head_value[i];
            ft_tensix_noc_write((uint32_t)(uintptr_t)line + PULL_HEAD_OFFSET,
                                ft_tensix_noc_address(xy & 0xFFFFu, xy >> 16,
                                                      PULL_QUEUE_CONTROL + PULL_HEAD_OFFSET),
                                4u);
        }
    }
    s->heads = 0u;
    /* LAZY (L10): nothing after a flush depends on the heads having LANDED — the next ring
     * slots go out behind the next flush's leading barrier, and an await is a read. Their
     * source lines stay allocated (scratch is recycled only behind a barrier: pull_scratch's
     * wrap, or the segment end below), so the trailing barrier is only a wait. */
    if (s->lazy_heads)
        return true;
    if (!s->dry)
        ft_tensix_noc_write_barrier();
    s->scratch_used = 0u;
    return true;
}

/** Source bytes for one write: a slice of scratch, recycled only after a barrier. */
static volatile uint32_t *pull_scratch(pull_state *s, uint32_t bytes)
{
    volatile uint32_t *at;

    bytes = (bytes + 15u) & ~15u;
    if (s->scratch_used + bytes > PULL_SCRATCH_BYTES) {
        if (!s->dry)
            ft_tensix_noc_write_barrier();
        s->scratch_used = 0u;
    }
    at = (volatile uint32_t *)(uintptr_t)(PULL_SCRATCH_BASE + s->scratch_used);
    s->scratch_used += bytes;
    return at;
}

static void pull_apply_patches(volatile uint32_t *words, const volatile uint32_t *patch,
                               uint32_t patches, uint32_t k)
{
    uint32_t p;

    for (p = 0; p < patches; p++)
        words[patch[2u * p]] += patch[2u * p + 1u] * k;
}

static bool pull_row_copy(const volatile uint32_t *a);

/** Run entries from `entry` until a segment end. @return 0 or an error code. */
static uint32_t pull_segment(pull_state *s, const volatile uint32_t *entry,
                             volatile uint32_t *control)
{
    for (;;) {
        const uint32_t head = entry[0];
        const uint32_t type = head & 0xFFu;
        const uint32_t count = (head >> 8) & 0xFFFu;
        const uint32_t patches = head >> 20;

        s->header[9] = (uint32_t)((uintptr_t)entry - (uintptr_t)s->header);
        if (control[kFieldStopRequest] != 0u)
            return kPlrErrSegment;
        switch (type) {
        case kPlrEntryWrite: {
            const uint32_t xy = entry[1], addr = entry[2];
            volatile uint32_t *words = pull_scratch(s, count * 4u);
            uint32_t i;

            for (i = 0; i < count; i++)
                words[i] = entry[3u + i];
            pull_apply_patches(words, entry + 3u + count, patches, s->patch_k);
            if (s->dry) {
                if (!pull_log(s, kPlrEntryWrite, xy, addr, words, count))
                    return kPlrErrLogFull;
            } else {
                ft_tensix_noc_write((uint32_t)(uintptr_t)words,
                                    ft_tensix_noc_address(xy & 0xFFFFu, xy >> 16, addr),
                                    count * 4u);
            }
            entry += 3u + count + 2u * patches;
            break;
        }
        case kPlrEntryRing: {
            const bool group = (entry[1] & 0x80000000u) != 0u;
            const volatile uint32_t *g = s->groups + 8u * (entry[1] & 0xFFFFu);
            const uint32_t q = group ? g[0] : entry[1];
            const uint32_t xy = group ? entry[1] : s->queues[4u * q];
            const uint32_t nonce = s->queues[4u * q + 1u];
            const uint32_t mask = s->queues[4u * q + 2u];
            const uint32_t index = entry[2] + entry[3] * s->k;
            const uint32_t words_n = 8u * count;
            const uint32_t addr = PULL_QUEUE_RING + (index & mask) * 32u;
            volatile uint32_t *words = pull_scratch(s, words_n * 4u);
            uint32_t i;

            for (i = 0; i < words_n; i++)
                words[i] = entry[4u + i];
            pull_apply_patches(words, entry + 4u + words_n, patches, s->k);
            for (i = 0; i < count; i++)
                words[8u * i + 1u] = pull_token(nonce, index + i);
            if (s->red_bad_token) {
                words[1] ^= 1u;
                s->red_bad_token = false;
            }
            if (s->dry) {
                if (!pull_log(s, kPlrEntryRing, xy, addr, words, words_n))
                    return kPlrErrLogFull;
            } else if (group) {
                uint64_t target;
                /* RED (bit 5): the first group slot misses its last member (x_hi - 1) — that
                 * member never sees the descriptor, member 0's pull waits on it: must REFUSE. */
                const bool shrink = s->red_group_token && g[5] >= 2u;
                const uint32_t x_hi = shrink ? g[3] - 1u : g[3];

                if (shrink)
                    s->red_group_token = false;
                if (ft_tensix_multicast_address(g[1], g[2], x_hi, g[4], addr, &target) !=
                        FT_TENSIX_MULTICAST_OK ||
                    ft_tensix_noc_write_multicast((uint32_t)(uintptr_t)words, target,
                                                  words_n * 4u, shrink ? g[5] - 1u : g[5]) !=
                        FT_TENSIX_MULTICAST_OK)
                    return kPlrErrEntry;
            } else {
                ft_tensix_noc_write((uint32_t)(uintptr_t)words,
                                    ft_tensix_noc_address(xy & 0xFFFFu, xy >> 16, addr),
                                    words_n * 4u);
            }
            entry += 4u + words_n + 2u * patches;
            break;
        }
        case kPlrEntryHead:
            if (s->heads == PULL_MAX_HEADS && !pull_flush_heads(s))
                return kPlrErrInline;
            s->head_xy[s->heads] = (entry[1] & 0x80000000u) != 0u ? entry[1]
                                                                   : s->queues[4u * entry[1]];
            s->head_value[s->heads] = entry[2] + entry[3] * s->k;
            s->heads++;
            entry += 4u;
            break;
        case kPlrEntryAwait: {
            const uint32_t q = entry[1];
            const uint32_t xy = s->queues[4u * q];
            const uint32_t nonce = s->queues[4u * q + 1u];
            const uint32_t wanted = entry[2] + entry[3] * s->k;
            volatile uint32_t *poll = (volatile uint32_t *)(uintptr_t)PULL_POLL_BUFFER;
            uint32_t spins = 0u;

            if (!pull_flush_heads(s))
                return kPlrErrInline;
            if (s->red_no_wait) {
                entry += 4u;
                break;
            }
            if (s->dry) {
                volatile uint32_t one[1];

                one[0] = wanted;
                if (!pull_log(s, kPlrEntryAwait, xy, PULL_QUEUE_CONTROL + 0x14u, one, 1u))
                    return kPlrErrLogFull;
                entry += 4u;
                break;
            }
            const uint32_t await_from = ft_bh_wall_clock_low();
            uint32_t step, mismatches = 0u;

            for (;;) {
                ft_tensix_noc_read(ft_tensix_noc_address(xy & 0xFFFFu, xy >> 16,
                                                         PULL_QUEUE_CONTROL + PULL_WINDOW_OFFSET),
                                   PULL_POLL_BUFFER, 16u);
                ft_tensix_noc_read_barrier();
                ft_bh_l1_cache_invalidate();
                step = ft_tt_puller_await_step(poll, wanted, nonce, &mismatches);
                if (step != FT_TT_PULLER_AWAIT_POLL)
                    break;
                if (control[kFieldStopRequest] != 0u || ++spins >= PULL_AWAIT_BOUND) {
                    s->header[11] = poll[1];
                    return kPlrErrAwaitTimeout;
                }
            }
            s->header[11] = poll[1];
            {
                const uint32_t waited = (ft_bh_wall_clock_low() - await_from) >> 4;

                s->header[13] += waited;
                control[63] += waited;
                if (q < PULL_AWAIT_QUEUES) {
                    volatile uint32_t *census =
                        (volatile uint32_t *)(uintptr_t)PULL_AWAIT_TABLE + 2u * q;

                    census[0] += waited;
                    census[1] += 1u;
                }
                if (mismatches != 0u) {
                    volatile uint32_t *retries =
                        (volatile uint32_t *)(uintptr_t)FT_TT_PULLER_AWAIT_RETRIES;

                    retries[0] += mismatches;
                    retries[1] += 1u;
                }
            }
            /* ft_tt_puller_await_step: the token is the count READ's (a step mark names an
             * older count than the queue's last), and a line read between the writer's token
             * and count stores is read again, not refused. */
            if (step == FT_TT_PULLER_AWAIT_POISON)
                return kPlrErrAwaitPoison;
            if (step == FT_TT_PULLER_AWAIT_TOKEN)
                return kPlrErrAwaitToken;
            entry += 4u;
            break;
        }
        case kPlrEntryWriteDirect: {
            const uint32_t pad = (head >> 20) & 3u;
            const volatile uint32_t *payload = entry + 3u + pad;

            if (!s->dry)
                ft_tensix_noc_write((uint32_t)(uintptr_t)payload,
                                    ft_tensix_noc_address(entry[1] & 0xFFFFu, entry[1] >> 16,
                                                          entry[2]),
                                    count * 4u);
            s->direct_pending = true;
            entry += 3u + pad + count;
            break;
        }
        case kPlrEntryRowCopy:
            /* The row copy stages through scratch: every queued write lands first. */
            if (!pull_flush_heads(s))
                return kPlrErrInline;
            if (!s->dry) {
                ft_tensix_noc_write_barrier();
                s->scratch_used = 0u;
                if (!pull_row_copy(entry + 1))
                    return kPlrErrEntry;
            }
            entry += 10u;
            break;
        case kPlrEntryFlush:
            if (!pull_flush_heads(s))
                return kPlrErrInline;
            entry += 1u;
            break;
        case kPlrEntryPrecond: {
            /* S2: one 16-byte sidecar entry, unicast or to a multicast group like a RING
             * slot (every member of a group shares the index and the mask). */
            const bool group = (entry[1] & 0x80000000u) != 0u;
            const volatile uint32_t *g = s->groups + 8u * (entry[1] & 0xFFFFu);
            const uint32_t q = group ? g[0] : entry[1];
            const uint32_t xy = group ? entry[1] : s->queues[4u * q];
            const uint32_t mask = s->queues[4u * q + 2u];
            const uint32_t index = entry[2] + entry[3] * s->k;
            const uint32_t producer = s->queues[4u * entry[4]];
            const uint32_t addr = PULL_QUEUE_CONTROL + ftq_pre_entry_offset(index, mask);
            volatile uint32_t *line = pull_scratch(s, 16u);

            line[kPreWordIndex] = index;
            line[kPreWordXy] = ftq_pre_xy_word(producer & 0xFFFFu, producer >> 16);
            line[kPreWordCount] = entry[5] + entry[6] * s->k;
            line[kPreWordNonce] = s->queues[4u * entry[4] + 1u];
            if (line[kPreWordXy] == 0u)
                return kPlrErrEntry;
            if (s->dry) {
                if (!pull_log(s, kPlrEntryPrecond, xy, addr, line, 4u))
                    return kPlrErrLogFull;
            } else if (group) {
                uint64_t target;

                if (ft_tensix_multicast_address(g[1], g[2], g[3], g[4], addr, &target) !=
                        FT_TENSIX_MULTICAST_OK ||
                    ft_tensix_noc_write_multicast((uint32_t)(uintptr_t)line, target, 16u, g[5]) !=
                        FT_TENSIX_MULTICAST_OK)
                    return kPlrErrEntry;
            } else {
                ft_tensix_noc_write((uint32_t)(uintptr_t)line,
                                    ft_tensix_noc_address(xy & 0xFFFFu, xy >> 16, addr), 16u);
            }
            entry += FT_TT_PULLER_PRECOND_WORDS;
            break;
        }
        case kPlrEntrySegment:
        case kPlrEntryEnd: {
            /* direct writes read the program: they land before its region can be reused;
             * and every head lands before the PULL completes (the next PULL reuses scratch) */
            const bool flushed = pull_flush_heads(s);

            if ((s->direct_pending || s->lazy_heads) && !s->dry)
                ft_tensix_noc_write_barrier();
            s->direct_pending = false;
            s->scratch_used = 0u;
            return flushed ? kPlrErrNone : kPlrErrInline;
        }
        default:
            return kPlrErrEntry;
        }
    }
}

#define PULL_MOVE_BYTES 0x8000u /* 32 KB of the 48 KB scratch a batch */

static bool pull_row_copy(const volatile uint32_t *a)
{
    const uint32_t src_x = a[0], src_y = a[1], src_addr = a[2], src_row = a[3];
    const uint32_t dst_x = a[4], dst_y = a[5], dst_addr = a[6], dst_row = a[7], tiles = a[8];
    const uint32_t src_base = (src_row < 16u ? 0u : 1024u) + (src_row % 16u) * 32u;
    const uint32_t dst_base = (dst_row < 16u ? 0u : 1024u) + (dst_row % 16u) * 32u;
    const uint32_t per_batch = PULL_MOVE_BYTES / 128u; /* tiles a batch: 2 x 64 B each */
    uint32_t t0;

    if (src_row >= 32u || dst_row >= 32u)
        return false;
    for (t0 = 0; t0 < tiles; t0 += per_batch) {
        const uint32_t n = (tiles - t0) < per_batch ? (tiles - t0) : per_batch;
        uint32_t t, f;

        for (t = 0; t < n; t++)
            for (f = 0; f < 2u; f++) {
                const uint32_t src = src_addr + (t0 + t) * 2048u + src_base + f * 512u;

                ft_tensix_noc_read(ft_tensix_noc_address(src_x, src_y, src & ~63u),
                                   PULL_SCRATCH_BASE + (2u * t + f) * 64u, 64u);
            }
        ft_tensix_noc_read_barrier();
        for (t = 0; t < n; t++)
            for (f = 0; f < 2u; f++) {
                const uint32_t src = src_addr + (t0 + t) * 2048u + src_base + f * 512u;

                ft_tensix_noc_write(PULL_SCRATCH_BASE + (2u * t + f) * 64u + (src & 32u),
                                    ft_tensix_noc_address(dst_x, dst_y,
                                                          dst_addr + (t0 + t) * 2048u +
                                                              dst_base + f * 512u),
                                    32u);
            }
        ft_tensix_noc_write_barrier();
    }
    return true;
}

/** FILL: see kOpFill. A zero-length or misaligned request is refused, never issued. */
static bool pull_fill(const volatile uint32_t *a)
{
    const uint32_t dst_x = a[0], dst_y = a[1], dst_addr = a[2], bytes = a[3], value = a[4];
    volatile uint32_t *source = (volatile uint32_t *)(uintptr_t)PULL_SCRATCH_BASE;
    uint32_t offset, word;

    if (bytes == 0u || (bytes & 31u) != 0u || (dst_addr & 31u) != 0u)
        return false;
    for (word = 0; word < PULL_MOVE_BYTES / 4u; word++)
        source[word] = value;
    for (offset = 0; offset < bytes; offset += PULL_MOVE_BYTES) {
        const uint32_t n = (bytes - offset) < PULL_MOVE_BYTES ? (bytes - offset) : PULL_MOVE_BYTES;

        ft_tensix_noc_write(PULL_SCRATCH_BASE, ft_tensix_noc_address(dst_x, dst_y, dst_addr + offset),
                            n);
    }
    ft_tensix_noc_write_barrier();
    return true;
}

/** COPY: see kOpCopy. A zero-length or misaligned request is refused, never issued. */
static bool pull_copy(const volatile uint32_t *a)
{
    const uint32_t src_x = a[0], src_y = a[1], src_addr = a[2];
    const uint32_t dst_x = a[3], dst_y = a[4], dst_addr = a[5], bytes = a[6];
    const uint32_t half = PULL_SCRATCH_BYTES / 2u;
    uint32_t offset, buffer = PULL_SCRATCH_BASE;

    if (bytes == 0u || ((src_addr | dst_addr | bytes) & 63u) != 0u)
        return false;
    ft_tensix_noc_read(ft_tensix_noc_address(src_x, src_y, src_addr), buffer,
                       bytes < half ? bytes : half);
    ft_tensix_noc_read_barrier();
    for (offset = 0; offset < bytes; offset += half) {
        const uint32_t n = (bytes - offset) < half ? (bytes - offset) : half;
        const uint32_t next = offset + half;
        const uint32_t other = buffer == PULL_SCRATCH_BASE ? PULL_SCRATCH_BASE + half
                                                           : PULL_SCRATCH_BASE;

        if (next < bytes)
            ft_tensix_noc_read(ft_tensix_noc_address(src_x, src_y, src_addr + next), other,
                               (bytes - next) < half ? (bytes - next) : half);
        ft_tensix_noc_write(buffer, ft_tensix_noc_address(dst_x, dst_y, dst_addr + offset), n);
        ft_tensix_noc_read_barrier();
        ft_tensix_noc_write_barrier();
        buffer = other;
    }
    return true;
}

void kernel_main(void)
{
    ftq_context ctx;

    ftq_context_init(&ctx);

    volatile FTQ_L1_PTR uint32_t *control = ctx.control;
    volatile FTQ_L1_PTR uint32_t *ring = ctx.ring;
    const uint32_t capacity_mask = ctx.capacity_mask;
    const uint32_t idle_spin_limit = ctx.idle_spin_limit;

    control[kFieldState] = kStateBoot;
    ftq_status_set(&ctx, kStateBoot);
    if (control[kFieldMagic] != kMagic || capacity_mask == 0u ||
        ((capacity_mask + 1u) & capacity_mask) != 0u ||
        control[kFieldCapacity] != capacity_mask + 1u || idle_spin_limit == 0u) {
        control[kFieldError] = kErrBadCapacity;
        control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    ((volatile uint32_t *)(uintptr_t)FT_TT_PULLER_AWAIT_RETRIES)[0] = 0u;
    ((volatile uint32_t *)(uintptr_t)FT_TT_PULLER_AWAIT_RETRIES)[1] = 0u;

    uint32_t tail = control[kFieldTail];
    uint32_t completed = control[kFieldCompletedCount];
    uint32_t idle_spins = 0u;
    uint32_t heartbeat = control[kFieldHeartbeat];
    uint32_t error = kErrNone;
    bool running = true;

    while (running) {
        control[kFieldHeartbeat] = ++heartbeat;
        if (control[kFieldStopRequest] != 0u)
            break;
        if (control[kFieldHead] == tail) {
            control[kFieldState] = kStateIdle;
            control[kFieldIdleSpins] = ++idle_spins;
            if (idle_spins >= idle_spin_limit) {
                error = kErrIdleTimeout;
                break;
            }
            continue;
        }
        idle_spins = 0u;
        control[kFieldState] = kStateExec;

        volatile FTQ_L1_PTR uint32_t *descriptor = ring + (tail & capacity_mask) * kDescriptorWords;
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t token = descriptor[kDescToken];
        bool item_completed = true;

        control[kFieldLastOpcode] = opcode;
        if (opcode == kOpPull) {
            const volatile uint32_t *args =
                (const volatile uint32_t *)(uintptr_t)descriptor[kDescA0];
            pull_state s;
            uint32_t segment, rc = kPlrErrNone;

            s.header = (volatile uint32_t *)(uintptr_t)args[0];
            s.k = args[1];
            s.dry = (args[4] & 1u) != 0u;
            s.patch_k = (args[4] & 2u) != 0u ? 0u : s.k;
            s.red_bad_token = (args[4] & 4u) != 0u;
            s.red_no_wait = (args[4] & 8u) != 0u;
            s.red_group_token = (args[4] & 32u) != 0u;
            s.lazy_heads = (args[4] & 64u) != 0u;
            s.scratch_used = 0u;
            s.direct_pending = false;
            s.heads = 0u;
            s.log = (volatile uint32_t *)(uintptr_t)s.header[6];
            s.log_words = 0u;
            s.log_capacity_words = s.header[7] / 4u;
            if (s.header[0] != kPlrMagic) {
                rc = kPlrErrMagic;
            } else {
                const volatile uint32_t *segments =
                    (const volatile uint32_t *)((uintptr_t)s.header + s.header[5]);

                s.queues = (const volatile uint32_t *)((uintptr_t)s.header + s.header[3]);
                s.groups = (const volatile uint32_t *)((uintptr_t)s.header + s.header[15]);
                const uint32_t started = ft_bh_wall_clock_low();

                for (segment = args[2]; rc == kPlrErrNone && segment <= args[3]; segment++) {
                    if (segment >= s.header[4]) {
                        rc = kPlrErrSegment;
                        break;
                    }
                    rc = pull_segment(&s, (const volatile uint32_t *)((uintptr_t)s.header +
                                                                     segments[segment]),
                                      control);
                }
                s.header[12] += (ft_bh_wall_clock_low() - started) >> 4;
                control[62] += (ft_bh_wall_clock_low() - started) >> 4;
            }
            s.header[8] = rc;
            s.header[10] = s.log_words * 4u;
            if (rc != kPlrErrNone) {
                /* A FAILED REPLAY IS NOT COMPLETED: the host sees the count stop and reads
                 * header[8] for why. The kernel stays resident for the host's recovery. */
                error = kErrBadParameters;
                item_completed = false;
                running = false;
            }
        } else if (opcode == kOpRowCopy) {
            const volatile uint32_t *args =
                (const volatile uint32_t *)(uintptr_t)descriptor[kDescA0];

            if (!pull_row_copy(args)) {
                error = kErrBadParameters;
                item_completed = false;
                running = false;
            }
        } else if (opcode == kOpFill) {
            if (!pull_fill((const volatile uint32_t *)(uintptr_t)descriptor[kDescA0])) {
                error = kErrBadParameters;
                item_completed = false;
                running = false;
            }
        } else if (opcode == kOpCopy) {
            if (!pull_copy((const volatile uint32_t *)(uintptr_t)descriptor[kDescA0])) {
                error = kErrBadParameters;
                item_completed = false;
                running = false;
            }
        } else if (opcode == kOpStop) {
            running = false;
        } else if (opcode != kOpNop) {
            error = kErrBadOpcode;
            item_completed = false;
            running = false;
        }
        if (item_completed) {
            tail++;
            control[kFieldTail] = tail;
            control[kFieldCompletedToken] = token;
            control[kFieldCompletedCount] = ++completed;
        }
    }
    control[kFieldError] = error;
    control[kFieldState] = kStateExit;
    ctx.error = error;
    ftq_status_set(&ctx, kStateExit);
}
