// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_reduce_protocol.h
//   The three PROGRAM words the REDUCE op family reads out of its parameter block, on top
//   of the thirteen resident_vector_protocol.h already defines. The host half is
//   ft_tt_op_vector.h; the two must agree word for word and tests/test_vector_ops.c pins
//   the pair off-hardware.
//
// ===========================================================================
// WHAT THE REDUCE FAMILY IS, AND WHY IT IS ONE BINARY
// ===========================================================================
// Softmax, RMS norm, layer norm and the MoE router's normalise are ONE SKELETON:
//
//     reduce along the row into DST column 0  ->  an SFPU verb on that column
//                                             ->  broadcast it back with *_bcast_cols
//
// They differ in three things and nothing else: HOW MANY reduce passes, WHICH pool and
// SFPU verb each pass uses, and WHICH per-tile steps the final scaling chain runs. All
// three are DESCRIPTOR FIELDS here, so the four ops are one resident binary and adding a
// fifth member is a host-side table entry rather than a kernel.
//
// The reason that matters is residency, not tidiness: a resident binary is linked at a
// fixed per-processor slab (ft_tt_op_l1_layout.h), so only ONE program is resident on a
// core at a time and swapping costs ~0.19 ms of image upload over the x1 link
// (docs/tt_bringup_log.md 20.4). A transformer block does an RMS norm, then attention's
// softmax, then another RMS norm; as separate binaries that is two image reloads per
// block. THAT is what decides a family (22.1), and it is why this is a MERGE of the
// Phase 4 one-shot rms_norm into softmax's resident triad rather than a second port.
//
// ===========================================================================
// THE PROGRAM, AND THE TWO THINGS IT DERIVES RATHER THAN CARRIES
// ===========================================================================
// Every field a pass needs is sent. Two are DERIVED, and both are family invariants
// rather than convenience:
//
//   - THE SCALE COLUMN IS ALWAYS THE LAST PASS'S OUTPUT. That is what "the last reduce
//     produces the scale" means, and it holds for all four members: softmax scales by
//     1/rowsum (pass 2), RMS norm by rsqrt(mean square + eps) (pass 1), layer norm by
//     rsqrt(var + eps) (pass 2), the router by 1/rowsum (pass 1).
//   - A MAP THAT SUBTRACTS ALWAYS SUBTRACTS COLUMN 0, the FIRST pass's output — softmax
//     subtracts the row max, layer norm the row mean. A map in pass 0 has no column to
//     read and never subtracts.
//
// Sending either would be a second field that has to agree with the first, and dev notes
// 3 has the catalogue of what two fields that must agree cost here.
//
// A word this family does not use is written 0 by the host and never read, exactly as
// resident_vector_protocol.h requires. kVecFunction carries no function id: the executable
// program is these three words. Since P3 that word is kRedSchedule — HOW the program is
// scheduled (streamed reads, half reads, chain chunk), never WHAT it computes; 0 is the
// pre-P3 schedule exactly.
//
#ifndef FOUNDRY_RESIDENT_REDUCE_PROTOCOL_H
#define FOUNDRY_RESIDENT_REDUCE_PROTOCOL_H

#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

/* The SFPI GCC fork's own pointer attribute — an access through it goes to L1 rather than
 * to ordinary memory, and dropping it changes the code that is emitted. Identical to
 * tt-metal's `tt_l1_ptr`; spelled here as well as in resident_queue_protocol.h so this
 * header does not depend on include order. */
#ifndef FTQ_L1_PTR
#define FTQ_L1_PTR __attribute__((rvtt_l1_ptr))
#endif

/* Header-only helper linkage, spelled the same way in both languages -- `inline` in C++ and
 * `static inline` in C. NOT cosmetic: internal linkage changes GCC's inlining and cloning
 * decisions, and the COMPUTE images that also include this header must not move for a
 * change that is about the data-movement ones. */
#ifndef FTQ_INLINE
#ifdef __cplusplus
#define FTQ_INLINE inline
#else
#define FTQ_INLINE static inline
#endif
#endif

#include "resident_vector_protocol.h"

// ---------------------------------------------------------------------------
// The three program words. Mirrors ft_tt_op_vector.h -- change both or neither.
// ---------------------------------------------------------------------------

/** Pass 0's program: byte 0 pool, byte 1 map, byte 2 SFPU verb, byte 3 add-eps. */
enum { kRedPass0 = 13 };

/** Pass 1's program, same layout. READ ONLY when the pass count is 2. */
enum { kRedPass1 = 14 };

/** byte 0 pass count (1 or 2), byte 1 the final chain's source stream, byte 2 the chain's
 *  step flags. */
enum { kRedFinal = 15 };

/** Where a byte lives in a program word. Bytes rather than bit fields because a field a
 *  human has to read out of a hex dump on a stuck core is worth a nibble pair. */
enum { kRedBytePool = 0 };
enum { kRedByteMap = 1 };
enum { kRedByteVerb = 2 };
enum { kRedByteAddEps = 3 };
enum { kRedBytePassCount = 0 };
enum { kRedByteSource = 1 };
enum { kRedByteChain = 2 };

/** Which reduction one pass performs. 0 is deliberately not a pool, so a zeroed parameter
 *  block is an invalid program rather than a plausible one. */
enum { kRedPoolSum = 1 };
enum { kRedPoolMax = 2 };

/** What a pass builds the stream it reduces over, out of `cb_in`. NONE reduces `cb_in`
 *  itself and touches no intermediate. */
enum { kRedMapNone = 0 };
enum { kRedMapSquareSelf = 1 };  // x*x            -- RMS norm's mean square
enum { kRedMapSubExp = 2 };      // exp(x - col0)  -- softmax's exponentials
enum { kRedMapSubSquare = 3 };   // (x - col0)^2   -- layer norm's variance

/** The SFPU verb applied to the reduce result before it is packed. */
enum { kRedVerbNone = 0 };
enum { kRedVerbRecip = 1 };
enum { kRedVerbRsqrt = 2 };

/** Which stream the final chain scales: the activations, or the intermediate the last map
 *  produced (softmax scales its exponentials, everything else scales x). */
enum { kRedSourceIn = 0 };
enum { kRedSourceMid = 1 };

/** The final chain's steps, applied in THIS order, each optional. The order is fixed and
 *  the multiplications commute, so it is a rounding choice rather than a semantic one --
 *  and it is gamma before scale because that is the order rms_norm_compute.c has been
 *  green in on silicon since Phase 4 (docs/tt_bringup_log.md 14). */
enum { kRedChainCenter = 0x1 };  // - col0           layer norm
enum { kRedChainGamma = 0x2 };   // * gamma[wt]      RMS norm, layer norm
enum { kRedChainScale = 0x4 };   // * last column    all four
enum { kRedChainBeta = 0x8 };    // + beta[wt]       layer norm
enum { kRedChainAll = 0xF };

/** The constants operand's tile layout, fixed for every member so the reader's offsets do
 *  not depend on the program. The eps slot exists even for a member that does not use it:
 *  a layout that moves with the flags is a wrong address for one descriptor in four. */
enum { kRedConstScalerTile = 0 };
enum { kRedConstEpsTile = 1 };
enum { kRedConstStreamTile = 2 };  // gamma tiles, then beta tiles, Wt each

/** One pass, decoded. */
typedef struct ftr_pass {
    uint32_t pool;
    uint32_t map;
    uint32_t verb;
    uint32_t add_eps;
} ftr_pass;

/** One descriptor's whole program, decoded once at the top of the op. */
typedef struct ftr_program {
    uint32_t pass_count;
    uint32_t source;
    uint32_t chain;
    ftr_pass pass[2];
} ftr_program;

FTQ_INLINE uint32_t ftr_byte(uint32_t word, uint32_t index)
{
    return (word >> (index * 8u)) & 0xFFu;
}

FTQ_INLINE ftr_pass ftr_decode_pass(uint32_t word)
{
    ftr_pass pass;

    pass.pool = ftr_byte(word, kRedBytePool);
    pass.map = ftr_byte(word, kRedByteMap);
    pass.verb = ftr_byte(word, kRedByteVerb);
    pass.add_eps = ftr_byte(word, kRedByteAddEps);
    return pass;
}

/**
 * L4 COLUMN SLICE (2026-10-05): byte 3 of kRedFinal = (members << 4) | member, 0 = whole.
 * Every member still reads and reduces the WHOLE row (the norm needs the full sum); the final
 * chain, the gamma/beta streams and the output cover only tiles [first, first + count) of
 * each row. `tile_cols` must divide by `members`.
 */
FTQ_INLINE void ftr_slice(uint32_t final_word, uint32_t tile_cols, uint32_t *first,
                          uint32_t *count)
{
    const uint32_t code = (final_word >> 24) & 0xFFu;
    const uint32_t members = code >> 4;
    const uint32_t member = code & 0xFu;

    if (members < 2u || member >= members || tile_cols % members != 0u) {
        *first = 0u;
        *count = tile_cols;
        return;
    }
    *count = tile_cols / members;
    *first = member * *count;
}

FTQ_INLINE ftr_program ftr_decode(volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t final_word = parameters[kRedFinal];
    ftr_program program;

    program.pass_count = ftr_byte(final_word, kRedBytePassCount);
    program.source = ftr_byte(final_word, kRedByteSource);
    program.chain = ftr_byte(final_word, kRedByteChain);
    program.pass[0] = ftr_decode_pass(parameters[kRedPass0]);
    program.pass[1] = ftr_decode_pass(parameters[kRedPass1]);
    return program;
}

/**
 * Is this a program this binary can execute?
 *
 * Checked on the DEVICE and not only on the host, because the failure it prevents is not a
 * wrong answer: a chain with no steps never writes cb_out, so the writer waits on an output
 * tile nobody produces and the whole triad blocks until its bound expires. A refusal names
 * that in one word instead.
 */
/* HOW A DECODED SHAPE IS PASSED, and why the name is bound twice. The body takes a
 * POINTER, which is what C wants. The name is then bound to a REFERENCE on the C++ arm and
 * to the pointer on the C arm, because the compute images that also include this header
 * were written against the reference and this round does not own them.
 *
 * All three forms were MEASURED against the pre-conversion images, and the ranking is not
 * obvious: pointer-plus-wrapper leaves 18 of the 21 compute images byte-identical, a body
 * that copies `*ptr` into a local leaves 15, and passing the struct BY VALUE -- which needs
 * no wrapper at all and reads identically in both languages -- leaves only 6. The wrapper
 * is the cost of the best of the three. All of it goes when the compute half of plan
 * Phase 6.6 lands. */
FTQ_INLINE bool ftr_program_is_valid(const ftr_program *program)
{
    if (program->pass_count == 0u || program->pass_count > 2u)
        return false;
    if (program->chain == 0u || program->chain > kRedChainAll)
        return false;
    if (program->source > kRedSourceMid)
        return false;
    for (uint32_t index = 0; index < program->pass_count; index++) {
        const ftr_pass pass = program->pass[index];

        if (pass.pool != kRedPoolSum && pass.pool != kRedPoolMax)
            return false;
        if (pass.map > kRedMapSubSquare)
            return false;
        if (pass.verb > kRedVerbRsqrt)
            return false;
        // A subtracting map reads column 0, which pass 0 has not produced yet.
        if (index == 0u && pass.map >= kRedMapSubExp)
            return false;
        // eps is one more accumulated row SUM; it is meaningless against a MAX.
        if (pass.add_eps != 0u && pass.pool != kRedPoolSum)
            return false;
    }
    return true;
}

/** Does any pass of this program read the eps tile? The reader pushes it only then, and the
 *  compute thread pops only what the reader pushed -- both from this one function. */
FTQ_INLINE bool ftr_program_has_eps(const ftr_program *program)
{
    for (uint32_t index = 0; index < program->pass_count; index++) {
        if (program->pass[index].add_eps != 0u)
            return true;
    }
    return false;
}

/** Does any pass build an intermediate stream? */
FTQ_INLINE bool ftr_program_has_mid(const ftr_program *program)
{
    for (uint32_t index = 0; index < program->pass_count; index++) {
        if (program->pass[index].map != kRedMapNone)
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// P3 SCHEDULE (word 0). Every bit changes WHEN a tile is read or computed; none changes the
// ops a tile sees, their order, or a rounding point — so a schedule is bit-exact against
// word 0 = 0 by construction (Pantheon docs/q38dec_p3_norm.md has the argument).
// ---------------------------------------------------------------------------

/** The schedule word. It is kVecFunction's slot, which the reduce family never used. */
enum { kRedSchedule = 0 };

/** Bits 0-7: final-chain tiles per step run (K). 0 or 1 = per tile, the pre-P3 order. */
enum { kRedSchedChunkMask = 0xFF };

/** STREAMED input: up to kRedStreamFlight blocks in flight on their own NoC transaction ids,
 *  the compute thread's first map waiting block by block, gamma/beta read after row 0. */
enum { kRedSchedStream = 0x100 };

/** HALF READS: read faces 0-1 only of the input and gamma. Honoured only where
 *  ftr_half_read() says the program never unpacks faces 2-3 of either. */
enum { kRedSchedHalfRead = 0x200 };

/** RED (proof only): a half read lands face 0 alone, so face 1 is stale and the output MUST
 *  differ. */
enum { kRedSchedRedFace0 = 0x400 };

/** Blocks the streamed reader keeps in flight; transaction ids 1..kRedStreamFlight. */
enum { kRedStreamFlight = 4 };

/** The final chain's scratch CBs. FIXED by reduce_float*.cbtable and FT_TT_OP_REDUCE_CB_STAGE*
 *  (the host test pins both); the reader has no compile-time arg for them and needs their
 *  page count only to refuse a chunk they cannot hold. */
enum { kRedCbStage0 = 8 };
enum { kRedCbStage1 = 9 };

/**
 * The final-chain chunk this descriptor asks for, checked against the scratch CBs' pages.
 *
 * A step run of K tiles packs K pages into a scratch CB before the next step consumes any,
 * so K past its page count is a pack that waits forever — a wedge, not a slow answer. The
 * READER refuses 0: it is the only processor that sees every CB's page count (MATH's
 * firmware sets up no CB records and UNPACK's no page counts). Compute treats 0 as 1.
 *
 * @return K >= 1, or 0 when the scratch CBs cannot hold it
 */
FTQ_INLINE uint32_t ftr_chain_chunk(uint32_t schedule, uint32_t stage_pages)
{
    const uint32_t chunk = schedule & (uint32_t)kRedSchedChunkMask;

    if (chunk <= 1u)
        return 1u;
    return chunk > stage_pages ? 0u : chunk;
}

/**
 * Does this program read cb_in and gamma ONLY through the faces-0-1 passes?
 *
 * Every pass maps x*x (so no reduce unpacks cb_in, and the map honours the half tile), the
 * chain scales cb_in rather than the intermediate, and it has no center or beta step (both
 * unpack whole tiles). RMS norm and L2 norm are; softmax, layer norm and the router are not.
 */
FTQ_INLINE bool ftr_program_is_norm_shaped(const ftr_program *program)
{
    if (program->pass_count == 0u || program->source != (uint32_t)kRedSourceIn)
        return false;
    if ((program->chain & ((uint32_t)kRedChainCenter | (uint32_t)kRedChainBeta)) != 0u)
        return false;
    for (uint32_t index = 0; index < program->pass_count; index++) {
        if (program->pass[index].map != (uint32_t)kRedMapSquareSelf)
            return false;
    }
    return true;
}

/** Half reads: asked for, the descriptor is a half-tile one, and the program is norm-shaped. */
FTQ_INLINE bool ftr_half_read(uint32_t schedule, uint32_t depth_word, const ftr_program *program)
{
    return (schedule & (uint32_t)kRedSchedHalfRead) != 0u && ftv_half_tile(depth_word) &&
           ftr_program_is_norm_shaped(program);
}

/** Bytes one input or gamma tile read lands: the whole tile, faces 0-1 (its first half — the
 *  four faces are back to back), or under the RED face 0 alone. */
FTQ_INLINE uint32_t ftr_tile_read_bytes(uint32_t tile_bytes, bool half, uint32_t schedule)
{
    if (!half)
        return tile_bytes;
    return (schedule & (uint32_t)kRedSchedRedFace0) != 0u ? tile_bytes / 4u : tile_bytes / 2u;
}

/** What the streamed reader does next. */
enum { kRedStreamIssue = 0 };   // start block `issued` behind the ones in flight
enum { kRedStreamRetire = 1 };  // wait for block `retired`'s id, then push it
enum { kRedStreamReserve = 2 }; // nothing in flight and no room: a blocking, stoppable wait
enum { kRedStreamDone = 3 };

/**
 * The streamed reader's next action — pure, so the host can run it against a modelled
 * consumer (tests/test_reduce_schedule.c).
 *
 * A block is issued only into pages already free BEYOND the blocks in flight, and the reader
 * never blocks while it holds an unpushed block: the compute thread may be waiting on exactly
 * that block, so blocking there is a deadlock rather than a stall. With nothing in flight a
 * blocking wait is safe — every page the consumer can be waiting on has been pushed.
 *
 * @param issued      blocks issued so far
 * @param retired     blocks pushed so far (retired <= issued)
 * @param total       blocks to move
 * @param free_pages  pages free in cb_in now, counted from the write pointer
 * @param depth       pages per block
 */
FTQ_INLINE uint32_t ftr_stream_action(uint32_t issued, uint32_t retired, uint32_t total,
                                      uint32_t free_pages, uint32_t depth)
{
    const uint32_t in_flight = issued - retired;

    if (retired >= total)
        return kRedStreamDone;
    if (issued < total && in_flight < (uint32_t)kRedStreamFlight &&
        free_pages >= (in_flight + 1u) * depth)
        return kRedStreamIssue;
    if (in_flight != 0u)
        return kRedStreamRetire;
    return kRedStreamReserve;
}

/* The three spellings the compute kernel uses. A C++ reference overload lived beside these
 * macros while `resident_reduce_compute.cpp` existed; plan Phase 6.6 removed the last C++
 * kernel in `tt/` on 2026-08-13, so the pointer form is the only one left. */
#define ftr_program_valid(program) ftr_program_is_valid(program)
#define ftr_program_uses_eps(program) ftr_program_has_eps(program)
#define ftr_program_uses_mid(program) ftr_program_has_mid(program)

#endif  // FOUNDRY_RESIDENT_REDUCE_PROTOCOL_H
