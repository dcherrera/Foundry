/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_weight_convert.h
 * @brief The Phase 3 weight conversion path: a GGUF checkpoint in, a tiled
 *   Blackhole weight artifact out, streaming, never full-model resident.
 *
 * ---------------------------------------------------------------------------
 * RUN IT ON THE CARD HOST, NOT THE MAC — MEASURED, and it reverses spec §8.4
 * ---------------------------------------------------------------------------
 * §8.4 put this on the Mac because it is bulk scalar work and the card host is 2
 * no-AVX cores at 1.8 GHz. That reasoning is about CPU and the binding
 * constraint turned out to be MEMORY: peak RSS is **76-215 MB on the card host
 * against 641 MB on macOS**, because `MADV_DONTNEED` is advisory-weak on Darwin
 * and the Mac holds the whole GGUF mapping instead of dropping it behind the
 * read cursor. The card host is also where the checkpoints already are, so nothing
 * crosses the LAN. **MiniCPM5-1B is ~45 s there; the full 50.9 GiB Qwen3.8 is
 * ~23 minutes.** Nothing here uses an SIMD intrinsic, so the same source builds
 * on both and the artifact is byte-identical either way — this is an operating
 * instruction, not a portability limit.
 *
 * ---------------------------------------------------------------------------
 * WHY IT NEVER HOLDS A TENSOR
 * ---------------------------------------------------------------------------
 * The unit of work is a **strip**: 32 logical rows of one tensor, the full
 * width. A strip of the widest tensor in MiniCPM5-1B is a few hundred KB, so
 * peak host memory is set by the strip buffers and not by the model. The GGUF
 * side is an mmap that is read once, sequentially, and advised away behind us.
 * The acceptance number the plan asks for — peak RSS under 8 GB for a full
 * model — is not a close call under this shape; it is measured anyway, because
 * "it should be small" is not a measurement.
 *
 * ---------------------------------------------------------------------------
 * THE ARTIFACT
 * ---------------------------------------------------------------------------
 * Little-endian throughout, which both the Mac and the card host are, and which the
 * card is.
 *
 *   magic 'FTTW' | version | tensor_count | flags
 *   directory_offset | data_offset                       (uint64 each)
 *   ... tensor data, each tensor 64-byte aligned ...
 *   ... directory: tensor_count fixed-size entries ...
 *
 * The directory is written LAST because a tensor's size is not known until it
 * has been converted — which is what lets the whole thing stream to a pipe or
 * a sequentially written file with no seeking back over the data.
 *
 * ---------------------------------------------------------------------------
 * TWO 2-D LAYOUTS, AND WHY A MATMUL WEIGHT GETS THE SECOND ONE
 * ---------------------------------------------------------------------------
 * `layout` in each directory entry says which, and it is a FORMAT field, not a
 * hint: read the wrong one and every index is in range and every answer is
 * wrong.
 *
 *   FT_TT_LAYOUT_SOURCE (0)
 *     GGUF's own order. ceil(rows/32) x ceil(cols/32) tiles in ROW-major tile
 *     order, `rows` = dims[1], `cols` = dims[0] (the fastest axis). This is
 *     what an EMBEDDING table wants, because it is indexed BY ROW: token id
 *     picks a row, and transposing it would turn a lookup into a gather.
 *
 *   FT_TT_LAYOUT_MATMUL_NMAJOR (1)
 *     A matmul weight. TRANSPOSED to K x N = [in_features x out_features] —
 *     the orientation Phase 4 decided (foundry/tt/ft_tt_op_matmul.h) — and its
 *     tile grid stored COLUMN-major, tile index nt*Kt + kt.
 *
 * The transpose has to happen here and nowhere else: a Bfp8_b/Bfp4_b tile
 * shares one exponent across 16 elements ALONG A ROW, so transposing a PACKED
 * tile means unpacking and repacking it. Upstream of the packer it is free.
 *
 * The column-major tile grid is the second half of the same change and it is a
 * measured one. The matmul reader fetches a block of `depth` K-tiles per NoC
 * barrier; in row-major tile order those are strided by Nt, so each is its own
 * transaction, and transaction COUNT is what the reader is bound by — the same
 * observation was worth 1.28x on the A side alone (docs/tt_bringup_log.md
 * §16.2). Column-major makes a block's B tiles consecutive, so the whole block
 * is one read.
 *
 * **It costs nothing to produce.** A conversion strip is 32 SOURCE rows, i.e.
 * one out-feature tile column, and the Kt target tiles it yields are exactly
 * the ones that are consecutive in N-major order. The streaming order the
 * converter already had IS the new order; the only added work is a 32x32
 * transpose per tile.
 *
 * Edge tiles are zero-padded, in whichever axis the padding lands after the
 * transpose.
 *
 * **1-D tensors are stored row-major and untiled, and that is a decision, not
 * an omission.** Blackhole broadcasts a norm weight in hardware, so it never
 * needs 32x replication in DRAM — but ROW broadcast wants a tile with a filled
 * 0-row and COL wants one with a filled 0-column, and which of the two RMSNorm
 * needs follows from the matmul orientation, which is Phase 4's to decide. The
 * untiled vector is the only form BOTH are derivable from, so the artifact
 * stores it and the loader or the reader kernel shapes it. 16 fp32 = 64 B =
 * one face row = the artifact's alignment, so that shaping is an aligned copy.
 * `docs/tt_bringup_log.md` §13.2.
 *
 * @status Converts MiniCPM5-1B end to end at every ladder rung.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_WEIGHT_CONVERT_H
#define FT_TT_WEIGHT_CONVERT_H

#include <stdint.h>
#include <stdio.h>

#include "ft_gguf_map.h"
#include "ft_tt_format_map.h"
#include "ft_tt_precision_ledger.h"

/** 'FTTW' little-endian. */
#define FT_TT_ARTIFACT_MAGIC 0x57545446u

/**
 * Bumped whenever the on-disk meaning changes.
 *
 *   1  every 2-D tensor in GGUF order, row-major tile grid, `reserved` zero
 *   2  `reserved` became `layout`, and a matmul weight is transposed to
 *      [in x out] with a column-major tile grid
 *   3  ROUTED-EXPERT STACKS and Bfp6_b. A 3-D `ffn_*_exps` tensor ([K, N, E] in GGML
 *      order) is E experts, each a contiguous N-major K x N slice; `tile_rows` x
 *      `tile_cols` is ONE expert's grid (Kt x Nt), and expert e starts at
 *      `e * tile_rows * tile_cols * tile_bytes` (ft_tt_fttw_expert_stride()). Format 4
 *      (FT_TT_FORMAT_BFP6_B, 832 B tiles) may appear. Every 2-D tensor means what it
 *      meant in v2, so a converter writes 3 ONLY when the file holds one of these and
 *      v2 otherwise — a dense model's artifact stays byte-identical and its cache hits.
 *
 * The bump is the point: a v2 artifact fed to a v1 consumer would be read with
 * the wrong orientation and the wrong tile stride, both in range and both
 * silent. A hard version refusal is the only thing that turns that into an
 * error. Readers here accept both and branch on `layout`.
 */
#define FT_TT_ARTIFACT_VERSION 3u

/** The version a file without expert stacks or Bfp6_b is written at (see above). */
#define FT_TT_ARTIFACT_VERSION_DENSE 2u

/** The oldest artifact version a reader in this tree still understands. */
#define FT_TT_ARTIFACT_VERSION_MIN 1u

/** How a 2-D tensor's tiles are arranged. See the header note. */
typedef enum {
    FT_TT_LAYOUT_SOURCE = 0,       /**< GGUF [out x in], row-major tile grid */
    FT_TT_LAYOUT_MATMUL_NMAJOR = 1 /**< [in x out], column-major tile grid */
} ft_tt_tensor_layout;

/**
 * Header `flags` bits. A reader that meets a bit it does not know must refuse
 * the file rather than ignore it: every flag here changes what the BYTES MEAN.
 */
/** `attn_q`/`attn_k` output rows are permuted from INTERLEAVED to SPLIT-HALF. */
#define FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF 0x00000001u

/**
 * The rotary PAIR INDEX is reflected: the slot a consumer calls pair `p` holds
 * the model's pair `half - 1 - p`.
 *
 * **ORTHOGONAL to the bit above, and deliberately so.** The two bits answer two
 * independent questions and the four combinations are all meaningful:
 *
 * | flags | WHERE a pair's partner is | WHICH pair a slot holds |
 * |---|---|---|
 * | `0x0` | adjacent (GGUF interleaved) | `p` |
 * | `0x1` | `+half` (split-half) | `p` |
 * | `0x2` | adjacent (GGUF interleaved) | `half - 1 - p` |
 * | `0x3` | `+half` (split-half) | `half - 1 - p` |
 *
 * The consumer owes exactly one thing for this bit: apply frequency
 * `half - 1 - p` at slot `p`. Do that and the attention score is EXACTLY the
 * unreflected one — the sum over pairs is the same set of terms in a different
 * order — and fail to and every rotary feature gets the wrong frequency, which
 * is why this is a header bit and not a naming convention.
 */
#define FT_TT_ARTIFACT_FLAG_ROPE_PAIRS_REVERSED 0x00000002u

/** Every flag bit this build understands. Anything else is from the future. */
#define FT_TT_ARTIFACT_FLAGS_KNOWN                                                                            \
    (FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF | FT_TT_ARTIFACT_FLAG_ROPE_PAIRS_REVERSED)

/**
 * Which rotary pairing `attn_q`/`attn_k` are stored for.
 *
 * ---------------------------------------------------------------------------
 * WHY A WEIGHT PERMUTATION IS A ROPE DECISION
 * ---------------------------------------------------------------------------
 * A head's rotary pairing is a property of the CHECKPOINT, not of the kernel.
 * HF publishes Llama-family q/k projections in SPLIT-HALF order — feature `j`
 * pairs with `j + rot/2` — and llama.cpp's HF->GGUF conversion PERMUTES the
 * output rows into INTERLEAVED order, pairing `(2i, 2i+1)`, because that is
 * what its own `rope_norm` wants. So a GGUF `llama` checkpoint is a permuted
 * checkpoint, and that permutation is the only reason our on-card RoPE kernel
 * cannot serve it: `ft_tt_op_rope.h` refuses interleaved pairing because an
 * interleaved partner is an adjacent column INSIDE a face, where a split-half
 * partner (with `rot/2` a multiple of 32) is the SAME position in a different
 * TILE — a permutation of tiles and the identity inside one.
 *
 * Running llama.cpp's permutation backwards in the converter therefore turns a
 * kernel capability gap into a build-time row shuffle.
 *
 * **IT CHANGES NO MATH, AND THAT IS PROVABLE RATHER THAN ASSERTED.** Attention
 * consumes q and k only through `q . k`, which is invariant under a permutation
 * applied to BOTH — and the same one is, because it is a function of the head
 * geometry, not of the tensor. `v` is not permuted and `attn_output` is not
 * permuted, so nothing downstream of the score ever sees a reordered feature
 * axis.
 *
 * What DOES move, and it is worth stating rather than discovering: a
 * block-float format shares one exponent across 16 elements of a stored tile
 * ROW, which after the N-major transpose is 16 consecutive OUTPUT features.
 * Permuting the output rows regroups those 16, so a Bfp8_b artifact is not
 * bit-identical to its unpermuted twin — it is a different, equally valid
 * quantization of the same weights. At the bf16 and fp32 rungs there are no
 * blocks and the permutation is EXACTLY inert, which is the rung the unit test
 * pins it at.
 *
 * ---------------------------------------------------------------------------
 * THE CONSUMER HAS TO BE TOLD, AND THE ARTIFACT IS WHERE
 * ---------------------------------------------------------------------------
 * A permuted `attn_q` fed to an interleaved RoPE is silently wrong in exactly
 * the way this converter's `layout` field exists to prevent, so the choice is
 * recorded in the header as FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF and the host
 * forward selects its rotary form from the FLAG, not from the architecture.
 */
/**
 * ---------------------------------------------------------------------------
 * AND WHY THERE ARE TWO MORE THAT NO CHECKPOINT ASKS FOR
 * ---------------------------------------------------------------------------
 * The last two values are INSTRUMENTS, not checkpoint formats, and they exist to
 * answer one question with a control rather than an argument.
 *
 * `--rope-pairing split-half` took the chat-parity gate's worst document from
 * **1.227205 to 1.032823** with nothing else moving — the artifacts are the same
 * size, the same tensors, the same ledger, and the only difference is which 16
 * output features share a block-float exponent. Two artifacts is a sample of
 * two, so `SPLIT_HALF_REVERSED` is a THIRD draw over the SAME two tensors: the
 * pair index is reflected, so the exponent blocks regroup again while the
 * attention score stays exact (`q . k` sums one term per pair, and reflecting
 * the labelling permutes the summands without changing the set).
 *
 * `INTERLEAVED_REVERSED` is the crossed control that separates the two things
 * split-half changed at once. Split-half did two things: it moved a pair's
 * partner from ADJACENT to `+half`, and it therefore stopped a pair's two
 * components sharing an exponent block. This value reflects the pair index while
 * KEEPING the partner adjacent — a different draw of the same shape as source.
 * So the four values isolate the variable:
 *
 * | value | partner | exponent block holds |
 * |---|---|---|
 * | `SOURCE` | adjacent | both components of 8 pairs |
 * | `SPLIT_HALF` | `+half` | one component of 16 pairs |
 * | `SPLIT_HALF_REVERSED` | `+half` | one component of 16 other pairs |
 * | `INTERLEAVED_REVERSED` | adjacent | both components of 8 other pairs |
 *
 * Neither instrument is loadable by the on-card RoPE kernel: its table is built
 * with the frequency in slot order, and it refuses an adjacent partner outright.
 * The host forward refuses both by name rather than serving them wrong.
 */
typedef enum {
    FT_TT_ROPE_PAIRING_SOURCE = 0,     /**< store `attn_q`/`attn_k` exactly as GGUF has them */
    FT_TT_ROPE_PAIRING_SPLIT_HALF = 1, /**< undo llama.cpp's permutation; NeoX/split-half order */
    FT_TT_ROPE_PAIRING_SPLIT_HALF_REVERSED = 2, /**< split-half, pair index reflected. Instrument. */
    FT_TT_ROPE_PAIRING_INTERLEAVED_REVERSED = 3 /**< GGUF order, pair index reflected. Instrument. */
} ft_tt_rope_pairing;

/**
 * Which tensors of a checkpoint one artifact holds.
 *
 * A checkpoint can carry a trailing MTP / NextN block (`blk.L.nextn.eh_proj.weight` in its
 * LAST block, Qwen3.6-35B-A3B's block 40). The trunk never runs it, and the card runs it
 * from a SEPARATE artifact (`<trunk minus .fttw>.mtp.fttw`) whose names are `mtp.*`, so:
 *
 *   TRUNK  every tensor except the MTP block's. The default; a checkpoint with no MTP
 *          block converts exactly as before.
 *   MTP    ONLY the MTP block, renamed `blk.L.X` -> `mtp.X` with `nextn.` dropped, and
 *          `eh_proj` [2n -> n] split along its INPUT into `mtp.eh_proj_e` (columns
 *          [0, n), applied to enorm(embedding)) and `mtp.eh_proj_h` ([n, 2n), applied to
 *          hnorm(hidden)) — the names and split Pantheon's MTP head binds. Refused when
 *          the checkpoint has no MTP block.
 *   ALL    every tensor under its own name.
 */
typedef enum {
    FT_TT_CONVERT_SCOPE_TRUNK = 0,
    FT_TT_CONVERT_SCOPE_MTP = 1,
    FT_TT_CONVERT_SCOPE_ALL = 2
} ft_tt_convert_scope;

/** Tensor payloads start on this boundary; the NoC's DRAM alignment is 64 B. */
#define FT_TT_ARTIFACT_ALIGNMENT 64u

/** Name field width in a directory entry. Matches ft_gguf_map's limit. */
#define FT_TT_ARTIFACT_NAME_BYTES 128

/** On-disk artifact header. 40 bytes, no padding by construction. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t tensor_count;
    uint32_t flags;
    uint64_t directory_offset;
    uint64_t data_offset;
    uint64_t total_bytes;
} ft_tt_artifact_header;

/** On-disk directory entry. Fixed size, so the directory is indexable. */
typedef struct {
    char name[FT_TT_ARTIFACT_NAME_BYTES];
    uint32_t format;    /**< ft_tt_tensor_format */
    uint32_t role;      /**< ft_tt_tensor_role, for diagnostics */
    uint32_t n_dims;
    /** The STORED tile grid, not the logical one. For FT_TT_LAYOUT_MATMUL_NMAJOR that
     *  means tile_rows = Kt = ceil(in/32) and tile_cols = Nt = ceil(out/32), i.e. the
     *  transpose of what `dims` says. `dims` stays logical and in GGML order. */
    uint32_t tile_rows; /**< 0 when the tensor is stored untiled */
    uint32_t tile_cols;
    uint32_t layout;    /**< ft_tt_tensor_layout; always 0 in a v1 artifact */
    uint64_t dims[4];   /**< logical, GGML order */
    uint64_t data_offset;
    uint64_t byte_count;
} ft_tt_artifact_entry;

/** The two on-disk structures are a format, so their sizes are pinned. */
#define FT_TT_ARTIFACT_HEADER_BYTES 40u
#define FT_TT_ARTIFACT_ENTRY_BYTES 200u

/** Per-tensor result, reported as the conversion runs. */
typedef struct {
    char name[FT_TT_ARTIFACT_NAME_BYTES];
    ft_tt_tensor_role role;
    ft_tt_tensor_format format;
    uint64_t rows;
    uint64_t cols;
    uint64_t source_bytes;
    uint64_t converted_bytes;
    double pcc;      /**< against the fp32 source; 1.0 when the format is fp32 */
    double max_abs_error;
    double rms_error;
    /**
     * rms_error divided by the RMS of the SOURCE, i.e. the error as a fraction
     * of the signal. This, not `rms_error`, is the number that compares across
     * tensor types: a norm weight and an ffn_down live at different scales, so
     * their absolute RMS errors are not on the same axis and ranking them by it
     * ranks them by magnitude. 0.0 for an all-zero tensor.
     */
    double relative_rms_error;
    /**
     * Relative RMS of the SOURCE against the reference, when
     * ft_tt_convert_measure_against() supplied one. Zero otherwise.
     *
     * This is the half of a double transcode that the on-card format is not
     * responsible for: converting a Q8_0 checkpoint measures
     * `relative_rms_error` against the bf16 the model was released as, and this
     * field says how much of that was already spent before the packer ran.
     */
    double source_relative_rms_error;
} ft_tt_convert_tensor_report;

/** Called once per tensor, in directory order. Return non-zero to abort. */
typedef int (*ft_tt_convert_progress_fn)(const ft_tt_convert_tensor_report *report, void *user);

/** Options for one conversion run. */
typedef struct {
    ft_tt_ledger_mode mode;
    int measure_pcc; /**< round-trip every tensor and report PCC */
    ft_tt_convert_progress_fn progress;
    void *progress_user;
    ft_tt_rope_pairing rope_pairing; /**< default 0 = FT_TT_ROPE_PAIRING_SOURCE */
    /**
     * A MEASUREMENT INSTRUMENT, and deliberately not a ledger rung.
     *
     * Every 2-D tensor whose name CONTAINS this substring is encoded at
     * `format_override` instead of the format the ledger chose. NULL disables
     * it, and 1-D tensors (norms, biases) are never touched — a shared exponent
     * over a normalization scale is not what this measures.
     *
     * It exists to answer a question the ledger cannot express: what share of
     * the artifact's total squared error does ONE tensor family carry? Degrade
     * that family alone, read the residual-divergence plateau, and the share
     * inverts directly, because divergence is linear in injected error. Naming a
     * SUBSTRING rather than a role is the whole point — `attn_output` and
     * `attn_q` are one role, and the question is about one of them.
     *
     * It is not a rung because an artifact built this way must never be mistaken
     * for a shipping one: it has no name on the ladder, the converter announces
     * every tensor it moves, and `ft_tt_fttw_list` reads the real formats back
     * out of the written file.
     */
    const char *format_override_name_contains;
    ft_tt_tensor_format format_override; /**< read only when the substring is set */
    ft_tt_convert_scope scope;           /**< default 0 = FT_TT_CONVERT_SCOPE_TRUNK */
    /**
     * Per-tensor formats on top of `mode` (ft_tt_format_map.h); NULL = none. A listed
     * tensor is encoded at its map format instead of the ledger's, matched by its GGUF
     * SOURCE name. Checked against the checkpoint before the output is created: a name
     * the checkpoint lacks, or a tensor ft_tt_ledger_map_refusal() refuses, fails the
     * conversion with FT_TT_CONVERT_EFORMATMAP and no file. A listed tensor outside
     * `scope` is legal (one map serves the trunk and MTP runs) and noted on stderr.
     * Refused together with `format_override_name_contains`: two override paths over one
     * tensor have no single obvious answer. NULL leaves the output byte-identical to a
     * conversion that never had the field.
     */
    const ft_tt_format_map *format_map;
} ft_tt_convert_options;

/** Status codes. Negative is failure. */
enum {
    FT_TT_CONVERT_OK = 0,
    FT_TT_CONVERT_ERROR = -1,
    FT_TT_CONVERT_ENOMEM = -2,
    FT_TT_CONVERT_EINVAL = -3,
    FT_TT_CONVERT_EIO = -4,
    FT_TT_CONVERT_EUNSUPPORTED = -8,   /**< a tensor type this path cannot read */
    FT_TT_CONVERT_EUNCLASSIFIED = -9,  /**< a 2-D tensor whose role no rule recognises */
    FT_TT_CONVERT_EROPEGEOMETRY = -10, /**< rope pairing asked for, head geometry does not allow it */
    FT_TT_CONVERT_EEXPERTGEOMETRY = -11, /**< an expert stack whose N does not tile per expert */
    FT_TT_CONVERT_ENOMTP = -12,          /**< SCOPE_MTP on a checkpoint with no MTP block */
    FT_TT_CONVERT_EFORMATMAP = -13       /**< a format map names a missing or unmappable tensor */
};

/**
 * Which layout a tensor is stored in, from its role and rank — or a REFUSAL.
 *
 * Only a MATMUL WEIGHT is transposed. The distinction that matters is
 * `token_embd`, which is 2-D and looks exactly like a weight: it is indexed BY
 * ROW by a token id, so transposing it would turn every embedding lookup into
 * a strided gather across the whole tensor. `output`/`lm_head` IS a matmul
 * weight and is transposed, even though it shares the embedding's shape.
 *
 * **AN UNRECOGNISED 2-D TENSOR HAS NO ANSWER AND THIS FUNCTION SAYS SO.** It
 * used to return FT_TT_LAYOUT_SOURCE, on the reasoning that a transpose we have
 * not reasoned about is the change that would be silently wrong. That reasoning
 * was half right and the half it missed is the dangerous one: NOT transposing a
 * tensor the kernels read N-major is equally silent and equally wrong, and it
 * is what happened to `ssm_out` — a real 6144->5120 matmul weight — the first
 * time this classifier met a GDN model. There is no safe default because BOTH
 * defaults are a guess about a kernel's expectations. So the guess is refused
 * and the caller has to add a rule.
 *
 * @param role    from ft_tt_ledger_role()
 * @param n_dims  tensor rank, as GGUF declares it
 * @param out     receives FT_TT_LAYOUT_SOURCE or FT_TT_LAYOUT_MATMUL_NMAJOR;
 *                untouched when the role has no layout
 * @return        FT_TT_CONVERT_OK, or FT_TT_CONVERT_EUNCLASSIFIED for a 2-D
 *                tensor of unknown role
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_convert_layout_for_checked(ft_tt_tensor_role role, uint32_t n_dims, ft_tt_tensor_layout *out);

/**
 * The tensor a TIED checkpoint's LM head is: `token_embd.weight` when the checkpoint has no
 * `output.weight` (Qwen3-0.6B, Llama-3.2-1B, ...), else NULL. The converter writes it a
 * second time under the name `output.weight` with the LM-head role and matmul layout, so a
 * forward binds one name whether the head is tied or not; the embedding's own copy stays in
 * SOURCE layout for the row lookup. Counted in the artifact's tensor count.
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: no
 */
const ft_gguf_tensor *ft_tt_convert_tied_head_source(const ft_gguf_map *map);

/**
 * Convert a GGUF checkpoint to a Blackhole weight artifact.
 *
 * @param gguf_path    input checkpoint
 * @param output_path  artifact to create, truncating any existing file
 * @param options      may be NULL for defaults (production ledger, no PCC)
 * @return             FT_TT_CONVERT_OK, or a negative status
 *
 * Thread-safe: yes (no shared state)
 * Reentrant: yes
 * Signal-safe: no
 */
int ft_tt_convert_gguf(const char *gguf_path, const char *output_path, const ft_tt_convert_options *options);

/**
 * The checkpoint's MTP / NextN block: the LAST block, when it carries
 * `blk.L.nextn.eh_proj.weight` (the rule Pantheon's qwen35moe loader uses to drop it).
 *
 * @return 1 and `*block` = L when there is one; 0 otherwise (`*block` untouched)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_convert_mtp_block(const ft_gguf_map *map, uint32_t *block);

/**
 * Whether a tensor is a routed-expert STACK this converter lays out per expert: an
 * EXPERT_UP / EXPERT_DOWN role on a 3-D tensor. Those are the tensors format v3 exists for.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_convert_is_expert_stack(const ft_gguf_tensor *tensor);

/**
 * How many directory entries converting `map` under `options` writes (the tied head and
 * the MTP eh_proj split included). The cache compares a trailer against it.
 *
 * @return the count, or 0 for SCOPE_MTP on a checkpoint with no MTP block
 */
uint32_t ft_tt_convert_tensor_count_for(const ft_gguf_map *map, const ft_tt_convert_options *options);

/**
 * The FTTW version converting `map` under `options` writes: FT_TT_ARTIFACT_VERSION when
 * an in-scope tensor is an expert stack or lands at Bfp6_b, FT_TT_ARTIFACT_VERSION_DENSE
 * otherwise.
 */
uint32_t ft_tt_convert_artifact_version_for(const ft_gguf_map *map, const ft_tt_convert_options *options);

/**
 * Measure only: walk every tensor and report the PCC each candidate format
 * would give, writing nothing. This is the Phase 3 quality gate.
 *
 * @param gguf_path  input checkpoint
 * @param format     the candidate format to score every 2-D tensor at
 * @param progress   called once per tensor
 * @param user       passed through to `progress`
 * @return           FT_TT_CONVERT_OK, or a negative status
 */
int ft_tt_convert_measure(const char *gguf_path, ft_tt_tensor_format format, ft_tt_convert_progress_fn progress,
                          void *user);

/**
 * Measure, but score the round trip against a SECOND checkpoint rather than
 * against the input's own values.
 *
 * The question this answers is the one a lossy source raises and a self-scoring
 * gate cannot: converting a Q8_0 checkpoint and comparing the result to the
 * Q8_0 values reports the block-float error only, and quietly treats the Q8_0
 * quantization as if it were the truth. Point `reference_gguf_path` at the
 * BF16 release the model was published as and the report is against the actual
 * weights, with `source_relative_rms_error` splitting out how much of it the
 * source had already spent.
 *
 * The two checkpoints must agree on every tensor's name and shape; a tensor
 * missing from the reference, or a shape mismatch, is an error rather than a
 * skip — silently scoring 90% of a model is how a regression hides.
 *
 * @param gguf_path            the checkpoint being converted
 * @param reference_gguf_path  the truth to score against; NULL means self-score
 * @param format               the candidate format for every 2-D tensor
 * @return                     FT_TT_CONVERT_OK, or a negative status
 */
int ft_tt_convert_measure_against(const char *gguf_path, const char *reference_gguf_path,
                                  ft_tt_tensor_format format, ft_tt_convert_progress_fn progress, void *user);

#endif /* FT_TT_WEIGHT_CONVERT_H */
