/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_fttw.h
 * @brief The FTTW loader: mmap a converted weight artifact, find a tensor, put it on the
 *   card. This is the Phase 3 follow-up "the artifact has no loader yet".
 *
 * ---------------------------------------------------------------------------
 * WHY A TENSOR UPLOAD IS A BYTE COPY
 * ---------------------------------------------------------------------------
 * The converter already did the work: a 2-D tensor is `ceil(rows/32) x ceil(cols/32)`
 * 32x32 tiles in row-major TILE order, face-ordered inside each tile, in the on-card
 * format, each payload 64-byte aligned — which is the NoC's DRAM alignment.
 * A Bfp8_b tile is the same 1088 bytes the FPU consumes
 * and the same 1088 bytes a CB page holds. So the loader neither converts nor re-lays-out
 * anything; it streams bytes from an mmap into DRAM. That is Route B's payoff on the host
 * side, and it is why this file is short.
 *
 * The one tensor class that is NOT a byte copy is the 1-D one. Norm weights are stored
 * untiled fp32 (they are 6 KiB each and block-floating a normalization scale buys
 * nothing), while the RMS-norm kernel wants them as full 32x32 tiles with the row
 * replicated 32 times — so ft_tt_fttw_upload_broadcast_rows() does that expansion here,
 * in the loader, rather than inflating the artifact 32x or spending a broadcast op in the
 * kernel every row block. That settles the open Phase 3 follow-up on 1-D layout.
 *
 * ---------------------------------------------------------------------------
 * ORIENTATION
 * ---------------------------------------------------------------------------
 * The artifact stores GGUF's order: rows = dims[1] = out_features, cols = dims[0] =
 * in_features. The matmul wants its weight operand as K x N = in x out, i.e. transposed.
 * THE LOADER DOES NOT TRANSPOSE, and cannot usefully: a Bfp8_b tile's shared exponent
 * covers 16 elements along a row, so transposing a packed tile means unpacking and
 * repacking it, which is conversion work in the wrong place. The transpose belongs in the
 * converter. See ft_tt_op_matmul.h for the decision and its reasons.
 *
 * @status Reads the production MiniCPM5-1B artifact and uploads real tensors to DRAM.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_FTTW_H
#define FT_TT_FTTW_H

#include <stddef.h>
#include <stdint.h>

#include "convert/ft_tt_weight_convert.h"
#include "ft_tt_launch.h"

/** An mmapped artifact. Caller-owned storage, like everything else in this backend. */
typedef struct ft_tt_fttw {
    const unsigned char *base; /**< the mapping */
    size_t bytes;
    ft_tt_artifact_header header;
    const ft_tt_artifact_entry *directory; /**< into the mapping; little-endian on disk */
    int fd;
} ft_tt_fttw;

/**
 * Map an artifact and validate its header and directory bounds.
 *
 * Validation is not a formality: the directory is written LAST by a streaming converter,
 * so a truncated artifact has a plausible header and a directory that is not there.
 *
 * @return FT_TT_OK; FT_TT_EIO if it cannot be opened or mapped; FT_TT_EINVAL for a bad
 *         magic, an unsupported version, or a directory that does not fit the file
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_fttw_open(const char *path, ft_tt_fttw *out_artifact);

/** Unmap and close. Safe on NULL and on a half-opened artifact. */
void ft_tt_fttw_close(ft_tt_fttw *artifact);

/** The directory entry for `name`, or NULL. Linear scan; 219 entries. */
const ft_tt_artifact_entry *ft_tt_fttw_find(const ft_tt_fttw *artifact, const char *name);

/** Bytes of one 32x32 tile in `format` (ft_tt_tensor_format), or 0 if unmodelled. */
uint32_t ft_tt_fttw_tile_bytes(uint32_t format);

/**
 * Experts in an entry: `dims[2]` for a routed-expert stack (format v3: role EXPERT_UP /
 * EXPERT_DOWN, 3-D), 1 for every other tensor.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_fttw_expert_count(const ft_tt_artifact_entry *entry);

/**
 * Bytes from one expert's first tile to the next's: `tile_rows * tile_cols * tile_bytes`.
 * In a v3 artifact an expert stack's grid is ONE expert's, so expert e of the stack is the
 * N-major K x N weight at `data_offset + e * stride`, and the card addresses it as
 * `base + idx * stride`. Every tile size is a multiple of 64 B, so every expert is
 * DRAM-aligned. For a non-expert tensor it is the whole tensor's bytes.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t ft_tt_fttw_expert_stride(const ft_tt_artifact_entry *entry);

/**
 * Copy a tensor's payload to `address` on the NoC endpoint `(x, y)` — normally a DRAM
 * bank from ft_tt_coord_dram_bank(), never a guessed coordinate (spec §7.3).
 *
 * @param chunk_bytes  how much to move per call into the accessor; 0 picks 64 KiB. The
 *                     accessor re-aims its window as the address moves, so this bounds
 *                     the host-side staging, not the transfer size
 * @return             FT_TT_OK, or the accessor's failure
 */
int ft_tt_fttw_upload(const ft_tt_fttw *artifact, const ft_tt_artifact_entry *entry,
                      const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t address,
                      uint32_t chunk_bytes);

/**
 * Copy PART of a tensor's payload — the byte run `[source_offset, source_offset+source_bytes)`
 * — to `address` on `(x, y)`.
 *
 * This is what makes a load-time BANK SHARD a copy rather than a re-tiling: an N-major
 * weight's column slice is an already-contiguous run of the canonical bytes
 * (ft_tt_dram_shard.h), so placing member c's slice in its own bank is exactly one call of
 * this. Nothing here knows about members or tiles; the caller does that arithmetic and this
 * refuses a run that leaves the tensor.
 *
 * @param source_offset  byte offset into THIS TENSOR's payload
 * @param source_bytes   how many; 0 is refused rather than treated as "the rest"
 * @return               FT_TT_OK; FT_TT_EINVAL for a NULL argument or a run outside the
 *                       tensor; otherwise the accessor's failure
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_fttw_upload_range(const ft_tt_fttw *artifact, const ft_tt_artifact_entry *entry,
                            uint64_t source_offset, uint64_t source_bytes,
                            const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            uint32_t address, uint32_t chunk_bytes);

/**
 * Upload a 1-D fp32 tensor as `ceil(n/32)` 32x32 tiles with the row replicated down every
 * row of every tile, converted to bf16 (Float16_b) — the layout a norm weight needs to be
 * a plain `mul_tiles` operand rather than a broadcast op.
 *
 * The expansion is 32x in bytes and that is the point: 1536 fp32 values become 48 tiles
 * of 2048 B, 96 KiB, which is nothing against L1's 1.5 MiB and saves a per-tile broadcast
 * reconfiguration in the innermost loop of every norm.
 *
 * Tiles are face-ordered, via ft_tt_tilize_matrix_strip(), so the same code lays them out
 * as the converter uses for everything else.
 *
 * @param scratch        caller-owned, at least 32*32*2 bytes per tile it will build; the
 *                       loader builds one tile at a time, so 2048 bytes is enough
 * @param scratch_bytes  its size
 * @return               FT_TT_OK; FT_TT_EINVAL if the entry is not 1-D fp32 or the
 *                       scratch is short
 */
int ft_tt_fttw_upload_broadcast_rows(const ft_tt_fttw *artifact,
                                     const ft_tt_artifact_entry *entry,
                                     const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                     uint32_t address, void *scratch, size_t scratch_bytes);

#endif /* FT_TT_FTTW_H */
