/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_tensor_codec.h
 * @brief  One block decoder over every coded mode (plain, delta, quant, quant delta, block
 *         delta, predict), so the random-access reader and the streaming decoder share one
 *         decode path.
 *
 * Version 2 and 3 payloads start with a frame: u64 prefix_bytes (frame + mode prefix, where the
 * block area starts), u64 block_count. Version 1 payloads have no frame (plain and delta only).
 * A predict payload needs its partner tensors (earlier entries) bound before any block.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_TENSOR_CODEC_H
#define FOUNDRY_INGOT_TENSOR_CODEC_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/container/ingot_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ingot_tensor_decoder ingot_tensor_decoder;

/**
 * Open a decoder on a mode prefix (after the frame, if any).
 *
 * @param e             Directory entry (mode, dtype, shape, size); must outlive the decoder
 * @param prefix        Mode prefix bytes
 * @param avail         Bytes available at prefix
 * @param out           Out: decoder, closed with ingot_tensor_decoder_close
 * @param prefix_bytes  Out: bytes of the mode prefix (the block area follows)
 * @return              0, -INGOT_EFORMAT (bad prefix, or raw mode), -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_tensor_decoder_open(const ingot_entry *e, const uint8_t *prefix, uint64_t avail,
                              ingot_tensor_decoder **out, uint64_t *prefix_bytes);

/**
 * Block count; the byte range [lo, hi) of block b in the tensor; whether blocks need the base
 * tensor's bytes of the same range.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_tensor_decoder_blocks(const ingot_tensor_decoder *d);
void ingot_tensor_decoder_range(const ingot_tensor_decoder *d, size_t b, uint64_t *lo,
                                uint64_t *hi);
int ingot_tensor_decoder_needs_base(const ingot_tensor_decoder *d);

/**
 * Partner entries a predict payload needs (directory indices, at most
 * INGOT_PREDICT_MAX_PARTNERS, written to entries); 0 for other modes.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint32_t ingot_tensor_decoder_partners(const ingot_tensor_decoder *d, uint32_t *entries);

/**
 * Bind partner j's decoded bytes (entry pe; data must outlive the block decodes).
 * @return 0 or -INGOT_EFORMAT (not a BF16 tensor of the declared shape)
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: yes
 */
int ingot_tensor_decoder_bind(ingot_tensor_decoder *d, uint32_t j, const ingot_entry *pe,
                              const uint8_t *data);

/**
 * Decode block b into out (hi - lo bytes, 2-aligned); base covers the same range (delta modes).
 *
 * @return 0, -INGOT_EFORMAT, -INGOT_ECORRUPT, -INGOT_EBASE or -INGOT_ENOMEM
 *
 * Thread-safe: yes (distinct out buffers)   Reentrant: yes   Signal-safe: no
 */
int ingot_tensor_decoder_block(const ingot_tensor_decoder *d, size_t b,
                               const ingot_block_view *v, const uint8_t *base, uint8_t *out);

/**
 * Close (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_tensor_decoder_close(ingot_tensor_decoder *d);

/** A partner tensor of a predict payload: its entry and decoded bytes. */
typedef struct {
    const ingot_entry *entry;
    const uint8_t *data;
} ingot_partner_data;

/**
 * Decode a whole coded payload held in memory (random access).
 *
 * @param e        Entry (coded mode)
 * @param payload  Payload bytes
 * @param size     Payload size
 * @param framed   1 for version 2 and 3 payloads, 0 for version 1
 * @param base     The base tensor's bytes (delta modes), else NULL
 * @param partners Predict mode: each partner, in the payload's order
 *                 (ingot_tensor_decoder_partners), else NULL
 * @param threads  Worker threads
 * @param out      e->source_size bytes, 2-aligned
 * @return         0 or an error as ingot_tensor_decoder_block
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_tensor_decode_payload(const ingot_entry *e, const uint8_t *payload, uint64_t size,
                                int framed, const uint8_t *base,
                                const ingot_partner_data *partners, int threads, uint8_t *out);

/**
 * Rows and columns of an entry for the row-based modes (cols = last dimension).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_entry_rows_cols(const ingot_entry *e, uint64_t *rows, uint64_t *cols);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_TENSOR_CODEC_H */
