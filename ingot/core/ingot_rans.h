/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_rans.h
 * @brief  Static 16-bit rANS: frequency tables (build, cost, serialize, parse), decode slot tables,
 *         and the 32-lane interleaved block encoder. Decoding is inlined by each codec.
 *
 * State x is 32-bit in [2^16, 2^32); renormalization moves one 16-bit word. Symbol i of a block
 * uses lane i % 32. The encoder runs i = n-1 .. 0 and the word stream is stored reversed, so the
 * decoder runs i = 0 .. n-1 and reads words forward: after decoding symbol i on its lane, if
 * x < 2^16 then x = (x << 16) | next word. Every lane ends back at 2^16 (a corruption check).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_RANS_H
#define FOUNDRY_INGOT_RANS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    INGOT_RANS_PROB_BITS = 16,
    INGOT_RANS_LANES = 32,
    INGOT_RANS_MAX_ALPHABET = 1024,
};

#define INGOT_RANS_TOTAL (1u << INGOT_RANS_PROB_BITS)
#define INGOT_RANS_LOW (1u << 16)

/** Frequency table: freq sums to INGOT_RANS_TOTAL over used symbols (or used == 0, empty). */
typedef struct {
    uint32_t alphabet;
    uint32_t used;
    uint32_t freq[INGOT_RANS_MAX_ALPHABET];
    uint32_t start[INGOT_RANS_MAX_ALPHABET];
} ingot_rans_table;

/**
 * Quantize symbol counts to a table: every counted symbol gets freq >= 1; the rest of the total
 * is handed out by largest fractional part. All counts zero gives an empty table.
 *
 * @param t         Output table
 * @param counts    [alphabet] symbol counts
 * @param alphabet  1..INGOT_RANS_MAX_ALPHABET
 * @return          0, or -INGOT_EINVAL (bad alphabet, or more used symbols than the total)
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_rans_table_build(ingot_rans_table *t, const uint64_t *counts, uint32_t alphabet);

/**
 * Exact coded size in bits of `counts` under table t: sum count * (16 - log2 freq). Excludes the
 * table itself and block framing.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
double ingot_rans_table_coded_bits(const ingot_rans_table *t, const uint64_t *counts);

/**
 * Serialized size: u16 entry count + used x (u16 symbol, u16 freq - 1).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_rans_table_size(const ingot_rans_table *t);

/**
 * Serialize t into dst (ingot_rans_table_size(t) bytes, little-endian).
 * @return bytes written
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_rans_table_write(const ingot_rans_table *t, uint8_t *dst);

/**
 * Parse a serialized table and validate it: symbols strictly ascending and < alphabet, freqs
 * summing to exactly INGOT_RANS_TOTAL (or zero entries).
 *
 * @param t         Output table
 * @param alphabet  Alphabet the table must fit
 * @param src       Serialized bytes
 * @param avail     Bytes available at src
 * @param consumed  Out: bytes the table occupied
 * @return          0 or -INGOT_EFORMAT
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_rans_table_read(ingot_rans_table *t, uint32_t alphabet, const uint8_t *src,
                          size_t avail, size_t *consumed);

/**
 * Fill a slot -> symbol map (INGOT_RANS_TOTAL entries) for decoding with table t (non-empty).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_rans_slots_build(const ingot_rans_table *t, uint16_t *slots);

/** One symbol to encode: the table it uses (NULL = position has no symbol) and its value. */
typedef struct {
    const ingot_rans_table *table;
    uint32_t symbol;
} ingot_rans_symbol;

/**
 * Encode one block's symbols with 32 interleaved lanes.
 *
 * @param syms        [n] symbols in decode order (entries with table == NULL are skipped)
 * @param n           Positions in the block
 * @param states      Out: [INGOT_RANS_LANES] decoder start states
 * @param words       Out: word stream in decoder read order; capacity >= n
 * @param word_count  Out: words written
 * @return            0, or -INGOT_EINVAL if a symbol has freq 0 in its table
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_rans_encode_block(const ingot_rans_symbol *syms, size_t n, uint32_t *states,
                            uint16_t *words, size_t *word_count);

/**
 * Decode one symbol on a lane: updates *x, may read one word. Returns the symbol, or -1 if a word
 * is needed but none is left.
 */
static inline int32_t ingot_rans_decode(uint32_t *x, const ingot_rans_table *t,
                                        const uint16_t *slots, const uint16_t *words,
                                        size_t word_count, size_t *wp) {
    uint32_t slot = *x & (INGOT_RANS_TOTAL - 1u);
    uint32_t s = slots[slot];
    uint32_t v = t->freq[s] * (*x >> INGOT_RANS_PROB_BITS) + slot - t->start[s];
    if (v < INGOT_RANS_LOW) {
        if (*wp >= word_count)
            return -1;
        v = (v << 16) | words[(*wp)++];
    }
    *x = v;
    return (int32_t)s;
}

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_RANS_H */
