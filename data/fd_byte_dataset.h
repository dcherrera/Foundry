/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_BYTE_DATASET_H
#define FOUNDRY_BYTE_DATASET_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * fd_byte_dataset — byte-level, mmap'd next-byte-prediction data loader.
 *
 * Built for RWKV-x byte-level training experiments. The corpus is a raw
 * byte file (enwik8); a BYTE IS A TOKEN, so the vocabulary is exactly 256 and
 * there is no tokenizer / no BPE. The file is mmap'd PROT_READ (no full-file
 * load; pages fault in on demand), and windows are sampled directly out of the
 * mapping.
 *
 * Sampling (next-byte prediction), for a window whose start byte is p:
 *     input [j] = bytes[p + j]        for j in [0, seq_len)
 *     target[j] = bytes[p + j + 1]    for j in [0, seq_len)   (input shifted +1)
 * so a window consumes seq_len+1 contiguous bytes and the last valid start is
 * (length - seq_len - 1). The number of valid window starts is:
 *     n_windows = length - seq_len
 *
 * DETERMINISM (mandatory for the reproducible A/B). The order in which window
 * start-positions are visited is a seeded permutation derived ONLY from
 * (seed, epoch); which windows land at a given (step, rank) is derived from
 * (step, rank, num_ranks, batch). Therefore:
 *     same (seed, epoch, step, rank, num_ranks, batch) ⇒ byte-identical windows.
 * The permutation is a cycle-walking Feistel network over the window-index
 * domain (a true bijection on [0, n_windows) for any n_windows), so no
 * O(n_windows) shuffle array is materialised. seed == FD_BYTE_SEED_UNSET (0) is
 * rejected (fail-loud) so an unseeded run can never silently be nondeterministic.
 */

/* Vocabulary is exactly 256: every byte value 0..255 is its own token id. */
#define FD_BYTE_VOCAB 256

/* Reserved sentinel: an explicit seed of 0 means "unseeded" and is rejected. */
#define FD_BYTE_SEED_UNSET 0ULL

/* Standard enwik8 split (bytes). first 90M train, next 5M val, last 5M test. */
#define FD_ENWIK8_SIZE       100000000UL
#define FD_ENWIK8_TRAIN_OFF          0UL
#define FD_ENWIK8_TRAIN_LEN   90000000UL
#define FD_ENWIK8_VAL_OFF     90000000UL
#define FD_ENWIK8_VAL_LEN      5000000UL
#define FD_ENWIK8_TEST_OFF    95000000UL
#define FD_ENWIK8_TEST_LEN     5000000UL

/* mmap'd byte corpus (the whole file, read-only). */
typedef struct {
    int            fd;         /* open file descriptor (kept for munmap accounting) */
    const uint8_t *data;       /* mmap base, PROT_READ */
    size_t         size;       /* file size in bytes */
} fd_byte_corpus;

/* A view onto one split of the corpus: a contiguous [offset, offset+length)
 * byte range plus the window length. Never reads outside its own range. */
typedef struct {
    const uint8_t *base;       /* corpus->data + offset (window byte 0 lives here) */
    size_t         offset;     /* byte offset of this split within the corpus */
    size_t         length;     /* split length in bytes */
    size_t         seq_len;    /* window length (tokens per input/target row) */
    size_t         n_windows;  /* # valid window starts = length - seq_len */
} fd_byte_split;

/* ------------------------------------------------------------------ */
/* Corpus (mmap)                                                      */
/* ------------------------------------------------------------------ */

/* mmap the byte file read-only. Returns 0 on success, -1 on failure
 * (errno set; nothing allocated). */
int  fd_byte_corpus_open(fd_byte_corpus *c, const char *path);

/* munmap + close. Safe on a zeroed / already-closed struct. */
void fd_byte_corpus_close(fd_byte_corpus *c);

/* ------------------------------------------------------------------ */
/* Splits                                                             */
/* ------------------------------------------------------------------ */

/* Initialise an arbitrary split view. Fails (-1) if [offset, offset+length)
 * exceeds the corpus, or if length <= seq_len (no full window fits). */
int fd_byte_split_init(fd_byte_split *s, const fd_byte_corpus *c,
                       size_t offset, size_t length, size_t seq_len);

/* Convenience: the three standard enwik8 splits from the offsets above.
 * Each requires the corpus to be exactly FD_ENWIK8_SIZE bytes. Returns 0/-1. */
int fd_byte_split_enwik8_train(fd_byte_split *s, const fd_byte_corpus *c, size_t seq_len);
int fd_byte_split_enwik8_val  (fd_byte_split *s, const fd_byte_corpus *c, size_t seq_len);
int fd_byte_split_enwik8_test (fd_byte_split *s, const fd_byte_corpus *c, size_t seq_len);

/* ------------------------------------------------------------------ */
/* Deterministic window sampling                                      */
/* ------------------------------------------------------------------ */

/* Map an ordinal position in the per-epoch stream to a window START BYTE
 * (relative to the split base). This is the seeded permutation applied to the
 * ordinal (mod n_windows). Deterministic in (seed, epoch, ordinal). Aborts if
 * seed == FD_BYTE_SEED_UNSET. Returned p satisfies 0 <= p < n_windows, so the
 * window [p, p+seq_len] stays strictly inside the split (no cross-split read). */
size_t fd_byte_window_pos(const fd_byte_split *s,
                          uint64_t seed, uint64_t epoch, uint64_t ordinal);

/* Fill one (batch, seq_len) minibatch of int token ids for this split.
 *   inputs / targets : caller-allocated, each batch*seq_len ints.
 *   inputs [b*seq_len + j] = byte[p_b + j]
 *   targets[b*seq_len + j] = byte[p_b + j + 1]
 * where p_b is the window start for global ordinal
 *   ordinal = (step * num_ranks + rank) * batch + b
 * dealt through the (seed, epoch) permutation. Every id is in [0, 256).
 * Returns 0 on success, -1 on bad args. Aborts (fail-loud) if unseeded. */
int fd_byte_get_batch(const fd_byte_split *s,
                      uint64_t seed, uint64_t epoch, uint64_t step,
                      uint32_t rank, uint32_t num_ranks, uint32_t batch,
                      int *inputs, int *targets);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_BYTE_DATASET_H */
