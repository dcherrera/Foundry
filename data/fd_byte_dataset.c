/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * fd_byte_dataset — byte-level mmap'd next-byte-prediction loader.
 * See fd_byte_dataset.h for the contract.
 */
#include "fd_byte_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------ */
/* splitmix64 — cheap, well-mixed PRNG for deriving round keys.        */
/* ------------------------------------------------------------------ */
static inline uint64_t fd_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* ------------------------------------------------------------------ */
/* Cycle-walking Feistel permutation over [0, n_windows).             */
/*                                                                    */
/* We build a balanced Feistel network over 2*hbits bits (a domain of */
/* size 2^(2*hbits) >= n_windows). Feistel with a bijective round is a */
/* bijection on that power-of-two domain; to get a bijection on the    */
/* exact range [0, n) we "cycle-walk": re-encrypt until the output     */
/* falls below n. This yields a true permutation of [0, n) for any n,  */
/* with no O(n) shuffle table, and is fully determined by the keys.    */
/* ------------------------------------------------------------------ */

#define FD_FEISTEL_ROUNDS 4

typedef struct {
    uint32_t hbits;                    /* bits per half */
    uint64_t hmask;                    /* (1<<hbits)-1 */
    uint64_t domain;                   /* 1 << (2*hbits) */
    uint64_t key[FD_FEISTEL_ROUNDS];   /* per-round keys */
} fd_feistel;

static void fd_feistel_init(fd_feistel *f, uint64_t n, uint64_t seed, uint64_t epoch) {
    /* smallest even bit-width whose domain covers n (n>=1). */
    uint32_t bits = 1;
    while ((1ULL << bits) < n) bits++;
    if (bits & 1u) bits++;             /* make it even so halves are equal */
    if (bits < 2) bits = 2;
    f->hbits  = bits / 2u;
    f->hmask  = (1ULL << f->hbits) - 1ULL;
    f->domain = 1ULL << bits;
    uint64_t s = fd_splitmix64(seed ^ 0xD1B54A32D192ED03ULL);
    s = fd_splitmix64(s ^ epoch);
    for (int r = 0; r < FD_FEISTEL_ROUNDS; r++) {
        s = fd_splitmix64(s + 0x9E3779B97F4A7C15ULL);
        f->key[r] = s;
    }
}

/* One pass of the Feistel network over the power-of-two domain. */
static uint64_t fd_feistel_encrypt(const fd_feistel *f, uint64_t x) {
    uint64_t L = (x >> f->hbits) & f->hmask;
    uint64_t R = x & f->hmask;
    for (int r = 0; r < FD_FEISTEL_ROUNDS; r++) {
        uint64_t Fr = fd_splitmix64((R + f->key[r]) ^ (uint64_t)r) & f->hmask;
        uint64_t nL = R;
        uint64_t nR = L ^ Fr;
        L = nL;
        R = nR;
    }
    return (L << f->hbits) | R;
}

/* Permute x in [0, n) -> a distinct value in [0, n). */
static uint64_t fd_feistel_permute(const fd_feistel *f, uint64_t n, uint64_t x) {
    uint64_t y = fd_feistel_encrypt(f, x);
    /* cycle-walk: guaranteed to land in [0,n) within finitely many steps
     * because the map is a bijection on the (>= n) power-of-two domain. */
    while (y >= n) y = fd_feistel_encrypt(f, y);
    return y;
}

/* ------------------------------------------------------------------ */
/* Corpus                                                             */
/* ------------------------------------------------------------------ */

int fd_byte_corpus_open(fd_byte_corpus *c, const char *path) {
    if (!c || !path) { errno = EINVAL; return -1; }
    c->fd = -1; c->data = NULL; c->size = 0;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return -1; }

    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { close(fd); return -1; }

    /* Advise the kernel we will stream/random-access; harmless if unsupported. */
#ifdef MADV_RANDOM
    madvise(m, (size_t)st.st_size, MADV_RANDOM);
#endif

    c->fd   = fd;
    c->data = (const uint8_t *)m;
    c->size = (size_t)st.st_size;
    return 0;
}

void fd_byte_corpus_close(fd_byte_corpus *c) {
    if (!c) return;
    if (c->data && c->size) munmap((void *)c->data, c->size);
    if (c->fd >= 0) close(c->fd);
    c->fd = -1; c->data = NULL; c->size = 0;
}

/* ------------------------------------------------------------------ */
/* Splits                                                             */
/* ------------------------------------------------------------------ */

int fd_byte_split_init(fd_byte_split *s, const fd_byte_corpus *c,
                       size_t offset, size_t length, size_t seq_len) {
    if (!s || !c || !c->data) return -1;
    if (seq_len == 0) return -1;
    if (offset > c->size || length > c->size - offset) return -1; /* no overflow */
    if (length <= seq_len) return -1;                             /* need seq_len+1 */

    s->base      = c->data + offset;
    s->offset    = offset;
    s->length    = length;
    s->seq_len   = seq_len;
    s->n_windows = length - seq_len;   /* last start = length-seq_len-1 */
    return 0;
}

int fd_byte_split_enwik8_train(fd_byte_split *s, const fd_byte_corpus *c, size_t seq_len) {
    if (!c || c->size != FD_ENWIK8_SIZE) return -1;
    return fd_byte_split_init(s, c, FD_ENWIK8_TRAIN_OFF, FD_ENWIK8_TRAIN_LEN, seq_len);
}
int fd_byte_split_enwik8_val(fd_byte_split *s, const fd_byte_corpus *c, size_t seq_len) {
    if (!c || c->size != FD_ENWIK8_SIZE) return -1;
    return fd_byte_split_init(s, c, FD_ENWIK8_VAL_OFF, FD_ENWIK8_VAL_LEN, seq_len);
}
int fd_byte_split_enwik8_test(fd_byte_split *s, const fd_byte_corpus *c, size_t seq_len) {
    if (!c || c->size != FD_ENWIK8_SIZE) return -1;
    return fd_byte_split_init(s, c, FD_ENWIK8_TEST_OFF, FD_ENWIK8_TEST_LEN, seq_len);
}

/* ------------------------------------------------------------------ */
/* Sampling                                                           */
/* ------------------------------------------------------------------ */

size_t fd_byte_window_pos(const fd_byte_split *s,
                          uint64_t seed, uint64_t epoch, uint64_t ordinal) {
    if (seed == FD_BYTE_SEED_UNSET) {
        fprintf(stderr,
            "fd_byte_dataset: FATAL — window sampled with seed=0 (unseeded). "
            "The reproducible A/B requires an explicit nonzero seed.\n");
        abort();
    }
    fd_feistel f;
    fd_feistel_init(&f, s->n_windows, seed, epoch);
    return (size_t)fd_feistel_permute(&f, s->n_windows, ordinal % s->n_windows);
}

int fd_byte_get_batch(const fd_byte_split *s,
                      uint64_t seed, uint64_t epoch, uint64_t step,
                      uint32_t rank, uint32_t num_ranks, uint32_t batch,
                      int *inputs, int *targets) {
    if (!s || !s->base || !inputs || !targets) return -1;
    if (num_ranks == 0 || batch == 0 || rank >= num_ranks) return -1;
    if (seed == FD_BYTE_SEED_UNSET) {
        fprintf(stderr,
            "fd_byte_dataset: FATAL — get_batch called with seed=0 (unseeded). "
            "The reproducible A/B requires an explicit nonzero seed.\n");
        abort();
    }

    /* One Feistel context for the whole (seed, epoch) permutation. */
    fd_feistel f;
    fd_feistel_init(&f, s->n_windows, seed, epoch);

    const uint64_t seq   = (uint64_t)s->seq_len;
    const uint64_t nwin  = (uint64_t)s->n_windows;
    const uint64_t base_ord = (step * (uint64_t)num_ranks + (uint64_t)rank) * (uint64_t)batch;

    for (uint32_t b = 0; b < batch; b++) {
        uint64_t ordinal = (base_ord + b) % nwin;
        uint64_t p = fd_feistel_permute(&f, nwin, ordinal);   /* window start byte */
        const uint8_t *w = s->base + p;
        int *in  = inputs  + (size_t)b * s->seq_len;
        int *tg  = targets + (size_t)b * s->seq_len;
        for (uint64_t j = 0; j < seq; j++) {
            in[j] = (int)w[j];
            tg[j] = (int)w[j + 1];   /* p+seq <= n_windows-1+seq = length-1: in range */
        }
    }
    return 0;
}
