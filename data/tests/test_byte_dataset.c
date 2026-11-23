/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_byte_dataset — verification for the byte-level enwik8 loader.
 *
 * Self-contained: builds small synthetic byte files in $TMPDIR so it needs no
 * network / no enwik8. If the real staged enwik8 is present at the experiment
 * repo's data dir it additionally sanity-checks the real corpus + splits.
 */
#include "../../lib/test.h"
#include "../fd_byte_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

/* Write `len` bytes produced by gen(i) to a fresh temp file; return path in
 * `out` (caller-owned static buffer). Returns 0/-1. */
static int write_tmp(const char *tag, size_t len,
                     unsigned char (*gen)(size_t), char *out, size_t out_sz) {
    snprintf(out, out_sz, "/tmp/fd_byte_%s_%d.bin", tag, (int)getpid());
    FILE *fp = fopen(out, "wb");
    if (!fp) return -1;
    unsigned char buf[65536];
    size_t written = 0;
    while (written < len) {
        size_t chunk = len - written;
        if (chunk > sizeof buf) chunk = sizeof buf;
        for (size_t i = 0; i < chunk; i++) buf[i] = gen(written + i);
        if (fwrite(buf, 1, chunk, fp) != chunk) { fclose(fp); return -1; }
        written += chunk;
    }
    fclose(fp);
    return 0;
}

static unsigned char gen_ramp(size_t i)   { return (unsigned char)(i & 0xFF); }

/* Region-marker generator for the leakage test (globals set per call). */
static size_t g_a_end, g_b_end;   /* [0,a_end)=A, [a_end,b_end)=B, [b_end,..)=C */
static unsigned char gen_regions(size_t i) {
    if (i < g_a_end) return (unsigned char)'A';
    if (i < g_b_end) return (unsigned char)'B';
    return (unsigned char)'C';
}

/* ---- window correctness: target = input shifted +1, matches the raw bytes ---- */
static void test_window_shift_and_bytes(void) {
    char path[256];
    ASSERT_INT_EQ(0, write_tmp("ramp", 100000, gen_ramp, path, sizeof path));

    fd_byte_corpus c;
    ASSERT_INT_EQ(0, fd_byte_corpus_open(&c, path));

    size_t seq = 128;
    fd_byte_split s;
    ASSERT_INT_EQ(0, fd_byte_split_init(&s, &c, 0, c.size, seq));
    ASSERT_TRUE(s.n_windows == c.size - seq);

    uint64_t seed = 0xABCDEF12345ULL, epoch = 0;
    int *in  = malloc(seq * sizeof(int));
    int *tg  = malloc(seq * sizeof(int));

    for (uint64_t step = 0; step < 50; step++) {
        ASSERT_INT_EQ(0, fd_byte_get_batch(&s, seed, epoch, step, 0, 1, 1, in, tg));
        size_t p = fd_byte_window_pos(&s, seed, epoch, step);   /* rank0,nr1,batch1 => ordinal=step */
        /* input/target match the underlying ramp bytes exactly */
        for (size_t j = 0; j < seq; j++) {
            ASSERT_INT_EQ((int)((p + j) & 0xFF),     in[j]);
            ASSERT_INT_EQ((int)((p + j + 1) & 0xFF), tg[j]);
        }
        /* target is input shifted by one within the row */
        for (size_t j = 0; j + 1 < seq; j++) ASSERT_INT_EQ(in[j + 1], tg[j]);
    }
    free(in); free(tg);
    fd_byte_corpus_close(&c);
    unlink(path);
}

/* ---- vocab is exactly 256 and every id is in range ---- */
static void test_vocab_range(void) {
    ASSERT_INT_EQ(256, FD_BYTE_VOCAB);

    char path[256];
    /* full 0..255 coverage so we know the range check is meaningful */
    ASSERT_INT_EQ(0, write_tmp("ramp2", 70000, gen_ramp, path, sizeof path));
    fd_byte_corpus c;
    ASSERT_INT_EQ(0, fd_byte_corpus_open(&c, path));

    size_t seq = 256, batch = 8;
    fd_byte_split s;
    ASSERT_INT_EQ(0, fd_byte_split_init(&s, &c, 0, c.size, seq));

    int *in = malloc(batch * seq * sizeof(int));
    int *tg = malloc(batch * seq * sizeof(int));
    int seen[256] = {0};
    for (uint64_t step = 0; step < 40; step++) {
        ASSERT_INT_EQ(0, fd_byte_get_batch(&s, 777ULL, 0, step, 0, 1, batch, in, tg));
        for (size_t k = 0; k < batch * seq; k++) {
            ASSERT_TRUE(in[k] >= 0 && in[k] < FD_BYTE_VOCAB);
            ASSERT_TRUE(tg[k] >= 0 && tg[k] < FD_BYTE_VOCAB);
            seen[in[k]] = 1;
        }
    }
    int distinct = 0; for (int i = 0; i < 256; i++) distinct += seen[i];
    ASSERT_INT_EQ(256, distinct);   /* every byte value actually observed */
    free(in); free(tg);
    fd_byte_corpus_close(&c);
    unlink(path);
}

/* ---- determinism: same seed ⇒ identical; different seed/epoch ⇒ different ---- */
static void test_determinism(void) {
    char path[256];
    ASSERT_INT_EQ(0, write_tmp("det", 200000, gen_ramp, path, sizeof path));
    fd_byte_corpus c;
    ASSERT_INT_EQ(0, fd_byte_corpus_open(&c, path));

    size_t seq = 64, batch = 4;
    fd_byte_split s;
    ASSERT_INT_EQ(0, fd_byte_split_init(&s, &c, 0, c.size, seq));

    size_t n = batch * seq;
    int *a_in = malloc(n*sizeof(int)), *a_tg = malloc(n*sizeof(int));
    int *b_in = malloc(n*sizeof(int)), *b_tg = malloc(n*sizeof(int));

    /* same seed/epoch/step/rank/num_ranks/batch across two independent handles */
    int diff_same = 0, diff_seed = 0, diff_epoch = 0, diff_rank = 0;
    for (uint64_t step = 0; step < 30; step++) {
        fd_byte_get_batch(&s, 12345ULL, 0, step, 0, 2, batch, a_in, a_tg);
        fd_byte_get_batch(&s, 12345ULL, 0, step, 0, 2, batch, b_in, b_tg);
        diff_same += memcmp(a_in,b_in,n*sizeof(int)) || memcmp(a_tg,b_tg,n*sizeof(int));

        fd_byte_get_batch(&s, 999ULL, 0, step, 0, 2, batch, b_in, b_tg);      /* diff seed */
        diff_seed += (memcmp(a_in,b_in,n*sizeof(int)) != 0);

        fd_byte_get_batch(&s, 12345ULL, 1, step, 0, 2, batch, b_in, b_tg);    /* diff epoch */
        diff_epoch += (memcmp(a_in,b_in,n*sizeof(int)) != 0);

        fd_byte_get_batch(&s, 12345ULL, 0, step, 1, 2, batch, b_in, b_tg);    /* diff rank */
        diff_rank += (memcmp(a_in,b_in,n*sizeof(int)) != 0);
    }
    ASSERT_INT_EQ(0, diff_same);       /* same params ⇒ byte-identical, always */
    ASSERT_TRUE(diff_seed  >= 29);     /* different seed ⇒ different stream */
    ASSERT_TRUE(diff_epoch >= 29);     /* different epoch ⇒ different stream */
    ASSERT_TRUE(diff_rank  >= 29);     /* different rank sees different windows */

    free(a_in); free(a_tg); free(b_in); free(b_tg);
    fd_byte_corpus_close(&c);
    unlink(path);
}

/* ---- no cross-split leakage: a split only ever reads its own bytes ---- */
static void test_no_leakage(void) {
    /* 30000 bytes: [0,18000)=A, [18000,24000)=B, [24000,30000)=C (90/5/5 shape) */
    size_t total = 30000, a = 18000, b = 24000;
    g_a_end = a; g_b_end = b;
    char path[256];
    ASSERT_INT_EQ(0, write_tmp("regions", total, gen_regions, path, sizeof path));
    fd_byte_corpus c;
    ASSERT_INT_EQ(0, fd_byte_corpus_open(&c, path));

    size_t seq = 128, batch = 4;
    fd_byte_split tr, va, te;
    ASSERT_INT_EQ(0, fd_byte_split_init(&tr, &c, 0,      a,           seq));
    ASSERT_INT_EQ(0, fd_byte_split_init(&va, &c, a,      b - a,       seq));
    ASSERT_INT_EQ(0, fd_byte_split_init(&te, &c, b,      total - b,   seq));

    size_t n = batch * seq;
    int *in = malloc(n*sizeof(int)), *tg = malloc(n*sizeof(int));
    struct { fd_byte_split *s; int marker; } cases[3] = {
        {&tr, 'A'}, {&va, 'B'}, {&te, 'C'}
    };
    for (int ci = 0; ci < 3; ci++) {
        int bad = 0;
        for (uint64_t step = 0; step < 200; step++) {
            fd_byte_get_batch(cases[ci].s, 55ULL, 0, step, 0, 1, batch, in, tg);
            for (size_t k = 0; k < n; k++)
                if (in[k] != cases[ci].marker || tg[k] != cases[ci].marker) bad++;
        }
        ASSERT_INT_EQ(0, bad);   /* never a byte from a neighbouring split */
    }
    free(in); free(tg);
    fd_byte_corpus_close(&c);
    unlink(path);
}

/* ---- throughput: report bytes/sec of window sampling (mmap should be fast) ---- */
static void test_throughput(void) {
    size_t total = 8u*1024u*1024u;   /* 8 MiB synthetic corpus */
    char path[256];
    ASSERT_INT_EQ(0, write_tmp("thru", total, gen_ramp, path, sizeof path));
    fd_byte_corpus c;
    ASSERT_INT_EQ(0, fd_byte_corpus_open(&c, path));

    size_t seq = 512, batch = 16;
    fd_byte_split s;
    ASSERT_INT_EQ(0, fd_byte_split_init(&s, &c, 0, c.size, seq));

    size_t n = batch * seq;
    int *in = malloc(n*sizeof(int)), *tg = malloc(n*sizeof(int));
    uint64_t steps = 2000;
    volatile long sink = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t step = 0; step < steps; step++) {
        fd_byte_get_batch(&s, 314159ULL, 0, step, 0, 1, batch, in, tg);
        sink += in[0] + tg[n-1];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    (void)sink;
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
    double bytes = (double)steps * (double)n;   /* input tokens sampled = bytes */
    printf("    [throughput] %.0f windows-bytes in %.4fs = %.1f MB/s (%.2fM windows/s)\n",
           bytes, secs, bytes/secs/1e6, (double)steps*batch/secs/1e6);
    ASSERT_TRUE(secs >= 0.0);   /* informational; just ensure it ran */

    free(in); free(tg);
    fd_byte_corpus_close(&c);
    unlink(path);
}

/* ---- optional: sanity-check the real staged enwik8 if present ---- */
static void test_real_enwik8_if_present(void) {
    /* FD_ENWIK8 names the corpus; otherwise data/enwik8 (gitignored) from the repo root. */
    const char *env_path = getenv("FD_ENWIK8");
    const char *cand = (env_path && env_path[0]) ? env_path : "data/enwik8";
    const char *path = (access(cand, R_OK) == 0) ? cand : NULL;

    if (!path) {
        printf("    [enwik8] not staged locally — skipping real-corpus check\n");
        return;
    }
    fd_byte_corpus c;
    ASSERT_INT_EQ(0, fd_byte_corpus_open(&c, path));
    ASSERT_TRUE(c.size == FD_ENWIK8_SIZE);   /* 100,000,000 bytes */

    size_t seq = 1024;
    fd_byte_split tr, va, te;
    ASSERT_INT_EQ(0, fd_byte_split_enwik8_train(&tr, &c, seq));
    ASSERT_INT_EQ(0, fd_byte_split_enwik8_val  (&va, &c, seq));
    ASSERT_INT_EQ(0, fd_byte_split_enwik8_test (&te, &c, seq));
    ASSERT_TRUE(tr.n_windows == FD_ENWIK8_TRAIN_LEN - seq);
    ASSERT_TRUE(va.n_windows == FD_ENWIK8_VAL_LEN   - seq);
    ASSERT_TRUE(te.n_windows == FD_ENWIK8_TEST_LEN  - seq);

    int *in = malloc(seq*sizeof(int)), *tg = malloc(seq*sizeof(int));
    ASSERT_INT_EQ(0, fd_byte_get_batch(&tr, 42ULL, 0, 0, 0, 1, 1, in, tg));
    for (size_t j = 0; j < seq; j++) {
        ASSERT_TRUE(in[j] >= 0 && in[j] < 256);
        ASSERT_TRUE(tg[j] >= 0 && tg[j] < 256);
        if (j + 1 < seq) ASSERT_INT_EQ(in[j+1], tg[j]);
    }
    printf("    [enwik8] real corpus OK: 100,000,000 bytes, splits 90M/5M/5M\n");
    free(in); free(tg);
    fd_byte_corpus_close(&c);
}

int main(void) {
    RUN_TEST(window_shift_and_bytes);
    RUN_TEST(vocab_range);
    RUN_TEST(determinism);
    RUN_TEST(no_leakage);
    RUN_TEST(throughput);
    RUN_TEST(real_enwik8_if_present);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
