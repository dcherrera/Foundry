/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_stream.c
 * @brief  Streaming: encode from a pipe and decode from a pipe (forward-only), truncated input,
 *         GGUF quant-delta files against a base read by pread and by pipe, base identity checks
 *         (header sha256, wrong base refused), and a fixed memory bound for `ingot decode` on a
 *         file much larger than the bound.
 */
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "lib/test.h"
#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/container/ingot_stream.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_ggml.h"
#include "ingot/tests/ingot_test_files.h"

#define DIR "build/test_data"

extern char **environ;

/* A read fd fed by a child process copying `path` into a pipe. *pid = the child. */
static int pipe_from_file(const char *path, pid_t *pid) {
    int fds[2];
    if (pipe(fds))
        return -1;
    *pid = fork();
    if (*pid == 0) {
        close(fds[0]);
        int in = open(path, O_RDONLY);
        char buf[1 << 16];
        ssize_t n = 0;
        while (in >= 0 && (n = read(in, buf, sizeof(buf))) > 0)
            if (write(fds[1], buf, (size_t)n) != n)
                break;
        _exit(0);
    }
    close(fds[1]);
    return fds[0];
}

static void reap(pid_t pid) {
    int status = 0;
    if (pid > 0)
        waitpid(pid, &status, 0);
}

/* A small Q8_0 GGUF: base model, and a tune with a fraction of blocks changed. */
static void write_pair(const char *base_path, const char *tune_path) {
    enum { R = 256, C = 2048, BYTES = R * C / 32 * 34 };
    test_rng = 21;
    uint8_t *a = test_q8_tensor(R, C, 0.02), *b = test_q8_tensor(R, C, 0.03);
    float norm[C];
    for (int i = 0; i < C; i++)
        norm[i] = (float)(1.0 + 0.1 * test_gauss());
    test_gguf_tensor t[3] = {
        {"blk.0.ffn_up.weight", INGOT_GGML_Q8_0, 2, {C, R}, a, BYTES},
        {"blk.0.ffn_norm.weight", INGOT_GGML_F32, 1, {C}, norm, sizeof(norm)},
        {"blk.0.ffn_down.weight", INGOT_GGML_Q8_0, 2, {C, R}, b, BYTES},
    };
    test_write_gguf(base_path, t, 3, 0);
    for (int blk = 0; blk < R * C / 32; blk += 50)
        b[blk * 34 + 3] ^= 0x01;
    norm[7] += 1.0f;
    test_write_gguf(tune_path, t, 3, 0);
    free(a);
    free(b);
}

static int encode_with_base(const char *src, const char *base, const char *dst,
                            ingot_encode_stats *st) {
    ingot_base_set *b = NULL;
    const char *paths[1] = {base};
    int rc = ingot_base_set_open(paths, 1, 2, &b);
    ingot_encode_options opt = {.threads = 2, .base = b};
    if (!rc)
        rc = ingot_encode_file(src, dst, &opt, st);
    ingot_base_set_close(b);
    return rc;
}

/* Decode `ingot` (a path, or through a pipe when piped) with an optional base path. */
static int decode(const char *ingot, int piped, const char *base, int check_base,
                  const char *out) {
    ingot_base_set *b = NULL;
    ingot_input *in = NULL;
    pid_t pid = -1;
    const char *paths[1] = {base};
    int rc = base ? ingot_base_set_open(paths, 1, 2, &b) : 0;
    if (!rc)
        rc = piped ? ingot_input_open_fd(pipe_from_file(ingot, &pid), 1, &in)
                   : ingot_input_open_path(ingot, &in);
    ingot_stream_options opt = {.threads = 2, .base = b, .check_base_sha256 = check_base};
    if (!rc)
        rc = ingot_stream_decode_to_file(in, out, &opt, NULL);
    ingot_input_close(in);
    ingot_base_set_close(b);
    reap(pid);
    return rc;
}

static void test_encode_and_decode_through_pipes(void) {
    mkdir(DIR, 0755);
    write_pair(DIR "/s_base.gguf", DIR "/s_tune.gguf");
    pid_t pid = -1;
    ingot_input *in = NULL;
    ASSERT_INT_EQ(0, ingot_input_open_fd(pipe_from_file(DIR "/s_base.gguf", &pid), 1, &in));
    ASSERT_INT_EQ(0, ingot_input_seekable(in));
    ingot_encode_stats st;
    ingot_encode_options opt = {.threads = 2};
    ASSERT_INT_EQ(0, ingot_encode_input(in, DIR "/s_base.gguf.ingot", &opt, &st));
    ingot_input_close(in);
    reap(pid);
    ASSERT_INT_EQ(2, (int)st.tensors[INGOT_MODE_QUANT]);
    ASSERT_INT_EQ(0, decode(DIR "/s_base.gguf.ingot", 1, NULL, 0, DIR "/s_base.out"));
    ASSERT_TRUE(test_files_equal(DIR "/s_base.gguf", DIR "/s_base.out"));
}

/* Red: a cut .ingot (any length short of the whole) fails as truncated; so does a cut source. */
static void test_truncated_input_fails(void) {
    long size = test_file_size(DIR "/s_base.gguf.ingot");
    long cuts[3] = {100, size / 2, size - 1};
    for (int i = 0; i < 3; i++) {
        ASSERT_INT_EQ(0, test_copy_flipped(DIR "/s_base.gguf.ingot", DIR "/cut.ingot", 0, 0,
                                           cuts[i]));
        int rc = decode(DIR "/cut.ingot", 1, NULL, 0, DIR "/cut.out");
        ASSERT_INT_EQ(-INGOT_ETRUNCATED, rc);
        ASSERT_TRUE(test_file_size(DIR "/cut.out") < 0); /* partial output removed */
    }
    ASSERT_INT_EQ(0, test_copy_flipped(DIR "/s_base.gguf", DIR "/cut.gguf", 0, 0,
                                       test_file_size(DIR "/s_base.gguf") / 2));
    pid_t pid = -1;
    ingot_input *in = NULL;
    ASSERT_INT_EQ(0, ingot_input_open_fd(pipe_from_file(DIR "/cut.gguf", &pid), 1, &in));
    ASSERT_INT_EQ(-INGOT_ETRUNCATED, ingot_encode_input(in, DIR "/cut.gguf.ingot", NULL, NULL));
    ingot_input_close(in);
    reap(pid);
}

static void test_quant_delta_with_file_and_pipe_base(void) {
    ingot_encode_stats st;
    ASSERT_INT_EQ(0, encode_with_base(DIR "/s_tune.gguf", DIR "/s_base.gguf",
                                      DIR "/s_tune.gguf.ingot", &st));
    ASSERT_INT_EQ(2, (int)st.tensors[INGOT_MODE_QUANT_DELTA]);
    ASSERT_INT_EQ(1, (int)st.tensors[INGOT_MODE_BLOCK_DELTA]);
    printf("  delta file: %llu of %llu bytes\n", (unsigned long long)st.output_bytes,
           (unsigned long long)st.source_bytes);
    ASSERT_TRUE(st.output_bytes * 10 < st.source_bytes);
    ingot_reader *r = NULL;
    ASSERT_INT_EQ(0, ingot_reader_open(DIR "/s_tune.gguf.ingot", &r));
    ASSERT_TRUE(ingot_header_has_base(ingot_reader_header(r)));
    ASSERT_TRUE(ingot_reader_header(r)->base_size == (uint64_t)test_file_size(DIR "/s_base.gguf"));
    ingot_reader_close(r);
    ASSERT_INT_EQ(0, decode(DIR "/s_tune.gguf.ingot", 0, DIR "/s_base.gguf", 1, DIR "/t.out"));
    ASSERT_TRUE(test_files_equal(DIR "/s_tune.gguf", DIR "/t.out"));
    remove(DIR "/t.out");
    /* base through a FIFO: read forward-only and hashed, header digest checked at the end */
    remove(DIR "/base.fifo");
    ASSERT_INT_EQ(0, mkfifo(DIR "/base.fifo", 0644));
    pid_t pid = fork();
    if (pid == 0) {
        int out = open(DIR "/base.fifo", O_WRONLY), in = open(DIR "/s_base.gguf", O_RDONLY);
        char buf[1 << 16];
        ssize_t n = 0;
        while ((n = read(in, buf, sizeof(buf))) > 0 && write(out, buf, (size_t)n) == n) {
        }
        _exit(0);
    }
    ASSERT_INT_EQ(0, decode(DIR "/s_tune.gguf.ingot", 1, DIR "/base.fifo", 0, DIR "/t.out"));
    reap(pid);
    ASSERT_TRUE(test_files_equal(DIR "/s_tune.gguf", DIR "/t.out"));
}

/* Red: no base, a different model as base, and a base that differs only outside the tensors
 * (caught by the header sha256 with --check-base) are all refused. */
static void test_wrong_base_refused(void) {
    ASSERT_INT_EQ(-INGOT_EBASE, decode(DIR "/s_tune.gguf.ingot", 0, NULL, 0, DIR "/w.out"));
    ASSERT_INT_EQ(-INGOT_EBASE, decode(DIR "/s_tune.gguf.ingot", 0, DIR "/s_tune.gguf", 0,
                                       DIR "/w.out"));
    /* flip a byte of the metadata value "ingot-test" (bytes 64..73) */
    ASSERT_INT_EQ(0, test_copy_flipped(DIR "/s_base.gguf", DIR "/s_base_meta.gguf", 66, 0x01,
                                       -1));
    ASSERT_INT_EQ(0, decode(DIR "/s_tune.gguf.ingot", 0, DIR "/s_base_meta.gguf", 0,
                            DIR "/w.out")); /* tensors equal: per-tensor checks pass */
    remove(DIR "/w.out");
    ASSERT_INT_EQ(-INGOT_EBASE, decode(DIR "/s_tune.gguf.ingot", 0, DIR "/s_base_meta.gguf", 1,
                                       DIR "/w.out"));
    ASSERT_TRUE(test_file_size(DIR "/w.out") < 0);
}

/* `ingot decode` of a 160+ MB file stays under a fixed RSS bound (a child process). */
static void test_decode_memory_bound(void) {
    enum { TENSORS = 20, R = 1024, C = 8192, BYTES = R * C / 32 * 34 };
    const uint64_t bound = 48ull << 20;
    test_gguf_tensor t[TENSORS];
    uint8_t *q = NULL;
    char names[TENSORS][32];
    test_rng = 31;
    q = test_q8_tensor(R, C, 0.02); /* the same blocks in every tensor: generation is slow */
    for (int i = 0; i < TENSORS; i++) {
        snprintf(names[i], sizeof(names[i]), "blk.%d.ffn_up.weight", i);
        t[i] = (test_gguf_tensor){names[i], INGOT_GGML_Q8_0, 2, {C, R}, q, BYTES};
    }
    ASSERT_INT_EQ(0, test_write_gguf(DIR "/big.gguf", t, TENSORS, 0));
    free(q);
    ingot_encode_options opt = {.threads = 4};
    ASSERT_INT_EQ(0, ingot_encode_file(DIR "/big.gguf", DIR "/big.gguf.ingot", &opt, NULL));
    remove(DIR "/big.out");
    char *cli = getenv("INGOT_CLI") ? getenv("INGOT_CLI") : "build/ingot";
    char *argv[] = {cli, "decode", "-q", "-j", "2", DIR "/big.gguf.ingot",
                    DIR "/big.out", NULL};
    pid_t pid = -1;
    ASSERT_INT_EQ(0, posix_spawn(&pid, argv[0], NULL, NULL, argv, environ));
    int status = 0;
    struct rusage ru;
    ASSERT_TRUE(wait4(pid, &status, 0, &ru) == pid);
    ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
#ifdef __APPLE__
    uint64_t maxrss = (uint64_t)ru.ru_maxrss;          /* bytes */
#else
    uint64_t maxrss = (uint64_t)ru.ru_maxrss * 1024u;  /* KiB */
#endif
    long out = test_file_size(DIR "/big.out");
    printf("  decode max RSS %.1f MiB for a %.1f MiB file (bound %.0f MiB)\n",
           (double)maxrss / 1048576.0, (double)out / 1048576.0, (double)bound / 1048576.0);
    ASSERT_TRUE(test_files_equal(DIR "/big.gguf", DIR "/big.out"));
    ASSERT_TRUE((uint64_t)out > 3 * bound);
    ASSERT_TRUE(maxrss < bound);
    remove(DIR "/big.out");
    remove(DIR "/big.gguf");
    remove(DIR "/big.gguf.ingot");
}

int main(void) {
    RUN_TEST(encode_and_decode_through_pipes);
    RUN_TEST(truncated_input_fails);
    RUN_TEST(quant_delta_with_file_and_pipe_base);
    RUN_TEST(wrong_base_refused);
    RUN_TEST(decode_memory_bound);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
