/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_pl_data.c
 * @brief Tests for potluck/pl_data: JSONL → tokens + loss-masked targets.
 *
 * Cases d1-d6 of the P1-data card (d7 = `make test-data`). Expectations are
 * literal strings decoded with rx_tokenizer, independent of token ids.
 * Needs the vocab: `bash rwkvx/data/fetch_rwkv_vocab.sh`. Run from the repo root.
 */
#include "lib/test.h"
#include "potluck/pl_data.h"
#include "rwkvx/rwkvx.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SCRATCH_DIR "build/potluck_test/data"
#define JSONL_PATH  SCRATCH_DIR "/in.jsonl"
#define VOCAB_PATH  "rwkvx/data/rwkv_vocab_v20230424.txt"
#define IGNORE_ID   65536

#define D1_LINE "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}," \
                "{\"role\":\"assistant\",\"content\":\"Hello there.\"}]}\n"

static rx_tokenizer *g_tok;
static int g_checks;

#define CHECK_INT(a, b) do { g_checks++; ASSERT_INT_EQ(a, b); } while (0)

/* ---- helpers ---- */

static void write_jsonl(const char *content) {
    FILE *f = fopen(JSONL_PATH, "w");
    if (!f) { perror(JSONL_PATH); exit(1); }
    fputs(content, f);
    fclose(f);
}

/* Write content and load it; returns pl_data_load_jsonl's rc. */
static int load(const char *content, pl_data_fmt fmt, int max_len, pl_example **exs, int *n,
                int *n_skipped) {
    write_jsonl(content);
    return pl_data_load_jsonl(JSONL_PATH, g_tok, fmt, max_len, IGNORE_ID, exs, n, n_skipped);
}

static char *decode_ids(const int *ids, int n) {
    uint32_t *u32 = malloc((size_t)(n > 0 ? n : 1) * sizeof(uint32_t));
    for (int i = 0; i < n; i++) u32[i] = (uint32_t)ids[i];
    char *text = n > 0 ? rx_tokenizer_decode(g_tok, u32, n) : NULL;
    free(u32);
    return text ? text : strdup("");
}

/* full = decode(tokens ++ last target token) */
static char *decode_full(const pl_example *ex) {
    int *ids = malloc((size_t)(ex->n + 1) * sizeof(int));
    memcpy(ids, ex->tokens, (size_t)ex->n * sizeof(int));
    ids[ex->n] = ex->targets[ex->n - 1];
    char *text = decode_ids(ids, ex->n + 1);
    free(ids);
    return text;
}

/* trained = decode of targets[t] over t with targets[t] != ignore_id */
static char *decode_trained(const pl_example *ex) {
    int *ids = malloc((size_t)ex->n * sizeof(int));
    int n = 0;
    for (int t = 0; t < ex->n; t++) if (ex->targets[t] != IGNORE_ID) ids[n++] = ex->targets[t];
    char *text = decode_ids(ids, n);
    free(ids);
    return text;
}

/* Load one record that must yield exactly one example; checks full/trained if non-NULL. */
static int check_one(const char *line, pl_data_fmt fmt, const char *want_full, const char *want_trained) {
    pl_example *exs = NULL; int n = 0, n_skipped = 0;
    int rc = load(line, fmt, 1000, &exs, &n, &n_skipped);
    int ok = rc == 0 && n == 1 && n_skipped == 0;
    if (!ok) printf("    load rc=%d n=%d skipped=%d\n", rc, n, n_skipped);
    if (ok && want_full) {
        char *full = decode_full(&exs[0]);
        if (strcmp(full, want_full) != 0) { printf("    full = \"%s\"\n", full); ok = 0; }
        free(full);
    }
    if (ok && want_trained) {
        char *trained = decode_trained(&exs[0]);
        if (strcmp(trained, want_trained) != 0) { printf("    trained = \"%s\"\n", trained); ok = 0; }
        free(trained);
    }
    pl_data_free(exs, n);
    return ok;
}

/* ---- d1-d4: literal decodes ---- */

static void test_d1_chat_single_turn(void) {
    CHECK_INT(1, check_one(D1_LINE, PL_FMT_CHAT, "User: Hi\n\nAssistant: Hello there.\n\n",
                           " Hello there.\n\n"));
}

static void test_d2_chat_two_turns(void) {
    const char *line = "{\"messages\":[{\"role\":\"user\",\"content\":\"Q1\"},"
                       "{\"role\":\"assistant\",\"content\":\"A1\"},"
                       "{\"role\":\"user\",\"content\":\"Q2\"},"
                       "{\"role\":\"assistant\",\"content\":\"A2\"}]}\n";
    CHECK_INT(1, check_one(line, PL_FMT_CHAT, NULL, " A1\n\n A2\n\n"));
}

static void test_d3_chat_escapes(void) {
    const char *line = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},"
                       "{\"role\":\"assistant\",\"content\":\"a \\\"b\\\"\\nc\"}]}\n";
    CHECK_INT(1, check_one(line, PL_FMT_CHAT, NULL, " a \"b\"\nc\n\n"));
}

static void test_d4_alpaca(void) {
    const char *line = "{\"instruction\":\"Add 2 and 3.\",\"input\":\"\",\"output\":\"5\"}\n";
    CHECK_INT(1, check_one(line, PL_FMT_ALPACA, "User: Add 2 and 3.\n\nAssistant: 5\n\n", " 5\n\n"));
}

/* ---- d5: text = plain shifted next tokens; truncation ---- */

static void test_d5_text_shift_and_truncate(void) {
    const char *line = "{\"text\":\"Hello world. Goodbye.\"}\n";
    uint32_t ids[64];
    int L = rx_tokenizer_encode(g_tok, "Hello world. Goodbye.", ids, 64);
    pl_example *exs = NULL; int n = 0, n_skipped = 0;
    CHECK_INT(0, load(line, PL_FMT_TEXT, 1000, &exs, &n, &n_skipped));
    CHECK_INT(1, n);
    CHECK_INT(L - 1, exs[0].n);
    for (int t = 0; t + 1 < exs[0].n; t++) CHECK_INT(exs[0].tokens[t + 1], exs[0].targets[t]);
    CHECK_INT((int)ids[L - 1], exs[0].targets[exs[0].n - 1]);
    pl_data_free(exs, n);

    CHECK_INT(1, L > 5);
    CHECK_INT(0, load(line, PL_FMT_TEXT, 4, &exs, &n, &n_skipped));
    CHECK_INT(1, n);
    CHECK_INT(4, exs[0].n);
    for (int t = 0; t < 4; t++) {
        CHECK_INT((int)ids[t], exs[0].tokens[t]);
        CHECK_INT((int)ids[t + 1], exs[0].targets[t]);
    }
    pl_data_free(exs, n);
}

/* ---- d6: public mapping, auto, errors, skips ---- */

static void test_d6_fmt_parse(void) {
    CHECK_INT(0, pl_data_fmt_parse("auto"));
    CHECK_INT(1, pl_data_fmt_parse("text"));
    CHECK_INT(2, pl_data_fmt_parse("alpaca"));
    CHECK_INT(3, pl_data_fmt_parse("chat"));
    CHECK_INT(-1, pl_data_fmt_parse("json"));
}

static void test_d6_errors_and_skips(void) {
    pl_example *exs = NULL; int n = 0, n_skipped = 0;
    CHECK_INT(-1, load("{\"foo\":\"bar\"}\n", PL_FMT_AUTO, 1000, &exs, &n, &n_skipped));
    CHECK_INT(-1, load("{\"messages\":[{\"role\":\"tool\",\"content\":\"x\"}]}\n", PL_FMT_CHAT, 1000,
                       &exs, &n, &n_skipped));
    CHECK_INT(-1, load("{\"instruction\":\"x\"}\n", PL_FMT_ALPACA, 1000, &exs, &n, &n_skipped));
    /* no assistant turn = no trained target; one-token text = < 2 tokens */
    CHECK_INT(0, load("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}\n{\"text\":\"a\"}\n"
                      D1_LINE, PL_FMT_AUTO, 1000, &exs, &n, &n_skipped));
    CHECK_INT(1, n);
    CHECK_INT(2, n_skipped);
    pl_data_free(exs, n);
}

static void test_d6_auto_matches_chat(void) {
    pl_example *a = NULL, *c = NULL; int na = 0, nc = 0, sa = 0, sc = 0;
    CHECK_INT(0, load(D1_LINE, PL_FMT_AUTO, 1000, &a, &na, &sa));
    CHECK_INT(0, load(D1_LINE, PL_FMT_CHAT, 1000, &c, &nc, &sc));
    CHECK_INT(1, na);
    CHECK_INT(1, nc);
    CHECK_INT(c[0].n, a[0].n);
    for (int t = 0; t < a[0].n; t++) {
        CHECK_INT(c[0].tokens[t], a[0].tokens[t]);
        CHECK_INT(c[0].targets[t], a[0].targets[t]);
    }
    pl_data_free(a, na);
    pl_data_free(c, nc);
}

int main(void) {
    printf("\n  pl_data tests\n\n");
    if (system("rm -rf " SCRATCH_DIR " && mkdir -p " SCRATCH_DIR) != 0) return 1;
    g_tok = rx_tokenizer_new(VOCAB_PATH);
    if (!g_tok) {
        printf("  cannot load %s (run: bash rwkvx/data/fetch_rwkv_vocab.sh)\n", VOCAB_PATH);
        return 1;
    }
    RUN_TEST(d1_chat_single_turn);
    RUN_TEST(d2_chat_two_turns);
    RUN_TEST(d3_chat_escapes);
    RUN_TEST(d4_alpaca);
    RUN_TEST(d5_text_shift_and_truncate);
    RUN_TEST(d6_fmt_parse);
    RUN_TEST(d6_errors_and_skips);
    RUN_TEST(d6_auto_matches_chat);
    rx_tokenizer_free(g_tok);
    if (system("rm -rf " SCRATCH_DIR) != 0) return 1;
    TEST_REPORT();
    printf("  %d checks\n\n", g_checks);
    return (__ft_failed > 0 || g_checks == 0) ? 1 : 0;
}
