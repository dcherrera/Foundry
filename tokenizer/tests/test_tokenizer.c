/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "../../lib/test.h"
#include "../tokenizer.h"

#include <string.h>

/* ============================================================= */
/* Creation Tests                                                */
/* ============================================================= */

static void test_create_tokenizer(void) {
    ft_tokenizer* tok = ft_new();
    ASSERT_NOT_NULL(tok);
    ASSERT_INT_EQ(0, ft_vocab_size(tok));
    ft_free(tok);
}

static void test_add_tokens(void) {
    ft_tokenizer* tok = ft_new();

    int id1 = ft_add_token(tok, "hello");
    int id2 = ft_add_token(tok, "world");

    ASSERT_INT_EQ(0, id1);
    ASSERT_INT_EQ(1, id2);
    ASSERT_INT_EQ(2, ft_vocab_size(tok));

    ft_free(tok);
}

static void test_duplicate_token_returns_same_id(void) {
    ft_tokenizer* tok = ft_new();

    int id1 = ft_add_token(tok, "hello");
    int id2 = ft_add_token(tok, "hello");

    ASSERT_INT_EQ(id1, id2);
    ASSERT_INT_EQ(1, ft_vocab_size(tok));

    ft_free(tok);
}

/* ============================================================= */
/* Token Lookup Tests                                            */
/* ============================================================= */

static void test_id_to_token(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "hello");
    ft_add_token(tok, "world");

    ASSERT_STR_EQ("hello", ft_id_to_token(tok, 0));
    ASSERT_STR_EQ("world", ft_id_to_token(tok, 1));
    ASSERT_NULL(ft_id_to_token(tok, 999));

    ft_free(tok);
}

static void test_token_to_id(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "hello");
    ft_add_token(tok, "world");

    ASSERT_INT_EQ(0, ft_token_to_id(tok, "hello"));
    ASSERT_INT_EQ(1, ft_token_to_id(tok, "world"));
    ASSERT_INT_EQ(-1, ft_token_to_id(tok, "unknown"));

    ft_free(tok);
}

/* ============================================================= */
/* Special Token Tests                                           */
/* ============================================================= */

static void test_special_tokens(void) {
    ft_tokenizer* tok = ft_new();

    /* Initially all -1 */
    ASSERT_INT_EQ(-1, ft_unk_id(tok));
    ASSERT_INT_EQ(-1, ft_bos_id(tok));
    ASSERT_INT_EQ(-1, ft_eos_id(tok));
    ASSERT_INT_EQ(-1, ft_pad_id(tok));

    /* Set special tokens */
    ft_set_unk_token(tok, "<unk>");
    ft_set_bos_token(tok, "<s>");
    ft_set_eos_token(tok, "</s>");
    ft_set_pad_token(tok, "<pad>");

    ASSERT_TRUE(ft_unk_id(tok) >= 0);
    ASSERT_TRUE(ft_bos_id(tok) >= 0);
    ASSERT_TRUE(ft_eos_id(tok) >= 0);
    ASSERT_TRUE(ft_pad_id(tok) >= 0);

    /* Verify tokens are in vocab */
    ASSERT_STR_EQ("<unk>", ft_id_to_token(tok, ft_unk_id(tok)));
    ASSERT_STR_EQ("<s>", ft_id_to_token(tok, ft_bos_id(tok)));

    ft_free(tok);
}

static void test_unk_for_unknown_tokens(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "hello");
    ft_set_unk_token(tok, "<unk>");

    /* Known token returns its ID */
    ASSERT_INT_EQ(0, ft_token_to_id(tok, "hello"));

    /* Unknown token returns UNK ID */
    ASSERT_INT_EQ(ft_unk_id(tok), ft_token_to_id(tok, "unknown"));

    ft_free(tok);
}

/* ============================================================= */
/* Encoding Tests                                                */
/* ============================================================= */

static void test_encode_simple(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "h");
    ft_add_token(tok, "e");
    ft_add_token(tok, "l");
    ft_add_token(tok, "o");

    int length;
    int* ids = ft_encode(tok, "hello", &length);

    ASSERT_NOT_NULL(ids);
    ASSERT_INT_EQ(5, length);  /* h e l l o */
    ASSERT_INT_EQ(0, ids[0]);  /* h */
    ASSERT_INT_EQ(1, ids[1]);  /* e */
    ASSERT_INT_EQ(2, ids[2]);  /* l */
    ASSERT_INT_EQ(2, ids[3]);  /* l */
    ASSERT_INT_EQ(3, ids[4]);  /* o */

    free(ids);
    ft_free(tok);
}

static void test_encode_empty(void) {
    ft_tokenizer* tok = ft_new();

    int length;
    int* ids = ft_encode(tok, "", &length);

    ASSERT_NOT_NULL(ids);
    ASSERT_INT_EQ(0, length);

    free(ids);
    ft_free(tok);
}

static void test_encode_with_bpe_merge(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "h");
    ft_add_token(tok, "e");
    ft_add_token(tok, "l");
    ft_add_token(tok, "o");
    ft_add_token(tok, "he");  /* Merged token */
    ft_add_token(tok, "ll");  /* Merged token */

    /* Add BPE merges */
    ft_add_merge(tok, "h", "e");
    ft_add_merge(tok, "l", "l");

    int length;
    int* ids = ft_encode(tok, "hello", &length);

    ASSERT_NOT_NULL(ids);
    ASSERT_INT_EQ(3, length);  /* "he" "ll" "o" */
    ASSERT_INT_EQ(4, ids[0]);  /* he */
    ASSERT_INT_EQ(5, ids[1]);  /* ll */
    ASSERT_INT_EQ(3, ids[2]);  /* o */

    free(ids);
    ft_free(tok);
}

static void test_encode_with_special_tokens(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "h");
    ft_add_token(tok, "i");
    ft_set_bos_token(tok, "<s>");
    ft_set_eos_token(tok, "</s>");

    int length;
    int* ids = ft_encode_with_special(tok, "hi", &length, true, true);

    ASSERT_NOT_NULL(ids);
    ASSERT_INT_EQ(4, length);  /* <s> h i </s> */
    ASSERT_INT_EQ(ft_bos_id(tok), ids[0]);
    ASSERT_INT_EQ(0, ids[1]);  /* h */
    ASSERT_INT_EQ(1, ids[2]);  /* i */
    ASSERT_INT_EQ(ft_eos_id(tok), ids[3]);

    free(ids);
    ft_free(tok);
}

/* ============================================================= */
/* Decoding Tests                                                */
/* ============================================================= */

static void test_decode_simple(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "hel");
    ft_add_token(tok, "lo");

    int ids[] = {0, 1};
    char* text = ft_decode(tok, ids, 2);

    ASSERT_NOT_NULL(text);
    ASSERT_STR_EQ("hello", text);

    free(text);
    ft_free(tok);
}

static void test_decode_empty(void) {
    ft_tokenizer* tok = ft_new();

    char* text = ft_decode(tok, NULL, 0);

    ASSERT_NOT_NULL(text);
    ASSERT_STR_EQ("", text);

    free(text);
    ft_free(tok);
}

static void test_decode_skip_special(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "hello");
    ft_set_bos_token(tok, "<s>");
    ft_set_eos_token(tok, "</s>");

    int ids[] = {ft_bos_id(tok), 0, ft_eos_id(tok)};
    char* text = ft_decode_skip_special(tok, ids, 3);

    ASSERT_NOT_NULL(text);
    ASSERT_STR_EQ("hello", text);

    free(text);
    ft_free(tok);
}

static void test_roundtrip(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "h");
    ft_add_token(tok, "e");
    ft_add_token(tok, "l");
    ft_add_token(tok, "o");
    ft_add_token(tok, " ");

    const char* original = "hello";

    int length;
    int* ids = ft_encode(tok, original, &length);
    char* decoded = ft_decode(tok, ids, length);

    ASSERT_STR_EQ(original, decoded);

    free(ids);
    free(decoded);
    ft_free(tok);
}

/* ============================================================= */
/* Batch Encoding Tests                                          */
/* ============================================================= */

static void test_encode_batch(void) {
    ft_tokenizer* tok = ft_new();
    ft_add_token(tok, "a");
    ft_add_token(tok, "b");
    ft_add_token(tok, "c");

    const char* texts[] = {"ab", "abc"};
    int* lengths;
    int** ids = ft_encode_batch(tok, texts, 2, &lengths);

    ASSERT_NOT_NULL(ids);
    ASSERT_NOT_NULL(lengths);

    ASSERT_INT_EQ(2, lengths[0]);
    ASSERT_INT_EQ(3, lengths[1]);

    ASSERT_INT_EQ(0, ids[0][0]);  /* a */
    ASSERT_INT_EQ(1, ids[0][1]);  /* b */

    ASSERT_INT_EQ(0, ids[1][0]);  /* a */
    ASSERT_INT_EQ(1, ids[1][1]);  /* b */
    ASSERT_INT_EQ(2, ids[1][2]);  /* c */

    free(ids[0]);
    free(ids[1]);
    free(ids);
    free(lengths);
    ft_free(tok);
}

/* ============================================================= */
/* Edge Cases                                                    */
/* ============================================================= */

static void test_null_safety(void) {
    ASSERT_INT_EQ(0, ft_vocab_size(NULL));
    ASSERT_NULL(ft_id_to_token(NULL, 0));
    ASSERT_INT_EQ(-1, ft_token_to_id(NULL, "test"));
    ASSERT_INT_EQ(-1, ft_unk_id(NULL));

    ft_free(NULL);  /* Should not crash */
    ASSERT_TRUE(1);
}

static void test_unicode_chars(void) {
    ft_tokenizer* tok = ft_new();

    /* Add some unicode characters */
    ft_add_token(tok, "é");
    ft_add_token(tok, "ñ");
    ft_add_token(tok, "日");

    ASSERT_INT_EQ(3, ft_vocab_size(tok));

    int length;
    int* ids = ft_encode(tok, "éñ", &length);

    ASSERT_INT_EQ(2, length);
    ASSERT_INT_EQ(0, ids[0]);  /* é */
    ASSERT_INT_EQ(1, ids[1]);  /* ñ */

    free(ids);
    ft_free(tok);
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(void) {
    printf("\n  Running tokenizer tests...\n\n");

    /* Creation tests */
    RUN_TEST(create_tokenizer);
    RUN_TEST(add_tokens);
    RUN_TEST(duplicate_token_returns_same_id);

    /* Lookup tests */
    RUN_TEST(id_to_token);
    RUN_TEST(token_to_id);

    /* Special token tests */
    RUN_TEST(special_tokens);
    RUN_TEST(unk_for_unknown_tokens);

    /* Encoding tests */
    RUN_TEST(encode_simple);
    RUN_TEST(encode_empty);
    RUN_TEST(encode_with_bpe_merge);
    RUN_TEST(encode_with_special_tokens);

    /* Decoding tests */
    RUN_TEST(decode_simple);
    RUN_TEST(decode_empty);
    RUN_TEST(decode_skip_special);
    RUN_TEST(roundtrip);

    /* Batch tests */
    RUN_TEST(encode_batch);

    /* Edge cases */
    RUN_TEST(null_safety);
    RUN_TEST(unicode_chars);

    TEST_REPORT();

    return __ft_failed > 0 ? 1 : 0;
}
