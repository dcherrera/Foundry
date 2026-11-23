/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_TOKENIZER_H
#define FOUNDRY_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fast text tokenization with BPE support.
 * Replaces HuggingFace tokenizers.
 */

typedef struct ft_tokenizer ft_tokenizer;

/* ============================================================= */
/* Tokenizer Creation                                            */
/* ============================================================= */

/* Load tokenizer from HuggingFace-style JSON file */
ft_tokenizer* ft_load(const char* path);

/* Load BPE tokenizer from vocab and merges files */
ft_tokenizer* ft_load_bpe(const char* vocab_path, const char* merges_path);

/* Create empty tokenizer (for manual construction) */
ft_tokenizer* ft_new(void);

/* Free tokenizer */
void ft_free(ft_tokenizer* tok);

/* ============================================================= */
/* Vocabulary Management                                         */
/* ============================================================= */

/* Add token to vocabulary, returns token ID */
int ft_add_token(ft_tokenizer* tok, const char* token);

/* Add BPE merge rule */
void ft_add_merge(ft_tokenizer* tok, const char* a, const char* b);

/* Get vocabulary size */
int ft_vocab_size(ft_tokenizer* tok);

/* Token <-> ID conversion */
const char* ft_id_to_token(ft_tokenizer* tok, int id);
int ft_token_to_id(ft_tokenizer* tok, const char* token);

/* ============================================================= */
/* Special Tokens                                                */
/* ============================================================= */

/* Set special tokens */
void ft_set_unk_token(ft_tokenizer* tok, const char* token);
void ft_set_bos_token(ft_tokenizer* tok, const char* token);
void ft_set_eos_token(ft_tokenizer* tok, const char* token);
void ft_set_pad_token(ft_tokenizer* tok, const char* token);

/* Get special token IDs (-1 if not set) */
int ft_unk_id(ft_tokenizer* tok);
int ft_bos_id(ft_tokenizer* tok);
int ft_eos_id(ft_tokenizer* tok);
int ft_pad_id(ft_tokenizer* tok);

/* ============================================================= */
/* Model Type (BPE vs Unigram/SentencePiece)                     */
/* ============================================================= */

/* Set tokenizer model type: 0=BPE (default), 1=UNIGRAM */
void ft_set_model_type(ft_tokenizer *tok, int type);
int ft_get_model_type(ft_tokenizer *tok);

/* BPE pre-tokenizer family code (tokenizer.ggml.pre); consumer-defined,
 * 0 = default/unset. Selects which regex pre-tokenization scanner to use. */
void ft_set_pre_type(ft_tokenizer *tok, int type);
int ft_get_pre_type(ft_tokenizer *tok);

/* Set unigram scores (copies the array). Also computes max_token_len. */
void ft_set_scores(ft_tokenizer *tok, const float *scores, int n);

/* ============================================================= */
/* Encoding                                                      */
/* ============================================================= */

/* Encode text to token IDs
 * Returns array of token IDs (caller must free)
 * Sets *length to number of tokens */
int* ft_encode(ft_tokenizer* tok, const char* text, int* length);

/* Encode with special tokens (BOS/EOS) */
int* ft_encode_with_special(ft_tokenizer* tok, const char* text, int* length,
                            bool add_bos, bool add_eos);

/* Batch encode multiple texts
 * Returns array of int arrays (caller must free each and the outer array)
 * Sets lengths[i] to length of each encoding */
int** ft_encode_batch(ft_tokenizer* tok, const char** texts, int n, int** lengths);

/* ============================================================= */
/* Decoding                                                      */
/* ============================================================= */

/* Decode token IDs to text (caller must free) */
char* ft_decode(ft_tokenizer* tok, int* ids, int length);

/* Decode skipping special tokens */
char* ft_decode_skip_special(ft_tokenizer* tok, int* ids, int length);

/* ============================================================= */
/* Merge Rules                                                   */
/* ============================================================= */

/* Get number of BPE merge rules */
int ft_num_merges(ft_tokenizer* tok);

/* Get merge rule by index. Sets *a and *b to the token strings.
 * Pointers are valid until tokenizer is freed. */
void ft_get_merge(ft_tokenizer* tok, int index, const char **a, const char **b);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Print tokenizer info */
void ft_print(ft_tokenizer* tok);

/* Print tokens with IDs (for debugging) */
void ft_print_tokens(ft_tokenizer* tok, int* ids, int length);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_TOKENIZER_H */
