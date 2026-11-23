/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pl_data.h
 * @brief JSONL (text / alpaca / chat) → tokens + loss-masked targets.
 *
 * Reads JSONL files, parses each line according to a format (text, alpaca,
 * chat, or auto-detect), tokenizes the text segments using an rx_tokenizer,
 * and produces loss-masked training examples: each example is a sequence of
 * tokens where targets[t] is either the next token (if it should be trained)
 * or ignore_id (if it should be skipped).
 */

#ifndef FOUNDRY_PL_DATA_H
#define FOUNDRY_PL_DATA_H

#include <stdint.h>

/* Forward declaration — the tokenizer lives in rwkvx/rwkvx.h */
typedef struct rx_tokenizer rx_tokenizer;

/** Data format for JSONL records. */
typedef enum {
    PL_FMT_AUTO = 0,    /**< auto-detect: messages → chat, instruction → alpaca, text → text */
    PL_FMT_TEXT = 1,    /**< {"text": "…"} */
    PL_FMT_ALPACA = 2,  /**< {"instruction": "…", "input": "…", "output": "…"} */
    PL_FMT_CHAT = 3     /**< {"messages": [{"role": "…", "content": "…"}, …]} */
} pl_data_fmt;

/** One training example: tokens[0..n) and per-position loss mask (targets). */
typedef struct {
    int     n;          /**< number of tokens (also number of targets) */
    int    *tokens;     /**< token ids, length n */
    int    *targets;    /**< targets[t] = next-token id or ignore_id if masked */
} pl_example;

/**
 * Parse a format name string into an enum value.
 * @param name "auto", "text", "alpaca", or "chat" (case-sensitive).
 * @return enum value, or −1 on unknown name.
 */
int pl_data_fmt_parse(const char *name);

/**
 * Load and process a JSONL file into loss-masked training examples.
 *
 * Each line is parsed as a JSON object, split into prompt / trained segments
 * per @p fmt (RWKV chat format "User: …\n\nAssistant: …\n\n"), each segment
 * tokenised separately with @p tok and concatenated. No BOS/EOS is added.
 * targets[t] = next token when that token is in a trained segment, else ignore_id.
 * Records with fewer than 2 tokens or no trained target are skipped and counted
 * in *n_skipped. Bad JSON, an unknown format/role or a missing field is fatal.
 *
 * @param path       Path to the JSONL file.
 * @param tok        Tokenizer instance.
 * @param fmt        Format to use for each record.
 * @param max_len    Maximum example length n (>= 1); the token stream is
 *                   truncated to max_len + 1 tokens before shifting.
 * @param ignore_id  Token id used for masked targets (usually vocab_size).
 * @param out        On success, malloc'd array of pl_example (caller frees).
 * @param n_out      On success, number of examples written to *out.
 * @param n_skipped  On success, number of records skipped.
 * @return 0 on success; −1 on fatal error (prints "pl_data: line <k>: <reason>").
 */
int pl_data_load_jsonl(const char *path, const rx_tokenizer *tok,
                       pl_data_fmt fmt, int max_len, int ignore_id,
                       pl_example **out, int *n_out, int *n_skipped);

/**
 * Free an array of examples produced by pl_data_load_jsonl.
 * @param exs  Array of examples (may be NULL).
 * @param n    Number of examples in the array.
 */
void pl_data_free(pl_example *exs, int n);

#endif /* FOUNDRY_PL_DATA_H */