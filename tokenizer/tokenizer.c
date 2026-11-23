/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define FT_MAX_VOCAB 100000
#define FT_MAX_MERGES 100000
#define FT_MAX_TOKEN_LEN 256
#define FT_HASH_SIZE 65536

/* ============================================================= */
/* Hash Table for Token Lookup                                   */
/* ============================================================= */

typedef struct ft_hash_entry {
    char* key;
    int value;
    struct ft_hash_entry* next;
} ft_hash_entry;

typedef struct {
    ft_hash_entry* buckets[FT_HASH_SIZE];
} ft_hash_table;

static uint32_t hash_string(const char* str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % FT_HASH_SIZE;
}

static ft_hash_table* hash_new(void) {
    ft_hash_table* ht = calloc(1, sizeof(ft_hash_table));
    return ht;
}

static void hash_free(ft_hash_table* ht) {
    if (!ht) return;
    for (int i = 0; i < FT_HASH_SIZE; i++) {
        ft_hash_entry* e = ht->buckets[i];
        while (e) {
            ft_hash_entry* next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }
    free(ht);
}

static void hash_set(ft_hash_table* ht, const char* key, int value) {
    uint32_t idx = hash_string(key);
    ft_hash_entry* e = ht->buckets[idx];

    while (e) {
        if (strcmp(e->key, key) == 0) {
            e->value = value;
            return;
        }
        e = e->next;
    }

    e = malloc(sizeof(ft_hash_entry));
    e->key = strdup(key);
    e->value = value;
    e->next = ht->buckets[idx];
    ht->buckets[idx] = e;
}

static int hash_get(ft_hash_table* ht, const char* key, int default_val) {
    uint32_t idx = hash_string(key);
    ft_hash_entry* e = ht->buckets[idx];

    while (e) {
        if (strcmp(e->key, key) == 0) {
            return e->value;
        }
        e = e->next;
    }

    return default_val;
}

/* ============================================================= */
/* BPE Merge Rule                                                */
/* ============================================================= */

typedef struct {
    char* a;
    char* b;
    char* merged;  /* a + b concatenated */
} ft_merge;

/* ============================================================= */
/* Tokenizer Structure                                           */
/* ============================================================= */

struct ft_tokenizer {
    /* Vocabulary */
    char** tokens;           /* id -> token string */
    int vocab_size;
    int vocab_capacity;
    ft_hash_table* token_to_id;

    /* BPE merges (in priority order) */
    ft_merge* merges;
    int num_merges;
    int merge_capacity;
    ft_hash_table* merge_lookup;  /* "a b" -> merge index */

    /* Special tokens */
    int unk_id;
    int bos_id;
    int eos_id;
    int pad_id;

    /* SentencePiece / Unigram fields */
    int model_type;         /* 0=BPE (default), 1=UNIGRAM */
    float *scores;          /* [vocab_size] log-prob per token, NULL for BPE */
    int max_token_len;      /* longest token in bytes (for Viterbi scan) */

    /* BPE pre-tokenizer family (tokenizer.ggml.pre); consumer-defined code,
     * 0 = default/unset. Selects the regex pre-tokenization scanner. */
    int pre_type;
};

/* ============================================================= */
/* Creation / Destruction                                        */
/* ============================================================= */

ft_tokenizer* ft_new(void) {
    ft_tokenizer* tok = calloc(1, sizeof(ft_tokenizer));
    if (!tok) return NULL;

    tok->vocab_capacity = 1024;
    tok->tokens = calloc(tok->vocab_capacity, sizeof(char*));
    tok->token_to_id = hash_new();

    tok->merge_capacity = 1024;
    tok->merges = calloc(tok->merge_capacity, sizeof(ft_merge));
    tok->merge_lookup = hash_new();

    tok->unk_id = -1;
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->pad_id = -1;

    return tok;
}

void ft_free(ft_tokenizer* tok) {
    if (!tok) return;

    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->tokens[i]);
    }
    free(tok->tokens);
    hash_free(tok->token_to_id);

    for (int i = 0; i < tok->num_merges; i++) {
        free(tok->merges[i].a);
        free(tok->merges[i].b);
        free(tok->merges[i].merged);
    }
    free(tok->merges);
    hash_free(tok->merge_lookup);

    free(tok->scores);
    free(tok);
}

/* ============================================================= */
/* Vocabulary Management                                         */
/* ============================================================= */

int ft_add_token(ft_tokenizer* tok, const char* token) {
    if (!tok || !token) return -1;

    /* Check if already exists */
    int existing = hash_get(tok->token_to_id, token, -1);
    if (existing >= 0) return existing;

    /* Expand if needed */
    if (tok->vocab_size >= tok->vocab_capacity) {
        tok->vocab_capacity *= 2;
        tok->tokens = realloc(tok->tokens, tok->vocab_capacity * sizeof(char*));
    }

    int id = tok->vocab_size;
    tok->tokens[id] = strdup(token);
    hash_set(tok->token_to_id, token, id);
    tok->vocab_size++;

    return id;
}

void ft_add_merge(ft_tokenizer* tok, const char* a, const char* b) {
    if (!tok || !a || !b) return;

    /* Expand if needed */
    if (tok->num_merges >= tok->merge_capacity) {
        tok->merge_capacity *= 2;
        tok->merges = realloc(tok->merges, tok->merge_capacity * sizeof(ft_merge));
    }

    ft_merge* m = &tok->merges[tok->num_merges];
    m->a = strdup(a);
    m->b = strdup(b);

    /* Create merged token */
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    m->merged = malloc(len_a + len_b + 1);
    memcpy(m->merged, a, len_a);
    memcpy(m->merged + len_a, b, len_b);
    m->merged[len_a + len_b] = '\0';

    /* Add to lookup */
    char key[FT_MAX_TOKEN_LEN * 2 + 2];
    snprintf(key, sizeof(key), "%s %s", a, b);
    hash_set(tok->merge_lookup, key, tok->num_merges);

    tok->num_merges++;
}

int ft_vocab_size(ft_tokenizer* tok) {
    return tok ? tok->vocab_size : 0;
}

const char* ft_id_to_token(ft_tokenizer* tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->tokens[id];
}

int ft_token_to_id(ft_tokenizer* tok, const char* token) {
    if (!tok || !token) return -1;
    int id = hash_get(tok->token_to_id, token, -1);
    /* Return unk_id for unknown tokens, but only if unk is set */
    if (id < 0 && tok->unk_id >= 0) {
        return tok->unk_id;
    }
    return id;
}

/* ============================================================= */
/* Special Tokens                                                */
/* ============================================================= */

/* Helper: check if token exists in vocab (without unk fallback) */
static int token_exists(ft_tokenizer* tok, const char* token) {
    return hash_get(tok->token_to_id, token, -1);
}

void ft_set_unk_token(ft_tokenizer* tok, const char* token) {
    if (!tok || !token) return;
    tok->unk_id = token_exists(tok, token);
    if (tok->unk_id < 0) tok->unk_id = ft_add_token(tok, token);
}

void ft_set_bos_token(ft_tokenizer* tok, const char* token) {
    if (!tok || !token) return;
    tok->bos_id = token_exists(tok, token);
    if (tok->bos_id < 0) tok->bos_id = ft_add_token(tok, token);
}

void ft_set_eos_token(ft_tokenizer* tok, const char* token) {
    if (!tok || !token) return;
    tok->eos_id = token_exists(tok, token);
    if (tok->eos_id < 0) tok->eos_id = ft_add_token(tok, token);
}

void ft_set_pad_token(ft_tokenizer* tok, const char* token) {
    if (!tok || !token) return;
    tok->pad_id = token_exists(tok, token);
    if (tok->pad_id < 0) tok->pad_id = ft_add_token(tok, token);
}

int ft_unk_id(ft_tokenizer* tok) { return tok ? tok->unk_id : -1; }
int ft_bos_id(ft_tokenizer* tok) { return tok ? tok->bos_id : -1; }
int ft_eos_id(ft_tokenizer* tok) { return tok ? tok->eos_id : -1; }
int ft_pad_id(ft_tokenizer* tok) { return tok ? tok->pad_id : -1; }

/* ============================================================= */
/* Model Type (BPE vs Unigram)                                   */
/* ============================================================= */

void ft_set_model_type(ft_tokenizer *tok, int type) {
    if (tok) tok->model_type = type;
}

int ft_get_model_type(ft_tokenizer *tok) {
    return tok ? tok->model_type : 0;
}

void ft_set_pre_type(ft_tokenizer *tok, int type) {
    if (tok) tok->pre_type = type;
}

int ft_get_pre_type(ft_tokenizer *tok) {
    return tok ? tok->pre_type : 0;
}

void ft_set_scores(ft_tokenizer *tok, const float *scores, int n) {
    if (!tok || !scores || n <= 0) return;

    free(tok->scores);
    tok->scores = malloc(n * sizeof(float));
    if (!tok->scores) return;
    memcpy(tok->scores, scores, n * sizeof(float));

    /* Compute max_token_len by scanning vocab */
    tok->max_token_len = 0;
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->tokens[i]) {
            int len = (int)strlen(tok->tokens[i]);
            if (len > tok->max_token_len) tok->max_token_len = len;
        }
    }
    /* Cap to something reasonable */
    if (tok->max_token_len > 64) tok->max_token_len = 64;
}

/* ============================================================= */
/* Unigram (SentencePiece) Encoding — Viterbi                    */
/* ============================================================= */

static int* encode_unigram(ft_tokenizer *tok, const char *text, int *length) {
    int text_len = (int)strlen(text);
    if (text_len == 0) {
        *length = 0;
        return calloc(1, sizeof(int));
    }

    int max_len = tok->max_token_len > 0 ? tok->max_token_len : 32;

    /* Viterbi DP arrays */
    float *best_score = malloc((text_len + 1) * sizeof(float));
    int *best_token  = malloc((text_len + 1) * sizeof(int));
    int *best_from   = malloc((text_len + 1) * sizeof(int));

    best_score[0] = 0.0f;
    for (int i = 1; i <= text_len; i++) {
        best_score[i] = -1e30f;
        best_token[i] = -1;
        best_from[i] = -1;
    }

    /* Temp buffer for substring lookups */
    char substr[256];

    /* Forward pass */
    for (int i = 0; i < text_len; i++) {
        if (best_score[i] <= -1e30f) continue;

        int remaining = text_len - i;
        int scan_len = remaining < max_len ? remaining : max_len;

        for (int len = 1; len <= scan_len; len++) {
            if (len >= (int)sizeof(substr)) break;

            memcpy(substr, text + i, len);
            substr[len] = '\0';

            /* Look up in hash table (without unk fallback) */
            int id = hash_get(tok->token_to_id, substr, -1);
            if (id < 0) continue;

            float score = tok->scores ? tok->scores[id] : 0.0f;
            float candidate = best_score[i] + score;

            if (candidate > best_score[i + len]) {
                best_score[i + len] = candidate;
                best_token[i + len] = id;
                best_from[i + len] = i;
            }
        }

        /* Byte fallback: if no token starts here, emit <0xXX> */
        if (best_score[i + 1] <= -1e30f) {
            unsigned char byte = (unsigned char)text[i];
            char hex_token[7];
            snprintf(hex_token, sizeof(hex_token), "<0x%02X>", byte);

            int id = hash_get(tok->token_to_id, hex_token, -1);
            if (id >= 0) {
                float score = tok->scores ? tok->scores[id] : -100.0f;
                best_score[i + 1] = best_score[i] + score;
                best_token[i + 1] = id;
                best_from[i + 1] = i;
            } else {
                /* Last resort: unk token */
                best_score[i + 1] = best_score[i] - 100.0f;
                best_token[i + 1] = tok->unk_id >= 0 ? tok->unk_id : 0;
                best_from[i + 1] = i;
            }
        }
    }

    /* Backtrack to collect token sequence */
    int *rev_tokens = malloc(text_len * sizeof(int));
    int n = 0;
    int pos = text_len;

    while (pos > 0) {
        rev_tokens[n++] = best_token[pos];
        pos = best_from[pos];
    }

    free(best_score);
    free(best_token);
    free(best_from);

    /* Reverse into output */
    int *ids = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        ids[i] = rev_tokens[n - 1 - i];
    }
    free(rev_tokens);

    *length = n;
    return ids;
}

/* ============================================================= */
/* BPE Encoding                                                  */
/* ============================================================= */

/* Split text into initial character-level tokens */
static char** split_to_chars(const char* text, int* count) {
    int len = (int)strlen(text);
    char** chars = malloc(len * sizeof(char*));
    int n = 0;

    const unsigned char* p = (const unsigned char*)text;
    while (*p) {
        int char_len = 1;

        /* Handle UTF-8 */
        if (*p >= 0xF0) char_len = 4;
        else if (*p >= 0xE0) char_len = 3;
        else if (*p >= 0xC0) char_len = 2;

        chars[n] = malloc(char_len + 1);
        memcpy(chars[n], p, char_len);
        chars[n][char_len] = '\0';
        n++;
        p += char_len;
    }

    *count = n;
    return chars;
}

/* Apply BPE merges to token list */
static void apply_bpe_merges(ft_tokenizer* tok, char** tokens, int* count) {
    bool changed = true;

    while (changed) {
        changed = false;
        int best_merge = -1;
        int best_pos = -1;

        /* Find highest priority merge that can be applied */
        for (int i = 0; i < *count - 1; i++) {
            char key[FT_MAX_TOKEN_LEN * 2 + 2];
            snprintf(key, sizeof(key), "%s %s", tokens[i], tokens[i + 1]);

            int merge_idx = hash_get(tok->merge_lookup, key, -1);
            if (merge_idx >= 0 && (best_merge < 0 || merge_idx < best_merge)) {
                best_merge = merge_idx;
                best_pos = i;
            }
        }

        /* Apply the merge */
        if (best_merge >= 0) {
            ft_merge* m = &tok->merges[best_merge];

            /* Replace tokens[best_pos] and tokens[best_pos+1] with merged */
            free(tokens[best_pos]);
            free(tokens[best_pos + 1]);
            tokens[best_pos] = strdup(m->merged);

            /* Shift remaining tokens */
            for (int i = best_pos + 1; i < *count - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            (*count)--;
            changed = true;
        }
    }
}

int* ft_encode(ft_tokenizer* tok, const char* text, int* length) {
    if (!tok || !text || !length) return NULL;

    if (strlen(text) == 0) {
        *length = 0;
        return calloc(1, sizeof(int));
    }

    /* Unigram / SentencePiece path */
    if (tok->model_type == 1) {
        /* SentencePiece expects a leading space for word boundary normalization */
        size_t len = strlen(text);
        char *spaced = malloc(len + 2);
        spaced[0] = ' ';
        memcpy(spaced + 1, text, len + 1);
        int *ids = encode_unigram(tok, spaced, length);
        free(spaced);
        return ids;
    }

    /* Split into characters */
    int char_count;
    char** chars = split_to_chars(text, &char_count);

    if (char_count == 0) {
        free(chars);
        *length = 0;
        return calloc(1, sizeof(int));
    }

    /* Apply BPE merges */
    if (tok->num_merges > 0) {
        apply_bpe_merges(tok, chars, &char_count);
    }

    /* Convert to token IDs */
    int* ids = malloc(char_count * sizeof(int));
    for (int i = 0; i < char_count; i++) {
        ids[i] = ft_token_to_id(tok, chars[i]);
        free(chars[i]);
    }
    free(chars);

    *length = char_count;
    return ids;
}

int* ft_encode_with_special(ft_tokenizer* tok, const char* text, int* length,
                            bool add_bos, bool add_eos) {
    int base_len;
    int* base_ids = ft_encode(tok, text, &base_len);
    if (!base_ids) return NULL;

    int extra = (add_bos ? 1 : 0) + (add_eos ? 1 : 0);
    int total_len = base_len + extra;

    int* ids = malloc(total_len * sizeof(int));
    int pos = 0;

    if (add_bos && tok->bos_id >= 0) {
        ids[pos++] = tok->bos_id;
    }

    memcpy(ids + pos, base_ids, base_len * sizeof(int));
    pos += base_len;

    if (add_eos && tok->eos_id >= 0) {
        ids[pos++] = tok->eos_id;
    }

    free(base_ids);
    *length = pos;
    return ids;
}

int** ft_encode_batch(ft_tokenizer* tok, const char** texts, int n, int** lengths) {
    if (!tok || !texts || !lengths || n <= 0) return NULL;

    int** results = malloc(n * sizeof(int*));
    *lengths = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        results[i] = ft_encode(tok, texts[i], &(*lengths)[i]);
    }

    return results;
}

/* ============================================================= */
/* Decoding                                                      */
/* ============================================================= */

/* Parse <0xHH> byte token, returns byte value or -1 */
static int parse_byte_token(const char *token) {
    if (!token) return -1;
    if (token[0] == '<' && token[1] == '0' && token[2] == 'x' &&
        token[5] == '>' && token[6] == '\0') {
        char hex[3] = { token[3], token[4], '\0' };
        return (int)strtol(hex, NULL, 16);
    }
    return -1;
}

/* Append token to buffer, handling SentencePiece ▁ and <0xHH> byte tokens */
static size_t append_token_sp(const char *token, char *buf, size_t pos) {
    const unsigned char *p = (const unsigned char *)token;

    while (*p) {
        /* Check for ▁ (U+2581 = 0xE2 0x96 0x81) — SentencePiece word separator */
        if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) {
            buf[pos++] = ' ';
            p += 3;
        } else {
            buf[pos++] = (char)*p;
            p++;
        }
    }
    return pos;
}

char* ft_decode(ft_tokenizer* tok, int* ids, int length) {
    if (!tok || !ids || length <= 0) {
        char* empty = malloc(1);
        empty[0] = '\0';
        return empty;
    }

    /* BPE path: simple concatenation */
    if (tok->model_type != 1) {
        size_t total_len = 0;
        for (int i = 0; i < length; i++) {
            const char* token = ft_id_to_token(tok, ids[i]);
            if (token) total_len += strlen(token);
        }

        char* result = malloc(total_len + 1);
        result[0] = '\0';

        for (int i = 0; i < length; i++) {
            const char* token = ft_id_to_token(tok, ids[i]);
            if (token) strcat(result, token);
        }

        return result;
    }

    /* Unigram/SentencePiece path: handle ▁ and <0xHH> byte tokens */
    size_t total_len = 0;
    for (int i = 0; i < length; i++) {
        const char* token = ft_id_to_token(tok, ids[i]);
        if (token) total_len += strlen(token);
    }

    char *result = malloc(total_len + 1);
    size_t pos = 0;

    for (int i = 0; i < length; i++) {
        const char *token = ft_id_to_token(tok, ids[i]);
        if (!token) continue;

        int byte_val = parse_byte_token(token);
        if (byte_val >= 0) {
            result[pos++] = (char)byte_val;
        } else {
            pos = append_token_sp(token, result, pos);
        }
    }

    result[pos] = '\0';
    return result;
}

char* ft_decode_skip_special(ft_tokenizer* tok, int* ids, int length) {
    if (!tok || !ids || length <= 0) {
        char* empty = malloc(1);
        empty[0] = '\0';
        return empty;
    }

    /* Calculate total length, skipping special tokens */
    size_t total_len = 0;
    for (int i = 0; i < length; i++) {
        if (ids[i] == tok->bos_id || ids[i] == tok->eos_id ||
            ids[i] == tok->pad_id) continue;

        const char* token = ft_id_to_token(tok, ids[i]);
        if (token) total_len += strlen(token);
    }

    char* result = malloc(total_len + 1);
    result[0] = '\0';

    for (int i = 0; i < length; i++) {
        if (ids[i] == tok->bos_id || ids[i] == tok->eos_id ||
            ids[i] == tok->pad_id) continue;

        const char* token = ft_id_to_token(tok, ids[i]);
        if (token) strcat(result, token);
    }

    return result;
}

/* ============================================================= */
/* File Loading                                                  */
/* ============================================================= */

/* Simple JSON string extraction (not a full parser) */
static char* json_get_string(const char* json, const char* key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char* pos = strstr(json, search);
    if (!pos) return NULL;

    pos += strlen(search);
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t')) pos++;

    if (*pos != '"') return NULL;
    pos++;

    const char* end = pos;
    while (*end && *end != '"') {
        if (*end == '\\') end++;
        if (*end) end++;
    }

    size_t len = end - pos;
    char* result = malloc(len + 1);
    memcpy(result, pos, len);
    result[len] = '\0';

    return result;
}

static char* read_file(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* content = malloc(size + 1);
    size_t read = fread(content, 1, size, f);
    content[read] = '\0';

    fclose(f);
    return content;
}

ft_tokenizer* ft_load_bpe(const char* vocab_path, const char* merges_path) {
    ft_tokenizer* tok = ft_new();
    if (!tok) return NULL;

    /* Load vocab file (JSON format: {"token": id, ...}) */
    char* vocab_content = read_file(vocab_path);
    if (vocab_content) {
        /* Parse simple JSON vocab */
        const char* p = vocab_content;
        while ((p = strchr(p, '"')) != NULL) {
            p++;
            const char* token_start = p;
            while (*p && *p != '"') {
                if (*p == '\\') p++;
                if (*p) p++;
            }
            if (!*p) break;

            size_t token_len = p - token_start;
            char* token = malloc(token_len + 1);
            memcpy(token, token_start, token_len);
            token[token_len] = '\0';

            p++;  /* Skip closing quote */

            /* Find colon and number */
            while (*p && *p != ':') p++;
            if (!*p) { free(token); break; }
            p++;

            while (*p && isspace(*p)) p++;

            int id = (int)strtol(p, NULL, 10);

            /* Ensure token is at correct position */
            while (tok->vocab_size <= id) {
                ft_add_token(tok, "");
            }
            free(tok->tokens[id]);
            tok->tokens[id] = token;
            hash_set(tok->token_to_id, token, id);
        }
        free(vocab_content);
    }

    /* Load merges file (one merge per line: "token1 token2") */
    char* merges_content = read_file(merges_path);
    if (merges_content) {
        char* line = strtok(merges_content, "\n");
        while (line) {
            /* Skip comments and empty lines */
            if (line[0] == '#' || line[0] == '\0') {
                line = strtok(NULL, "\n");
                continue;
            }

            /* Split on space */
            char* space = strchr(line, ' ');
            if (space) {
                *space = '\0';
                char* a = line;
                char* b = space + 1;

                /* Trim trailing whitespace */
                char* end = b + strlen(b) - 1;
                while (end > b && isspace(*end)) *end-- = '\0';

                if (strlen(a) > 0 && strlen(b) > 0) {
                    ft_add_merge(tok, a, b);

                    /* Add merged token to vocab if not present */
                    char merged[FT_MAX_TOKEN_LEN];
                    snprintf(merged, sizeof(merged), "%s%s", a, b);
                    if (ft_token_to_id(tok, merged) < 0) {
                        ft_add_token(tok, merged);
                    }
                }
            }

            line = strtok(NULL, "\n");
        }
        free(merges_content);
    }

    return tok;
}

ft_tokenizer* ft_load(const char* path) {
    char* content = read_file(path);
    if (!content) return NULL;

    ft_tokenizer* tok = ft_new();

    /* Try to parse as HuggingFace tokenizer.json format */
    /* This is simplified - real implementation would need full JSON parser */

    /* Look for vocab in "model" -> "vocab" */
    const char* vocab_start = strstr(content, "\"vocab\"");
    if (vocab_start) {
        vocab_start = strchr(vocab_start, '{');
        if (vocab_start) {
            /* Parse vocab entries */
            const char* p = vocab_start + 1;
            while (*p && *p != '}') {
                while (*p && *p != '"') p++;
                if (!*p || *p == '}') break;

                p++;
                const char* token_start = p;
                while (*p && *p != '"') {
                    if (*p == '\\') p++;
                    if (*p) p++;
                }

                size_t token_len = p - token_start;
                char* token = malloc(token_len + 1);
                memcpy(token, token_start, token_len);
                token[token_len] = '\0';

                /* Handle escape sequences */
                /* (simplified - just handle common ones) */

                p++;
                while (*p && *p != ':') p++;
                p++;
                while (*p && isspace(*p)) p++;

                int id = (int)strtol(p, (char**)&p, 10);

                while (tok->vocab_size <= id) {
                    ft_add_token(tok, "");
                }
                free(tok->tokens[id]);
                tok->tokens[id] = token;
                hash_set(tok->token_to_id, token, id);

                while (*p && *p != ',' && *p != '}') p++;
                if (*p == ',') p++;
            }
        }
    }

    /* Look for special tokens */
    char* unk = json_get_string(content, "unk_token");
    char* bos = json_get_string(content, "bos_token");
    char* eos = json_get_string(content, "eos_token");
    char* pad = json_get_string(content, "pad_token");

    if (unk) { ft_set_unk_token(tok, unk); free(unk); }
    if (bos) { ft_set_bos_token(tok, bos); free(bos); }
    if (eos) { ft_set_eos_token(tok, eos); free(eos); }
    if (pad) { ft_set_pad_token(tok, pad); free(pad); }

    free(content);
    return tok;
}

/* ============================================================= */
/* Merge Rules                                                   */
/* ============================================================= */

int ft_num_merges(ft_tokenizer* tok) {
    return tok ? tok->num_merges : 0;
}

void ft_get_merge(ft_tokenizer* tok, int index, const char **a, const char **b) {
    if (!tok || index < 0 || index >= tok->num_merges) {
        if (a) *a = NULL;
        if (b) *b = NULL;
        return;
    }
    if (a) *a = tok->merges[index].a;
    if (b) *b = tok->merges[index].b;
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

void ft_print(ft_tokenizer* tok) {
    if (!tok) {
        printf("tokenizer(NULL)\n");
        return;
    }

    printf("tokenizer(vocab_size=%d, merges=%d)\n", tok->vocab_size, tok->num_merges);
    printf("  special: unk=%d, bos=%d, eos=%d, pad=%d\n",
           tok->unk_id, tok->bos_id, tok->eos_id, tok->pad_id);
}

void ft_print_tokens(ft_tokenizer* tok, int* ids, int length) {
    if (!tok || !ids) return;

    printf("[");
    for (int i = 0; i < length; i++) {
        const char* token = ft_id_to_token(tok, ids[i]);
        printf("%d:'%s'%s", ids[i], token ? token : "?", i < length - 1 ? ", " : "");
    }
    printf("]\n");
}
