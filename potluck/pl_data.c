/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pl_data.c
 * @brief JSONL (text / alpaca / chat) → tokens + loss-masked targets.
 *
 * Each record becomes a list of text segments with a trained flag. Segments are
 * encoded separately with rx_tokenizer and concatenated; targets are shifted
 * next tokens, kept only where the NEXT token's segment is trained.
 */
#include "pl_data.h"
#include "data/data.h"
#include "rwkvx/rwkvx.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- format names ---- */

int pl_data_fmt_parse(const char *name) {
    if (!name) return -1;
    if (strcmp(name, "auto") == 0)   return PL_FMT_AUTO;
    if (strcmp(name, "text") == 0)   return PL_FMT_TEXT;
    if (strcmp(name, "alpaca") == 0) return PL_FMT_ALPACA;
    if (strcmp(name, "chat") == 0)   return PL_FMT_CHAT;
    return -1;
}

/* ---- segment list ---- */

typedef struct { char *text; int trained; } pl_segment;
typedef struct { pl_segment *items; int n; int cap; } pl_segment_list;

static void segments_free(pl_segment_list *list) {
    for (int i = 0; i < list->n; i++) free(list->items[i].text);
    free(list->items);
    list->items = NULL; list->n = list->cap = 0;
}

/* Append prefix + body + suffix as one segment. */
static void segments_push(pl_segment_list *list, const char *prefix, const char *body,
                          const char *suffix, int trained) {
    if (list->n == list->cap) {
        list->cap = list->cap ? list->cap * 2 : 8;
        list->items = realloc(list->items, (size_t)list->cap * sizeof(pl_segment));
    }
    size_t lp = strlen(prefix), lb = strlen(body), ls = strlen(suffix);
    char *text = malloc(lp + lb + ls + 1);
    memcpy(text, prefix, lp);
    memcpy(text + lp, body, lb);
    memcpy(text + lp + lb, suffix, ls + 1);
    list->items[list->n].text = text;
    list->items[list->n].trained = trained;
    list->n++;
}

static int line_error(int line_no, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "pl_data: line %d: ", line_no);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    return -1;
}

/* ---- per-format segment builders (0 ok, -1 error printed) ---- */

static int build_text(pl_segment_list *out, fd_sample *s, int line_no) {
    const char *text = fd_sample_get(s, "text");
    if (!text) return line_error(line_no, "missing field \"text\"");
    segments_push(out, "", text, "", 1);
    return 0;
}

static int build_alpaca(pl_segment_list *out, fd_sample *s, int line_no) {
    const char *instruction = fd_sample_get(s, "instruction");
    const char *input = fd_sample_get(s, "input");
    const char *output = fd_sample_get(s, "output");
    if (!instruction) return line_error(line_no, "missing field \"instruction\"");
    if (!output) return line_error(line_no, "missing field \"output\"");
    if (input && *input) {
        size_t li = strlen(instruction), lx = strlen(input);
        char *joined = malloc(li + 1 + lx + 1);
        memcpy(joined, instruction, li);
        joined[li] = '\n';
        memcpy(joined + li + 1, input, lx + 1);
        segments_push(out, "User: ", joined, "\n\nAssistant:", 0);
        free(joined);
    } else {
        segments_push(out, "User: ", instruction, "\n\nAssistant:", 0);
    }
    segments_push(out, " ", output, "\n\n", 1);
    return 0;
}

/* Copy the next top-level {…} of a raw JSON array into *obj (string-aware).
 * Returns 1 when an object was copied, 0 at the end of the array, -1 on malformed input. */
static int next_array_object(const char **cursor, char **obj) {
    const char *p = *cursor;
    p += strspn(p, " \t\r\n,");
    if (*p == ']') return 0;
    if (*p != '{') return -1;
    const char *start = p;
    int depth = 0;
    do {
        if (*p == '"') {
            for (p++; *p && *p != '"'; p++) if (*p == '\\' && p[1]) p++;
        } else if (*p == '{') {
            depth++;
        } else if (*p == '}') {
            depth--;
        }
        if (!*p) return -1;
        p++;
    } while (depth > 0);
    size_t len = (size_t)(p - start);
    *obj = malloc(len + 1);
    memcpy(*obj, start, len);
    (*obj)[len] = '\0';
    *cursor = p;
    return 1;
}

static int build_chat_message(pl_segment_list *out, fd_sample *msg, int line_no) {
    const char *role = fd_sample_get(msg, "role");
    const char *content = fd_sample_get(msg, "content");
    if (!role) return line_error(line_no, "message missing field \"role\"");
    if (!content) return line_error(line_no, "message missing field \"content\"");
    if (strcmp(role, "system") == 0) {
        segments_push(out, "System: ", content, "\n\n", 0);
    } else if (strcmp(role, "user") == 0) {
        segments_push(out, "User: ", content, "\n\n", 0);
    } else if (strcmp(role, "assistant") == 0) {
        segments_push(out, "Assistant:", "", "", 0);
        segments_push(out, " ", content, "\n\n", 1);
    } else {
        return line_error(line_no, "unknown role \"%s\"", role);
    }
    return 0;
}

static int build_chat(pl_segment_list *out, fd_sample *s, int line_no) {
    const char *messages = fd_sample_get(s, "messages");
    if (!messages) return line_error(line_no, "missing field \"messages\"");
    if (*messages != '[') return line_error(line_no, "\"messages\" is not an array");
    const char *cursor = messages + 1;
    char *obj = NULL;
    int got;
    while ((got = next_array_object(&cursor, &obj)) == 1) {
        fd_sample *msg = fd_parse_json_object(obj);
        free(obj);
        int rc = msg ? build_chat_message(out, msg, line_no)
                     : line_error(line_no, "bad message JSON");
        if (msg) fd_sample_free(msg);
        if (rc != 0) return -1;
    }
    if (got < 0) return line_error(line_no, "malformed \"messages\" array");
    return 0;
}

static int build_segments(pl_segment_list *out, fd_sample *s, pl_data_fmt fmt, int line_no) {
    if (fmt == PL_FMT_AUTO) {
        if (fd_sample_has(s, "messages"))         fmt = PL_FMT_CHAT;
        else if (fd_sample_has(s, "instruction")) fmt = PL_FMT_ALPACA;
        else if (fd_sample_has(s, "text"))        fmt = PL_FMT_TEXT;
        else return line_error(line_no, "unknown format (no messages/instruction/text)");
    }
    switch (fmt) {
        case PL_FMT_TEXT:   return build_text(out, s, line_no);
        case PL_FMT_ALPACA: return build_alpaca(out, s, line_no);
        case PL_FMT_CHAT:   return build_chat(out, s, line_no);
        default:            return line_error(line_no, "unknown format %d", (int)fmt);
    }
}

/* ---- encode + concatenate + targets ---- */

/* Encode segments into one example truncated to max_len + 1 tokens.
 * Returns 1 when *ex was filled, 0 when the record is skipped (< 2 tokens or no trained target). */
static int encode_example(const pl_segment_list *segs, const rx_tokenizer *tok, int max_len,
                          int ignore_id, pl_example *ex) {
    int cap = max_len + 1, L = 0;
    uint32_t *toks = malloc((size_t)cap * sizeof(uint32_t));
    int *trained = malloc((size_t)cap * sizeof(int));
    for (int i = 0; i < segs->n && L < cap; i++) {
        int n_enc = rx_tokenizer_encode(tok, segs->items[i].text, toks + L, cap - L);
        for (int j = 0; j < n_enc; j++) trained[L + j] = segs->items[i].trained;
        L += n_enc;
    }
    int n = L - 1, n_trained = 0;
    if (n >= 1) {
        ex->n = n;
        ex->tokens = malloc((size_t)n * sizeof(int));
        ex->targets = malloc((size_t)n * sizeof(int));
        for (int t = 0; t < n; t++) {
            ex->tokens[t] = (int)toks[t];
            ex->targets[t] = trained[t + 1] ? (int)toks[t + 1] : ignore_id;
            n_trained += trained[t + 1];
        }
        if (n_trained == 0) { free(ex->tokens); free(ex->targets); }
    }
    free(toks);
    free(trained);
    return n >= 1 && n_trained > 0;
}

/* ---- loader ---- */

int pl_data_load_jsonl(const char *path, const rx_tokenizer *tok, pl_data_fmt fmt, int max_len,
                       int ignore_id, pl_example **out, int *n_out, int *n_skipped) {
    if (!path || !tok || !out || !n_out || !n_skipped) return -1;
    *out = NULL; *n_out = 0; *n_skipped = 0;
    if (max_len < 1) { fprintf(stderr, "pl_data: max_len must be >= 1\n"); return -1; }
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "pl_data: cannot open %s\n", path); return -1; }

    pl_example *examples = NULL;
    int n_examples = 0, cap = 0, line_no = 0, rc = 0;
    char *line = NULL; size_t line_cap = 0; ssize_t len;
    while ((len = getline(&line, &line_cap, f)) != -1) {
        line_no++;
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (line[strspn(line, " \t")] == '\0') continue;
        fd_sample *s = fd_parse_json_object(line);
        if (!s) { rc = line_error(line_no, "bad JSON"); break; }
        pl_segment_list segs = { 0 };
        rc = build_segments(&segs, s, fmt, line_no);
        fd_sample_free(s);
        if (rc == 0) {
            if (n_examples == cap) {
                cap = cap ? cap * 2 : 16;
                examples = realloc(examples, (size_t)cap * sizeof(pl_example));
            }
            if (encode_example(&segs, tok, max_len, ignore_id, &examples[n_examples])) n_examples++;
            else (*n_skipped)++;
        }
        segments_free(&segs);
        if (rc != 0) break;
    }
    free(line);
    fclose(f);
    if (rc != 0) { pl_data_free(examples, n_examples); *n_skipped = 0; return -1; }
    *out = examples;
    *n_out = n_examples;
    return 0;
}

void pl_data_free(pl_example *ex, int n) {
    if (!ex) return;
    for (int i = 0; i < n; i++) { free(ex[i].tokens); free(ex[i].targets); }
    free(ex);
}
