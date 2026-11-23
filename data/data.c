/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================= */
/* Random state                                                  */
/* ============================================================= */

static uint64_t fd_rng_state = 0;

void fd_seed(uint64_t seed) {
    fd_rng_state = seed;
}

static uint64_t fd_rand(void) {
    if (fd_rng_state == 0) {
        fd_rng_state = (uint64_t)time(NULL);
    }
    /* xorshift64 */
    fd_rng_state ^= fd_rng_state << 13;
    fd_rng_state ^= fd_rng_state >> 7;
    fd_rng_state ^= fd_rng_state << 17;
    return fd_rng_state;
}

/* ============================================================= */
/* Sample                                                        */
/* ============================================================= */

fd_sample* fd_sample_new(void) {
    fd_sample* s = calloc(1, sizeof(fd_sample));
    return s;
}

void fd_sample_free(fd_sample* s) {
    if (!s) return;

    for (int i = 0; i < s->num_fields; i++) {
        free(s->keys[i]);
        free(s->values[i]);
    }
    free(s->keys);
    free(s->values);
    free(s);
}

void fd_sample_set(fd_sample* s, const char* key, const char* value) {
    if (!s || !key || !value) return;

    /* Check if key already exists */
    for (int i = 0; i < s->num_fields; i++) {
        if (strcmp(s->keys[i], key) == 0) {
            free(s->values[i]);
            s->values[i] = strdup(value);
            return;
        }
    }

    /* Add new field */
    s->num_fields++;
    s->keys = realloc(s->keys, s->num_fields * sizeof(char*));
    s->values = realloc(s->values, s->num_fields * sizeof(char*));
    s->keys[s->num_fields - 1] = strdup(key);
    s->values[s->num_fields - 1] = strdup(value);
}

const char* fd_sample_get(fd_sample* s, const char* key) {
    if (!s || !key) return NULL;

    for (int i = 0; i < s->num_fields; i++) {
        if (strcmp(s->keys[i], key) == 0) {
            return s->values[i];
        }
    }
    return NULL;
}

bool fd_sample_has(fd_sample* s, const char* key) {
    return fd_sample_get(s, key) != NULL;
}

fd_sample* fd_sample_clone(fd_sample* s) {
    if (!s) return NULL;

    fd_sample* clone = fd_sample_new();
    for (int i = 0; i < s->num_fields; i++) {
        fd_sample_set(clone, s->keys[i], s->values[i]);
    }
    return clone;
}

void fd_sample_print(fd_sample* s) {
    if (!s) {
        printf("(null sample)\n");
        return;
    }

    printf("{");
    for (int i = 0; i < s->num_fields; i++) {
        if (i > 0) printf(", ");
        printf("\"%s\": \"%s\"", s->keys[i], s->values[i]);
    }
    printf("}\n");
}

/* ============================================================= */
/* Dataset                                                       */
/* ============================================================= */

struct fd_dataset {
    fd_sample** samples;
    int num_samples;
    int capacity;
};

fd_dataset* fd_new(void) {
    fd_dataset* ds = calloc(1, sizeof(fd_dataset));
    if (!ds) return NULL;

    ds->capacity = 16;
    ds->samples = calloc(ds->capacity, sizeof(fd_sample*));
    return ds;
}

void fd_free(fd_dataset* ds) {
    if (!ds) return;

    for (int i = 0; i < ds->num_samples; i++) {
        fd_sample_free(ds->samples[i]);
    }
    free(ds->samples);
    free(ds);
}

void fd_add_sample(fd_dataset* ds, fd_sample* sample) {
    if (!ds || !sample) return;

    if (ds->num_samples >= ds->capacity) {
        ds->capacity *= 2;
        ds->samples = realloc(ds->samples, ds->capacity * sizeof(fd_sample*));
    }

    ds->samples[ds->num_samples++] = sample;
}

int fd_len(fd_dataset* ds) {
    return ds ? ds->num_samples : 0;
}

fd_sample* fd_get(fd_dataset* ds, int idx) {
    if (!ds || idx < 0 || idx >= ds->num_samples) return NULL;
    return ds->samples[idx];
}

void fd_shuffle(fd_dataset* ds) {
    if (!ds || ds->num_samples <= 1) return;

    /* Fisher-Yates shuffle */
    for (int i = ds->num_samples - 1; i > 0; i--) {
        int j = fd_rand() % (i + 1);
        fd_sample* tmp = ds->samples[i];
        ds->samples[i] = ds->samples[j];
        ds->samples[j] = tmp;
    }
}

void fd_split(fd_dataset* ds, float ratio, fd_dataset** train, fd_dataset** test) {
    if (!ds || !train || !test || ratio <= 0.0f || ratio >= 1.0f) return;

    *train = fd_new();
    *test = fd_new();

    int split_idx = (int)(ds->num_samples * ratio);

    for (int i = 0; i < ds->num_samples; i++) {
        fd_sample* clone = fd_sample_clone(ds->samples[i]);
        if (i < split_idx) {
            fd_add_sample(*train, clone);
        } else {
            fd_add_sample(*test, clone);
        }
    }
}

void fd_print(fd_dataset* ds) {
    if (!ds) {
        printf("(null dataset)\n");
        return;
    }

    printf("Dataset: %d samples\n", ds->num_samples);

    if (ds->num_samples > 0) {
        printf("Fields: ");
        fd_sample* s = ds->samples[0];
        for (int i = 0; i < s->num_fields; i++) {
            if (i > 0) printf(", ");
            printf("%s", s->keys[i]);
        }
        printf("\n");
    }
}

/* ============================================================= */
/* JSON Parsing (minimal)                                        */
/* ============================================================= */

static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static char* parse_json_string(const char** pp) {
    const char* p = *pp;
    if (*p != '"') return NULL;
    p++;

    /* Find end of string */
    const char* start = p;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p+1)) p++;
        p++;
    }

    size_t len = p - start;
    char* s = malloc(len + 1);
    if (!s) return NULL;

    /* Copy with escape handling */
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (start[i] == '\\' && i + 1 < len) {
            i++;
            switch (start[i]) {
                case 'n': s[j++] = '\n'; break;
                case 't': s[j++] = '\t'; break;
                case 'r': s[j++] = '\r'; break;
                case '"': s[j++] = '"'; break;
                case '\\': s[j++] = '\\'; break;
                default: s[j++] = start[i]; break;
            }
        } else {
            s[j++] = start[i];
        }
    }
    s[j] = '\0';

    if (*p == '"') p++;
    *pp = p;
    return s;
}

/* Skip a JSON value */
static void skip_json_value(const char** pp) {
    const char* p = *pp;
    p = skip_ws(p);

    if (*p == '"') {
        p++;
        while (*p && *p != '"') {
            if (*p == '\\') p++;
            if (*p) p++;
        }
        if (*p == '"') p++;
    } else if (*p == '[') {
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') p++;
                    if (*p) p++;
                }
            }
            if (*p) p++;
        }
    } else if (*p == '{') {
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') p++;
                    if (*p) p++;
                }
            }
            if (*p) p++;
        }
    } else {
        /* Number, bool, null */
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
            p++;
        }
    }
    *pp = p;
}

/* Parse a JSON value as a string (for non-string values) */
static char* parse_json_value_as_string(const char** pp) {
    const char* p = *pp;
    p = skip_ws(p);

    if (*p == '"') {
        return parse_json_string(pp);
    }

    /* For non-string values, capture the raw text */
    const char* start = p;

    if (*p == '[' || *p == '{') {
        /* Capture entire object/array as string */
        skip_json_value(pp);
        size_t len = *pp - start;
        char* s = malloc(len + 1);
        memcpy(s, start, len);
        s[len] = '\0';
        return s;
    }

    /* Number, bool, null */
    while (*p && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
        p++;
    }

    size_t len = p - start;
    char* s = malloc(len + 1);
    memcpy(s, start, len);
    s[len] = '\0';
    *pp = p;
    return s;
}

static fd_sample* parse_json_object(const char* line) {
    const char* p = line;
    p = skip_ws(p);

    if (*p != '{') return NULL;
    p++;

    fd_sample* s = fd_sample_new();

    while (*p && *p != '}') {
        p = skip_ws(p);
        if (*p != '"') break;

        char* key = parse_json_string(&p);
        if (!key) break;

        p = skip_ws(p);
        if (*p == ':') p++;
        p = skip_ws(p);

        char* value = parse_json_value_as_string(&p);

        if (key && value) {
            fd_sample_set(s, key, value);
        }

        free(key);
        free(value);

        p = skip_ws(p);
        if (*p == ',') p++;
    }

    return s;
}

/* Public wrapper — forwards to the static parse_json_object */
fd_sample* fd_parse_json_object(const char* text) {
    return parse_json_object(text);
}

fd_dataset* fd_load_jsonl(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    fd_dataset* ds = fd_new();
    if (!ds) {
        fclose(f);
        return NULL;
    }

    char* line = NULL;
    size_t len = 0;

    while (getline(&line, &len, f) != -1) {
        /* Skip empty lines */
        const char* p = line;
        p = skip_ws(p);
        if (*p == '\0' || *p == '\n') continue;

        fd_sample* s = parse_json_object(line);
        if (s && s->num_fields > 0) {
            fd_add_sample(ds, s);
        } else {
            fd_sample_free(s);
        }
    }

    free(line);
    fclose(f);
    return ds;
}

/* ============================================================= */
/* Data Loader                                                   */
/* ============================================================= */

struct fd_loader {
    fd_dataset* ds;
    int batch_size;
    bool shuffle;
    int current_idx;
    int* indices;  /* Shuffled indices */
};

fd_loader* fd_loader_new(fd_dataset* ds, int batch_size, bool shuffle) {
    if (!ds || batch_size <= 0) return NULL;

    fd_loader* loader = calloc(1, sizeof(fd_loader));
    if (!loader) return NULL;

    loader->ds = ds;
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->current_idx = 0;

    /* Create index array */
    loader->indices = malloc(ds->num_samples * sizeof(int));
    for (int i = 0; i < ds->num_samples; i++) {
        loader->indices[i] = i;
    }

    /* Shuffle if requested */
    if (shuffle && ds->num_samples > 1) {
        for (int i = ds->num_samples - 1; i > 0; i--) {
            int j = fd_rand() % (i + 1);
            int tmp = loader->indices[i];
            loader->indices[i] = loader->indices[j];
            loader->indices[j] = tmp;
        }
    }

    return loader;
}

void fd_loader_free(fd_loader* loader) {
    if (!loader) return;
    free(loader->indices);
    free(loader);
}

void fd_loader_reset(fd_loader* loader) {
    if (!loader) return;

    loader->current_idx = 0;

    /* Re-shuffle if shuffle is enabled */
    if (loader->shuffle && loader->ds->num_samples > 1) {
        for (int i = loader->ds->num_samples - 1; i > 0; i--) {
            int j = fd_rand() % (i + 1);
            int tmp = loader->indices[i];
            loader->indices[i] = loader->indices[j];
            loader->indices[j] = tmp;
        }
    }
}

fd_batch* fd_loader_next(fd_loader* loader) {
    if (!loader) return NULL;
    if (loader->current_idx >= loader->ds->num_samples) return NULL;

    int remaining = loader->ds->num_samples - loader->current_idx;
    int batch_size = (remaining < loader->batch_size) ? remaining : loader->batch_size;

    fd_batch* batch = calloc(1, sizeof(fd_batch));
    batch->samples = calloc(batch_size, sizeof(fd_sample*));
    batch->size = batch_size;

    for (int i = 0; i < batch_size; i++) {
        int idx = loader->indices[loader->current_idx + i];
        batch->samples[i] = loader->ds->samples[idx];
    }

    loader->current_idx += batch_size;
    return batch;
}

void fd_batch_free(fd_batch* batch) {
    if (!batch) return;
    /* Note: we don't free the samples, they belong to the dataset */
    free(batch->samples);
    free(batch);
}

int fd_loader_num_batches(fd_loader* loader) {
    if (!loader || loader->batch_size <= 0) return 0;
    return (loader->ds->num_samples + loader->batch_size - 1) / loader->batch_size;
}

/* ============================================================= */
/* SQLite Chat History Loading                                    */
/* ============================================================= */

#ifdef HAS_SQLITE
#include <sqlite3.h>

fd_dataset* fd_load_chat_sqlite(const char *db_path, int max_conversations) {
    if (!db_path) return NULL;

    sqlite3 *db = NULL;
    if (sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
        if (db) sqlite3_close(db);
        return NULL;
    }

    fd_dataset *ds = fd_new();
    if (!ds) { sqlite3_close(db); return NULL; }

    /* Query all conversations with their messages ordered by timestamp */
    const char *sql =
        "SELECT c.id, m.role, m.content "
        "FROM messages m "
        "JOIN conversations c ON m.conversation_id = c.id "
        "WHERE m.role IN ('user', 'assistant') "
        "  AND m.content IS NOT NULL "
        "  AND length(m.content) > 0 "
        "ORDER BY c.started_at, c.id, m.timestamp";

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        sqlite3_close(db);
        fd_free(ds);
        return NULL;
    }

    /* Accumulate messages per conversation, format as chat template */
    char *current_conv_id = NULL;
    char *text_buf = NULL;
    size_t text_len = 0;
    size_t text_cap = 0;
    int msg_count = 0;
    int conv_count = 0;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *conv_id = (const char*)sqlite3_column_text(stmt, 0);
        const char *role = (const char*)sqlite3_column_text(stmt, 1);
        const char *content = (const char*)sqlite3_column_text(stmt, 2);
        if (!conv_id || !role || !content) continue;

        /* New conversation? Flush the previous one. */
        if (!current_conv_id || strcmp(current_conv_id, conv_id) != 0) {
            if (text_buf && msg_count >= 2) {
                fd_sample *s = fd_sample_new();
                fd_sample_set(s, "text", text_buf);
                fd_add_sample(ds, s);
                conv_count++;
                if (max_conversations > 0 && conv_count >= max_conversations)
                    break;
            }
            free(current_conv_id);
            current_conv_id = strdup(conv_id);
            text_len = 0;
            msg_count = 0;
            if (!text_buf) { text_cap = 4096; text_buf = (char*)malloc(text_cap); }
            text_buf[0] = '\0';
        }

        /* Format: <|role|>\ncontent\n */
        size_t role_len = strlen(role);
        size_t content_len = strlen(content);
        size_t needed = text_len + 4 + role_len + 2 + content_len + 1 + 1;
        if (needed > text_cap) {
            text_cap = needed * 2;
            text_buf = (char*)realloc(text_buf, text_cap);
        }
        text_len += snprintf(text_buf + text_len, text_cap - text_len,
                              "<|%s|>\n%s\n", role, content);
        msg_count++;
    }

    /* Flush last conversation */
    if (text_buf && msg_count >= 2 &&
        (max_conversations <= 0 || conv_count < max_conversations)) {
        fd_sample *s = fd_sample_new();
        fd_sample_set(s, "text", text_buf);
        fd_add_sample(ds, s);
    }

    free(current_conv_id);
    free(text_buf);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return ds;
}

#else /* !HAS_SQLITE */

fd_dataset* fd_load_chat_sqlite(const char *db_path, int max_conversations) {
    (void)db_path;
    (void)max_conversations;
    fprintf(stderr, "fd_load_chat_sqlite: compiled without sqlite3 support\n");
    return NULL;
}

#endif /* HAS_SQLITE */
