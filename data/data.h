/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_DATA_H
#define FOUNDRY_DATA_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Dataset loading and batching.
 * Replaces HuggingFace datasets.
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Single data sample */
typedef struct {
    char** keys;          /* Field names */
    char** values;        /* Field values (all stored as strings) */
    int num_fields;
} fd_sample;

/* Dataset */
typedef struct fd_dataset fd_dataset;

/* Data loader for batching */
typedef struct fd_loader fd_loader;

/* Batch of samples */
typedef struct {
    fd_sample** samples;
    int size;
} fd_batch;

/* ============================================================= */
/* Dataset Loading                                               */
/* ============================================================= */

/* Load dataset from JSONL file */
fd_dataset* fd_load_jsonl(const char* path);

/* Public wrapper over the static parse_json_object in data.c */
fd_sample* fd_parse_json_object(const char* text);

/* Load conversations from TeamIDE sqlite database.
 * Formats as RWKV chat template: <|user|>\n...\n<|assistant|>\n...
 * Each conversation becomes one sample with a "text" field.
 * max_conversations: limit number of conversations (0 = all).
 * Requires linking with -lsqlite3. */
fd_dataset* fd_load_chat_sqlite(const char *db_path, int max_conversations);

/* Create empty dataset */
fd_dataset* fd_new(void);

/* Free dataset */
void fd_free(fd_dataset* ds);

/* Add sample to dataset */
void fd_add_sample(fd_dataset* ds, fd_sample* sample);

/* Get number of samples */
int fd_len(fd_dataset* ds);

/* Get sample by index */
fd_sample* fd_get(fd_dataset* ds, int idx);

/* Shuffle dataset in-place */
void fd_shuffle(fd_dataset* ds);

/* Split dataset into train/test */
void fd_split(fd_dataset* ds, float ratio, fd_dataset** train, fd_dataset** test);

/* ============================================================= */
/* Sample Operations                                             */
/* ============================================================= */

/* Create sample with fields */
fd_sample* fd_sample_new(void);

/* Free sample */
void fd_sample_free(fd_sample* s);

/* Add field to sample */
void fd_sample_set(fd_sample* s, const char* key, const char* value);

/* Get field from sample (returns NULL if not found) */
const char* fd_sample_get(fd_sample* s, const char* key);

/* Check if field exists */
bool fd_sample_has(fd_sample* s, const char* key);

/* Clone sample */
fd_sample* fd_sample_clone(fd_sample* s);

/* ============================================================= */
/* Data Loader                                                   */
/* ============================================================= */

/* Create data loader with batching */
fd_loader* fd_loader_new(fd_dataset* ds, int batch_size, bool shuffle);

/* Free loader */
void fd_loader_free(fd_loader* loader);

/* Reset loader to beginning */
void fd_loader_reset(fd_loader* loader);

/* Get next batch (returns NULL when exhausted) */
fd_batch* fd_loader_next(fd_loader* loader);

/* Free batch */
void fd_batch_free(fd_batch* batch);

/* Get number of batches */
int fd_loader_num_batches(fd_loader* loader);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Print dataset info */
void fd_print(fd_dataset* ds);

/* Print sample */
void fd_sample_print(fd_sample* s);

/* Seed random number generator for shuffling */
void fd_seed(uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_DATA_H */
