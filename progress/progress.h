/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_PROGRESS_H
#define FOUNDRY_PROGRESS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Progress bar and training metrics reporting.
 * Replaces Python's tqdm.
 */

typedef struct fp_progress fp_progress;

/* Training state for logging */
typedef struct {
    int epoch;
    int step;
    float loss;
    float lr;
    float throughput;  /* samples/sec or tokens/sec */
} fp_train_state;

/* Create/destroy progress bar */
fp_progress* fp_new(int64_t total, const char* desc);
fp_progress* fp_new_nested(fp_progress* parent, int64_t total, const char* desc);
void fp_free(fp_progress* p);

/* Update progress */
void fp_update(fp_progress* p, int64_t n);
void fp_set(fp_progress* p, int64_t n);
void fp_set_postfix(fp_progress* p, const char* fmt, ...);

/* Finish and print final state */
void fp_close(fp_progress* p);

/* Accessors */
int64_t fp_total(fp_progress* p);
int64_t fp_current(fp_progress* p);
double fp_elapsed(fp_progress* p);
double fp_rate(fp_progress* p);
double fp_eta(fp_progress* p);
float fp_percent(fp_progress* p);

/* Training-specific logging */
void fp_log_step(fp_train_state* state);
void fp_log_epoch(fp_train_state* state);

/* Configuration */
void fp_set_width(fp_progress* p, int width);
void fp_set_unit(fp_progress* p, const char* unit);
void fp_disable(fp_progress* p);

/* Refresh display (usually called automatically) */
void fp_refresh(fp_progress* p);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_PROGRESS_H */
