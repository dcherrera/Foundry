/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_OPTIM_H
#define FOUNDRY_OPTIM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../tensor/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Optimizers for training neural networks.
 * Includes SGD, Adam, AdamW with learning rate scheduling.
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

typedef enum {
    FO_SGD,
    FO_ADAM,
    FO_ADAMW,
} fo_optim_type;

/* SGD configuration */
typedef struct {
    float lr;           /* Learning rate */
    float momentum;     /* Momentum factor (0 = no momentum) */
    float weight_decay; /* L2 regularization */
    float dampening;    /* Dampening for momentum */
    bool nesterov;      /* Use Nesterov momentum */
} fo_sgd_config;

/* Adam configuration */
typedef struct {
    float lr;           /* Learning rate */
    float beta1;        /* Exponential decay for first moment (default: 0.9) */
    float beta2;        /* Exponential decay for second moment (default: 0.999) */
    float eps;          /* Numerical stability (default: 1e-8) */
    float weight_decay; /* Weight decay (0 for Adam, >0 for AdamW) */
    bool amsgrad;       /* Use AMSGrad variant */
} fo_adam_config;

/* Learning rate scheduler types */
typedef enum {
    FO_SCHED_NONE,
    FO_SCHED_STEP,
    FO_SCHED_COSINE,
    FO_SCHED_LINEAR,
    FO_SCHED_WARMUP_COSINE,
} fo_scheduler_type;

/* Scheduler configuration */
typedef struct {
    fo_scheduler_type type;
    int warmup_steps;       /* Steps for warmup */
    int total_steps;        /* Total training steps */
    float min_lr;           /* Minimum learning rate */
    float step_size;        /* For step scheduler */
    float gamma;            /* Decay factor for step scheduler */
} fo_scheduler_config;

/* Optimizer state (opaque) */
typedef struct fo_optimizer fo_optimizer;

/* ============================================================= */
/* Default Configurations                                        */
/* ============================================================= */

fo_sgd_config fo_sgd_default(float lr);
fo_adam_config fo_adam_default(float lr);
fo_scheduler_config fo_scheduler_default(void);

/* ============================================================= */
/* Optimizer Creation                                            */
/* ============================================================= */

/* Create SGD optimizer */
fo_optimizer* fo_sgd_new(tm_tensor** params, int num_params, fo_sgd_config* config);

/* Create Adam optimizer */
fo_optimizer* fo_adam_new(tm_tensor** params, int num_params, fo_adam_config* config);

/* Create AdamW optimizer (Adam with decoupled weight decay) */
fo_optimizer* fo_adamw_new(tm_tensor** params, int num_params, fo_adam_config* config);

/* Free optimizer */
void fo_free(fo_optimizer* opt);

/* ============================================================= */
/* Optimizer Operations                                          */
/* ============================================================= */

/* Perform optimization step (updates parameters using gradients) */
void fo_step(fo_optimizer* opt);

/* Zero all gradients */
void fo_zero_grad(fo_optimizer* opt);

/* Get current learning rate */
float fo_get_lr(fo_optimizer* opt);

/* Set learning rate */
void fo_set_lr(fo_optimizer* opt, float lr);

/* Get step count */
int fo_get_step(fo_optimizer* opt);

/* ============================================================= */
/* Learning Rate Scheduling                                      */
/* ============================================================= */

/* Attach scheduler to optimizer */
void fo_set_scheduler(fo_optimizer* opt, fo_scheduler_config* config);

/* Update learning rate based on scheduler (call after each step) */
void fo_scheduler_step(fo_optimizer* opt);

/* Get scheduled learning rate for given step */
float fo_get_scheduled_lr(fo_optimizer* opt, int step);

/* ============================================================= */
/* Gradient Clipping                                             */
/* ============================================================= */

/* Clip gradients by global norm, returns the norm before clipping */
float fo_clip_grad_norm(fo_optimizer* opt, float max_norm);

/* Clip gradients by value */
void fo_clip_grad_value(fo_optimizer* opt, float clip_value);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_OPTIM_H */
