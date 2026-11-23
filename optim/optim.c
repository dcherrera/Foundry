/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "optim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================= */
/* Optimizer Structure                                           */
/* ============================================================= */

struct fo_optimizer {
    fo_optim_type type;
    tm_tensor** params;
    int num_params;
    int step_count;
    float lr;

    /* SGD state */
    fo_sgd_config sgd;
    tm_tensor** momentum_buffers;  /* For SGD momentum */

    /* Adam state */
    fo_adam_config adam;
    tm_tensor** m;  /* First moment estimates */
    tm_tensor** v;  /* Second moment estimates */
    tm_tensor** v_max;  /* For AMSGrad */

    /* Scheduler */
    fo_scheduler_config scheduler;
    float base_lr;
};

/* ============================================================= */
/* Default Configurations                                        */
/* ============================================================= */

fo_sgd_config fo_sgd_default(float lr) {
    fo_sgd_config cfg = {0};
    cfg.lr = lr;
    cfg.momentum = 0.0f;
    cfg.weight_decay = 0.0f;
    cfg.dampening = 0.0f;
    cfg.nesterov = false;
    return cfg;
}

fo_adam_config fo_adam_default(float lr) {
    fo_adam_config cfg = {0};
    cfg.lr = lr;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.eps = 1e-8f;
    cfg.weight_decay = 0.0f;
    cfg.amsgrad = false;
    return cfg;
}

fo_scheduler_config fo_scheduler_default(void) {
    fo_scheduler_config cfg = {0};
    cfg.type = FO_SCHED_NONE;
    cfg.warmup_steps = 0;
    cfg.total_steps = 0;
    cfg.min_lr = 0.0f;
    cfg.step_size = 1;
    cfg.gamma = 0.1f;
    return cfg;
}

/* ============================================================= */
/* SGD Optimizer                                                 */
/* ============================================================= */

fo_optimizer* fo_sgd_new(tm_tensor** params, int num_params, fo_sgd_config* config) {
    if (!params || num_params <= 0 || !config) return NULL;

    fo_optimizer* opt = calloc(1, sizeof(fo_optimizer));
    if (!opt) return NULL;

    opt->type = FO_SGD;
    opt->params = params;
    opt->num_params = num_params;
    opt->lr = config->lr;
    opt->base_lr = config->lr;
    opt->sgd = *config;
    opt->step_count = 0;
    opt->scheduler = fo_scheduler_default();

    /* Allocate momentum buffers if needed */
    if (config->momentum != 0.0f) {
        opt->momentum_buffers = calloc(num_params, sizeof(tm_tensor*));
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                opt->momentum_buffers[i] = tm_zeros(params[i]->shape, params[i]->ndim, params[i]->dtype);
            }
        }
    }

    return opt;
}

static void sgd_step(fo_optimizer* opt) {
    float lr = opt->lr;
    float momentum = opt->sgd.momentum;
    float weight_decay = opt->sgd.weight_decay;
    float dampening = opt->sgd.dampening;
    bool nesterov = opt->sgd.nesterov;

    for (int i = 0; i < opt->num_params; i++) {
        tm_tensor* p = opt->params[i];
        if (!p || !p->grad) continue;

        float* data = tm_data_f32(p);
        float* grad = tm_data_f32(p->grad);

        /* Weight decay (L2 regularization) */
        if (weight_decay != 0.0f) {
            for (int64_t j = 0; j < p->numel; j++) {
                grad[j] += weight_decay * data[j];
            }
        }

        if (momentum != 0.0f && opt->momentum_buffers) {
            float* buf = tm_data_f32(opt->momentum_buffers[i]);

            if (opt->step_count == 0) {
                /* First step: initialize momentum buffer */
                memcpy(buf, grad, p->numel * sizeof(float));
            } else {
                /* Update momentum: buf = momentum * buf + (1 - dampening) * grad */
                for (int64_t j = 0; j < p->numel; j++) {
                    buf[j] = momentum * buf[j] + (1.0f - dampening) * grad[j];
                }
            }

            if (nesterov) {
                /* Nesterov: use grad + momentum * buf */
                for (int64_t j = 0; j < p->numel; j++) {
                    data[j] -= lr * (grad[j] + momentum * buf[j]);
                }
            } else {
                for (int64_t j = 0; j < p->numel; j++) {
                    data[j] -= lr * buf[j];
                }
            }
        } else {
            /* Simple SGD without momentum */
            for (int64_t j = 0; j < p->numel; j++) {
                data[j] -= lr * grad[j];
            }
        }
    }
}

/* ============================================================= */
/* Adam Optimizer                                                */
/* ============================================================= */

fo_optimizer* fo_adam_new(tm_tensor** params, int num_params, fo_adam_config* config) {
    if (!params || num_params <= 0 || !config) return NULL;

    fo_optimizer* opt = calloc(1, sizeof(fo_optimizer));
    if (!opt) return NULL;

    opt->type = FO_ADAM;
    opt->params = params;
    opt->num_params = num_params;
    opt->lr = config->lr;
    opt->base_lr = config->lr;
    opt->adam = *config;
    opt->step_count = 0;
    opt->scheduler = fo_scheduler_default();

    /* Allocate moment buffers */
    opt->m = calloc(num_params, sizeof(tm_tensor*));
    opt->v = calloc(num_params, sizeof(tm_tensor*));

    for (int i = 0; i < num_params; i++) {
        if (params[i]) {
            opt->m[i] = tm_zeros(params[i]->shape, params[i]->ndim, params[i]->dtype);
            opt->v[i] = tm_zeros(params[i]->shape, params[i]->ndim, params[i]->dtype);
        }
    }

    if (config->amsgrad) {
        opt->v_max = calloc(num_params, sizeof(tm_tensor*));
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                opt->v_max[i] = tm_zeros(params[i]->shape, params[i]->ndim, params[i]->dtype);
            }
        }
    }

    return opt;
}

fo_optimizer* fo_adamw_new(tm_tensor** params, int num_params, fo_adam_config* config) {
    fo_optimizer* opt = fo_adam_new(params, num_params, config);
    if (opt) {
        opt->type = FO_ADAMW;
    }
    return opt;
}

static void adam_step(fo_optimizer* opt) {
    float lr = opt->lr;
    float beta1 = opt->adam.beta1;
    float beta2 = opt->adam.beta2;
    float eps = opt->adam.eps;
    float weight_decay = opt->adam.weight_decay;
    bool amsgrad = opt->adam.amsgrad;
    bool is_adamw = (opt->type == FO_ADAMW);

    int t = opt->step_count + 1;  /* Step starts at 1 for bias correction */

    /* Bias correction factors */
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);

    for (int i = 0; i < opt->num_params; i++) {
        tm_tensor* p = opt->params[i];
        if (!p || !p->grad) continue;

        float* data = tm_data_f32(p);
        float* grad = tm_data_f32(p->grad);
        float* m = tm_data_f32(opt->m[i]);
        float* v = tm_data_f32(opt->v[i]);

        for (int64_t j = 0; j < p->numel; j++) {
            float g = grad[j];

            /* For Adam (not AdamW), add weight decay to gradient */
            if (!is_adamw && weight_decay != 0.0f) {
                g += weight_decay * data[j];
            }

            /* Update biased first moment estimate */
            m[j] = beta1 * m[j] + (1.0f - beta1) * g;

            /* Update biased second moment estimate */
            v[j] = beta2 * v[j] + (1.0f - beta2) * g * g;

            /* Bias-corrected estimates */
            float m_hat = m[j] / bc1;
            float v_hat = v[j] / bc2;

            if (amsgrad && opt->v_max) {
                float* vmax = tm_data_f32(opt->v_max[i]);
                if (v_hat > vmax[j]) vmax[j] = v_hat;
                v_hat = vmax[j];
            }

            /* Update parameter */
            float update = lr * m_hat / (sqrtf(v_hat) + eps);

            /* AdamW: decoupled weight decay */
            if (is_adamw && weight_decay != 0.0f) {
                data[j] -= lr * weight_decay * data[j];
            }

            data[j] -= update;
        }
    }
}

/* ============================================================= */
/* Common Operations                                             */
/* ============================================================= */

void fo_step(fo_optimizer* opt) {
    if (!opt) return;

    switch (opt->type) {
        case FO_SGD:
            sgd_step(opt);
            break;
        case FO_ADAM:
        case FO_ADAMW:
            adam_step(opt);
            break;
    }

    opt->step_count++;
}

void fo_zero_grad(fo_optimizer* opt) {
    if (!opt) return;

    for (int i = 0; i < opt->num_params; i++) {
        tm_tensor* p = opt->params[i];
        if (p && p->grad) {
            float* grad = tm_data_f32(p->grad);
            memset(grad, 0, p->grad->numel * sizeof(float));
        }
    }
}

float fo_get_lr(fo_optimizer* opt) {
    return opt ? opt->lr : 0.0f;
}

void fo_set_lr(fo_optimizer* opt, float lr) {
    if (opt) opt->lr = lr;
}

int fo_get_step(fo_optimizer* opt) {
    return opt ? opt->step_count : 0;
}

void fo_free(fo_optimizer* opt) {
    if (!opt) return;

    if (opt->momentum_buffers) {
        for (int i = 0; i < opt->num_params; i++) {
            tm_release(opt->momentum_buffers[i]);
        }
        free(opt->momentum_buffers);
    }

    if (opt->m) {
        for (int i = 0; i < opt->num_params; i++) {
            tm_release(opt->m[i]);
        }
        free(opt->m);
    }

    if (opt->v) {
        for (int i = 0; i < opt->num_params; i++) {
            tm_release(opt->v[i]);
        }
        free(opt->v);
    }

    if (opt->v_max) {
        for (int i = 0; i < opt->num_params; i++) {
            tm_release(opt->v_max[i]);
        }
        free(opt->v_max);
    }

    free(opt);
}

/* ============================================================= */
/* Learning Rate Scheduling                                      */
/* ============================================================= */

void fo_set_scheduler(fo_optimizer* opt, fo_scheduler_config* config) {
    if (!opt || !config) return;
    opt->scheduler = *config;
}

float fo_get_scheduled_lr(fo_optimizer* opt, int step) {
    if (!opt) return 0.0f;

    float base_lr = opt->base_lr;
    fo_scheduler_config* sched = &opt->scheduler;

    switch (sched->type) {
        case FO_SCHED_NONE:
            return base_lr;

        case FO_SCHED_STEP: {
            int num_decays = step / sched->step_size;
            return base_lr * powf(sched->gamma, (float)num_decays);
        }

        case FO_SCHED_COSINE: {
            if (sched->total_steps <= 0) return base_lr;
            float progress = (float)step / sched->total_steps;
            if (progress > 1.0f) progress = 1.0f;
            return sched->min_lr + 0.5f * (base_lr - sched->min_lr) *
                   (1.0f + cosf((float)M_PI * progress));
        }

        case FO_SCHED_LINEAR: {
            if (sched->total_steps <= 0) return base_lr;
            float progress = (float)step / sched->total_steps;
            if (progress > 1.0f) progress = 1.0f;
            return base_lr + (sched->min_lr - base_lr) * progress;
        }

        case FO_SCHED_WARMUP_COSINE: {
            if (step < sched->warmup_steps) {
                /* Linear warmup */
                return base_lr * ((float)(step + 1) / sched->warmup_steps);
            } else {
                /* Cosine decay after warmup */
                int decay_steps = sched->total_steps - sched->warmup_steps;
                if (decay_steps <= 0) return base_lr;
                float progress = (float)(step - sched->warmup_steps) / decay_steps;
                if (progress > 1.0f) progress = 1.0f;
                return sched->min_lr + 0.5f * (base_lr - sched->min_lr) *
                       (1.0f + cosf((float)M_PI * progress));
            }
        }
    }

    return base_lr;
}

void fo_scheduler_step(fo_optimizer* opt) {
    if (!opt) return;
    opt->lr = fo_get_scheduled_lr(opt, opt->step_count);
}

/* ============================================================= */
/* Gradient Clipping                                             */
/* ============================================================= */

float fo_clip_grad_norm(fo_optimizer* opt, float max_norm) {
    if (!opt || max_norm <= 0.0f) return 0.0f;

    /* Compute global norm */
    float total_norm = 0.0f;
    for (int i = 0; i < opt->num_params; i++) {
        tm_tensor* p = opt->params[i];
        if (!p || !p->grad) continue;

        float* grad = tm_data_f32(p->grad);
        for (int64_t j = 0; j < p->grad->numel; j++) {
            total_norm += grad[j] * grad[j];
        }
    }
    total_norm = sqrtf(total_norm);

    /* Clip if necessary */
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (int i = 0; i < opt->num_params; i++) {
            tm_tensor* p = opt->params[i];
            if (!p || !p->grad) continue;

            float* grad = tm_data_f32(p->grad);
            for (int64_t j = 0; j < p->grad->numel; j++) {
                grad[j] *= clip_coef;
            }
        }
    }

    return total_norm;
}

void fo_clip_grad_value(fo_optimizer* opt, float clip_value) {
    if (!opt || clip_value <= 0.0f) return;

    for (int i = 0; i < opt->num_params; i++) {
        tm_tensor* p = opt->params[i];
        if (!p || !p->grad) continue;

        float* grad = tm_data_f32(p->grad);
        for (int64_t j = 0; j < p->grad->numel; j++) {
            if (grad[j] > clip_value) grad[j] = clip_value;
            else if (grad[j] < -clip_value) grad[j] = -clip_value;
        }
    }
}
