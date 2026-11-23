/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Foundry Training Example: MLP with Adam Optimizer
 *
 * Demonstrates the complete training pipeline:
 * - Tensor operations with autograd
 * - Adam optimizer with learning rate scheduling
 * - MSE loss with automatic differentiation
 * - Progress tracking
 */

#include "../tensor/tensor.h"
#include "../optim/optim.h"
#include "../progress/progress.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ============================================================= */
/* Simple MLP Model                                              */
/* ============================================================= */

typedef struct {
    /* Layer 1: 2 -> 16 */
    tm_tensor* W1;
    tm_tensor* b1;

    /* Layer 2: 16 -> 1 */
    tm_tensor* W2;
    tm_tensor* b2;

    /* Parameter array for optimizer */
    tm_tensor* params[4];
    int num_params;
} mlp_model;

/* Note: We now use tm_add with broadcasting for bias addition */

mlp_model* mlp_new(int hidden_size) {
    mlp_model* m = calloc(1, sizeof(mlp_model));
    if (!m) return NULL;

    int64_t w1_shape[] = {2, hidden_size};
    int64_t b1_shape[] = {hidden_size};
    int64_t w2_shape[] = {hidden_size, 1};
    int64_t b2_shape[] = {1};

    m->W1 = tm_randn(w1_shape, 2, TM_F32);
    m->b1 = tm_zeros(b1_shape, 1, TM_F32);
    m->W2 = tm_randn(w2_shape, 2, TM_F32);
    m->b2 = tm_zeros(b2_shape, 1, TM_F32);

    /* Scale weights (Xavier initialization) */
    float scale1 = sqrtf(2.0f / 2.0f);
    float scale2 = sqrtf(2.0f / hidden_size);
    float* w1_data = tm_data_f32(m->W1);
    float* w2_data = tm_data_f32(m->W2);
    for (int64_t i = 0; i < m->W1->numel; i++) w1_data[i] *= scale1;
    for (int64_t i = 0; i < m->W2->numel; i++) w2_data[i] *= scale2;

    /* Enable gradients */
    tm_set_requires_grad(m->W1, true);
    tm_set_requires_grad(m->b1, true);
    tm_set_requires_grad(m->W2, true);
    tm_set_requires_grad(m->b2, true);

    /* Setup params array for optimizer */
    m->params[0] = m->W1;
    m->params[1] = m->b1;
    m->params[2] = m->W2;
    m->params[3] = m->b2;
    m->num_params = 4;

    return m;
}

void mlp_free(mlp_model* m) {
    if (!m) return;
    tm_release(m->W1);
    tm_release(m->b1);
    tm_release(m->W2);
    tm_release(m->b2);
    free(m);
}

/* Forward pass: x -> linear -> relu -> linear -> sigmoid */
tm_tensor* mlp_forward(mlp_model* m, tm_tensor* x) {
    /* Layer 1: h = relu(x @ W1 + b1) - using broadcasting for bias */
    tm_tensor* z1_pre = tm_matmul(x, m->W1);
    tm_tensor* z1 = tm_add(z1_pre, m->b1);  /* Broadcasting: [batch, hidden] + [hidden] */
    tm_tensor* h = tm_relu(z1);

    /* Layer 2: out = sigmoid(h @ W2 + b2) - using broadcasting for bias */
    tm_tensor* z2_pre = tm_matmul(h, m->W2);
    tm_tensor* z2 = tm_add(z2_pre, m->b2);  /* Broadcasting: [batch, 1] + [1] */
    tm_tensor* out = tm_sigmoid(z2);

    tm_release(z1_pre);
    tm_release(z1);
    tm_release(h);
    tm_release(z2_pre);
    tm_release(z2);

    return out;
}

/* Manual backward pass (since we don't have full autograd for matmul+bias yet) */
void mlp_backward(mlp_model* m, tm_tensor* x, tm_tensor* pred, tm_tensor* target) {
    int batch_size = (int)x->shape[0];
    int hidden_size = (int)m->W1->shape[1];

    float* pred_data = tm_data_f32(pred);
    float* target_data = tm_data_f32(target);

    /* d_loss/d_pred for MSE: 2 * (pred - target) / n */
    float* d_out = malloc(batch_size * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        d_out[i] = 2.0f * (pred_data[i] - target_data[i]) / batch_size;
    }

    /* Sigmoid backward: d_sigmoid = sigmoid * (1 - sigmoid) * d_out */
    float* d_z2 = malloc(batch_size * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        float s = pred_data[i];
        d_z2[i] = s * (1.0f - s) * d_out[i];
    }

    /* Recompute hidden layer */
    tm_tensor* z1_pre = tm_matmul(x, m->W1);
    tm_tensor* z1 = tm_add(z1_pre, m->b1);  /* Broadcasting */
    tm_tensor* h = tm_relu(z1);
    float* h_data = tm_data_f32(h);

    /* Ensure gradients exist */
    if (!m->W2->grad) m->W2->grad = tm_zeros(m->W2->shape, m->W2->ndim, TM_F32);
    if (!m->b2->grad) m->b2->grad = tm_zeros(m->b2->shape, m->b2->ndim, TM_F32);
    if (!m->W1->grad) m->W1->grad = tm_zeros(m->W1->shape, m->W1->ndim, TM_F32);
    if (!m->b1->grad) m->b1->grad = tm_zeros(m->b1->shape, m->b1->ndim, TM_F32);

    /* Gradient for W2: h^T @ d_z2 */
    float* W2_grad = tm_data_f32(m->W2->grad);
    for (int j = 0; j < hidden_size; j++) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += h_data[i * hidden_size + j] * d_z2[i];
        }
        W2_grad[j] += sum;
    }

    /* Gradient for b2 */
    float* b2_grad = tm_data_f32(m->b2->grad);
    for (int i = 0; i < batch_size; i++) {
        b2_grad[0] += d_z2[i];
    }

    /* Gradient for hidden layer: d_z2 @ W2^T */
    float* W2_data = tm_data_f32(m->W2);
    float* d_h = malloc(batch_size * hidden_size * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            d_h[i * hidden_size + j] = d_z2[i] * W2_data[j];
        }
    }

    /* ReLU backward */
    float* z1_data = tm_data_f32(z1);
    float* d_z1 = malloc(batch_size * hidden_size * sizeof(float));
    for (int i = 0; i < batch_size * hidden_size; i++) {
        d_z1[i] = z1_data[i] > 0 ? d_h[i] : 0.0f;
    }

    /* Gradient for W1: x^T @ d_z1 */
    float* x_data = tm_data_f32(x);
    float* W1_grad = tm_data_f32(m->W1->grad);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                sum += x_data[b * 2 + i] * d_z1[b * hidden_size + j];
            }
            W1_grad[i * hidden_size + j] += sum;
        }
    }

    /* Gradient for b1 */
    float* b1_grad = tm_data_f32(m->b1->grad);
    for (int j = 0; j < hidden_size; j++) {
        for (int i = 0; i < batch_size; i++) {
            b1_grad[j] += d_z1[i * hidden_size + j];
        }
    }

    free(d_out);
    free(d_z2);
    free(d_h);
    free(d_z1);
    tm_release(z1_pre);
    tm_release(z1);
    tm_release(h);
}

/* Compute MSE loss */
float compute_mse(tm_tensor* pred, tm_tensor* target) {
    float* p = tm_data_f32(pred);
    float* t = tm_data_f32(target);
    float mse = 0.0f;
    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = p[i] - t[i];
        mse += diff * diff;
    }
    return mse / pred->numel;
}

/* ============================================================= */
/* Main Training Loop                                            */
/* ============================================================= */

int main(void) {
    printf("\n");
    printf("===========================================\n");
    printf("  Foundry Training Example: XOR with Adam\n");
    printf("===========================================\n\n");

    srand((unsigned int)time(NULL));

    /* XOR dataset */
    static float X_data[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    static float Y_data[] = {0.0f, 1.0f, 1.0f, 0.0f};

    int64_t x_shape[] = {4, 2};
    int64_t y_shape[] = {4, 1};

    tm_tensor* X = tm_from_data(X_data, x_shape, 2, TM_F32);
    tm_tensor* Y = tm_from_data(Y_data, y_shape, 2, TM_F32);

    /* Create model */
    int hidden_size = 16;
    mlp_model* model = mlp_new(hidden_size);
    if (!model) {
        printf("Failed to create model\n");
        return 1;
    }

    /* Create Adam optimizer */
    fo_adam_config adam_cfg = fo_adam_default(0.01f);
    fo_optimizer* optimizer = fo_adam_new(model->params, model->num_params, &adam_cfg);

    /* Training hyperparameters */
    int num_epochs = 500;
    int print_every = 50;

    printf("Model: 2-layer MLP (2 -> %d -> 1)\n", hidden_size);
    printf("Optimizer: Adam (lr=%.3f, beta1=%.1f, beta2=%.3f)\n",
           adam_cfg.lr, adam_cfg.beta1, adam_cfg.beta2);
    printf("Epochs: %d\n\n", num_epochs);

    /* Training loop */
    fp_progress* progress = fp_new(num_epochs, "Training");

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        /* Zero gradients */
        fo_zero_grad(optimizer);

        /* Forward pass */
        tm_tensor* pred = mlp_forward(model, X);

        /* Compute loss */
        float loss = compute_mse(pred, Y);

        if (epoch == 0) initial_loss = loss;
        if (epoch == num_epochs - 1) final_loss = loss;

        /* Backward pass */
        mlp_backward(model, X, pred, Y);

        /* Optimizer step */
        fo_step(optimizer);

        /* Progress update */
        fp_update(progress, 1);

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            printf("Epoch %4d | Loss: %.6f | LR: %.4f\n",
                   epoch + 1, loss, fo_get_lr(optimizer));
        }

        tm_release(pred);
    }

    fp_close(progress);

    /* Final evaluation */
    printf("\n");
    printf("===========================================\n");
    printf("  Training Complete!\n");
    printf("===========================================\n\n");

    printf("Initial loss: %.6f\n", initial_loss);
    printf("Final loss:   %.6f\n", final_loss);
    printf("Improvement:  %.2fx\n\n", initial_loss / (final_loss + 1e-8f));

    printf("Predictions vs Targets:\n");
    printf("-----------------------\n");

    tm_tensor* final_pred = mlp_forward(model, X);
    float* pred_data = tm_data_f32(final_pred);

    const char* inputs[] = {"0 XOR 0", "0 XOR 1", "1 XOR 0", "1 XOR 1"};
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        float rounded = pred_data[i] > 0.5f ? 1.0f : 0.0f;
        bool is_correct = (rounded == Y_data[i]);
        if (is_correct) correct++;
        printf("  %s = %.3f (target: %.0f) %s\n",
               inputs[i], pred_data[i], Y_data[i],
               is_correct ? "[OK]" : "[MISS]");
    }

    printf("\nAccuracy: %d/4 (%.0f%%)\n", correct, 100.0f * correct / 4.0f);

    if (correct == 4) {
        printf("\n[SUCCESS] Model learned XOR perfectly!\n");
    } else if (final_loss < initial_loss * 0.1f) {
        printf("\n[PROGRESS] Significant improvement. May need more epochs.\n");
    }

    /* Cleanup */
    tm_release(final_pred);
    tm_release(X);
    tm_release(Y);
    fo_free(optimizer);
    mlp_free(model);

    printf("\n");
    return 0;
}
