/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * rfa_lora_merge — CLI tool for averaging multiple LoRA adapters
 *
 * Loads N adapter safetensors files, averages A/B weights element-wise,
 * saves the merged adapter. Used for cross-machine parallel training.
 *
 * Usage:
 *   ./examples/rfa_lora_merge \
 *     --output merged_adapter.safetensors \
 *     adapter_0.safetensors adapter_1.safetensors ... adapter_N.safetensors
 */

#include "rwkvx/rfa_train.h"
#include "lora/lora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ADAPTERS 64

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --output OUTPUT ADAPTER1 ADAPTER2 [... ADAPTER_N]\n\n", prog);
    fprintf(stderr, "Averages multiple LoRA adapter files element-wise.\n");
    fprintf(stderr, "All adapters must have the same structure (same layers/dimensions).\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --output PATH    Output merged adapter path (required)\n");
    fprintf(stderr, "  --help           Show this help message\n");
}

int main(int argc, char **argv) {
    const char *output_path = NULL;
    const char *adapter_paths[MAX_ADAPTERS];
    int n_adapters = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else {
            if (n_adapters >= MAX_ADAPTERS) {
                fprintf(stderr, "Error: max %d adapters supported\n", MAX_ADAPTERS);
                return 1;
            }
            adapter_paths[n_adapters++] = argv[i];
        }
    }

    if (!output_path || n_adapters < 2) {
        fprintf(stderr, "Error: need --output and at least 2 adapter files\n\n");
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "Loading %d adapters...\n", n_adapters);

    /* Load all adapters */
    fl_config lora_cfg = fl_config_default();
    fl_adapter **adapters = (fl_adapter**)calloc(n_adapters, sizeof(fl_adapter*));
    for (int i = 0; i < n_adapters; i++) {
        adapters[i] = fl_adapter_load(adapter_paths[i], &lora_cfg);
        if (!adapters[i]) {
            fprintf(stderr, "Error: failed to load %s\n", adapter_paths[i]);
            for (int j = 0; j < i; j++) fl_adapter_free(adapters[j]);
            free(adapters);
            return 1;
        }
        fprintf(stderr, "  [%d] %s: %d layers, %lld params\n",
                i, adapter_paths[i],
                fl_adapter_size(adapters[i]),
                (long long)fl_adapter_num_params(adapters[i]));
    }

    /* Verify all have same structure */
    int n_layers = fl_adapter_size(adapters[0]);
    for (int i = 1; i < n_adapters; i++) {
        if (fl_adapter_size(adapters[i]) != n_layers) {
            fprintf(stderr, "Error: adapter %d has %d layers (expected %d)\n",
                    i, fl_adapter_size(adapters[i]), n_layers);
            for (int j = 0; j < n_adapters; j++) fl_adapter_free(adapters[j]);
            free(adapters);
            return 1;
        }
    }

    /* Average */
    fprintf(stderr, "Averaging %d adapters (%d layers each)...\n",
            n_adapters, n_layers);
    fl_adapter *merged = rfa_lora_average(adapters, n_adapters);
    if (!merged) {
        fprintf(stderr, "Error: averaging failed\n");
        for (int j = 0; j < n_adapters; j++) fl_adapter_free(adapters[j]);
        free(adapters);
        return 1;
    }

    /* Save */
    if (fl_adapter_save(merged, output_path)) {
        fprintf(stderr, "Saved: %s (%d layers, %lld params)\n",
                output_path, fl_adapter_size(merged),
                (long long)fl_adapter_num_params(merged));
    } else {
        fprintf(stderr, "Error: failed to save %s\n", output_path);
    }

    /* Cleanup */
    fl_adapter_free(merged);
    for (int i = 0; i < n_adapters; i++)
        fl_adapter_free(adapters[i]);
    free(adapters);

    return 0;
}
