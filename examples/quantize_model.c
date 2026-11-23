/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../quantization/export.h"

/* ============================================================= */
/* Progress display                                              */
/* ============================================================= */

static void progress_callback(int current, int total, const char* name, void* ud) {
    (void)ud;
    int bar_width = 40;
    float pct = (float)current / total;
    int filled = (int)(pct * bar_width);

    printf("\r  [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %3d%% (%d/%d) %s", (int)(pct * 100), current, total, name);
    fflush(stdout);

    if (current == total) printf("\n");
}

/* ============================================================= */
/* Usage                                                         */
/* ============================================================= */

static void usage(const char* prog) {
    printf("Usage: %s \\\n", prog);
    printf("  --input model.safetensors \\\n");
    printf("  --output model-quantized.safetensors \\\n");
    printf("  --dtype int8|int4|ternary \\\n");
    printf("  [--granularity per_tensor|per_channel|per_group] \\\n");
    printf("  [--group-size 128] \\\n");
    printf("  [--threshold 0.0]\n");
    printf("\nQuantizes a safetensors model to INT8, INT4, or ternary format.\n");
}

/* ============================================================= */
/* Main                                                          */
/* ============================================================= */

int main(int argc, char** argv) {
    const char* input = NULL;
    const char* output = NULL;
    const char* dtype_str = NULL;
    const char* gran_str = NULL;
    int group_size = 128;
    float threshold = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            dtype_str = argv[++i];
        } else if (strcmp(argv[i], "--granularity") == 0 && i + 1 < argc) {
            gran_str = argv[++i];
        } else if (strcmp(argv[i], "--group-size") == 0 && i + 1 < argc) {
            group_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!input || !output || !dtype_str) {
        fprintf(stderr, "Error: --input, --output, and --dtype are required\n\n");
        usage(argv[0]);
        return 1;
    }

    /* Parse dtype */
    fq_export_config cfg = fq_export_config_defaults();

    if (strcmp(dtype_str, "int8") == 0) {
        cfg.target_dtype = FQ_INT8;
    } else if (strcmp(dtype_str, "int4") == 0) {
        cfg.target_dtype = FQ_INT4;
        cfg.granularity = FQ_PER_GROUP; /* INT4 defaults to per-group */
    } else if (strcmp(dtype_str, "ternary") == 0) {
        cfg.target_dtype = FQ_TERNARY;
    } else {
        fprintf(stderr, "Error: unknown dtype '%s' (use int8, int4, or ternary)\n", dtype_str);
        return 1;
    }

    /* Parse granularity */
    if (gran_str) {
        if (strcmp(gran_str, "per_tensor") == 0) {
            cfg.granularity = FQ_PER_TENSOR;
        } else if (strcmp(gran_str, "per_channel") == 0) {
            cfg.granularity = FQ_PER_CHANNEL;
        } else if (strcmp(gran_str, "per_group") == 0) {
            cfg.granularity = FQ_PER_GROUP;
        } else {
            fprintf(stderr, "Error: unknown granularity '%s'\n", gran_str);
            return 1;
        }
    }

    cfg.group_size = group_size;
    cfg.ternary_threshold = threshold;
    cfg.progress_cb = progress_callback;

    /* Get input file size */
    struct stat st;
    if (stat(input, &st) != 0) {
        fprintf(stderr, "Error: cannot stat '%s'\n", input);
        return 1;
    }
    size_t input_size = st.st_size;

    printf("Quantizing model:\n");
    printf("  Input:       %s (%.1f MB)\n", input, input_size / (1024.0 * 1024.0));
    printf("  Output:      %s\n", output);
    printf("  Dtype:       %s\n", fq_dtype_name(cfg.target_dtype));
    printf("  Granularity: %s\n",
           cfg.granularity == FQ_PER_TENSOR ? "per_tensor" :
           cfg.granularity == FQ_PER_CHANNEL ? "per_channel" : "per_group");
    if (cfg.granularity == FQ_PER_GROUP) {
        printf("  Group size:  %d\n", cfg.group_size);
    }
    if (cfg.target_dtype == FQ_TERNARY && cfg.ternary_threshold > 0.0f) {
        printf("  Threshold:   %.4f\n", cfg.ternary_threshold);
    }
    printf("\n");

    bool ok = fq_export_quantized(input, output, &cfg);

    if (!ok) {
        fprintf(stderr, "\nError: quantization failed\n");
        return 1;
    }

    /* Report compression ratio */
    if (stat(output, &st) == 0) {
        size_t output_size = st.st_size;
        printf("\n  Input:  %.1f MB\n", input_size / (1024.0 * 1024.0));
        printf("  Output: %.1f MB\n", output_size / (1024.0 * 1024.0));
        printf("  Ratio:  %.1fx compression\n", (double)input_size / output_size);
    }

    printf("\nDone.\n");
    return 0;
}
