/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_ggml.c
 * @brief  ggml type table (see header). Geometry as in ggml's type traits.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_ggml.h"

#include <stddef.h>

#include "ingot/core/ingot_error.h"

typedef struct {
    const char *name;
    uint16_t weights;
    uint16_t bytes;
} ggml_type_info;

/* Indexed by ggml type id; name NULL = unassigned or removed id. */
static const ggml_type_info TYPES[] = {
    [0] = {"F32", 1, 4},          [1] = {"F16", 1, 2},          [2] = {"Q4_0", 32, 18},
    [3] = {"Q4_1", 32, 20},       [6] = {"Q5_0", 32, 22},       [7] = {"Q5_1", 32, 24},
    [8] = {"Q8_0", 32, 34},       [9] = {"Q8_1", 32, 36},       [10] = {"Q2_K", 256, 84},
    [11] = {"Q3_K", 256, 110},    [12] = {"Q4_K", 256, 144},    [13] = {"Q5_K", 256, 176},
    [14] = {"Q6_K", 256, 210},    [15] = {"Q8_K", 256, 292},    [16] = {"IQ2_XXS", 256, 66},
    [17] = {"IQ2_XS", 256, 74},   [18] = {"IQ3_XXS", 256, 98},  [19] = {"IQ1_S", 256, 50},
    [20] = {"IQ4_NL", 32, 18},    [21] = {"IQ3_S", 256, 110},   [22] = {"IQ2_S", 256, 82},
    [23] = {"IQ4_XS", 256, 136},  [24] = {"I8", 1, 1},          [25] = {"I16", 1, 2},
    [26] = {"I32", 1, 4},         [27] = {"I64", 1, 8},         [28] = {"F64", 1, 8},
    [29] = {"IQ1_M", 256, 56},    [30] = {"BF16", 1, 2},        [34] = {"TQ1_0", 256, 54},
    [35] = {"TQ2_0", 256, 66},    [39] = {"MXFP4", 32, 17},
};

enum { TYPE_COUNT = sizeof(TYPES) / sizeof(TYPES[0]) };

int ingot_ggml_geometry(uint32_t type, uint32_t *weights, uint32_t *bytes) {
    if (type >= TYPE_COUNT || !TYPES[type].name)
        return -INGOT_EUNSUPPORTED;
    *weights = TYPES[type].weights;
    *bytes = TYPES[type].bytes;
    return 0;
}

const char *ingot_ggml_name(uint32_t type) {
    return type < TYPE_COUNT ? TYPES[type].name : NULL;
}

ingot_dtype ingot_ggml_dtype(uint32_t type) {
    switch (type) {
    case 0: return INGOT_DTYPE_F32;
    case 1: return INGOT_DTYPE_F16;
    case 24: return INGOT_DTYPE_I8;
    case 25: return INGOT_DTYPE_I16;
    case 26: return INGOT_DTYPE_I32;
    case 27: return INGOT_DTYPE_I64;
    case 28: return INGOT_DTYPE_F64;
    case 30: return INGOT_DTYPE_BF16;
    default: return (ingot_dtype)(INGOT_DTYPE_GGML + type);
    }
}

int ingot_dtype_ggml_type(ingot_dtype dtype, uint32_t *type) {
    switch (dtype) {
    case INGOT_DTYPE_F32: *type = 0; return 0;
    case INGOT_DTYPE_F16: *type = 1; return 0;
    case INGOT_DTYPE_I8: *type = 24; return 0;
    case INGOT_DTYPE_I16: *type = 25; return 0;
    case INGOT_DTYPE_I32: *type = 26; return 0;
    case INGOT_DTYPE_I64: *type = 27; return 0;
    case INGOT_DTYPE_F64: *type = 28; return 0;
    case INGOT_DTYPE_BF16: *type = 30; return 0;
    default: break;
    }
    if (dtype < INGOT_DTYPE_GGML || dtype >= INGOT_DTYPE_GGML + TYPE_COUNT ||
        !TYPES[dtype - INGOT_DTYPE_GGML].name)
        return -INGOT_EUNSUPPORTED;
    *type = (uint32_t)(dtype - INGOT_DTYPE_GGML);
    return 0;
}
