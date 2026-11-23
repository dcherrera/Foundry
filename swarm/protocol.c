/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "protocol.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

/* ============================================================= */
/* Message Construction                                          */
/* ============================================================= */

uint8_t* sw_msg_alloc(sw_msg_type type, uint32_t payload_size) {
    if (payload_size > SW_MAX_MESSAGE_SIZE - SW_HEADER_SIZE) {
        return NULL;
    }

    uint32_t total = SW_HEADER_SIZE + payload_size;
    uint8_t* msg = calloc(1, total);
    if (!msg) return NULL;

    sw_msg_header* hdr = (sw_msg_header*)msg;
    hdr->magic = SW_PROTOCOL_MAGIC;
    hdr->type = (uint8_t)type;
    hdr->length = payload_size;

    return msg;
}

void sw_msg_free(uint8_t* msg) {
    free(msg);
}

sw_msg_header* sw_msg_get_header(uint8_t* msg) {
    if (!msg) return NULL;
    return (sw_msg_header*)msg;
}

void* sw_msg_get_payload(uint8_t* msg) {
    if (!msg) return NULL;
    return msg + SW_HEADER_SIZE;
}

uint32_t sw_msg_total_size(uint8_t* msg) {
    if (!msg) return 0;
    sw_msg_header* hdr = sw_msg_get_header(msg);
    return SW_HEADER_SIZE + hdr->length;
}

bool sw_msg_validate(uint8_t* msg, size_t received_size) {
    if (!msg || received_size < SW_HEADER_SIZE) {
        return false;
    }

    sw_msg_header* hdr = sw_msg_get_header(msg);

    /* Check magic */
    if (hdr->magic != SW_PROTOCOL_MAGIC) {
        return false;
    }

    /* Check size */
    uint32_t expected_size = SW_HEADER_SIZE + hdr->length;
    if (received_size < expected_size) {
        return false;
    }

    /* Check payload size limit */
    if (hdr->length > SW_MAX_MESSAGE_SIZE - SW_HEADER_SIZE) {
        return false;
    }

    return true;
}

/* ============================================================= */
/* Serialization Helpers                                         */
/* ============================================================= */

/* Simple FP16 conversion (no denormals, limited range) */
static uint16_t float_to_fp16(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;

    if (exp <= 0) {
        return (uint16_t)sign;  /* Zero or denormal → zero */
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);  /* Inf */
    }
    return (uint16_t)(sign | (exp << 10) | mant);
}

static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        return 0.0f;
    } else if (exp == 31) {
        uint32_t inf = sign | 0x7F800000;
        return *(float*)&inf;
    }

    exp = exp - 15 + 127;
    uint32_t result = sign | (exp << 23) | (mant << 13);
    return *(float*)&result;
}

/* Comparison for qsort - sort by absolute value descending */
typedef struct {
    uint32_t index;
    float abs_value;
} indexed_value;

static int compare_indexed(const void* a, const void* b) {
    float va = ((indexed_value*)a)->abs_value;
    float vb = ((indexed_value*)b)->abs_value;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

size_t sw_serialize_tensor(uint8_t* buf, float* data, uint64_t numel,
                           sw_compress_type compression, float topk_ratio) {
    if (!buf || !data || numel == 0) return 0;

    size_t written = 0;

    switch (compression) {
        case SW_COMPRESS_NONE: {
            memcpy(buf, data, numel * sizeof(float));
            written = numel * sizeof(float);
            break;
        }

        case SW_COMPRESS_FP16: {
            uint16_t* out = (uint16_t*)buf;
            for (uint64_t i = 0; i < numel; i++) {
                out[i] = float_to_fp16(data[i]);
            }
            written = numel * sizeof(uint16_t);
            break;
        }

        case SW_COMPRESS_TOPK: {
            /* Find top K values by absolute magnitude */
            uint32_t k = (uint32_t)(numel * topk_ratio);
            if (k == 0) k = 1;
            if (k > numel) k = (uint32_t)numel;

            /* Create indexed array for sorting */
            indexed_value* indexed = malloc(numel * sizeof(indexed_value));
            if (!indexed) return 0;

            for (uint64_t i = 0; i < numel; i++) {
                indexed[i].index = (uint32_t)i;
                indexed[i].abs_value = fabsf(data[i]);
            }

            /* Partial sort - only need top K */
            qsort(indexed, numel, sizeof(indexed_value), compare_indexed);

            /* Write: K (4 bytes), then K pairs of (index, value) */
            uint32_t* k_ptr = (uint32_t*)buf;
            *k_ptr = k;
            written += sizeof(uint32_t);

            uint8_t* ptr = buf + sizeof(uint32_t);
            for (uint32_t i = 0; i < k; i++) {
                uint32_t idx = indexed[i].index;
                float val = data[idx];
                memcpy(ptr, &idx, sizeof(uint32_t));
                ptr += sizeof(uint32_t);
                memcpy(ptr, &val, sizeof(float));
                ptr += sizeof(float);
                written += sizeof(uint32_t) + sizeof(float);
            }

            free(indexed);
            break;
        }

        case SW_COMPRESS_QSGD: {
            /* Quantize to 8-bit with scale factor */
            float max_val = 0.0f;
            for (uint64_t i = 0; i < numel; i++) {
                float absv = fabsf(data[i]);
                if (absv > max_val) max_val = absv;
            }

            /* Write scale factor first */
            memcpy(buf, &max_val, sizeof(float));
            written += sizeof(float);

            /* Quantize to int8 */
            int8_t* out = (int8_t*)(buf + sizeof(float));
            float scale = (max_val > 0) ? (127.0f / max_val) : 0.0f;
            for (uint64_t i = 0; i < numel; i++) {
                int32_t q = (int32_t)roundf(data[i] * scale);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                out[i] = (int8_t)q;
            }
            written += numel;
            break;
        }
    }

    return written;
}

size_t sw_deserialize_tensor(uint8_t* buf, float* data, uint64_t numel,
                             sw_compress_type compression) {
    if (!buf || !data || numel == 0) return 0;

    size_t read = 0;

    switch (compression) {
        case SW_COMPRESS_NONE: {
            memcpy(data, buf, numel * sizeof(float));
            read = numel * sizeof(float);
            break;
        }

        case SW_COMPRESS_FP16: {
            uint16_t* in = (uint16_t*)buf;
            for (uint64_t i = 0; i < numel; i++) {
                data[i] = fp16_to_float(in[i]);
            }
            read = numel * sizeof(uint16_t);
            break;
        }

        case SW_COMPRESS_TOPK: {
            /* Zero out array first */
            memset(data, 0, numel * sizeof(float));

            /* Read K */
            uint32_t k = *(uint32_t*)buf;
            read += sizeof(uint32_t);

            /* Read K pairs of (index, value) */
            uint8_t* ptr = buf + sizeof(uint32_t);
            for (uint32_t i = 0; i < k; i++) {
                uint32_t idx;
                float val;
                memcpy(&idx, ptr, sizeof(uint32_t));
                ptr += sizeof(uint32_t);
                memcpy(&val, ptr, sizeof(float));
                ptr += sizeof(float);
                read += sizeof(uint32_t) + sizeof(float);

                if (idx < numel) {
                    data[idx] = val;
                }
            }
            break;
        }

        case SW_COMPRESS_QSGD: {
            /* Read scale factor */
            float max_val;
            memcpy(&max_val, buf, sizeof(float));
            read += sizeof(float);

            /* Dequantize */
            int8_t* in = (int8_t*)(buf + sizeof(float));
            float scale = max_val / 127.0f;
            for (uint64_t i = 0; i < numel; i++) {
                data[i] = (float)in[i] * scale;
            }
            read += numel;
            break;
        }
    }

    return read;
}

size_t sw_estimate_compressed_size(uint64_t numel, sw_compress_type compression,
                                   float topk_ratio) {
    switch (compression) {
        case SW_COMPRESS_NONE:
            return numel * sizeof(float);

        case SW_COMPRESS_FP16:
            return numel * sizeof(uint16_t);

        case SW_COMPRESS_TOPK: {
            uint32_t k = (uint32_t)(numel * topk_ratio);
            if (k == 0) k = 1;
            /* K header + K*(index + value) */
            return sizeof(uint32_t) + k * (sizeof(uint32_t) + sizeof(float));
        }

        case SW_COMPRESS_QSGD:
            /* Scale factor + int8 values */
            return sizeof(float) + numel;
    }

    return numel * sizeof(float);
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

const char* sw_msg_type_name(sw_msg_type type) {
    switch (type) {
        case SW_MSG_REGISTER:       return "REGISTER";
        case SW_MSG_REGISTER_ACK:   return "REGISTER_ACK";
        case SW_MSG_DEREGISTER:     return "DEREGISTER";
        case SW_MSG_HEARTBEAT:      return "HEARTBEAT";
        case SW_MSG_HEARTBEAT_ACK:  return "HEARTBEAT_ACK";
        case SW_MSG_WEIGHTS_REQ:    return "WEIGHTS_REQ";
        case SW_MSG_WEIGHTS:        return "WEIGHTS";
        case SW_MSG_WEIGHTS_META:   return "WEIGHTS_META";
        case SW_MSG_GRADIENTS:      return "GRADIENTS";
        case SW_MSG_GRADIENTS_ACK:  return "GRADIENTS_ACK";
        case SW_MSG_SYNC:           return "SYNC";
        case SW_MSG_CHECKPOINT:     return "CHECKPOINT";
        case SW_MSG_SHUTDOWN:       return "SHUTDOWN";
        case SW_MSG_PAUSE:          return "PAUSE";
        case SW_MSG_RESUME:         return "RESUME";
        case SW_MSG_STATUS_REQ:     return "STATUS_REQ";
        case SW_MSG_STATUS:         return "STATUS";
        case SW_MSG_ERROR:          return "ERROR";
        default:                    return "UNKNOWN";
    }
}

const char* sw_profile_name(sw_hardware_profile profile) {
    switch (profile) {
        case SW_PROFILE_RPI3:           return "rpi3";
        case SW_PROFILE_RPI4_2G:        return "rpi4-2g";
        case SW_PROFILE_RPI4_4G:        return "rpi4-4g";
        case SW_PROFILE_RPI4_8G:        return "rpi4-8g";
        case SW_PROFILE_LAPTOP_LIGHT:   return "laptop-light";
        case SW_PROFILE_LAPTOP_MID:     return "laptop-mid";
        case SW_PROFILE_DESKTOP_MID:    return "desktop-mid";
        case SW_PROFILE_DESKTOP_HEAVY:  return "desktop-heavy";
        case SW_PROFILE_SERVER:         return "server";
        default:                        return "unknown";
    }
}

static uint64_t get_total_memory_mb(void) {
#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return (info.totalram * info.mem_unit) / (1024 * 1024);
    }
#endif

#ifdef __APPLE__
    int64_t memsize;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
        return (uint64_t)memsize / (1024 * 1024);
    }
#endif

    return 0;
}

static int get_num_cores(void) {
#ifdef __linux__
    return get_nprocs();
#endif

#ifdef __APPLE__
    int ncpu;
    size_t len = sizeof(ncpu);
    if (sysctlbyname("hw.ncpu", &ncpu, &len, NULL, 0) == 0) {
        return ncpu;
    }
#endif

    return 1;
}

sw_hardware_profile sw_detect_profile(void) {
    uint64_t mem_mb = get_total_memory_mb();
    int cores = get_num_cores();

    /* Check for Raspberry Pi (ARM with specific memory sizes) */
#ifdef __aarch64__
    if (mem_mb <= 1024) return SW_PROFILE_RPI3;
    if (mem_mb <= 2048) return SW_PROFILE_RPI4_2G;
    if (mem_mb <= 4096) return SW_PROFILE_RPI4_4G;
    if (mem_mb <= 8192 && cores <= 4) return SW_PROFILE_RPI4_8G;
#endif

    /* x86 profiles based on memory and cores */
    if (mem_mb >= 64000 && cores >= 16) return SW_PROFILE_SERVER;
    if (mem_mb >= 32000 && cores >= 8) return SW_PROFILE_DESKTOP_HEAVY;
    if (mem_mb >= 16000 && cores >= 4) return SW_PROFILE_DESKTOP_MID;
    if (mem_mb >= 16000) return SW_PROFILE_LAPTOP_MID;
    if (mem_mb >= 8000) return SW_PROFILE_LAPTOP_LIGHT;

    return SW_PROFILE_UNKNOWN;
}

uint32_t sw_profile_batch_size(sw_hardware_profile profile) {
    switch (profile) {
        case SW_PROFILE_RPI3:           return 2;
        case SW_PROFILE_RPI4_2G:        return 4;
        case SW_PROFILE_RPI4_4G:        return 8;
        case SW_PROFILE_RPI4_8G:        return 16;
        case SW_PROFILE_LAPTOP_LIGHT:   return 16;
        case SW_PROFILE_LAPTOP_MID:     return 32;
        case SW_PROFILE_DESKTOP_MID:    return 64;
        case SW_PROFILE_DESKTOP_HEAVY:  return 128;
        case SW_PROFILE_SERVER:         return 256;
        default:                        return 8;
    }
}

uint32_t sw_profile_accumulation_steps(sw_hardware_profile profile) {
    switch (profile) {
        case SW_PROFILE_RPI3:           return 100;
        case SW_PROFILE_RPI4_2G:        return 100;
        case SW_PROFILE_RPI4_4G:        return 50;
        case SW_PROFILE_RPI4_8G:        return 50;
        case SW_PROFILE_LAPTOP_LIGHT:   return 50;
        case SW_PROFILE_LAPTOP_MID:     return 25;
        case SW_PROFILE_DESKTOP_MID:    return 25;
        case SW_PROFILE_DESKTOP_HEAVY:  return 10;
        case SW_PROFILE_SERVER:         return 5;
        default:                        return 50;
    }
}

/* ============================================================= */
/* Compression Helpers                                           */
/* ============================================================= */

void sw_fp32_to_fp16(float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = float_to_fp16(src[i]);
    }
}

void sw_fp16_to_fp32(uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fp16_to_float(src[i]);
    }
}

void sw_topk_compress(float* values, size_t n, size_t k,
                      uint32_t* out_indices, float* out_values) {
    if (k > n) k = n;
    if (k == 0) return;

    /* Create indexed array for sorting */
    indexed_value* indexed = malloc(n * sizeof(indexed_value));
    if (!indexed) return;

    for (size_t i = 0; i < n; i++) {
        indexed[i].index = (uint32_t)i;
        indexed[i].abs_value = fabsf(values[i]);
    }

    /* Sort by absolute value (descending) */
    qsort(indexed, n, sizeof(indexed_value), compare_indexed);

    /* Output top K */
    for (size_t i = 0; i < k; i++) {
        out_indices[i] = indexed[i].index;
        out_values[i] = values[indexed[i].index];
    }

    free(indexed);
}

void sw_topk_decompress(uint32_t* indices, float* values, size_t k,
                        float* out, size_t n) {
    /* Zero output first */
    memset(out, 0, n * sizeof(float));

    /* Scatter values */
    for (size_t i = 0; i < k; i++) {
        if (indices[i] < n) {
            out[indices[i]] = values[i];
        }
    }
}
