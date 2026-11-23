/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_SWARM_PROTOCOL_H
#define FOUNDRY_SWARM_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Swarm Protocol - Message format for coordinator/worker communication.
 *
 * All messages use a simple binary format:
 * ┌─────────┬─────────┬─────────────┬──────────────────┐
 * │ Magic   │ Type    │ Length      │ Payload          │
 * │ 4 bytes │ 1 byte  │ 4 bytes     │ Variable         │
 * └─────────┴─────────┴─────────────┴──────────────────┘
 */

/* Protocol magic number: "FDTC" (Foundry Distributed Training Cluster) */
#define SW_PROTOCOL_MAGIC 0x46445443

/* Protocol version */
#define SW_PROTOCOL_VERSION 1

/* Maximum message size (256 MB) */
#define SW_MAX_MESSAGE_SIZE (256 * 1024 * 1024)

/* Maximum worker ID length */
#define SW_MAX_WORKER_ID 64

/* Maximum path length */
#define SW_MAX_PATH 256

/* ============================================================= */
/* Message Types                                                 */
/* ============================================================= */

typedef enum {
    /* Registration */
    SW_MSG_REGISTER         = 0x01,  /* Worker → Coordinator: I exist */
    SW_MSG_REGISTER_ACK     = 0x02,  /* Coordinator → Worker: Welcome */
    SW_MSG_DEREGISTER       = 0x03,  /* Worker → Coordinator: I'm leaving */

    /* Health */
    SW_MSG_HEARTBEAT        = 0x10,  /* Worker → Coordinator: I'm alive */
    SW_MSG_HEARTBEAT_ACK    = 0x11,  /* Coordinator → Worker: Noted */

    /* Weights */
    SW_MSG_WEIGHTS_REQ      = 0x20,  /* Worker → Coordinator: Send weights */
    SW_MSG_WEIGHTS          = 0x21,  /* Coordinator → Worker: Here are weights */
    SW_MSG_WEIGHTS_META     = 0x22,  /* Coordinator → Worker: Weight metadata */

    /* Gradients */
    SW_MSG_GRADIENTS        = 0x30,  /* Worker → Coordinator: Here are gradients */
    SW_MSG_GRADIENTS_ACK    = 0x31,  /* Coordinator → Worker: Got them */

    /* Control */
    SW_MSG_START            = 0x40,  /* Coordinator → Worker: Start training */
    SW_MSG_SYNC             = 0x45,  /* Coordinator → Worker: Sync point */
    SW_MSG_CHECKPOINT       = 0x46,  /* Coordinator → Worker: Checkpoint now */
    SW_MSG_SHUTDOWN         = 0x47,  /* Coordinator → Worker: Training done */
    SW_MSG_PAUSE            = 0x48,  /* Coordinator → Worker: Pause training */
    SW_MSG_RESUME           = 0x49,  /* Coordinator → Worker: Resume training */

    /* Status */
    SW_MSG_STATUS_REQ       = 0x50,  /* Coordinator → Worker: Report status */
    SW_MSG_STATUS           = 0x51,  /* Worker → Coordinator: Current status */

    /* Error */
    SW_MSG_ERROR            = 0xFF,  /* Either direction: Error occurred */
} sw_msg_type;

/* ============================================================= */
/* Log Levels                                                    */
/* ============================================================= */

typedef enum {
    SW_LOG_DEBUG,
    SW_LOG_INFO,
    SW_LOG_WARN,
    SW_LOG_ERROR,
} sw_log_level;

/* ============================================================= */
/* Message Header                                                */
/* ============================================================= */

typedef struct {
    uint32_t magic;         /* SW_PROTOCOL_MAGIC */
    uint8_t type;           /* sw_msg_type */
    uint32_t length;        /* Payload length in bytes */
} __attribute__((packed)) sw_msg_header;

#define SW_HEADER_SIZE sizeof(sw_msg_header)

/* ============================================================= */
/* Payload Structures                                            */
/* ============================================================= */

/* Worker hardware profile */
typedef enum {
    SW_PROFILE_RPI3,        /* Raspberry Pi 3 */
    SW_PROFILE_RPI4_2G,     /* Raspberry Pi 4, 2GB */
    SW_PROFILE_RPI4_4G,     /* Raspberry Pi 4, 4GB */
    SW_PROFILE_RPI4_8G,     /* Raspberry Pi 4, 8GB */
    SW_PROFILE_LAPTOP_LIGHT,
    SW_PROFILE_LAPTOP_MID,
    SW_PROFILE_DESKTOP_MID,
    SW_PROFILE_DESKTOP_HEAVY,
    SW_PROFILE_SERVER,
    SW_PROFILE_UNKNOWN,
} sw_hardware_profile;

/* Compression type */
typedef enum {
    SW_COMPRESS_NONE,       /* No compression */
    SW_COMPRESS_FP16,       /* Float32 → Float16 */
    SW_COMPRESS_TOPK,       /* Top K% of gradients */
    SW_COMPRESS_QSGD,       /* Quantized SGD */
} sw_compress_type;

/* SW_MSG_REGISTER payload */
typedef struct {
    char worker_id[SW_MAX_WORKER_ID];
    uint8_t protocol_version;
    sw_hardware_profile profile;
    uint32_t num_cores;
    uint64_t memory_mb;
    uint32_t suggested_batch_size;
} __attribute__((packed)) sw_register_payload;

/* SW_MSG_REGISTER_ACK payload */
typedef struct {
    bool accepted;
    uint32_t assigned_shard;
    uint32_t total_shards;
    uint32_t batch_size;
    uint32_t accumulation_steps;
    sw_compress_type compression;
    float topk_ratio;           /* For SW_COMPRESS_TOPK */
    uint32_t global_step;       /* Current training step */
} __attribute__((packed)) sw_register_ack_payload;

/* SW_MSG_HEARTBEAT payload */
typedef struct {
    char worker_id[SW_MAX_WORKER_ID];
    uint32_t local_step;
    float current_loss;
    float samples_per_sec;
} __attribute__((packed)) sw_heartbeat_payload;

/* SW_MSG_WEIGHTS_META payload (sent before weights) */
typedef struct {
    uint32_t num_tensors;
    uint32_t global_step;
    float learning_rate;
    /* Followed by: for each tensor, sw_tensor_meta */
} __attribute__((packed)) sw_weights_meta_payload;

/* Tensor metadata */
typedef struct {
    char name[64];
    uint32_t ndim;
    uint64_t shape[8];
    uint64_t numel;
    uint8_t dtype;          /* tm_dtype */
} __attribute__((packed)) sw_tensor_meta;

/* SW_MSG_GRADIENTS payload header */
typedef struct {
    char worker_id[SW_MAX_WORKER_ID];
    uint32_t local_step;
    uint32_t num_tensors;
    sw_compress_type compression;
    uint32_t uncompressed_size;
    /* Followed by: compressed gradient data */
} __attribute__((packed)) sw_gradients_payload;

/* SW_MSG_STATUS payload */
typedef struct {
    char worker_id[SW_MAX_WORKER_ID];
    uint32_t local_step;
    uint32_t samples_processed;
    float current_loss;
    float avg_loss;
    float samples_per_sec;
    uint64_t memory_used_mb;
    bool is_training;
} __attribute__((packed)) sw_status_payload;

/* SW_MSG_ERROR payload */
typedef struct {
    uint32_t error_code;
    char message[256];
} __attribute__((packed)) sw_error_payload;

/* ============================================================= */
/* Message Construction                                          */
/* ============================================================= */

/* Allocate a message buffer with header */
uint8_t* sw_msg_alloc(sw_msg_type type, uint32_t payload_size);

/* Free a message buffer */
void sw_msg_free(uint8_t* msg);

/* Get header from message buffer */
sw_msg_header* sw_msg_get_header(uint8_t* msg);

/* Get payload pointer from message buffer */
void* sw_msg_get_payload(uint8_t* msg);

/* Get total message size (header + payload) */
uint32_t sw_msg_total_size(uint8_t* msg);

/* Validate message header */
bool sw_msg_validate(uint8_t* msg, size_t received_size);

/* ============================================================= */
/* Serialization Helpers                                         */
/* ============================================================= */

/* Write tensor data to buffer, returns bytes written */
size_t sw_serialize_tensor(uint8_t* buf, float* data, uint64_t numel,
                           sw_compress_type compression, float topk_ratio);

/* Read tensor data from buffer, returns bytes read */
size_t sw_deserialize_tensor(uint8_t* buf, float* data, uint64_t numel,
                             sw_compress_type compression);

/* Estimate compressed size */
size_t sw_estimate_compressed_size(uint64_t numel, sw_compress_type compression,
                                   float topk_ratio);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Get string name for message type */
const char* sw_msg_type_name(sw_msg_type type);

/* Get string name for hardware profile */
const char* sw_profile_name(sw_hardware_profile profile);

/* Auto-detect hardware profile */
sw_hardware_profile sw_detect_profile(void);

/* Get recommended batch size for profile */
uint32_t sw_profile_batch_size(sw_hardware_profile profile);

/* Get recommended accumulation steps for profile */
uint32_t sw_profile_accumulation_steps(sw_hardware_profile profile);

/* ============================================================= */
/* Compression Helpers                                           */
/* ============================================================= */

/* Convert FP32 to FP16 */
void sw_fp32_to_fp16(float* src, uint16_t* dst, size_t n);

/* Convert FP16 to FP32 */
void sw_fp16_to_fp32(uint16_t* src, float* dst, size_t n);

/* TopK compression: select top K values by magnitude */
void sw_topk_compress(float* values, size_t n, size_t k,
                      uint32_t* out_indices, float* out_values);

/* TopK decompression: scatter values back */
void sw_topk_decompress(uint32_t* indices, float* values, size_t k,
                        float* out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_SWARM_PROTOCOL_H */
