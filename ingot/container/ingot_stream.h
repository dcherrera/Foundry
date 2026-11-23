/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_stream.h
 * @brief  Streaming decoder: reads a version 2 or 3 .ingot file once, in order (a download pipe
 *         works), and writes the exact source file in order through a write callback. Checks
 *         every tensor hash, every base-tensor hash, the file size and the SHA-256 of the
 *         rebuilt file against the header.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Memory: a small multiple of the 2^18-weight block size per thread, the directory, and each
 * tensor's decode tables (largest: a BF16 MoE tensor with per-expert tables, 128 KB per expert).
 * It does not grow with model or tensor size. Version 3 adds, whole: each tensor decoded ahead of
 * its place in the output (payloads in decode order) and each predict partner until its last
 * dependent is decoded; for a Qwen3.8 layer about gate + up (0.36 GB) at most.
 */
#ifndef FOUNDRY_INGOT_STREAM_H
#define FOUNDRY_INGOT_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_format.h"
#include "ingot/container/ingot_header.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/core/ingot_input.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Output callback: write all n bytes in order. @return 0 or a negated ingot_error */
typedef int (*ingot_write_fn)(void *ctx, const void *buf, size_t n);

typedef struct {
    int threads;
    ingot_base_set *base;        /* required when the file has delta tensors */
    int check_base_sha256;       /* also hash the base file and compare it to the header
                                    (always done for a forward-only base) */
    ingot_progress_fn progress;  /* optional: output bytes so far, source size */
    void *progress_ctx;
} ingot_stream_options;

/** What was rebuilt. */
typedef struct {
    ingot_source_kind kind;
    uint64_t source_bytes;
    uint8_t sha256[32];          /* of the rebuilt file (= the header's) */
    uint64_t coded_bytes;        /* source bytes of coded (non-raw) tensors */
} ingot_stream_result;

/**
 * Decode `in` (at position 0) into the write callback.
 *
 * @param in         .ingot input
 * @param write      Output callback
 * @param write_ctx  Passed to write
 * @param opt        Options (NULL: 1 thread, no base)
 * @param result     Out (may be NULL)
 * @return           0, -INGOT_EFORMAT, -INGOT_EUNSUPPORTED (version 1 or unknown version: use
 *                   the random-access reader for version 1), -INGOT_ETRUNCATED (input ended
 *                   early), -INGOT_ECORRUPT (a check failed), -INGOT_EBASE (base missing or
 *                   wrong), -INGOT_EIO, -INGOT_ENOMEM, or the write callback's error
 *
 * Thread-safe: yes, with distinct inputs and base sets   Reentrant: yes   Signal-safe: no
 */
int ingot_stream_decode(ingot_input *in, ingot_write_fn write, void *write_ctx,
                        const ingot_stream_options *opt, ingot_stream_result *result);

/**
 * ingot_stream_decode into a file (created or truncated; removed on any failure).
 *
 * Thread-safe: yes, with distinct inputs and base sets   Reentrant: yes   Signal-safe: no
 */
int ingot_stream_decode_to_file(ingot_input *in, const char *dst_path,
                                const ingot_stream_options *opt, ingot_stream_result *result);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_STREAM_H */
