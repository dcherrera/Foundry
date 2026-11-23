/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cli.h
 * @brief  Pieces of the `ingot` CLI shared between its source files.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_CLI_H
#define FOUNDRY_INGOT_CLI_H

#include <stdint.h>

#include "ingot/container/ingot_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Print why opening or decoding `path` failed with rc, in words a user can act on (not an .ingot
 * file, unsupported version with the version number, truncated download, corrupt data, missing
 * or wrong base with the base identity from the header, ...).
 *
 * Thread-safe: no (writes stderr)   Reentrant: no   Signal-safe: no
 */
void ingot_cli_error(const char *cmd, const char *path, int rc);

/**
 * Progress line on stderr ("\r" updates, only when stderr is a terminal and not quiet).
 * ctx: the command name (const char *).
 *
 * Thread-safe: no (writes stderr)   Reentrant: no   Signal-safe: no
 */
void ingot_cli_progress(void *ctx, uint64_t done, uint64_t total);

/**
 * End the progress line (newline if one was drawn).
 *
 * Thread-safe: no (writes stderr)   Reentrant: no   Signal-safe: no
 */
void ingot_cli_progress_end(void);

/**
 * Lower-case hex of 32 bytes into out (65 bytes).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_cli_hex(const uint8_t digest[32], char out[65]);

/**
 * Name of a payload mode ("raw", "plain", ... "predict"; "?" if unknown).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const char *ingot_cli_mode_name(ingot_mode m);

/**
 * Encode entry report for --tensor-log (ctx: a FILE *): name, dtype, shape, mode,
 * payload_bytes, v2_bytes, prefix_bytes, tab-separated.
 *
 * Thread-safe: no (writes ctx)   Reentrant: no   Signal-safe: no
 */
void ingot_cli_log_entry(void *ctx, const ingot_entry *e, uint64_t v2_bytes,
                         uint64_t prefix_bytes);

/**
 * `ingot info FILE`: header (version, source kind and size, sha256, base) and per-tensor mode,
 * size and bits/element. Returns the exit code.
 *
 * Thread-safe: no (writes stdout)   Reentrant: no   Signal-safe: no
 */
int ingot_cli_info(const char *path);

/**
 * `ingot st-ls HEADER`: "name dtype shape begin end" per tensor of a (header-only) model file.
 *
 * Thread-safe: no (writes stdout)   Reentrant: no   Signal-safe: no
 */
int ingot_cli_st_ls(const char *path);

/**
 * `ingot st-wrap NAME DTYPE AxB NBYTES`: one-tensor safetensors header to stdout.
 *
 * Thread-safe: no (writes stdout)   Reentrant: no   Signal-safe: no
 */
int ingot_cli_st_wrap(const char *name, const char *dtype, const char *shape,
                      const char *nbytes);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_CLI_H */
