/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_input.h
 * @brief  Byte source for the streaming encoder and decoder: a regular file (seekable), a pipe or
 *         stdin (forward-only), or a caller's read callback (forward-only, e.g. a download that
 *         reconnects). Optionally SHA-256s every byte consumed in order.
 *
 * Sequential reads (ingot_input_read / ingot_input_skip) advance the position. Random reads
 * (ingot_input_read_at) use pread on a seekable source and leave the position alone; on a
 * forward-only source they skip forward to the offset, so offsets must not go backwards.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_INPUT_H
#define FOUNDRY_INGOT_INPUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ingot_input ingot_input;

/**
 * Read callback: fill up to n bytes of buf, set *got (0 = end of input).
 * @return 0 or a negated ingot_error
 */
typedef int (*ingot_read_fn)(void *ctx, void *buf, size_t n, size_t *got);

/**
 * Open a path ("-" = stdin). Regular files are seekable; pipes, FIFOs and stdin are forward-only.
 *
 * @param path  File path or "-"
 * @param out   Out: input, closed with ingot_input_close
 * @return      0, -INGOT_EIO or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_input_open_path(const char *path, ingot_input **out);

/**
 * Wrap an open file descriptor (seekable if it is a regular file).
 *
 * @param fd     Descriptor
 * @param owned  Non-zero: ingot_input_close closes fd
 * @param out    Out: input
 * @return       0 or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_input_open_fd(int fd, int owned, ingot_input **out);

/**
 * Wrap a read callback (forward-only).
 *
 * @return 0 or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_input_open_callback(ingot_read_fn fn, void *ctx, ingot_input **out);

/**
 * Close (NULL is a no-op).
 *
 * Thread-safe: no (no other call on the handle may run)   Reentrant: yes   Signal-safe: no
 */
void ingot_input_close(ingot_input *in);

/**
 * 1 if read_at uses pread (regular file), else 0.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_input_seekable(const ingot_input *in);

/**
 * Size of a seekable source at open, else 0.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_input_size(const ingot_input *in);

/**
 * Bytes consumed sequentially so far.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_input_pos(const ingot_input *in);

/**
 * Read exactly n bytes at the position.
 *
 * @return 0, -INGOT_ETRUNCATED (input ended first) or -INGOT_EIO
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
int ingot_input_read(ingot_input *in, void *buf, size_t n);

/**
 * Consume n bytes (hashed like a read).
 *
 * @return 0, -INGOT_ETRUNCATED or -INGOT_EIO
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
int ingot_input_skip(ingot_input *in, uint64_t n);

/**
 * Read n bytes at absolute offset `at` (see the file comment for forward-only sources).
 *
 * @return 0, -INGOT_ETRUNCATED, -INGOT_EIO, or -INGOT_EUNSUPPORTED (forward-only source and
 *         `at` is behind the position)
 *
 * Thread-safe: seekable sources yes (pread), forward-only no   Reentrant: yes   Signal-safe: no
 */
int ingot_input_read_at(ingot_input *in, uint64_t at, void *buf, size_t n);

/**
 * Start SHA-256 over every byte consumed from now on (call at position 0 to hash the source).
 *
 * @return 0 or -INGOT_EINVAL (position is not 0)
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
int ingot_input_hash_start(ingot_input *in);

/**
 * Consume the rest of the input to its end, then write the SHA-256 of all of it.
 *
 * @param in      Input with hashing started
 * @param digest  Out: 32 bytes
 * @return        0, -INGOT_EINVAL (hashing not started) or -INGOT_EIO
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
int ingot_input_hash_finish(ingot_input *in, uint8_t digest[32]);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_INPUT_H */
