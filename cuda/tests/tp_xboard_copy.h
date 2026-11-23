/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tp_xboard_copy.h
 * @brief Cross-board die-to-die copies for the TP benches (no P2P across K80 boards).
 *
 * Bank-1's boards each sit behind a PCIe gen2 x1 root port, and cross-board peer copies
 * arrive corrupt, so every cross-board byte is staged through host memory. This prices
 * the ways to do that, each ordered by device events only (no host sync):
 *
 *   drv   cuMemcpyPeerAsync between unpeered contexts (the driver stages it).
 *   ce    copy engines: D2H on the source die's stream into pinned host memory, event,
 *         H2D on the destination die's stream.
 *   zc    zero-copy kernels: the source die's SMs write mapped pinned host memory
 *         (fc_copy_2d), event, the destination die's SMs read it (fc_copy_2d).
 *   zcce  zc write, ce read.
 *   cezc  ce write, zc read.
 *
 * Pinned memory is portable + device-mapped (cuMemHostAlloc via dlsym); under UVA its
 * device address is the host address in every context. Bench-only: it wraps that
 * memory in an fc_buffer by mirroring the struct (see xb_wrap in the .c).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef TP_XBOARD_COPY_H
#define TP_XBOARD_COPY_H

#include "cuda/fc_cuda.h"

#include <stddef.h>

enum { XB_DRV = 0, XB_CE, XB_ZC, XB_ZCCE, XB_CEZC, XB_N };

/* Method by name ("drv", "ce", ...), or -1. */
int xb_parse(const char *name);
const char *xb_name(int method);

/* Two pinned slots of slot_bytes per die (ping-pong) and an event ring per die.
 * Returns 0, or -1 (no libcuda symbols / allocation failed). */
int xb_init(fc_cuda_ctx *const *ctx, int nd, size_t slot_bytes);

/* Copy bytes from src (die a, byte offset src_off) to dst (die b, dst_off). Die a's
 * producer must be complete at ev_ready (recorded on a's stream; NULL = record one
 * now); the copy lands in b's stream order. slot (0/1) picks a's staging slot: a
 * symmetric dependent exchange alternates it. Returns 0, or -1. */
int xb_copy(int method, int a, fc_buffer *src, size_t src_off, int b, fc_buffer *dst,
            size_t dst_off, size_t bytes, fc_event *ev_ready, int slot);

/* mode xbv: every method across boards (dies 0<->2, 1<->3 of 4): a verified copy each
 * way, then a dependent swap chain on one cross pair and on both at once, and one-way
 * bandwidth, over the message sizes TP4 sends. Returns 0, or 1 on a failed verify. */
int xb_mode_verify(fc_cuda_ctx *const *ctx, int nd);

#endif
