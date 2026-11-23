/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * barswap.c — Userspace library for BAR Swap kernel module v2
 *
 * Opens /dev/bar_swap, issues ioctls (QUERY, ACQUIRE, RELEASE), and
 * provides VRAM mmap via /dev/mem.
 */

#include "barswap.h"
#include "bar_swap_ko/bar_swap.h"

#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

struct bs_context {
    int fd;
    struct bs_query_info info;
};

bs_context* bs_init(void)
{
    bs_context *ctx = calloc(1, sizeof(bs_context));
    if (!ctx)
        return NULL;

    ctx->fd = open("/dev/bar_swap", O_RDWR);
    if (ctx->fd < 0) {
        /* Try read-only as fallback */
        ctx->fd = open("/dev/bar_swap", O_RDONLY);
        if (ctx->fd < 0) {
            free(ctx);
            return NULL;
        }
    }

    /* Query current state */
    memset(&ctx->info, 0, sizeof(ctx->info));
    if (ioctl(ctx->fd, BAR_SWAP_QUERY, &ctx->info) < 0) {
        close(ctx->fd);
        free(ctx);
        return NULL;
    }

    return ctx;
}

void bs_free(bs_context *ctx)
{
    if (!ctx)
        return;
    if (ctx->fd >= 0)
        close(ctx->fd);
    free(ctx);
}

uint32_t bs_num_dies(const bs_context *ctx)
{
    if (!ctx) return 0;
    return ctx->info.n_dies;
}

uint32_t bs_num_slots(const bs_context *ctx)
{
    if (!ctx) return 0;
    return ctx->info.n_slots;
}

int bs_is_mapped(const bs_context *ctx, int die_id)
{
    if (!ctx) return 0;
    if (die_id < 0 || die_id >= (int)ctx->info.n_dies) return 0;
    if (die_id >= BS_MAX_DIES) return 0;
    return ctx->info.mapped[die_id] ? 1 : 0;
}

bs_gpu* bs_acquire(bs_context *ctx, int die_id)
{
    struct bs_acquire_req req;
    bs_gpu *gpu;

    if (!ctx || ctx->fd < 0)
        return NULL;

    memset(&req, 0, sizeof(req));
    req.die_id = (uint32_t)die_id;
    req.flags = 0;

    if (ioctl(ctx->fd, BAR_SWAP_ACQUIRE, &req) < 0)
        return NULL;

    gpu = calloc(1, sizeof(bs_gpu));
    if (!gpu)
        return NULL;

    gpu->die_id = die_id;
    gpu->bar1_addr = req.bar1_addr;
    gpu->bar1_size = req.bar1_size;
    gpu->vram = NULL;
    gpu->vram_mapped_size = 0;
    gpu->mem_fd = -1;

    /* mmap VRAM via /dev/mem */
    gpu->mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (gpu->mem_fd >= 0) {
        gpu->vram = mmap(NULL, (size_t)gpu->bar1_size,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         gpu->mem_fd, (off_t)gpu->bar1_addr);
        if (gpu->vram == MAP_FAILED) {
            gpu->vram = NULL;
        } else {
            gpu->vram_mapped_size = (size_t)gpu->bar1_size;
        }
    }

    /* Refresh cached state */
    bs_refresh(ctx);

    return gpu;
}

void bs_release(bs_context *ctx, bs_gpu *gpu)
{
    struct bs_release_req req;

    if (!gpu)
        return;

    /* Unmap VRAM */
    if (gpu->vram && gpu->vram_mapped_size > 0)
        munmap(gpu->vram, gpu->vram_mapped_size);

    /* Close /dev/mem */
    if (gpu->mem_fd >= 0)
        close(gpu->mem_fd);

    /* Issue RELEASE ioctl */
    if (ctx && ctx->fd >= 0) {
        memset(&req, 0, sizeof(req));
        req.die_id = (uint32_t)gpu->die_id;
        req.flags = 0;
        ioctl(ctx->fd, BAR_SWAP_RELEASE, &req);

        /* Refresh cached state */
        bs_refresh(ctx);
    }

    free(gpu);
}

int bs_active_die(bs_context *ctx)
{
    struct bs_query_info info;

    if (!ctx || ctx->fd < 0)
        return -1;

    memset(&info, 0, sizeof(info));
    if (ioctl(ctx->fd, BAR_SWAP_QUERY, &info) < 0)
        return -1;

    if (info.active_die == 0xFFFFFFFF)
        return -1;

    return (int)info.active_die;
}

int bs_refresh(bs_context *ctx)
{
    if (!ctx || ctx->fd < 0)
        return -1;

    memset(&ctx->info, 0, sizeof(ctx->info));
    if (ioctl(ctx->fd, BAR_SWAP_QUERY, &ctx->info) < 0)
        return -1;

    return 0;
}
