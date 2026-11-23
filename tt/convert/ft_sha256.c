/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_sha256.c
 * @brief Implementation of SHA-256, from FIPS 180-4 §4.1.2, §4.2.2, §5.1.1 and §6.2.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_sha256.h"

#include <string.h>

/* FIPS 180-4 §4.2.2: the first 32 bits of the fractional parts of the cube roots of the
 * first 64 primes. */
static const uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static uint32_t sha256_rotr(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32u - n));
}

/** One 64-byte block, §6.2.2. */
static void sha256_compress(uint32_t state[8], const unsigned char block[64])
{
    uint32_t w[64];
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (unsigned t = 0; t < 16u; t++)
        w[t] = ((uint32_t)block[4u * t] << 24) | ((uint32_t)block[4u * t + 1u] << 16) |
               ((uint32_t)block[4u * t + 2u] << 8) | (uint32_t)block[4u * t + 3u];
    for (unsigned t = 16; t < 64u; t++) {
        uint32_t s0 = sha256_rotr(w[t - 15u], 7) ^ sha256_rotr(w[t - 15u], 18) ^ (w[t - 15u] >> 3);
        uint32_t s1 = sha256_rotr(w[t - 2u], 17) ^ sha256_rotr(w[t - 2u], 19) ^ (w[t - 2u] >> 10);

        w[t] = w[t - 16u] + s0 + w[t - 7u] + s1;
    }
    for (unsigned t = 0; t < 64u; t++) {
        uint32_t big_s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t t1 = h + big_s1 + choose + SHA256_K[t] + w[t];
        uint32_t big_s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = big_s0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void ft_sha256_init(ft_sha256_ctx *ctx)
{
    /* §5.3.3: the first 32 bits of the fractional parts of the square roots of the first
     * eight primes. */
    static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    memcpy(ctx->state, initial, sizeof(ctx->state));
    ctx->total_bytes = 0;
    ctx->block_used = 0;
}

void ft_sha256_update(ft_sha256_ctx *ctx, const void *data, size_t bytes)
{
    const unsigned char *cursor = (const unsigned char *)data;

    ctx->total_bytes += bytes;
    if (ctx->block_used) {
        size_t take = sizeof(ctx->block) - ctx->block_used;

        if (take > bytes)
            take = bytes;
        memcpy(ctx->block + ctx->block_used, cursor, take);
        ctx->block_used += take;
        cursor += take;
        bytes -= take;
        if (ctx->block_used < sizeof(ctx->block))
            return;
        sha256_compress(ctx->state, ctx->block);
        ctx->block_used = 0;
    }
    while (bytes >= sizeof(ctx->block)) {
        sha256_compress(ctx->state, cursor);
        cursor += sizeof(ctx->block);
        bytes -= sizeof(ctx->block);
    }
    memcpy(ctx->block, cursor, bytes);
    ctx->block_used = bytes;
}

void ft_sha256_final(ft_sha256_ctx *ctx, unsigned char out[FT_SHA256_BYTES])
{
    uint64_t bit_length = ctx->total_bytes * 8u;

    /* §5.1.1: a 1 bit, zeros to 56 mod 64, then the 64-bit big-endian bit length. */
    ctx->block[ctx->block_used++] = 0x80u;
    if (ctx->block_used > 56u) {
        memset(ctx->block + ctx->block_used, 0, sizeof(ctx->block) - ctx->block_used);
        sha256_compress(ctx->state, ctx->block);
        ctx->block_used = 0;
    }
    memset(ctx->block + ctx->block_used, 0, 56u - ctx->block_used);
    for (unsigned i = 0; i < 8u; i++)
        ctx->block[56u + i] = (unsigned char)(bit_length >> (56u - 8u * i));
    sha256_compress(ctx->state, ctx->block);

    for (unsigned i = 0; i < 8u; i++) {
        out[4u * i] = (unsigned char)(ctx->state[i] >> 24);
        out[4u * i + 1u] = (unsigned char)(ctx->state[i] >> 16);
        out[4u * i + 2u] = (unsigned char)(ctx->state[i] >> 8);
        out[4u * i + 3u] = (unsigned char)ctx->state[i];
    }
}

void ft_sha256(const void *data, size_t bytes, unsigned char out[FT_SHA256_BYTES])
{
    ft_sha256_ctx ctx;

    ft_sha256_init(&ctx);
    ft_sha256_update(&ctx, data, bytes);
    ft_sha256_final(&ctx, out);
}

void ft_sha256_hex(const unsigned char digest[FT_SHA256_BYTES], char out[2u * FT_SHA256_BYTES + 1u])
{
    static const char digits[] = "0123456789abcdef";

    for (unsigned i = 0; i < FT_SHA256_BYTES; i++) {
        out[2u * i] = digits[digest[i] >> 4];
        out[2u * i + 1u] = digits[digest[i] & 0x0fu];
    }
    out[2u * FT_SHA256_BYTES] = '\0';
}
