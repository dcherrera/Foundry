/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_hash.c
 * @brief  Per-tensor 64-bit content hash (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_hash.h"

#include <string.h>

static inline uint64_t mix_word(uint64_t h, uint64_t w) {
    h = (h ^ w) * 0x9E3779B97F4A7C15ull;
    return h ^ (h >> 32);
}

uint64_t ingot_hash64(const void *data, size_t size) {
    ingot_hash64_state s;
    ingot_hash64_begin(&s, size);
    ingot_hash64_update(&s, data, size);
    return ingot_hash64_end(&s);
}

void ingot_hash64_begin(ingot_hash64_state *s, uint64_t size) {
    s->h = 0x6A09E667F3BCC909ull ^ size;
    s->tail_bytes = 0;
}

void ingot_hash64_update(ingot_hash64_state *s, const void *data, size_t n) {
    const unsigned char *p = data;
    while (n && s->tail_bytes) {
        s->tail[s->tail_bytes++] = *p++;
        n--;
        if (s->tail_bytes == 8) {
            uint64_t w = 0;
            memcpy(&w, s->tail, sizeof(w));
            s->h = mix_word(s->h, w);
            s->tail_bytes = 0;
        }
    }
    for (; n >= 8; n -= 8, p += 8) {
        uint64_t w = 0;
        memcpy(&w, p, sizeof(w));
        s->h = mix_word(s->h, w);
    }
    memcpy(s->tail + s->tail_bytes, p, n);
    s->tail_bytes += (unsigned)n;
}

uint64_t ingot_hash64_end(ingot_hash64_state *s) {
    if (s->tail_bytes) {
        uint64_t w = 0;
        memcpy(&w, s->tail, s->tail_bytes);
        s->h = mix_word(s->h, w);
        s->tail_bytes = 0;
    }
    return s->h;
}
