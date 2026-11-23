/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_libc.c
 * @brief Foundry's C library for Tensix images. Default build: the four symbols every image
 *   links -- `atexit`, `exit`, a word
 *   zeroing routine and a word copy routine. Linked into all five firmware ELFs and all five
 *   kernel ELFs.
 *
 * WHY THESE FOUR AND NO OTHERS. They are the symbol set the firmware and kernel entries link
 * (measured with `nm`, 2026-10-08):
 *   - `atexit`: kept as a no-op registration; a Tensix core never exits.
 *   - `exit`: never reached in normal operation; it parks the core in a loop.
 *   - `wzerorange` / the word copy: bss zeroing and local-data initialisation, called by the
 *     firmware and by the kernel entries (`ft_copy_words_to_local_memory`).
 *
 * Built with loop-pattern distribution disabled so neither loop is turned into a call to a
 * `memset`/`memcpy` that nothing provides.
 *
 * RUNTIME BUILD (-DFT_LIBC_RUNTIME): `memcpy` only, for the support ARCHIVE. GCC emits calls
 * to `memcpy` for struct and array copies, and every link is -nostdlib, so this is the only
 * memcpy any image can get. It lives in the archive, compiled without LTO, so it links only
 * into an image that calls it (today `topk/ncrisc`) and every other image stays byte-for-byte
 * what it was. Word copies when both pointers and the length are word-aligned, bytes
 * otherwise.
 *
 * @status Clean-room replacement of the staged tree's support object, plan Phase 2; memcpy
 *   added 2026-10-08 (toolchain C library removed).
 * @issues None
 * @todo Drop the mangled link name once no C++ firmware links this (plan Phases 5-6).
 */

#include <stddef.h>
#include <stdint.h>

#define FT_LIBC_PLAIN_LOOP __attribute__((optimize("no-tree-loop-distribute-patterns")))

#ifdef FT_LIBC_RUNTIME

void *memcpy(void *destination, const void *source, size_t count);

/** Copy `count` bytes; the regions must not overlap. */
FT_LIBC_PLAIN_LOOP void *memcpy(void *destination, const void *source, size_t count)
{
    uint8_t *to = destination;
    const uint8_t *from = source;

    if ((((uintptr_t)to | (uintptr_t)from | count) & 3u) == 0u) {
        uint32_t *to_word = destination;
        const uint32_t *from_word = source;

        for (count >>= 2; count != 0u; count--)
            *to_word++ = *from_word++;
        return destination;
    }
    while (count-- != 0u)
        *to++ = *from++;
    return destination;
}

#else /* the default build: the four symbols every image links */

void wzerorange(uint32_t *start, uint32_t *end);
void ft_copy_words_to_local_memory(uint32_t *destination, const uint32_t *source,
                                   int32_t word_count);

/** Static-destructor registration: a Tensix core never exits, so nothing is recorded. */
int atexit(void (*function)(void))
{
    (void)function;
    return 0;
}

/** Park the core. Reaching this means firmware returned, which it never does. */
void exit(int status)
{
    (void)status;
    for (;;) {
    }
}

/** Zero the words in [start, end). */
FT_LIBC_PLAIN_LOOP void wzerorange(uint32_t *start, uint32_t *end)
{
    while (start < end)
        *start++ = 0u;
}

/** Copy `word_count` 32-bit words from L1 into local data memory. */
FT_LIBC_PLAIN_LOOP void ft_copy_words_to_local_memory(uint32_t *destination,
                                                      const uint32_t *source,
                                                      int32_t word_count)
{
    while (word_count-- > 0)
        *destination++ = *source++;
}

#endif /* FT_LIBC_RUNTIME */
