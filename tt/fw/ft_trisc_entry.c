/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_trisc_entry.c
 * @brief `_start` of a TRISC kernel image: the code the firmware calls to run one compute
 *   kernel on TRISC0 (unpack), TRISC1 (math) or TRISC2 (pack). Compiled once per thread with
 *   `-DCOMPILE_FOR_TRISC=<0|1|2>` and LTO-linked with the kernel body, so `kernel_main` is
 *   inlined into it.
 *
 * THE CONTRACT, IN ORDER (every step measured from the images this replaces, 2026-10-08):
 *   1. Zero the kernel's local `.bss` and copy its initialised local data from L1
 *      (`__kernel_data_lma`), using the linker-script symbols.
 *   2. TRISC0 only: write 0 to debug register 0xFFB12068. The register's name and field are
 *      not in the public docs (spec §2.6); the value and address are carried from the
 *      measured image.
 *   3. Wait for the go signal: `go_messages[go_message_index].signal == FT_TT_RUN_MSG_GO`,
 *      the host/firmware mailbox protocol in ft_tt_mailbox.h, with a fence between polls.
 *   4. Put this thread's coprocessor state in the kernel's expected starting state:
 *      TRISC0 `ZEROSRC`; TRISC1 three `SETC16` writes of 0 (ThreadConfig indices 13, 29, 48,
 *      measured; names pending spec §2.6) then `ZEROACC`; TRISC2 nothing.
 *   5. `kernel_main()`, then return 0 to the firmware.
 *
 * @status Clean-room replacement of the staged tree's TRISC kernel entry, plan Phase 2.
 * @issues None
 * @todo Name the 0xFFB12068 register and the three SETC16 indices once a public source exists.
 */

#include <stdint.h>

#include "ft_tt_mailbox.h"

#ifndef COMPILE_FOR_TRISC
#error "ft_trisc_entry.c is built once per thread with -DCOMPILE_FOR_TRISC=0, 1 or 2"
#endif

/* Linker-script symbols: the kernel's local data image and where it lives in L1. */
extern uint32_t __ldm_bss_start[], __ldm_bss_end[];
extern uint32_t __ldm_data_start[], __ldm_data_end[];
extern uint32_t __kernel_data_lma[];

/* Per-thread sync state the compute headers (kernels/ft_tensix_compute.h) declare extern and
 * this translation unit owns. Zero at kernel start (they live in .bss). A thread that does not
 * use one loses it at LTO. */
uint32_t unp_cfg_context;
uint32_t pack_sync_tile_dst_ptr;
uint32_t math_sync_tile_dst_index;

void wzerorange(uint32_t *start, uint32_t *end);
void ft_copy_words_to_local_memory(uint32_t *destination, const uint32_t *source,
                                   int32_t word_count);
void kernel_main(void);

/** TRISC0's start-of-kernel debug register write (step 2). */
#define FT_TRISC0_ENTRY_DEBUG_REGISTER 0xFFB12068u

static inline void initialise_kernel_local_memory(void)
{
    wzerorange(__ldm_bss_start, __ldm_bss_end);
    if ((const uint32_t *)__kernel_data_lma != (const uint32_t *)__ldm_data_start)
        ft_copy_words_to_local_memory(__ldm_data_start, __kernel_data_lma,
                                      (int32_t)(__ldm_data_end - __ldm_data_start));
}

static inline void wait_for_go_signal(void)
{
    uint32_t index = *(volatile uint32_t *)FT_TT_MB_GO_MESSAGE_INDEX_ADDR;
    volatile uint8_t *signal =
        (volatile uint8_t *)(FT_TT_MB_GO_SLOT_ADDR(index) + offsetof(ft_tt_go_msg, signal));

    while (*signal != FT_TT_RUN_MSG_GO)
        __asm__ volatile("fence" ::: "memory");
}

static inline void reset_thread_coprocessor_state(void)
{
#if COMPILE_FOR_TRISC == 0
    __asm__ volatile("ttzerosrc 0, 0, 1, 3" ::: "memory");
#elif COMPILE_FOR_TRISC == 1
    __asm__ volatile("ttsetc16 13, 0\n\t"
                     "ttsetc16 29, 0\n\t"
                     "ttsetc16 48, 0\n\t"
                     "ttzeroacc 3, 0, 0, 1, 0" ::: "memory");
#endif
}

/* In `.start`, which the kernel linker script places first, so the entry is the first
 * instruction of the kernel text whatever order LTO emits the other functions in. */
__attribute__((section(".start"))) int _start(void)
{
    initialise_kernel_local_memory();
#if COMPILE_FOR_TRISC == 0
    *(volatile uint32_t *)FT_TRISC0_ENTRY_DEBUG_REGISTER = 0u;
#endif
    wait_for_go_signal();
    reset_thread_coprocessor_state();
    kernel_main();
    return 0;
}
