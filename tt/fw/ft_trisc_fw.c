/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_trisc_fw.c
 * @brief Firmware for one TRISC (TRISC0 unpack, TRISC1 math, TRISC2 pack), built once per
 *   thread with `-DCOMPILE_FOR_TRISC=<0|1|2>`. It readies the core, then serves launches
 *   forever: wait for this thread's GO, set up what the kernel reads, call the kernel, wait
 *   for the coprocessor to drain, report done.
 *
 * THE CONTRACT (measured from the images this replaces, 2026-10-08; registers cited to
 * tt-isa-documentation BlackholeA0/TensixTile; the mailbox is ft_tt_mailbox.h, the host's own
 * definition of the protocol):
 *   1. `cfg0` (CSR 0x7C0, BabyRISCV/CSRs.md): set DisTriscCache (bit 18) with the branch
 *      predictor briefly disabled around it, then set DisLowCash (bit 3).
 *   2. Zero .bss; copy .data from this TRISC's L1 staging area (the host stages it there).
 *   3. Zero the 64 Tensix GPRs (REGFILE_BASE, BabyRISCV/README.md); `cfg_state_id = 0`;
 *      write 0 to TENSIX_CFG_BASE + 0x2E8 (unnamed in the docs, spec §2.6; carried as
 *      measured); wait 600 wall-clock ticks (RISCV_DEBUG_REG_WALL_CLOCK_L/H).
 *   4. Report done (sync byte 0), then loop on `subordinate_sync[1 + thread]`:
 *      GO (0x80) runs a launch; on TRISC0, 3 resets the 64 CB stream page counters.
 *   5. A launch reads `launch[launch_msg_rd_ptr]`: TRISC0/TRISC2 fill `cb_interface` from the
 *      host's 4-word local CB records (in 16-byte units: TRISC0 the reader fields, TRISC2 the
 *      writer fields); set `rta_l1_base`; call the kernel at kernel_config_base[0] +
 *      kernel_text_offset[this processor]; then the CoprocessorDoneCheck sequence
 *      (BabyRISCV/ManualTTSync.md) so "done" means the coprocessor has drained this thread.
 *
 * WHAT IS DELIBERATELY ABSENT, because no Foundry kernel or host path uses it: remote
 * circular buffers (the host always sends min_remote_cb_start_index = 64; anything else
 * halts here with `ebreak`), the CRTA base, and the logical/relative coordinate globals.
 *
 * @status Clean-room replacement of the staged tree's TRISC firmware, plan Phase 5.
 * @issues None
 * @todo Name the TENSIX_CFG_BASE + 0x2E8 field once a public source exists (spec §2.6).
 */

#include <stddef.h>
#include <stdint.h>

#include "ft_tensix_circular_buffer.h"
#include "ft_tt_mailbox.h"

#ifndef COMPILE_FOR_TRISC
#error "ft_trisc_fw.c is built once per thread with -DCOMPILE_FOR_TRISC=0, 1 or 2"
#endif

#define FT_TRISC_PROCESSOR_INDEX (2 + COMPILE_FOR_TRISC) /* B, NC, T0, T1, T2 */

/* Hardware (tt-isa-documentation BlackholeA0/TensixTile). */
#define FT_FW_REGFILE_BASE           0xFFE00000u /* Tensix GPRs, 64 x 32-bit */
#define FT_FW_TENSIX_GPR_COUNT       64u
#define FT_FW_TENSIX_CFG_BASE        0xFFEF0000u
#define FT_FW_TENSIX_CFG_RESET_WORD  0x2E8u      /* measured; field unnamed (spec §2.6) */
#define FT_FW_PC_BUF_BASE            0xFFE80000u
#define FT_FW_COPROCESSOR_DONE_CHECK (FT_FW_PC_BUF_BASE + 4u)
#define FT_FW_WALL_CLOCK_L           0xFFB121F0u
#define FT_FW_WALL_CLOCK_H           0xFFB121F8u
#define FT_FW_START_DELAY_TICKS      600u
#define FT_FW_CFG0_DIS_LOW_CACHE     (1u << 3)
#define FT_FW_CFG0_DIS_BRANCH_PRED   (1u << 1)
#define FT_FW_CFG0_DIS_TRISC_CACHE   (1u << 18)

/* The host-side protocol. */
#define FT_FW_SYNC_RESET_STREAM_COUNTERS 3u
#define FT_FW_SYNC_BYTE \
    ((volatile uint8_t *)(FT_TT_MB_SUBORDINATE_SYNC_ADDR + 1u + (uint32_t)COMPILE_FOR_TRISC))
#define FT_FW_LOCAL_CB_RECORD_WORDS FT_TT_WORDS_PER_LOCAL_CB_CONFIG

/* Globals the kernels reach through --just-symbols. EXPORTED: the firmware link is whole-
 * program LTO and nothing in it reads these, so without `externally_visible` LTO internalises
 * them and deletes the stores (measured 2026-10-08: .bss 0 and no CB setup in the image). */
#define FT_FW_EXPORTED __attribute__((used, externally_visible))
FT_FW_EXPORTED uint32_t ft_tensix_cfg_state_id __asm__("tensix_cfg_state_id");
FT_FW_EXPORTED uint32_t ft_tensix_dest_offset_id __asm__("tensix_dest_offset_id");
FT_FW_EXPORTED uint32_t *rta_l1_base;
#if COMPILE_FOR_TRISC != 1
FT_FW_EXPORTED ft_tensix_cb_interface cb_interface[FT_TENSIX_NUM_CIRCULAR_BUFFERS];
#endif

extern uint32_t __ldm_bss_start[], __ldm_bss_end[];
extern uint32_t __ldm_data_start[], __ldm_data_end[];
void wzerorange(uint32_t *start, uint32_t *end);
void ft_copy_words_to_local_memory(uint32_t *destination, const uint32_t *source,
                                   int32_t word_count);

static const uint32_t k_local_memory_staging[3] = {
    FT_TT_MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH,
    FT_TT_MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH,
    FT_TT_MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH,
};

static inline void configure_core(void)
{
    __asm__ volatile("csrs 0x7c0, %0\n\t"
                     "fence\n\t"
                     "csrs 0x7c0, %1\n\t"
                     "csrc 0x7c0, %0\n\t"
                     "fence\n\t"
                     "fence\n\t"
                     "csrs 0x7c0, %2"
                     :
                     : "r"(FT_FW_CFG0_DIS_BRANCH_PRED), "r"(FT_FW_CFG0_DIS_TRISC_CACHE),
                       "r"(FT_FW_CFG0_DIS_LOW_CACHE)
                     : "memory");
}

static inline void initialise_local_memory(void)
{
    wzerorange(__ldm_bss_start, __ldm_bss_end);
    ft_copy_words_to_local_memory(__ldm_data_start,
                                  (const uint32_t *)k_local_memory_staging[COMPILE_FOR_TRISC],
                                  (int32_t)(__ldm_data_end - __ldm_data_start));
}

static inline void reset_coprocessor_state(void)
{
    for (uint32_t index = 0; index < FT_FW_TENSIX_GPR_COUNT; index++)
        ((volatile uint32_t *)FT_FW_REGFILE_BASE)[index] = 0u;
    ft_tensix_cfg_state_id = 0u;
    *(volatile uint32_t *)(FT_FW_TENSIX_CFG_BASE + FT_FW_TENSIX_CFG_RESET_WORD) = 0u;
}

static inline uint64_t read_wall_clock(void)
{
    uint32_t low = *(volatile uint32_t *)FT_FW_WALL_CLOCK_L;
    uint32_t high = *(volatile uint32_t *)FT_FW_WALL_CLOCK_H;
    return ((uint64_t)high << 32) | low;
}

static inline void wait_wall_clock_ticks(uint32_t ticks)
{
    uint64_t until = read_wall_clock() + ticks;
    while (read_wall_clock() < until) {
    }
}

#if COMPILE_FOR_TRISC == 0
/** The host's "reset sync registers": zero every CB stream's acked and received page counts. */
static inline void reset_stream_page_counters(void)
{
    for (uint32_t stream = 0; stream < FT_TENSIX_NUM_CIRCULAR_BUFFERS; stream++) {
        *(volatile uint32_t *)FT_BH_STREAM_REG_ADDR(stream,
                FT_BH_STREAM_PAGES_RECEIVED_INDEX) = 0u;
        *(volatile uint32_t *)FT_BH_STREAM_REG_ADDR(stream,
                FT_BH_STREAM_PAGES_ACKED_INDEX) = 0u;
    }
}
#endif

#if COMPILE_FOR_TRISC != 1
/** Fill `cb_interface` from the host's local CB records, in the 16-byte units the compute
 *  threads address L1 in. TRISC0 reads a CB (read pointer), TRISC2 writes one (write pointer,
 *  page count, write-tile pointer). */
static void set_up_local_circular_buffers(const ft_tt_kernel_config_msg *config, uint32_t base)
{
    const uint32_t *record = (const uint32_t *)(base + config->local_cb_offset);
    uint64_t mask = config->local_cb_mask;

    for (uint32_t index = 0; mask != 0u; index++, mask >>= 1, record += FT_FW_LOCAL_CB_RECORD_WORDS) {
        if (!(mask & 1u))
            continue;
        ft_tensix_cb_interface *cb = &cb_interface[index];
        uint32_t address = record[0] >> 4;
        uint32_t size = record[1] >> 4;

        cb->fifo_size = size;
        cb->fifo_limit = address + size;
        cb->fifo_page_size = record[3] >> 4;
        cb->tiles.tiles_acked_received_init = 0u;
#if COMPILE_FOR_TRISC == 0
        cb->fifo_read_pointer = address;
#else
        cb->fifo_num_pages = record[2];
        cb->fifo_write_pointer = address;
        cb->fifo_write_tile_pointer = 0u;
#endif
    }
}
#endif

static inline void wait_for_coprocessor_drain(void)
{
    volatile uint32_t *check = (volatile uint32_t *)FT_FW_COPROCESSOR_DONE_CHECK;
    uint32_t value;

    *check = 0u;
    value = *check;
    __asm__ volatile("and zero, zero, %0" : : "r"(value) : "memory");
}

static void run_launch(void)
{
    uint32_t slot = *(volatile uint32_t *)FT_TT_MB_LAUNCH_RD_PTR_ADDR;
    const ft_tt_kernel_config_msg *config =
        (const ft_tt_kernel_config_msg *)FT_TT_MB_LAUNCH_SLOT_ADDR(slot);
    uint32_t base = config->kernel_config_base[0];

    if (config->min_remote_cb_start_index != FT_TENSIX_NUM_CIRCULAR_BUFFERS)
        __asm__ volatile("ebreak"); /* remote CBs: not supported, see the file header */
#if COMPILE_FOR_TRISC != 1
    set_up_local_circular_buffers(config, base);
#endif
    rta_l1_base = (uint32_t *)(base + config->rta_offset[FT_TRISC_PROCESSOR_INDEX].rta_offset);
#ifdef FT_FW_RED_WRONG_RTA_OFFSET
    /* RED ONLY (the Phase 5 card gate's red): runtime args one word off, a wrong VALUE taken
     * from a wrong mailbox-derived offset. Never in a shipped build. */
    rta_l1_base += 1;
#endif
    ((void (*)(void))(base + config->kernel_text_offset[FT_TRISC_PROCESSOR_INDEX]))();
    wait_for_coprocessor_drain();
}

int main(void)
{
    configure_core();
    initialise_local_memory();
    reset_coprocessor_state();
    wait_wall_clock_ticks(FT_FW_START_DELAY_TICKS);

    *FT_FW_SYNC_BYTE = FT_TT_RUN_SYNC_MSG_DONE;
    for (;;) {
        uint8_t sync = *FT_FW_SYNC_BYTE;

        if (sync == FT_TT_RUN_SYNC_MSG_GO) {
            run_launch();
            *FT_FW_SYNC_BYTE = FT_TT_RUN_SYNC_MSG_DONE;
            continue;
        }
#if COMPILE_FOR_TRISC == 0
        if (sync == FT_FW_SYNC_RESET_STREAM_COUNTERS) {
            reset_stream_page_counters();
            *FT_FW_SYNC_BYTE = FT_TT_RUN_SYNC_MSG_DONE;
        }
#endif
        __asm__ volatile("fence" ::: "memory");
    }
}
