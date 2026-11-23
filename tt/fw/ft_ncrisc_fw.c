/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_ncrisc_fw.c
 * @brief Firmware for NCRISC, the second data-movement processor. It readies the core, then
 *   serves launches forever: wait for LOAD or GO, set up the circular buffers the kernel
 *   reads, wait for GO, call the kernel, report done.
 *
 * THE CONTRACT (measured from the image this replaces, 2026-10-08; registers cited to
 * tt-isa-documentation BlackholeA0; the mailbox is ft_tt_mailbox.h):
 *   1. `cfg0` (CSR 0x7C0): as on the TRISCs (ft_trisc_fw.c step 1).
 *   2. Zero .bss; copy .data from NCRISC's L1 staging area.
 *   3. Report done (sync byte 0), then loop on `subordinate_sync[0]`: LOAD (1) or GO (0x80)
 *      starts a launch. The launch fills `cb_interface` from the host's 4-word local CB
 *      records (in bytes; both the read and write pointers, since a data-movement kernel may
 *      do either), sets `rta_l1_base`, waits for GO if it was woken by LOAD, calls the
 *      kernel at kernel_config_base[0] + kernel_text_offset[NCRISC] and reports done.
 *
 * WHAT IS DELIBERATELY ABSENT, because no Foundry kernel or host path uses it: remote
 * circular buffers (min_remote_cb_start_index must be 64, else `ebreak`), the CRTA base, the
 * per-NoC node coordinates, the logical/relative coordinates and the DRAM/L1 bank tables.
 * NCRISC kernels keep their own NoC counters (fw/ft_dataflow_entry.c).
 *
 * @status Clean-room replacement of the staged tree's NCRISC firmware, plan Phase 5.
 * @issues None
 * @todo None
 */

#include <stddef.h>
#include <stdint.h>

#include "ft_tensix_circular_buffer.h"
#include "ft_tt_mailbox.h"

#ifndef COMPILE_FOR_NCRISC
#error "ft_ncrisc_fw.c is built with -DCOMPILE_FOR_NCRISC"
#endif

#define FT_NCRISC_PROCESSOR_INDEX 1u /* B, NC, T0, T1, T2 */
#define FT_FW_CFG0_DIS_LOW_CACHE   (1u << 3)
#define FT_FW_CFG0_DIS_BRANCH_PRED (1u << 1)
#define FT_FW_CFG0_DIS_TRISC_CACHE (1u << 18)
#define FT_FW_RUN_SYNC_MSG_LOAD    1u
#define FT_FW_SYNC_BYTE            ((volatile uint8_t *)FT_TT_MB_SUBORDINATE_SYNC_ADDR)

/* Globals the kernel reaches through --just-symbols. EXPORTED for the reason ft_trisc_fw.c
 * gives: whole-program LTO would otherwise delete them and their stores. */
#define FT_FW_EXPORTED __attribute__((used, externally_visible))
FT_FW_EXPORTED uint32_t *rta_l1_base;
FT_FW_EXPORTED ft_tensix_cb_interface cb_interface[FT_TENSIX_NUM_CIRCULAR_BUFFERS];

extern uint32_t __ldm_bss_start[], __ldm_bss_end[];
extern uint32_t __ldm_data_start[], __ldm_data_end[];
void wzerorange(uint32_t *start, uint32_t *end);
void ft_copy_words_to_local_memory(uint32_t *destination, const uint32_t *source,
                                   int32_t word_count);

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
                                  (const uint32_t *)FT_TT_MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH,
                                  (int32_t)(__ldm_data_end - __ldm_data_start));
}

/** Fill `cb_interface` from the host's local CB records, in bytes. */
static void set_up_local_circular_buffers(const ft_tt_kernel_config_msg *config, uint32_t base)
{
    const uint32_t *record = (const uint32_t *)(base + config->local_cb_offset);
    uint64_t mask = config->local_cb_mask;

    for (uint32_t index = 0; mask != 0u;
         index++, mask >>= 1, record += FT_TT_WORDS_PER_LOCAL_CB_CONFIG) {
        if (!(mask & 1u))
            continue;
        ft_tensix_cb_interface *cb = &cb_interface[index];
        uint32_t address = record[0];

        cb->fifo_size = record[1];
        cb->fifo_limit = address + record[1];
        cb->fifo_page_size = record[3];
        cb->fifo_num_pages = record[2];
        cb->fifo_read_pointer = address;
        cb->fifo_write_pointer = address;
        cb->tiles.tiles_acked_received_init = 0u;
    }
}

static void run_launch(void)
{
    uint32_t slot = *(volatile uint32_t *)FT_TT_MB_LAUNCH_RD_PTR_ADDR;
    const ft_tt_kernel_config_msg *config =
        (const ft_tt_kernel_config_msg *)FT_TT_MB_LAUNCH_SLOT_ADDR(slot);
    uint32_t base = config->kernel_config_base[0];

    if (config->min_remote_cb_start_index != FT_TENSIX_NUM_CIRCULAR_BUFFERS)
        __asm__ volatile("ebreak"); /* remote CBs: not supported, see the file header */
    set_up_local_circular_buffers(config, base);
    rta_l1_base = (uint32_t *)(base + config->rta_offset[FT_NCRISC_PROCESSOR_INDEX].rta_offset);

    while (*FT_FW_SYNC_BYTE != FT_TT_RUN_SYNC_MSG_GO)
        __asm__ volatile("fence" ::: "memory");
    ((void (*)(void))(base + config->kernel_text_offset[FT_NCRISC_PROCESSOR_INDEX]))();
}

int main(void)
{
    configure_core();
    initialise_local_memory();

    *FT_FW_SYNC_BYTE = FT_TT_RUN_SYNC_MSG_DONE;
    for (;;) {
        uint8_t sync = *FT_FW_SYNC_BYTE;

        if (sync == FT_TT_RUN_SYNC_MSG_GO || sync == FT_FW_RUN_SYNC_MSG_LOAD) {
            run_launch();
            *FT_FW_SYNC_BYTE = FT_TT_RUN_SYNC_MSG_DONE;
            continue;
        }
        __asm__ volatile("fence" ::: "memory");
    }
}
