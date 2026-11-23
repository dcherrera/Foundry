/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_brisc_fw.c
 * @brief Firmware for BRISC, the core's first processor: it boots the tile (the subordinate
 *   processors, the Tensix coprocessor's starting state, the NoC interfaces) and then serves
 *   the host's launches, driving the four subordinates through their sync bytes.
 *
 * THE CONTRACT is spec §7.4, measured from the image this replaces and cited there to
 * tt-isa-documentation BlackholeA0. In outline:
 *   boot: cfg0; local memory; reset unnamed debug state; NIU config; subordinate reset PCs;
 *         zero the L1 zeros region; Tensix init (ZEROACC, SFPU setup, semaphores); release the
 *         subordinates and wait for their DONE; tell the host (go signal DONE); NoC command-
 *         buffer presets and counters on both NoCs; ask TRISC0 to reset the stream counters.
 *   loop: wait for GO; read the launch; start the enabled subordinates; set up BRISC's CBs
 *         and run its kernel if enabled; wait for every subordinate; report DONE.
 *
 * ONLY THE HOST DISPATCH MODE IS SERVED. Foundry's host always launches in host mode with the
 * dedicated NoC mode and no remote circular buffers (ft_tt_launch.c), so the device-dispatch
 * completion path (a NoC write to a dispatch core), dynamic NoC mode, the read-pointer
 * cycling and the fast-dispatch go signals are absent; remote CBs halt with `ebreak`.
 *
 * @status Clean-room replacement of the staged tree's BRISC firmware, plan Phase 6.
 * @issues None
 * @todo Name the four unnamed values (spec §2.6) once a public source exists.
 */

#include <stddef.h>
#include <stdint.h>

#include "ft_noc.h"
#include "ft_tensix_circular_buffer.h"
#include "ft_tt_mailbox.h"

#ifndef COMPILE_FOR_BRISC
#error "ft_brisc_fw.c is built with -DCOMPILE_FOR_BRISC"
#endif

#define FT_BRISC_PROCESSOR_INDEX 0u

/* Processor enable bits in the launch message's `enables` (B, NC, T0..T2 as one group). */
#define FT_FW_ENABLE_BRISC  (1u << 0)
#define FT_FW_ENABLE_NCRISC (1u << 1)
#define FT_FW_ENABLE_TRISCS (1u << 2)

/* Sync byte values the subordinate firmware understands (fw/ft_*_fw.c). */
#define FT_FW_SYNC_LOAD                  1u
#define FT_FW_SYNC_RESET_STREAM_COUNTERS 3u

/* Hardware (BlackholeA0/TensixTile). */
#define FT_FW_CFG0_DIS_LOW_CACHE     (1u << 3)
#define FT_FW_CFG0_DIS_BRANCH_PRED   (1u << 1)
#define FT_FW_CFG0_DIS_TRISC_CACHE   (1u << 18)
#define FT_FW_DEBUG_REG(offset)      ((volatile uint32_t *)(0xFFB12000u + (offset)))
#define FT_FW_SOFT_RESET_0           0x1B0u /* SoftReset.md */
#define FT_FW_TRISC0_RESET_PC        0x228u
#define FT_FW_TRISC1_RESET_PC        0x22Cu
#define FT_FW_TRISC2_RESET_PC        0x230u
#define FT_FW_TRISC_RESET_PC_OVERRIDE 0x234u
#define FT_FW_NCRISC_RESET_PC        0x238u
#define FT_FW_NCRISC_RESET_PC_OVERRIDE 0x23Cu
#define FT_FW_DEBUG_REG_0X240        0x240u /* unnamed; measured: written 0 at boot */
#define FT_FW_TDMA_RISC_REG_0X24     ((volatile uint32_t *)0xFFB11024u) /* unnamed; 63 */
#define FT_FW_TENSIX_CFG(word)       ((volatile uint32_t *)(0xFFEF0000u + 4u * (word)))
#define FT_FW_TENSIX_CFG_WORD_185    185u /* unnamed; measured: 31 at boot and every launch */
#define FT_FW_TENSIX_CFG_WORD_3      3u   /* unnamed; bits 0, 1 and field 3..13 = 0x100 */
#define FT_FW_INSTRN_BUF             ((volatile uint32_t *)0xFFE40000u) /* PushTensixInstruction.md */

/* Tensix instructions pushed at boot, encodings per tt-isa-documentation Diagrams/Src/Bits32.lua. */
#define FT_FW_TT_ZEROACC_MODE3   0x10180000u /* ZEROACC Mode=3 */
#define FT_FW_TT_SFPENCC         0x8A00300Au /* SFPENCC (measured operands) */
#define FT_FW_TT_NOP             0x02000000u
#define FT_FW_TT_SFPLOADI        0x7100BF80u /* SFPLOADI (measured operands) */
#define FT_FW_TT_SFPCONFIG       0x910000B0u /* SFPCONFIG (measured operands) */
#define FT_FW_TT_SEMINIT(mask)   (0xA3000000u | (1u << 20) | ((uint32_t)(mask) << 2)) /* Max 1, Value 0 */

#define FT_FW_SYNC_WORD ((volatile uint32_t *)FT_TT_MB_SUBORDINATE_SYNC_ADDR)
#define FT_FW_SYNC_BYTE(index) ((volatile uint8_t *)(FT_TT_MB_SUBORDINATE_SYNC_ADDR + (index)))

/* Globals the kernels reach through --just-symbols; EXPORTED for the reason ft_trisc_fw.c gives. */
#define FT_FW_EXPORTED __attribute__((used, externally_visible))
FT_FW_EXPORTED uint32_t noc_reads_num_issued[FT_NOC_COUNT];
FT_FW_EXPORTED uint32_t noc_nonposted_writes_num_issued[FT_NOC_COUNT];
FT_FW_EXPORTED uint32_t noc_nonposted_writes_acked[FT_NOC_COUNT];
FT_FW_EXPORTED uint32_t noc_nonposted_atomics_acked[FT_NOC_COUNT];
FT_FW_EXPORTED uint32_t noc_posted_writes_num_issued[FT_NOC_COUNT];
FT_FW_EXPORTED uint32_t *rta_l1_base;
FT_FW_EXPORTED ft_tensix_cb_interface cb_interface[FT_TENSIX_NUM_CIRCULAR_BUFFERS];

static const ft_noc_software_counters k_software_counters = {
    noc_reads_num_issued, noc_nonposted_writes_num_issued, noc_nonposted_writes_acked,
    noc_nonposted_atomics_acked, noc_posted_writes_num_issued,
};

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

static inline void fence(void)
{
    __asm__ volatile("fence" ::: "memory");
}

static void initialise_local_memory(void)
{
    wzerorange(__ldm_bss_start, __ldm_bss_end);
    ft_copy_words_to_local_memory(__ldm_data_start,
                                  (const uint32_t *)FT_TT_MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH,
                                  (int32_t)(__ldm_data_end - __ldm_data_start));
}

static void set_subordinate_reset_pcs(void)
{
    *FT_FW_DEBUG_REG(FT_FW_NCRISC_RESET_PC) = FT_TT_MEM_NCRISC_FIRMWARE_BASE;
    *FT_FW_DEBUG_REG(FT_FW_TRISC0_RESET_PC) = FT_TT_MEM_TRISC0_FIRMWARE_BASE;
    *FT_FW_DEBUG_REG(FT_FW_TRISC1_RESET_PC) = FT_TT_MEM_TRISC1_FIRMWARE_BASE;
    *FT_FW_DEBUG_REG(FT_FW_TRISC2_RESET_PC) = FT_TT_MEM_TRISC2_FIRMWARE_BASE;
    *FT_FW_DEBUG_REG(FT_FW_TRISC_RESET_PC_OVERRIDE) = 0x7u; /* all three TRISCs */
    *FT_FW_DEBUG_REG(FT_FW_NCRISC_RESET_PC_OVERRIDE) = 0x1u;
}

static void initialise_tensix(void)
{
    volatile uint32_t *word3 = FT_FW_TENSIX_CFG(FT_FW_TENSIX_CFG_WORD_3);

    *FT_FW_TENSIX_CFG(FT_FW_TENSIX_CFG_WORD_185) = 31u;
    *FT_FW_INSTRN_BUF = FT_FW_TT_ZEROACC_MODE3;
    *FT_FW_INSTRN_BUF = FT_FW_TT_SFPENCC;
    *FT_FW_INSTRN_BUF = FT_FW_TT_NOP;
    *FT_FW_INSTRN_BUF = FT_FW_TT_SFPLOADI;
    *FT_FW_INSTRN_BUF = FT_FW_TT_SFPCONFIG;
    *word3 |= 1u;
    *word3 |= 2u;
    *word3 = (*word3 & ~0x3FF8u) | (0x100u << 3);
    *FT_FW_INSTRN_BUF = FT_FW_TT_SEMINIT(1u << 1);
    *FT_FW_INSTRN_BUF = FT_FW_TT_SEMINIT(1u << 2);
    *FT_FW_INSTRN_BUF = FT_FW_TT_SEMINIT(1u << 7);
    *FT_FW_INSTRN_BUF = FT_FW_TT_SEMINIT(1u << 4);
}

static void wait_for_all_subordinates_done(void)
{
    while (*FT_FW_SYNC_WORD != FT_TT_RUN_SYNC_MSG_ALL_SUBORDINATES_DONE)
        fence();
}

static volatile uint8_t *go_signal(void)
{
    uint32_t index = *(volatile uint32_t *)FT_TT_MB_GO_MESSAGE_INDEX_ADDR;
    return (volatile uint8_t *)(FT_TT_MB_GO_SLOT_ADDR(index) + offsetof(ft_tt_go_msg, signal));
}

static void boot(void)
{
    configure_core();
    initialise_local_memory();
    *(volatile uint32_t *)FT_TT_MB_LAUNCH_RD_PTR_ADDR = 0u;
    *FT_FW_DEBUG_REG(FT_FW_DEBUG_REG_0X240) = 0u;
    *FT_FW_TDMA_RISC_REG_0X24 = 63u;
    ft_noc_set_niu_and_router_config();
    set_subordinate_reset_pcs();
    wzerorange((uint32_t *)FT_TT_MEM_ZEROS_BASE,
               (uint32_t *)(FT_TT_MEM_ZEROS_BASE + FT_TT_MEM_ZEROS_SIZE));
    initialise_tensix();

    *(volatile uint32_t *)FT_TT_MEM_MAILBOX_BASE = 0u;
    *FT_FW_SYNC_WORD = FT_TT_RUN_SYNC_MSG_ALL_INIT;
    *FT_FW_DEBUG_REG(FT_FW_SOFT_RESET_0) = 0u; /* release the subordinates */
    wait_for_all_subordinates_done();
    *(volatile uint8_t *)(FT_TT_MB_GO_SLOT_ADDR(0) + offsetof(ft_tt_go_msg, signal)) =
        FT_TT_RUN_MSG_DONE;

    for (uint32_t noc = 0; noc < FT_NOC_COUNT; noc++)
        ft_noc_preset_command_buffers(noc, FT_TT_MEM_NOC_ATOMIC_RET_VAL_ADDR);
    ft_noc_seed_software_counters(0u, &k_software_counters);
    for (uint32_t noc = 0; noc < FT_NOC_COUNT; noc++)
        ft_noc_clear_packet_tags(noc);
    *FT_FW_SYNC_BYTE(1) = FT_FW_SYNC_RESET_STREAM_COUNTERS;
}

/** BRISC's own CBs: in bytes, both pointers, as NCRISC (fw/ft_ncrisc_fw.c). */
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
    uint32_t enables = config->enables;

    if (config->mode != FT_TT_DISPATCH_MODE_HOST || config->brisc_noc_mode != 0u ||
        config->min_remote_cb_start_index != FT_TENSIX_NUM_CIRCULAR_BUFFERS)
        __asm__ volatile("ebreak"); /* only the host dispatch mode is served, see the header */

    if (enables & FT_FW_ENABLE_NCRISC)
        *FT_FW_SYNC_BYTE(0) = FT_FW_SYNC_LOAD;
    rta_l1_base = (uint32_t *)(base + config->rta_offset[FT_BRISC_PROCESSOR_INDEX].rta_offset);
    *FT_FW_TENSIX_CFG(FT_FW_TENSIX_CFG_WORD_185) = 31u;

    while (*FT_FW_SYNC_BYTE(1) != FT_TT_RUN_SYNC_MSG_DONE) /* TRISC0's stream reset */
        fence();
    if (enables & FT_FW_ENABLE_TRISCS) {
        *FT_FW_SYNC_BYTE(1) = FT_TT_RUN_SYNC_MSG_GO;
        *FT_FW_SYNC_BYTE(2) = FT_TT_RUN_SYNC_MSG_GO;
        *FT_FW_SYNC_BYTE(3) = FT_TT_RUN_SYNC_MSG_GO;
    }
    ft_noc_seed_software_counters(config->brisc_noc_id, &k_software_counters);
    if (enables & FT_FW_ENABLE_NCRISC)
        *FT_FW_SYNC_BYTE(0) = FT_TT_RUN_SYNC_MSG_GO;

    if (enables & FT_FW_ENABLE_BRISC) {
        set_up_local_circular_buffers(config, base);
        ((void (*)(void))(base + config->kernel_text_offset[FT_BRISC_PROCESSOR_INDEX]))();
    }
    wait_for_all_subordinates_done();
    *FT_FW_SYNC_BYTE(1) = FT_FW_SYNC_RESET_STREAM_COUNTERS;
}

int main(void)
{
    boot();
    for (;;) {
        volatile uint8_t *signal = go_signal();

        if (*signal != FT_TT_RUN_MSG_GO) {
            fence();
            continue;
        }
        run_launch();
#ifdef FT_FW_RED_WRONG_GO_SLOT
        /* RED ONLY (the Phase 6 card gate's red): DONE reported to the wrong go-message slot, a
         * wrong VALUE in the host handshake. Chosen because it moves no NoC traffic (a wrong
         * NoC coordinate can hang the chip). Never in a shipped build. */
        signal += sizeof(ft_tt_go_msg);
#endif
        *signal = FT_TT_RUN_MSG_DONE;
    }
}
