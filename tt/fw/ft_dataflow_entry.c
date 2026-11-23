/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_dataflow_entry.c
 * @brief `_start` of a BRISC or NCRISC kernel image: the code the firmware calls to run one
 *   data-movement kernel. One source for both processors, compiled with
 *   `-DCOMPILE_FOR_BRISC` or `-DCOMPILE_FOR_NCRISC` and `-DNOC_INDEX=<0|1>`, LTO-linked with
 *   the kernel body so `kernel_main` is inlined into it.
 *
 * THE CONTRACT, IN ORDER (every step measured from the images this replaces, 2026-10-08):
 *   1. Zero the kernel's local `.bss` and copy its initialised local data from L1.
 *   2. Seed the software NoC counters for this kernel's NoC from the NIU's hardware counters
 *      (tt-isa-documentation, BlackholeA0/NoC/Counters.md, at `NIU_BASE + 0x200`), so the
 *      kernel's barriers compare like with like. The counters are the contract with
 *      kernels/ft_tensix_noc.h. On BRISC they belong to the firmware and resolve against it;
 *      NCRISC's firmware keeps its own, so the kernel defines a private set here.
 *   3. Wait for the go signal, as on TRISC (ft_trisc_entry.c step 3).
 *   4. `kernel_main()`.
 *   5. Debug checks, active with `LIGHTWEIGHT_KERNEL_ASSERTS`: `ebreak` (halting the core
 *      where a debugger can see it) if any software counter disagrees with its hardware
 *      counter, i.e. the kernel returned with a NoC transaction unaccounted for, or if request
 *      initiator 0, 2 or 3 still holds a transaction tag (`NOC_PACKET_TAG`,
 *      BlackholeA0/NoC/MemoryMap.md).
 *   6. Return 0 to the firmware.
 *
 * @status Clean-room replacement of the staged tree's BRISC and NCRISC kernel entries, plan
 *   Phase 2.
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tt_mailbox.h"

#if !defined(COMPILE_FOR_BRISC) && !defined(COMPILE_FOR_NCRISC)
#error "ft_dataflow_entry.c is built with -DCOMPILE_FOR_BRISC or -DCOMPILE_FOR_NCRISC"
#endif
#ifndef NOC_INDEX
#error "ft_dataflow_entry.c needs -DNOC_INDEX=0 or 1"
#endif

#define FT_ENTRY_NOC_COUNT 2

/* NIU registers (BlackholeA0/NoC/MemoryMap.md, Counters.md). */
#define FT_ENTRY_NIU_BASE(noc)            (0xFFB20000u + (uint32_t)(noc) * 0x10000u)
#define FT_ENTRY_NIU_REQUEST_INITIATOR(i) ((uint32_t)(i) * 0x800u)
#define FT_ENTRY_NOC_PACKET_TAG           0x18u
#define FT_ENTRY_NIU_COUNTER(index)       (0x200u + 4u * (uint32_t)(index))
#define FT_ENTRY_ATOMIC_RESP_RECEIVED     FT_ENTRY_NIU_COUNTER(0)
#define FT_ENTRY_WR_ACK_RECEIVED          FT_ENTRY_NIU_COUNTER(1)
#define FT_ENTRY_RD_RESP_RECEIVED         FT_ENTRY_NIU_COUNTER(2)
#define FT_ENTRY_NONPOSTED_WR_REQ_SENT    FT_ENTRY_NIU_COUNTER(10)
#define FT_ENTRY_POSTED_WR_REQ_SENT       FT_ENTRY_NIU_COUNTER(11)

#ifdef COMPILE_FOR_NCRISC
uint32_t noc_reads_num_issued[FT_ENTRY_NOC_COUNT];
uint32_t noc_nonposted_writes_num_issued[FT_ENTRY_NOC_COUNT];
uint32_t noc_nonposted_writes_acked[FT_ENTRY_NOC_COUNT];
uint32_t noc_nonposted_atomics_acked[FT_ENTRY_NOC_COUNT];
uint32_t noc_posted_writes_num_issued[FT_ENTRY_NOC_COUNT];
#else
extern uint32_t noc_reads_num_issued[FT_ENTRY_NOC_COUNT];
extern uint32_t noc_nonposted_writes_num_issued[FT_ENTRY_NOC_COUNT];
extern uint32_t noc_nonposted_writes_acked[FT_ENTRY_NOC_COUNT];
extern uint32_t noc_nonposted_atomics_acked[FT_ENTRY_NOC_COUNT];
extern uint32_t noc_posted_writes_num_issued[FT_ENTRY_NOC_COUNT];
#endif

extern uint32_t __ldm_bss_start[], __ldm_bss_end[];
extern uint32_t __ldm_data_start[], __ldm_data_end[];
extern uint32_t __kernel_data_lma[];

void wzerorange(uint32_t *start, uint32_t *end);
void ft_copy_words_to_local_memory(uint32_t *destination, const uint32_t *source,
                                   int32_t word_count);
void kernel_main(void);

static inline uint32_t read_niu_register(uint32_t offset)
{
    return *(volatile uint32_t *)(FT_ENTRY_NIU_BASE(NOC_INDEX) + offset);
}

static inline void initialise_kernel_local_memory(void)
{
    wzerorange(__ldm_bss_start, __ldm_bss_end);
    if ((const uint32_t *)__kernel_data_lma != (const uint32_t *)__ldm_data_start)
        ft_copy_words_to_local_memory(__ldm_data_start, __kernel_data_lma,
                                      (int32_t)(__ldm_data_end - __ldm_data_start));
}

static inline void seed_software_noc_counters(void)
{
#ifdef FT_ENTRY_RED_WRONG_NOC_SEED
    /* RED ONLY (the clean-room card gate's non-vacuous red): one read too many, a wrong VALUE
     * in state no kernel re-initialises; the kernel's first read barrier can never complete.
     * Never in a shipped build. */
    noc_reads_num_issued[NOC_INDEX] = read_niu_register(FT_ENTRY_RD_RESP_RECEIVED) + 1u;
#else
    noc_reads_num_issued[NOC_INDEX] = read_niu_register(FT_ENTRY_RD_RESP_RECEIVED);
#endif
    noc_nonposted_writes_num_issued[NOC_INDEX] = read_niu_register(FT_ENTRY_NONPOSTED_WR_REQ_SENT);
    noc_nonposted_writes_acked[NOC_INDEX] = read_niu_register(FT_ENTRY_WR_ACK_RECEIVED);
    noc_nonposted_atomics_acked[NOC_INDEX] = read_niu_register(FT_ENTRY_ATOMIC_RESP_RECEIVED);
    noc_posted_writes_num_issued[NOC_INDEX] = read_niu_register(FT_ENTRY_POSTED_WR_REQ_SENT);
}

static inline void wait_for_go_signal(void)
{
    uint32_t index = *(volatile uint32_t *)FT_TT_MB_GO_MESSAGE_INDEX_ADDR;
    volatile uint8_t *signal =
        (volatile uint8_t *)(FT_TT_MB_GO_SLOT_ADDR(index) + offsetof(ft_tt_go_msg, signal));

    while (*signal != FT_TT_RUN_MSG_GO)
        __asm__ volatile("fence" ::: "memory");
}

#ifdef LIGHTWEIGHT_KERNEL_ASSERTS
static inline void halt_unless(int condition)
{
    if (!condition)
        __asm__ volatile("ebreak" ::: "memory");
}

static inline void check_kernel_left_noc_clean(void)
{
    halt_unless(read_niu_register(FT_ENTRY_RD_RESP_RECEIVED) == noc_reads_num_issued[NOC_INDEX]);
    halt_unless(read_niu_register(FT_ENTRY_NONPOSTED_WR_REQ_SENT) ==
                noc_nonposted_writes_num_issued[NOC_INDEX]);
    halt_unless(read_niu_register(FT_ENTRY_ATOMIC_RESP_RECEIVED) ==
                noc_nonposted_atomics_acked[NOC_INDEX]);
    halt_unless(read_niu_register(FT_ENTRY_POSTED_WR_REQ_SENT) ==
                noc_posted_writes_num_issued[NOC_INDEX]);
    halt_unless(read_niu_register(FT_ENTRY_NIU_REQUEST_INITIATOR(0) + FT_ENTRY_NOC_PACKET_TAG) == 0u &&
                read_niu_register(FT_ENTRY_NIU_REQUEST_INITIATOR(2) + FT_ENTRY_NOC_PACKET_TAG) == 0u &&
                read_niu_register(FT_ENTRY_NIU_REQUEST_INITIATOR(3) + FT_ENTRY_NOC_PACKET_TAG) == 0u);
}
#else
static inline void check_kernel_left_noc_clean(void)
{
}
#endif

/* In `.start`, which the kernel linker script places first, so the entry is the first
 * instruction of the kernel text whatever order LTO emits the other functions in. */
__attribute__((section(".start"))) int _start(void)
{
    initialise_kernel_local_memory();
    seed_software_noc_counters();
    wait_for_go_signal();
    kernel_main();
    check_kernel_left_noc_clean();
    return 0;
}
