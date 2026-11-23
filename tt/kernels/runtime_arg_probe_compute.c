/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file runtime_arg_probe_compute.c
 * @brief Reads two runtime args and stores one at the L1 address named by the other. This
 *   is the first kernel whose effect the host can SEE, so it does two jobs at once
 *   (docs/tt_launch_notes.md §6 step 10):
 *
 *     1. proves the rta_offset encoding -- the launch message's 16-bit per-processor offset
 *        and the 0xFFFF "no args" sentinel -- by round-tripping a value the host planted;
 *     2. proves OUR kernel text actually executed, which the empty noop_compute.c cannot:
 *        an empty body and a silently-empty link are indistinguishable.
 *
 * Each TRISC gets its OWN runtime-arg block, so the host plants a different
 * (address, value) pair for each and reads three distinct words back. That turns
 * "rta_offset works" into "rta_offset is per-processor", in one launch -- and it is what
 * makes this the kernel that proves the C build path's THREE-WAY compilation really
 * produced three distinct images rather than three copies of one.
 *
 * Runtime args:
 *   0: L1 destination address, 4-byte aligned
 *   1: value to store there
 * Compile-time args: none.
 *
 * The store is a plain L1 write: the baby RISC-Vs address L1 directly at 0, so no NoC
 * transaction is issued and the blast radius stays inside this core. TRISC firmware runs
 * tensix_sync() before it reports DONE, so the write has landed by the time
 * the host sees completion.
 *
 * Build:
 *   CTARGS=0 bash foundry/tt/build_tensix_kernel.sh \
 *       foundry/tt/kernels/runtime_arg_probe_compute.c /tmp/ttk_rta
 *
 * @status RUN ON HARDWARE 2026-08-11 as C++ on Tensix 0 of the p150a; all three TRISCs
 *   wrote their own value. Converted to C99 2026-08-13; .text byte-identical to the C++
 *   build on all three TRISC images.
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_kernel_args.h"

void kernel_main(void)
{
    const uint32_t destination_address = ft_tensix_runtime_arg(0);
    const uint32_t value = ft_tensix_runtime_arg(1);

    volatile FT_BH_L1_PTR uint32_t *destination =
        (volatile FT_BH_L1_PTR uint32_t *)(uintptr_t)destination_address;
    destination[0] = value;
}
