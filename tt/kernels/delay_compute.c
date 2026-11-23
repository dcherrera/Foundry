/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file delay_compute.c
 * @brief A compute kernel that spins for a runtime-arg number of iterations and then
 *   returns. It exists for ONE reason: docs/tt_launch_notes.md §6 step 8 asks for a
 *   deliberate timeout, and noop_compute.c finishes in microseconds -- far inside any
 *   timeout the host can express, so it can never prove the timeout path fires.
 *
 * This kernel makes the run genuinely longer than the timeout, so
 * ft_tt_launch_wait_done(timeout_ms = 1) MUST return FT_TT_EIO and print its dump. It also
 * gives the recovery path a rehearsal: the core is still running when the host gives up,
 * and a second wait picks the completion up cleanly.
 *
 * Bounded, never infinite. A kernel that spins forever is recoverable only by asserting
 * reset, and rehearsing the timeout is not worth rehearsing that.
 *
 * Runtime args:
 *   0: iteration count
 * Compile-time args: none.
 *
 * Build:
 *   CTARGS=0 bash foundry/tt/build_tensix_kernel.sh \
 *       foundry/tt/kernels/delay_compute.c /tmp/ttk_delay
 *
 * @status RUN ON HARDWARE 2026-08-11 as C++ on Tensix 0 of the p150a. Measured rate in
 *   docs/tt_bringup_log.md. Converted to C99 2026-08-13; .text byte-identical to the C++
 *   build on all three TRISC images.
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_kernel_args.h"

void kernel_main(void)
{
    const uint32_t iterations = ft_tensix_runtime_arg(0);

    /* volatile so -O3 -flto cannot delete the loop; the whole point is the time it takes,
     * and a kernel optimised down to nothing would silently un-prove step 8. */
    volatile uint32_t counter = 0;
    for (uint32_t iteration = 0; iteration < iterations; iteration++)
        counter = counter + 1;
}
