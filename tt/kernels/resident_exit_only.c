/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_exit_only.c
 * @brief The idle processors of a one-processor residency (the puller): initialise the
 *        queue context, report EXIT in this processor's status slot, and return — so every
 *        processor of the core runs and finishes a kernel on every launch, as in every triad.
 *
 * @status STEP 3 (A): the puller family's NCRISC and TRISC kernels.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "resident_queue_protocol.h"

void kernel_main(void)
{
    ftq_context ctx;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateExit);
}
