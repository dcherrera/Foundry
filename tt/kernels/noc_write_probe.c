/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file noc_write_probe.c
 * @brief One NoC write, to an endpoint and address the host names at run time. No circular
 *   buffers, no compute, no loop.
 *
 * C99, on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 *
 * It exists for NoC sanitize (plan Phase 2 follow-up: "turn on the rest of the watcher
 * family"). The acceptance case is a kernel pointed at a deliberately wrong address, and the
 * check has to be reachable in a kernel small enough to still fit its firmware hole:
 * `WATCHER_ENABLED` without `WATCHER_DISABLE_NOC_SANITIZE` inlines `get_core_type()` plus the
 * per-space bounds checks into EVERY noc call site, which grew the vector-add writer from 552
 * to 1552 bytes — 288 bytes MORE than NCRISC's whole kernel budget (2560 B hole, 1288 B of
 * firmware in it). This kernel carries one call site and runs on BRISC, whose hole is 8704
 * bytes.
 *
 * A WRITE is the deliberate choice. It is the only direction with a violation that is
 * harmless if the check is absent and the transaction really is issued: a write into the
 * read-only region below MEM_MAP_READ_ONLY_END is a POSTED write to a VALID, MAPPED endpoint,
 * so it always completes and can never leave a transaction dangling on the NoC. An
 * out-of-range READ would be the other obvious violation and is not safe to issue: a
 * non-posted read at an unmapped address may never be answered.
 *
 * Compile-time args: none (CTARGS is passed but unused — the builder requires one).
 * Runtime args:
 *   0: destination NoC x    1: destination NoC y   2: destination address
 *   3: bytes                4: source address in THIS core's L1
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0 NOC=0 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/noc_write_probe.c brisc /tmp/ttk_probe
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_dataflow.h"

void kernel_main(void)
{
    const uint32_t destination_x = ft_tensix_runtime_arg(0);
    const uint32_t destination_y = ft_tensix_runtime_arg(1);
    const uint32_t destination_address = ft_tensix_runtime_arg(2);
    const uint32_t bytes = ft_tensix_runtime_arg(3);
    const uint32_t source_address = ft_tensix_runtime_arg(4);

    ft_tensix_noc_write(source_address,
                        ft_tensix_noc_address(destination_x, destination_y, destination_address),
                        bytes);
    ft_tensix_noc_write_barrier();
}
