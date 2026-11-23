/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file noop_compute.c
 * @brief The smallest thing a Tensix can be asked to run: a compute kernel with an empty
 *   body. It isolates the LAUNCH PROTOCOL from everything else -- no circular buffers, no
 *   destination registers, no runtime args, no NoC transaction.
 *
 * If a launch of this hangs, the fault is in the launch message or the mailbox handshake,
 * and the timeout dump's subordinate_sync bytes say which TRISC never started
 * (docs/tt_launch_notes.md §6 step 7).
 *
 * Compiled three times, once per TRISC, like every compute kernel. All three run. As a C
 * kernel that is three separate compilations of this file rather than three #includes of it
 * into trisck.cc (build_tensix_kernel.sh's C KERNEL PATH block) -- which for an empty body
 * makes no difference and is exactly why this kernel goes first.
 *
 * WHAT THIS KERNEL CANNOT PROVE: that OUR code executed. An empty body and a silently-empty
 * --just-symbols link (docs/tt_kernel_toolchain.md §4) both complete instantly.
 * runtime_arg_probe_compute.c is the kernel that leaves evidence behind; this one only
 * proves the protocol terminates.
 *
 * Compile-time args: none. Runtime args: none.
 *
 * Build:
 *   CTARGS=0 bash foundry/tt/build_tensix_kernel.sh \
 *       foundry/tt/kernels/noop_compute.c /tmp/ttk_noop
 *
 * @status RUN ON HARDWARE 2026-08-11 as C++ on Tensix 0 of the p150a -- 1000 consecutive
 *   launches, no wedge. Converted to C99 2026-08-13; .text byte-identical to the C++ build
 *   on all three TRISC images.
 * @issues None
 * @todo None
 */

void kernel_main(void)
{
}
