#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
check_tensix_reduce_exit_state.py
  Enforces the two source-level invariants that keep the REDUCE family's exit state safe.
  Both were settled ON SILICON on 2026-08-13 (cores 60-63), each with its own red, and both
  are invisible to every other gate in this tree: `diff_tensix_kernel_instructions.sh` is
  perfectly happy with either violation -- one of them REMOVES instructions and the other
  removes two stores that no reduce kernel issues -- and `size` cannot see them either.

  Usage:  check_tensix_reduce_exit_state.py [kernels-dir]      (default ../kernels)
  Needs:  nothing. No card, no toolchain, no tt-metal source.

  ---------------------------------------------------------------------------
  INVARIANT 1 -- a kernel that programs the reduce packer mask must clear it
  ---------------------------------------------------------------------------
  `ft_tensix_pack_reduce_row_mask_config()` makes the packer write ONE datum per row and
  ZERO everywhere else. `ft_tensix_pack_reduce_mask_clear()` is the ONLY
  thing that undoes it -- the stock reduce uninit has an empty body and the UNPACK side
  of `reduce_uninit` has no statement at all -- so EVERY pack issued between the two is
  masked, whether or not it belongs to the reduce.

  MEASURED: deleting the single `ft_tensix_pack_reduce_mask_clear()` line from
  `rms_norm_compute.c` -- nothing else; trisc0/trisc1/brisc/ncrisc stay byte-identical and
  trisc2 shrinks 2548 -> 2500 B -- takes rms_norm from PCC 0.999978735 / max-abs 0.034142 to
  **PCC 0.195355069 / max-abs 3.65988**, because the second half of the kernel,
  `y = (x*gamma)*scale`, then packs through the reduce mask.

  The count is WRAPPER-AWARE: a same-file `static` helper whose body contains the clear
  (`attention_reduce_uninit()` in `resident_attention_compute.c`) counts as a clear at each
  of ITS call sites. A flat grep reports that kernel as a violation and it is not one.

  ---------------------------------------------------------------------------
  INVARIANT 2 -- ft_tensix_pack_hw_configure() must re-establish the packer baseline
  ---------------------------------------------------------------------------
  The mask state is per-core and outlives the program. What stops it poisoning the NEXT
  kernel image is exactly two full-word writes in `ft_tensix_pack_hw_configure()`:

      cfg[PCK_EDGE_OFFSET_SEC0_mask]              = 0xffff  /* mask through, selectors 0 */
      cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0] = 0x0    /* every row -> row set 0    */

  Only SEC0 carries the four `tile_row_set_select_packN` bits, so zeroing them sends every
  packer row back to TILE_ROW_SET_MAPPING_0, which the second line zeroes. The stale
  `PCK_EDGE_OFFSET_SEC1 = 0x0001` (cfg 25) and `TILE_ROW_SET_MAPPING_1 = 0x55555555`
  (cfg 21) a reduce leaves behind become UNREACHABLE rather than merely stale. Neither cfg
  25 nor cfg 21 is written by the startup path, upstream or here.

  MEASURED as a 2x2, on cores the experiment had never touched, same process and same core
  with no reset and no re-boot between the two ops:

      leaky rms_norm + stock matmul      -> matmul PCC 0.999998638          (repaired)
      good  rms_norm + no-repair matmul  -> matmul PCC 0.999998638          (mutation alone
                                                                             is harmless)
      leaky rms_norm + no-repair matmul  -> matmul PCC 0.24-0.26 on ALL SEVEN shapes,
                                            max-abs to 22.6, 8 checks failed

  So the poison is real, those two lines are what pay for it, and deleting them is a silent
  4x accuracy loss on the next op rather than a build error. This script is that build error.

  WHAT THIS DELIBERATELY DOES NOT DO: it changes no emitted instruction, and it does not
  "fix" upstream. Upstream really does leave MATH's ADDR_MOD_0..3, its MOP template, its
  replay slots 16-23, UNPACK's haloize bit, both unpackers' X-end and the UNPACK MOP at
  replay slots 0-1 live past `reduce_uninit`, and we reproduce that faithfully. Every one of
  those is reprogrammed by the next op's own init, which is why they are a documented hazard
  and not a check. The packer mask is the one an op init does NOT reprogram, so it is the one
  that gets a gate.

  Status: both invariants proven load-bearing on silicon 2026-08-13; both violations
          reproduced red by this script.
  Issues: None
  Todo:   None
"""

import re
import sys
from pathlib import Path

MASK_CONFIG = "ft_tensix_pack_reduce_row_mask_config"
MASK_CLEAR = "ft_tensix_pack_reduce_mask_clear"
PACK_HEADER_NAME = "ft_tensix_compute_pack.h"

BASELINE_WRITES = (
    ("PCK_EDGE_OFFSET_SEC0_mask <- 0xffff",
     re.compile(r"cfg\[FT_BH_CFG_PACKER_EDGE_MASK0_WORD\]\s*=\s*0xffffu?\s*;"),
     "a reduce that exits with its mask set will poison the next kernel image on that core"),
    ("TILE_ROW_SET_MAPPING_0 <- 0x0",
     re.compile(r"cfg\[FT_BH_CFG_PACKER_TILE_ROW_MAP0_WORD\]\s*=\s*0x0u?\s*;"),
     "the stale TILE_ROW_SET_MAPPING_1 = 0x55555555 a reduce leaves behind becomes reachable"),
)


def strip_comments(text):
    """Remove /* */ and // comments, preserving line count so reported lines stay true."""
    def blank_but_keep_newlines(match):
        return re.sub(r"[^\n]", " ", match.group(0))

    text = re.sub(r"/\*.*?\*/", blank_but_keep_newlines, text, flags=re.S)
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return text


def helper_functions_containing(text, callee, excluding):
    """Names of same-file functions whose body calls `callee` but never `excluding`.

    A reduce kernel may wrap the clear in a local `reduce_uninit`-shaped helper and call it
    from several places; those call sites are clears too. Bodies are delimited by a brace at
    column 0, which is this tree's C style (best-practices: K&R, function brace on its own
    line at column 0).

    `excluding` is the mask CONFIG. Without it `kernel_main` itself qualifies as a "helper"
    -- it contains the clear -- and its whole body is then skipped, hiding the very call the
    check is counting. That made `rms_norm_compute.c`, the one kernel this was written for,
    report a false violation.
    """
    names = []
    definition = re.compile(r"^(?:static\s+)?[A-Za-z_][\w \t*]*?\b(\w+)\s*\([^;]*?\)\s*$",
                            re.M)
    lines = text.split("\n")
    for index, line in enumerate(lines):
        match = definition.match(line)
        if not match or index + 1 >= len(lines) or lines[index + 1].rstrip() != "{":
            continue
        end = index + 2
        while end < len(lines) and lines[end] != "}":
            end += 1
        body = "\n".join(lines[index + 2:end])
        if callee + "(" in body and excluding + "(" not in body:
            names.append((match.group(1), index + 1, end + 1))
    return names


def count_calls(text, name, skip_ranges=()):
    """Call sites of `name(`, ignoring lines inside `skip_ranges` (1-based, inclusive)."""
    hits = []
    for index, line in enumerate(text.split("\n"), start=1):
        if name + "(" not in line:
            continue
        if any(start <= index <= end for start, end in skip_ranges):
            continue
        hits.append(index)
    return hits


def check_kernel_source(path):
    """Returns (checked, failure_message_or_None, info)."""
    text = strip_comments(path.read_text())
    configs = count_calls(text, MASK_CONFIG)
    if not configs:
        return False, None, None

    helpers = helper_functions_containing(text, MASK_CLEAR, excluding=MASK_CONFIG)
    skip = [(start, end) for _, start, end in helpers]
    clears = count_calls(text, MASK_CLEAR, skip_ranges=skip)
    for helper_name, _, _ in helpers:
        clears += count_calls(text, helper_name, skip_ranges=skip)

    detail = "%d config(s) at %s, %d clear(s) at %s" % (
        len(configs), ",".join(str(n) for n in configs),
        len(clears), ",".join(str(n) for n in sorted(clears)))
    if helpers:
        detail += " (via %s)" % ", ".join(name for name, _, _ in helpers)

    if len(clears) < len(configs):
        return True, "%s programs the reduce packer mask more often than it clears it -- %s" % (
            path.name, detail), None
    return True, None, "%s: %s" % (path.name, detail)


def main():
    kernels = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "kernels"
    failures = 0
    checks = 0

    print("=== invariant 1: every kernel that programs the reduce packer mask clears it")
    sources = sorted(kernels.glob("*.c"))
    if not sources:
        print("    FAIL: no kernel sources under %s" % kernels)
        failures += 1
    for source in sources:
        checked, failure, info = check_kernel_source(source)
        if not checked:
            continue
        checks += 1
        if failure:
            print("    FAIL: " + failure)
            failures += 1
        else:
            print("    ok:   " + info)
    if checks == 0:
        print("    FAIL: no kernel under %s programs the reduce packer mask -- this check "
              "cannot fail, which makes it worthless" % kernels)
        failures += 1

    print("=== invariant 2: ft_tensix_pack_hw_configure re-establishes the packer baseline")
    header = kernels / PACK_HEADER_NAME
    if not header.is_file():
        print("    FAIL: %s does not exist" % header)
        failures += 1
    else:
        body = strip_comments(header.read_text())
        for label, pattern, consequence in BASELINE_WRITES:
            checks += 1
            if pattern.search(body):
                print("    ok:   %s is present" % label)
            else:
                print("    FAIL: %s no longer writes %s -- %s" % (PACK_HEADER_NAME, label,
                                                                  consequence))
                failures += 1

    print("=== %d check(s), %d failure(s)" % (checks, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
