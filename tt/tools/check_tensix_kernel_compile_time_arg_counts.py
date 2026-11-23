#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
check_tensix_kernel_compile_time_arg_counts.py
  Enforces the ONE guard `ft_tensix_compile_time_arg()` does not have. It indexes
  `static const uint32_t values[] = {KERNEL_COMPILE_TIME_ARGS}` with no bounds check
  (kernels/ft_tensix_kernel_args.h) -- MEASURED on the card's toolchain (SFPI GCC 15.1.0,
  a second card host, 2026-08-18): an out-of-range read folds to a literal `0` at `-Os` and at `-O3`,
  with NO diagnostic at any `-Warray-bounds` level. So `-Werror` catches nothing here --
  there is no warning to promote -- and the argument COUNT written beside each kernel's
  documented build line is the only thing standing between a short build and a kernel that
  silently reads zero. This script is that guard, made real.

  WHAT IT CHECKS. For every `kernels/*.c` source: the highest N in any
  `ft_tensix_compile_time_arg(N)` it reads (directly or through a `#define` wrapper) versus
  the number of comma-separated values the kernel's own documented `Build:` comment passes
  as `CTARGS=` / `READER_CTARGS=` / `COMPUTE_CTARGS=` / `WRITER_CTARGS=`. A build line that
  supplies fewer than N+1 values is a silent zero waiting to happen -- that nearly shipped
  once already: resident_gdn_reader.c reads CONV_DEPTH at index 7, and a seven-argument
  READER_CTARGS line (kernels/resident_gdn_reader.c's own header describes exactly this
  build) would have made `conv_block_depth()` return 0 and refused every CONV descriptor.

  Usage:  check_tensix_kernel_compile_time_arg_counts.py [kernels-dir]  (default ../kernels)
  Needs:  nothing. No card, no toolchain, no tt-metal source -- this reads C comments.

  ---------------------------------------------------------------------------
  HOW THE BUILD LINE IS FOUND, AND WHY IT IS NOT A SIMPLE GREP
  ---------------------------------------------------------------------------
  Every kernel documents its build under a `Build:` or `Build (...):` comment label
  (build_tensix_kernel.sh / build_tensix_dataflow_kernel.sh for one kernel,
  build_tensix_program.sh for a reader/compute/writer triad -- see those three scripts'
  own headers). The label starts a STANZA that ends at the next blank comment line, which
  matters: resident_gdn_reader.c's Build stanza is followed, after a blank line, by a prose
  correction quoting a STALE, one-argument-short `READER_CTARGS=` value it does NOT build
  with any more ("The round's spec wrote ... The 7 is corrected to 4 above"). Reading to the
  next blank line, not to the end of the comment, is what keeps that stale value out of the
  count this script trusts.

  Inside a stanza, the actual invocation is whatever script name follows the literal word
  `bash` -- not any `build_tensix_*.sh` mentioned in prose (vector_add_compute.c's stanza
  opens with "Build (or build_tensix_program.sh for the whole triad at once):", which names
  a script this kernel is NOT built with in that line). For a `build_tensix_program.sh`
  stanza, which of READER_CTARGS/COMPUTE_CTARGS/WRITER_CTARGS belongs to THIS kernel is
  read off the invocation's own positional `reader.c compute.c writer.c` argument order,
  not assumed from the kernel's filename.

  Every `KERNEL_COMPILE_TIME_ARGS`-shaped `NAME=` line inside a stanza is collected, and the
  SMALLEST comma-count among them is what gets compared against N+1 -- a kernel with two
  documented arms (resident_queue.c's RESIDENT and ONE-SHOT builds) must clear the guard in
  BOTH, not just the first one this script happens to read.

  ---------------------------------------------------------------------------
  WHAT THIS DELIBERATELY DOES NOT DO
  ---------------------------------------------------------------------------
  It does not run a compiler and it does not open the card -- it is a doc-comment audit, so
  it can be wrong if a build line drifts from what someone actually typed on the card host. It
  narrows that gap the same way the rest of this tree does: `-DKERNEL_COMPILE_TIME_ARGS=`
  IS the build line, so a kernel that changes it without updating the comment beside it is
  the exact failure mode this script exists to catch on the next person's read, even if it
  cannot catch it on the card host's own command line.

  A `ft_tensix_compile_time_arg(N)` whose N is not a bare integer literal (a macro, an
  expression) is reported and treated as a FAILURE rather than skipped -- this codebase has
  none today, and silently passing an index this script cannot evaluate is exactly the
  "quiet skip that reads as coverage" this program does not accept from a checker. Same
  treatment for a kernel whose Build stanza cannot be resolved to a single CTARGS-shaped
  value: it fails loudly, unless it is named in NO_BUILD_LINE_ALLOWLIST below with a reason,
  which is the one case (resident_vector_writer_blocked.c) known to be a real gap today.

  A kernel that reads NO compile-time arg passes without a build line: it has nothing a short
  build could under-supply (dispatch_gate.c, resident_exit_only.c, resident_puller.c).

  Status: written 2026-08-18. Verified to fail: a build line temporarily shortened by one
          argument was caught, naming the kernel and the missing index, then restored --
          see the commit this file landed in.
  Issues: None
  Todo:   None
"""

import re
import sys
from pathlib import Path

ARG_READ_RE = re.compile(r"ft_tensix_compile_time_arg\s*\(\s*([^)]*?)\s*\)")
BUILD_LABEL_RE = re.compile(r"^\s*\*+\s*Build\s*[:(]")
BLANK_COMMENT_LINE_RE = re.compile(r"^\s*\*+\s*$")
COMMENT_PREFIX_RE = re.compile(r"^\s*\*+\s?")
CTARGS_ASSIGN_RE = re.compile(r"\b(CTARGS|READER_CTARGS|COMPUTE_CTARGS|WRITER_CTARGS)=([0-9,]+)")
BASH_INVOKE_RE = re.compile(r"\bbash\s+(\S+)")
SINGLE_KERNEL_SCRIPTS = ("build_tensix_kernel.sh", "build_tensix_dataflow_kernel.sh")
PROGRAM_SCRIPT = "build_tensix_program.sh"
PROGRAM_ROLE_VARS = ("READER_CTARGS", "COMPUTE_CTARGS", "WRITER_CTARGS")

# Kernels whose header documents named compile-time args but no build invocation to check
# them against. Silently skipping these is the failure mode that makes a checker worthless,
# so each one must be named here with why, and shows up as a WARN on every run rather than
# vanishing from the report.
NO_BUILD_LINE_ALLOWLIST = {
    "resident_vector_writer_blocked.c": (
        "a measurement arm (Phase 6 fusion round), never shipped -- its own header lists "
        "the five compile-time args by name but documents no build invocation for them"
    ),
}


def strip_comments(text):
    """Blank out /* */ and // comments, keeping line numbers intact for reporting."""
    def blank_but_keep_newlines(match):
        return re.sub(r"[^\n]", " ", match.group(0))

    text = re.sub(r"/\*.*?\*/", blank_but_keep_newlines, text, flags=re.S)
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return text


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def highest_compile_time_arg_index(raw_text):
    """(max_index_or_None, [(bad_expr, line), ...]) over CODE only -- comments stripped so a
    prose mention of ft_tensix_compile_time_arg() (several kernels discuss the very bug this
    script guards against) is never mistaken for a real read."""
    code = strip_comments(raw_text)
    literal_indices = []
    bad = []
    for match in ARG_READ_RE.finditer(code):
        expr = match.group(1).strip()
        if re.fullmatch(r"\d+", expr):
            literal_indices.append(int(expr))
        else:
            bad.append((expr if expr else "<empty>", line_of(code, match.start())))
    return (max(literal_indices) if literal_indices else None), bad


def build_stanzas(raw_text):
    """Each documented Build block: (start_line, normalized_one_line_text)."""
    lines = raw_text.split("\n")
    stanzas = []
    i = 0
    while i < len(lines):
        if BUILD_LABEL_RE.match(lines[i]):
            start = i
            block = []
            while i < len(lines) and not BLANK_COMMENT_LINE_RE.match(lines[i]):
                block.append(COMMENT_PREFIX_RE.sub("", lines[i]))
                i += 1
            normalized = " ".join(block).replace("\\", " ")
            normalized = re.sub(r"\s+", " ", normalized).strip()
            stanzas.append((start + 1, normalized))
        else:
            i += 1
    return stanzas


def resolve_stanza(stanza_text, kernel_basename):
    """One stanza -> ('ok', count, detail) | ('cannot_verify', None, detail) | (None, ...)
    if the stanza names no build_tensix_*.sh invocation at all (not this kernel's stanza)."""
    invoke = BASH_INVOKE_RE.search(stanza_text)
    script = invoke.group(1).rsplit("/", 1)[-1] if invoke else None
    if script not in SINGLE_KERNEL_SCRIPTS and script != PROGRAM_SCRIPT:
        return None, None, None

    assigns = {}
    for var, values in CTARGS_ASSIGN_RE.findall(stanza_text):
        count = len([tok for tok in values.split(",") if tok != ""])
        assigns.setdefault(var, []).append(count)

    if script == PROGRAM_SCRIPT:
        after = stanza_text[invoke.end():]
        files = [Path(tok).name for tok in re.findall(r"(\S+\.c)\b", after)]
        if kernel_basename not in files:
            return "cannot_verify", None, (
                "build_tensix_program.sh invocation found but does not name %s in its "
                "reader/compute/writer argument list (%s)" % (kernel_basename, files or "none"))
        role = files.index(kernel_basename)
        if role >= len(PROGRAM_ROLE_VARS):
            return "cannot_verify", None, (
                "%s appears at position %d in the program invocation, past reader/compute/"
                "writer" % (kernel_basename, role))
        target_var = PROGRAM_ROLE_VARS[role]
    else:
        target_var = "CTARGS"

    counts = assigns.get(target_var)
    if not counts:
        return "cannot_verify", None, (
            "build_tensix_%s invocation found but no %s=<values> beside it"
            % ("program.sh" if script == PROGRAM_SCRIPT else "kernel/dataflow_kernel.sh",
               target_var))
    return "ok", min(counts), "%s=%s (%d value(s))" % (
        target_var, ",".join(str(c) for c in counts), min(counts))


def check_kernel(path):
    """Returns a dict describing one kernel's result; see main() for the fields used."""
    raw_text = path.read_text()
    max_index, bad_exprs = highest_compile_time_arg_index(raw_text)
    needed = 0 if max_index is None else max_index + 1

    if bad_exprs:
        return {"status": "FAIL", "reason": "non-literal ft_tensix_compile_time_arg() index: "
                "%s" % ", ".join("'%s' at line %d" % (e, ln) for e, ln in bad_exprs)}

    stanzas = build_stanzas(raw_text)
    resolved = None
    cannot_verify_reason = None
    for _, stanza_text in stanzas:
        status, count, detail = resolve_stanza(stanza_text, path.name)
        if status == "ok":
            resolved = (count, detail)
            break
        if status == "cannot_verify" and cannot_verify_reason is None:
            cannot_verify_reason = detail

    if resolved is None:
        if cannot_verify_reason:
            return {"status": "FAIL", "reason": "build line found but not resolvable: %s"
                    % cannot_verify_reason}
        if max_index is None:
            return {"status": "OK", "reason": "reads no compile-time arg; nothing to under-supply"}
        if path.name in NO_BUILD_LINE_ALLOWLIST:
            return {"status": "WARN", "reason": (
                "no discoverable build line to check against (needs %d compile-time "
                "arg(s)) -- allowlisted: %s" % (needed, NO_BUILD_LINE_ALLOWLIST[path.name]))}
        return {"status": "FAIL", "reason": (
            "no discoverable build line -- reads up to compile-time arg index %s (needs %d "
            "value(s)) and is not in NO_BUILD_LINE_ALLOWLIST"
            % (max_index, needed))}

    supplied, detail = resolved
    if supplied < needed:
        return {"status": "FAIL", "reason": (
            "reads ft_tensix_compile_time_arg(%d) (needs %d value(s)) but its build line "
            "supplies only %d: %s" % (max_index, needed, supplied, detail))}
    return {"status": "OK", "reason": "needs %d, build line supplies %s" % (needed, detail)}


def main():
    kernels = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "kernels"
    sources = sorted(kernels.glob("*.c"))
    if not sources:
        print("FAIL: no kernel sources under %s" % kernels)
        return 1

    failures = 0
    warnings = 0
    for source in sources:
        result = check_kernel(source)
        if result["status"] == "FAIL":
            failures += 1
            print("FAIL %-42s %s" % (source.name, result["reason"]))
        elif result["status"] == "WARN":
            warnings += 1
            print("WARN %-42s %s" % (source.name, result["reason"]))
        else:
            print("ok   %-42s %s" % (source.name, result["reason"]))

    print("=== %d kernel(s), %d failure(s), %d warning(s)" %
          (len(sources), failures, warnings))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
