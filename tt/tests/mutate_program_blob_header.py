#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
mutate_program_blob_header.py
    Damage a generated ft_tt_programs.h in one specific, named way, so that the gate aimed
    at that damage can be watched failing.

    A gate nobody has seen fail is not evidence (coding-best-practices.md 9), and four
    vacuous gates have already been caught in this project. Every check in
    tests/test_program_blob.c has a mutation here whose only job is to make that check red.

    This is a TEXT transform over the emitted header, in Python. The detectors are in C and
    were written from the format, not from this file — so a mutation and its detector never
    share an implementation.

    Usage:  mutate_program_blob_header.py <mode> <input.h> <output.h>

    Modes:
      flip-byte           one payload byte of prog_alpha's DM0 kernel changes; the recorded
                          digest does not     -> the digest check must fire
      truncate-image      the last twelve bytes of that array are deleted
                          -> the digest check must fire, and the size stops matching disk
      break-elf-magic     that array's ELF magic is destroyed AND its digest is recomputed
                          to match -> the digest is happy and ft_tt_elf_parse must refuse
      swap-kernel-images  prog_alpha's DM0 and DM1 rows exchange payload AND digest, so
                          every image is internally consistent and two of them are in the
                          wrong slot -> only the comparison against the files on disk can
                          see it. This is the mutation that proves that comparison is doing
                          work.
      merge-firmware      prog_gamma is pointed at prog_alpha's firmware set -> the dedup
                          check must refuse to call two different sets the same
      drop-program        prog_beta is removed from the lookup table -> the lookup must
                          notice, rather than the count being whatever it happens to be

    Status:  Runs on the Mac; used by tests/run_program_blob_gates.sh.
    Issues:  None
    Todo:    None
"""

import re
import sys

FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
FNV64_MASK = 0xFFFFFFFFFFFFFFFF

ARRAY_PATTERN = re.compile(
    r"static const unsigned char (\w+)\[\] = \{\n(.*?)\n\};\n", re.S
)


def fnv1a_64(data):
    digest = FNV64_OFFSET
    for byte in data:
        digest ^= byte
        digest = (digest * FNV64_PRIME) & FNV64_MASK
    return digest


def array_body(text, symbol):
    for match in ARRAY_PATTERN.finditer(text):
        if match.group(1) == symbol:
            return match
    raise SystemExit("mutate: no array named %r in the header" % symbol)


def array_bytes(body):
    return bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", body))


def render_body(payload):
    lines = []
    for start in range(0, len(payload), 12):
        chunk = payload[start:start + 12]
        lines.append("    " + " ".join("0x%02x," % byte for byte in chunk))
    return "\n".join(lines)


def replace_array(text, symbol, payload):
    match = array_body(text, symbol)
    return text[:match.start(2)] + render_body(payload) + text[match.end(2):]


def kernel_row(text, program, processor):
    """The one line of `ft_tt_program_<program>_kernels[]` for one processor."""
    table = re.search(
        r"static const ft_tt_program_blob_image ft_tt_program_%s_kernels\[\] = \{\n(.*?)\n\};"
        % program, text, re.S)
    if not table:
        raise SystemExit("mutate: no kernel table for %r" % program)
    for line in table.group(1).split("\n"):
        if ("FT_TT_PROC_%s" % processor) in line:
            return line
    raise SystemExit("mutate: %r has no %s row" % (program, processor))


def row_symbol(row):
    match = re.search(r"FT_TT_PROC_\w+, (\w+), sizeof", row)
    if not match:
        raise SystemExit("mutate: cannot read the payload symbol out of %r" % row)
    return match.group(1)


def set_row_digest(text, symbol, digest):
    """Every row that references `symbol` gets the new digest — a payload is shared across
    programs when its bytes are identical, and leaving one row behind would make the result
    a mutation of something other than what was asked for."""
    pattern = re.compile(r"(%s, sizeof\(%s\), )0x[0-9a-f]{16}ull" % (symbol, symbol))
    (text, count) = pattern.subn(r"\g<1>0x%016xull" % digest, text)
    if count == 0:
        raise SystemExit("mutate: no row references %r" % symbol)
    return text


def mutate_flip_byte(text):
    symbol = row_symbol(kernel_row(text, "prog_alpha", "DM0"))
    payload = bytearray(array_bytes(array_body(text, symbol).group(2)))
    index = len(payload) // 2
    payload[index] ^= 0xFF
    return replace_array(text, symbol, bytes(payload))


def mutate_truncate_image(text):
    symbol = row_symbol(kernel_row(text, "prog_alpha", "DM0"))
    payload = array_bytes(array_body(text, symbol).group(2))
    return replace_array(text, symbol, payload[:-12])


def mutate_break_elf_magic(text):
    symbol = row_symbol(kernel_row(text, "prog_alpha", "DM0"))
    payload = bytearray(array_bytes(array_body(text, symbol).group(2)))
    payload[0] = 0x00
    text = replace_array(text, symbol, bytes(payload))
    return set_row_digest(text, symbol, fnv1a_64(bytes(payload)))


def mutate_swap_kernel_images(text):
    left = kernel_row(text, "prog_alpha", "DM0")
    right = kernel_row(text, "prog_alpha", "DM1")
    tail = re.compile(r"(FT_TT_PROC_\w+, )(\w+, sizeof\(\w+\), 0x[0-9a-f]{16}ull)")
    left_tail = tail.search(left).group(2)
    right_tail = tail.search(right).group(2)
    text = text.replace(left, tail.sub(r"\g<1>" + right_tail, left, count=1), 1)
    return text.replace(right, tail.sub(r"\g<1>" + left_tail, right, count=1), 1)


def mutate_merge_firmware(text):
    sets = re.findall(r"static const ft_tt_program_blob_firmware (\w+) =", text)
    if len(sets) < 2:
        raise SystemExit("mutate: the fixture blob must hold two firmware sets to merge")
    alpha = re.search(r"ft_tt_program_prog_alpha = \{.*?&(\w+),", text, re.S).group(1)
    gamma_entry = re.search(r"(ft_tt_program_prog_gamma = \{.*?&)(\w+)(,)", text, re.S)
    if gamma_entry.group(2) == alpha:
        raise SystemExit("mutate: alpha and gamma already share a firmware set")
    return text[:gamma_entry.start(2)] + alpha + text[gamma_entry.end(2):]


def mutate_drop_program(text):
    (text, count) = re.subn(r"\n    &ft_tt_program_prog_beta,", "", text)
    if count != 1:
        raise SystemExit("mutate: prog_beta is not in the table exactly once")
    # The entry itself stays defined and merely unreferenced, which -Wunused would reject
    # under -Werror; a build failure is not the red this mutation is aiming for. Referencing
    # it from a comment is not enough, so it is kept alive by a second table nothing reads.
    return text.replace(
        "static const ft_tt_program_blob_entry *const *const ft_tt_program_blob_table =",
        "static const ft_tt_program_blob_entry *const ft_tt_program_dropped[] = {\n"
        "    &ft_tt_program_prog_beta,\n};\n"
        "static const ft_tt_program_blob_entry *const *const ft_tt_program_dropped_ref =\n"
        "    ft_tt_program_dropped;\n"
        "static const ft_tt_program_blob_entry *const *const ft_tt_program_blob_table =",
        1)


MODES = {
    "flip-byte": mutate_flip_byte,
    "truncate-image": mutate_truncate_image,
    "break-elf-magic": mutate_break_elf_magic,
    "swap-kernel-images": mutate_swap_kernel_images,
    "merge-firmware": mutate_merge_firmware,
    "drop-program": mutate_drop_program,
}


def main(argv):
    if len(argv) != 3 or argv[0] not in MODES:
        raise SystemExit("usage: mutate_program_blob_header.py <%s> <in.h> <out.h>"
                         % "|".join(sorted(MODES)))
    with open(argv[1]) as handle:
        text = handle.read()
    mutated = MODES[argv[0]](text)
    if mutated == text:
        raise SystemExit("mutate: %r changed nothing -- the mutation itself is vacuous"
                         % argv[0])
    with open(argv[2], "w") as handle:
        handle.write(mutated)
    print("mutated %s -> %s (%s)" % (argv[1], argv[2], argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
