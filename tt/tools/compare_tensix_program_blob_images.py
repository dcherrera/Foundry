#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
compare_tensix_program_blob_images.py
    Prove a regenerated ft_tt_programs.h moved NO image of any program it already carried:
    for every program in the OLD header, every image row (source, processor, payload bytes,
    size, FNV digest) and the program's firmware set must be byte-identical in the NEW one.
    The provenance text beside a kernel row is NOT compared -- a provenance-only refresh
    (a source edited under a define the program does not set) changes it and nothing else;
    it is reported, per program, as a count of changed records.

    Programs only in the new header must be exactly the --new list; a program dropped from
    the old one is a failure.

    Usage:
      compare_tensix_program_blob_images.py OLD.h NEW.h [--new NAME ...]

    Exit: 0 no existing image moved and the program sets are as declared; 1 otherwise;
          2 usage / unreadable header.

    Status: added with the attention fast-math variants (run_attn_fastmath_regen_mac.sh).
    Issues: None
    Todo:   None
"""

import argparse
import re
import sys

ARRAY_RE = re.compile(r"static const unsigned char (ft_tt_program_image_\w+)\[\] = \{(.*?)\};",
                      re.S)
KERNELS_RE = re.compile(r"static const ft_tt_program_blob_image ft_tt_program_(\w+)_kernels\[\] "
                        r"= \{(.*?)\n\};", re.S)
FIRMWARE_RE = re.compile(r"static const ft_tt_program_blob_firmware (ft_tt_program_firmware_\w+) = "
                         r"\{(.*?)\n\};", re.S)
ENTRY_RE = re.compile(r'static const ft_tt_program_blob_entry ft_tt_program_(\w+) = \{\n.*?\n'
                      r'    &(ft_tt_program_firmware_\w+),')
ROW_RE = re.compile(r'\{"([^"]*)", (\w+), (ft_tt_program_image_\w+), sizeof\(\w+\), '
                    r'(0x[0-9a-f]+ull),\n(.*?)\n *\},', re.S)


def parse(path):
    with open(path) as handle:
        text = handle.read()
    arrays = {name: bytes(int(b, 16) for b in re.findall(r"0x([0-9a-fA-F]{2})", body))
              for name, body in ARRAY_RE.findall(text)}

    def rows(body):
        return [((source, processor, arrays.get(symbol), digest), provenance)
                for source, processor, symbol, digest, provenance in ROW_RE.findall(body)]

    firmware = {name: rows(body) for name, body in FIRMWARE_RE.findall(text)}
    programs = {name: {"kernels": rows(body)} for name, body in KERNELS_RE.findall(text)}
    for name, set_name in ENTRY_RE.findall(text):
        programs.setdefault(name, {})["firmware"] = firmware.get(set_name, [])
    return programs


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--new", dest="added", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        old, new = parse(args.old), parse(args.new)
    except OSError as error:
        print("unreadable: %s" % error, file=sys.stderr)
        return 2
    if not old or not new:
        print("no program tables parsed (old %u, new %u)" % (len(old), len(new)), file=sys.stderr)
        return 2

    failures = []
    for name in sorted(old):
        if name not in new:
            failures.append("%s: DROPPED from the new header" % name)
            continue
        refreshed = 0
        for kind in ("kernels", "firmware"):
            a, b = old[name].get(kind, []), new[name].get(kind, [])
            if len(a) != 5 or any(row[2] is None for row, _ in a):
                failures.append("%s: %s rows did not parse (%u) -- nothing was compared"
                                % (name, kind, len(a)))
            if [row for row, _ in a] != [row for row, _ in b]:
                failures.append("%s: %s images MOVED" % (name, kind))
            refreshed += sum(1 for (_, pa), (_, pb) in zip(a, b) if pa != pb)
        print("%-26s images IDENTICAL%s" % (name, "  (%u provenance record(s) refreshed)"
                                           % refreshed if refreshed else ""))
    added = sorted(set(new) - set(old))
    for name in added:
        print("%-26s NEW (%u kernel images)" % (name, len(new[name].get("kernels", []))))
    if added != sorted(set(args.added)):
        failures.append("new programs %s, declared %s" % (added, sorted(set(args.added))))

    for line in failures:
        print("FAIL: " + line)
    print("%u existing program(s): %s" % (len(old), "every image IDENTICAL" if not failures
                                           else "%u failure(s)" % len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
