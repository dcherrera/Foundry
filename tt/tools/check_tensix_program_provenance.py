#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
check_tensix_program_provenance.py
    Refuse a Tensix PROGRAM -- on disk or already embedded in the blob -- whose recorded
    build inputs are not the files in this tree.

    This is the CONTENT check that replaces the mtime one. `check_tensix_program_sources.sh`
    asks "is the named source newer than the ELF"; an `explode` rewrites every file and
    every answer becomes `current`. This asks "does the content this image was built from
    still hash to what it hashed then", which nothing about file timestamps can change.
    tensix_image_provenance.py's header carries the full argument and the two shipped
    artifacts it was written against.

    Usage:
      check_tensix_program_provenance.py [--tt-root DIR] <program-dir> [more dirs...]
      check_tensix_program_provenance.py [--tt-root DIR] --header ../ft_tt_programs.h

    --tt-root  the foundry/tt directory the recorded input paths are relative to.
               Defaults to the parent of this script's directory.
    --header   check the GENERATED BLOB instead of directories. This is the one that
               matters, and it needs no program directories, no toolchain and no card: it
               is what the binary will actually run, checked against the sources beside it.
               It runs on the Mac.
    --quiet    print only the failures and the summary.

    Exit: 0 every image is current
          1 at least one image is stale, unverifiable, or carries no provenance
          2 usage

    A MISSING PROVENANCE RECORD IS A FAILURE, not a skip. An image nobody can vouch for is
    exactly the state that shipped a reader with no cooperative code for a day, and a
    checker that passes what it cannot check is worth less than no checker -- it converts
    "unknown" into "green".

    Status: added 2026-08-19. Verified to fail before it was believed: run against the
            pre-regeneration `programs/gdn` it named the five GDN sources by content; run
            against `programs/rope`, whose sources had not moved, it passed.
    Issues: None
    Todo:   None
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tensix_image_provenance import (  # noqa: E402
    FORMAT_NAME,
    FORMAT_VERSION,
    ProvenanceError,
    check_record,
    parse_record,
)

KERNEL_FILE_NAMES = ["brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf"]

RECORD_OPENER = '"%s %u\\n"' % (FORMAT_NAME, FORMAT_VERSION)
STRING_LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
PROGRAM_TABLE_RE = re.compile(r"ft_tt_program_(\w+)_kernels\[\] = \{")


def unescape(text):
    return text.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def records_from_header(path):
    """
    Every provenance record embedded in a generated ft_tt_programs.h, as
    [(program_name, record_text)].

    The blob emits each record as consecutive C string literals inside its image row, so
    this walks from each opener to the literal carrying `build_id` and joins them. It is
    the emitter's own format, read back by the same tree that writes it.
    """
    with open(path) as handle:
        text = handle.read()

    program_at = [(match.start(), match.group(1)) for match in PROGRAM_TABLE_RE.finditer(text)]

    def program_for(position):
        name = "?"
        for (start, candidate) in program_at:
            if start < position:
                name = candidate
            else:
                break
        return name

    found = []
    position = text.find(RECORD_OPENER)
    while position != -1:
        pieces = []
        cursor = position
        while True:
            match = STRING_LITERAL_RE.search(text, cursor)
            if match is None:
                raise ProvenanceError("%s: a provenance record at offset %u never ends"
                                      % (path, position))
            pieces.append(unescape(match.group(1)))
            cursor = match.end()
            if pieces[-1].startswith("build_id"):
                break
        found.append((program_for(position), "".join(pieces)))
        position = text.find(RECORD_OPENER, cursor)
    return found


def check_program_directory(directory, tt_root, quiet):
    """Findings for one build_tensix_program.sh output directory."""
    problems = []
    checked = 0
    if not os.path.isdir(directory):
        return ["  %s: MISSING" % directory]

    for file_name in KERNEL_FILE_NAMES:
        image = os.path.join(directory, file_name)
        if not os.path.isfile(image):
            continue
        sidecar = image + ".prov"
        if not os.path.isfile(sidecar):
            problems.append("  %s: NO PROVENANCE for %s -- built by something that does not "
                            "record what it was built from, so it cannot be vouched for"
                            % (directory, file_name))
            continue
        try:
            with open(sidecar) as handle:
                record = parse_record(handle.read(), sidecar)
        except ProvenanceError as error:
            problems.append("  %s" % error)
            continue
        checked += 1
        problems.extend("  " + line for line in check_record(record, tt_root, directory))

    if checked == 0 and not problems:
        problems.append("  %s: holds no kernel images" % directory)
    if not problems and not quiet:
        print("  %s: current (%u image(s), content-checked)" % (directory, checked))
    return problems


def check_header(path, tt_root, quiet):
    """
    Findings for a generated ft_tt_programs.h.

    THE BLOB IS THE ONE THAT MATTERS AND IT IS THE ONE THAT WENT STALE. `tt/programs/` is
    gitignored, so the committed header is the ONLY thing that carries a Tensix kernel across
    a checkout: a commit that changes a kernel source without regenerating the blob ships a
    binary running yesterday's kernel on every machine that clones the repo. Measured
    2026-08-26 on foundry main: `gdnc` carries the PRE-`first_row` writer (b014330) and the
    four `rtriad_*` plus `probe` carry the pre-row-block matmul (71c0b7a) -- both landed with
    the blob untouched, and the C = 16 chunkwise prefill that ships by default therefore runs
    a writer that cannot honour its own sub-chunk offset.

    Build intermediates a checkout does not carry are COUNTED rather than reported, because
    66 of the 94 findings on that same blob were only that -- and a check that is red on a
    fresh clone for a reason that is not a defect is a check nobody gates on.
    """
    problems = []
    by_program = {}
    unverifiable = []
    for (program, text) in records_from_header(path):
        try:
            record = parse_record(text, "%s [%s]" % (path, program))
        except ProvenanceError as error:
            problems.append("  %s" % error)
            continue
        found = check_record(record, tt_root, "%s [%s]" % (path, program), unverifiable)
        by_program.setdefault(program, [0, []])
        by_program[program][0] += 1
        by_program[program][1].extend(found)

    if not by_program:
        return ["  %s: NO PROVENANCE anywhere in the blob -- this header predates the "
                "provenance format and nothing in it can be vouched for" % path]

    for program in sorted(by_program):
        (count, found) = by_program[program]
        if found:
            problems.extend("  " + line for line in found)
        elif not quiet:
            print("  %s: current (%u image(s), content-checked)" % (program, count))
    if unverifiable and not quiet:
        print("  %u recorded input(s) are build intermediates under programs/ that this "
              "checkout does not carry (gitignored); every TRACKED input was hashed"
              % len(unverifiable))
    return problems


def main(argv):
    tt_root = os.path.abspath(os.path.join(HERE, ".."))
    quiet = False
    header = None
    directories = []

    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--tt-root":
            index += 1
            tt_root = argv[index]
        elif argument == "--header":
            index += 1
            header = argv[index]
        elif argument == "--quiet":
            quiet = True
        elif argument in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            directories.append(argument)
        index += 1

    if not header and not directories:
        print(__doc__)
        return 2
    if not os.path.isdir(tt_root):
        print("REFUSED: no tree at --tt-root %s -- nothing can be checked against it"
              % tt_root, file=sys.stderr)
        return 1

    problems = []
    try:
        if header:
            problems.extend(check_header(header, tt_root, quiet))
        for directory in directories:
            problems.extend(check_program_directory(directory, tt_root, quiet))
    except ProvenanceError as error:
        print("REFUSED: %s" % error, file=sys.stderr)
        return 1

    if problems:
        print("\n".join(problems))
        print("\nPROVENANCE CHECK FAILED: %u finding(s). Rebuild before believing any result "
              "from these images -- a stale directory on disk costs a re-run, a stale "
              "directory baked into the binary survives one." % len(problems))
        return 1
    print("\nprovenance: every image is current against %s" % tt_root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
