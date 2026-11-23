#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
record_tensix_image_provenance.py
    Write the `<image>.prov` record beside a Tensix ELF the two builders just linked.

    This is the WRITE half of tensix_image_provenance.py; check_tensix_program_provenance.py
    is the read half. It is called by build_tensix_kernel.sh and
    build_tensix_dataflow_kernel.sh at the point each `.src` sidecar is written, because
    that is the only place in this tree that knows BOTH the recipe it was handed and the
    depfile the compiler just produced.

    Usage:
      record_tensix_image_provenance.py --tt-root ../ --image OUT/brisc.elf \\
          --source kernels/resident_gdn_reader.c \\
          --depfile OUT/gen_brisc/kernel_body.d [--depfile ...] \\
          [--extra-input kernels/cb_tables/gdn_float32.cbtable] \\
          --recipe ctargs=0,1,2 --recipe noc=0 ...

    THE INPUT SET COMES FROM THE COMPILER, NOT FROM A SCAN. `--depfile` takes the `-MMD`
    output of the compile that produced this image, so the recorded closure is the files
    the compiler actually opened. A hand-written list of "what a kernel includes" would be
    a second, drifting model of the include graph, and the whole point of this record is
    that it cannot drift.

    `--extra-input` is for an input the COMPILER NEVER SEES: a `.cbtable` is read by
    tools/emit_tensix_cb_formats.py, which writes the `ft_tensix_cb_formats.h` the compute
    images are built against, so the table's content is part of those images and appears in
    no depfile. The generated header itself is deliberately NOT recorded -- it lives in the
    scratch build directory, not in the tree, so re-hashing it later would check a file
    that no longer exists.

    Files OUTSIDE --tt-root are dropped: the SFPI toolchain headers are pinned
    and are not in this tree, so recording them would make the check unrunnable anywhere
    but the card host. tensix_image_provenance.py's header states that limit.

    Status: added 2026-08-19 with the provenance format.
    Issues: None
    Todo:   None
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tensix_image_provenance import RECIPE_KEYS, build_record, hash_file  # noqa: E402


def parse_depfile(path):
    """Every prerequisite in a `-MMD` depfile, as written. The target before the first
    colon is dropped; `\\`-continuations are joined."""
    with open(path) as handle:
        text = handle.read()
    text = text.replace("\\\n", " ")
    prerequisites = []
    for line in text.splitlines():
        _, separator, rest = line.partition(":")
        if not separator:
            continue
        prerequisites.extend(word for word in rest.split() if word)
    return prerequisites


def in_tree_inputs(paths, tt_root):
    """{tt-relative path: content hash} for the paths that are real files inside tt_root."""
    inputs = {}
    root = os.path.realpath(tt_root)
    for path in paths:
        full = os.path.realpath(path)
        if not os.path.isfile(full):
            continue
        if not (full == root or full.startswith(root + os.sep)):
            continue
        inputs[os.path.relpath(full, root)] = hash_file(full)
    return inputs


def parse_arguments(argv):
    options = {
        "tt_root": None,
        "image": None,
        "source": None,
        "depfiles": [],
        "extra": [],
        "recipe": {},
    }
    index = 0
    while index < len(argv):
        argument = argv[index]
        index += 1
        if argument == "--tt-root":
            options["tt_root"] = argv[index]
        elif argument == "--image":
            options["image"] = argv[index]
        elif argument == "--source":
            options["source"] = argv[index]
        elif argument == "--depfile":
            options["depfiles"].append(argv[index])
        elif argument == "--extra-input":
            options["extra"].append(argv[index])
        elif argument == "--recipe":
            key, _, value = argv[index].partition("=")
            key = key.strip()
            # REFUSED RATHER THAN DROPPED. `format_recipe()` writes RECIPE_KEYS and nothing
            # else, so a key that is not on that tuple used to be accepted here and then
            # vanish -- which is how a SyncFull variant came to record the shipped recipe
            # and rebuild itself into a SyncHalf kernel. A build that passes a field the
            # record cannot carry must stop, not be quietly recorded as something else.
            if key not in RECIPE_KEYS:
                raise SystemExit("record-provenance: --recipe %s is not a recorded field. "
                                 "Add it to RECIPE_KEYS in tensix_image_provenance.py, or "
                                 "the image would be vouched for by a recipe that does not "
                                 "mention it." % key)
            options["recipe"][key] = value.strip()
        elif argument in ("-h", "--help"):
            print(__doc__)
            raise SystemExit(0)
        else:
            raise SystemExit("record-provenance: unrecognised argument %r" % argument)
        index += 1

    for required in ("tt_root", "image", "source"):
        if not options[required]:
            raise SystemExit("record-provenance: --%s is required" % required.replace("_", "-"))
    return options


def main(argv):
    options = parse_arguments(argv)
    image_path = options["image"]
    if not os.path.isfile(image_path):
        raise SystemExit("record-provenance: no image at %s -- nothing to vouch for"
                         % image_path)

    prerequisites = list(options["extra"])
    prerequisites.append(options["source"])
    for depfile in options["depfiles"]:
        if not os.path.isfile(depfile):
            raise SystemExit("record-provenance: no depfile at %s. The recorded closure "
                             "comes from the compiler; without it this record would hash "
                             "only the kernel source and miss every header it includes."
                             % depfile)
        prerequisites.extend(parse_depfile(depfile))

    inputs = in_tree_inputs(prerequisites, options["tt_root"])
    if not inputs:
        raise SystemExit("record-provenance: none of the %u prerequisites is inside %s -- "
                         "refusing to write a record that hashes nothing"
                         % (len(prerequisites), options["tt_root"]))

    record = build_record(os.path.basename(image_path),
                          os.path.basename(options["source"]),
                          options["recipe"], inputs)
    with open(image_path + ".prov", "w") as handle:
        handle.write(record)
    print("    provenance: %u input(s), build_id %s"
          % (len(inputs), record.rsplit("build_id ", 1)[1].strip()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
