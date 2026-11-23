#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
tensix_image_provenance.py
    The CONTENT record of how one Tensix ELF was built, and the check that it still
    describes the tree on disk.

    ------------------------------------------------------------------------------
    WHY THIS EXISTS -- the exact failure it is written against
    ------------------------------------------------------------------------------
    `tools/check_tensix_program_sources.sh` decides a program directory is current by
    comparing MTIMES: the `.src` sidecar names a kernel source, and the directory is stale
    if that source is NEWER than the ELF beside it. That check is defeated by
    `explode_tensix_program_blob`, which writes every image and every sidecar out fresh --
    so all 21 program directories report `current` the instant they are exploded,
    whatever the blob inside them was built from.

    That is not a hypothetical. The four `rtriad_*` readers' `brisc.elf` predated the
    cooperative multicast (`2037181`, +388 lines in `kernels/resident_matmul_reader.c`) by
    1588 B and rebuilt byte-identical at that commit's PARENT -- so the shipped matmul ran
    a reader with no cooperative code, and a recorded closure ("cooperative multicast
    cannot arm on Ornith") was measured on an image that could not cooperate. `programs/gdn`
    went stale the same way behind five commits of GDN kernel work.

    ------------------------------------------------------------------------------
    WHAT REPLACES THE MTIME, AND WHY IT CANNOT BE RESET BY TOUCHING FILES
    ------------------------------------------------------------------------------
    Every image records, BESIDE ITSELF, a `<image>.prov` naming:

      * the kernel source it was built from;
      * the BUILD LINE -- every environment value that reaches code generation
        (compile-time args, NoC index, watcher, assert flag, text base, CB table). A
        kernel's own header is the authority on its recipe, and a short `CTARGS` list
        folds to a literal 0 under SFPI GCC 15.1.0 with no diagnostic, so the recipe is
        part of the identity of the image and not a comment about it;
      * a CONTENT HASH of every in-tree file the compiler actually opened, taken from that
        compile's own `-MMD` depfile rather than from a guess about what a kernel includes.

    The check re-hashes those files where they sit now. A hash does not care about mtimes,
    so an `explode` cannot launder it, `touch` cannot clear it, and a rebuild inside the
    same second cannot hide behind GNU make 3.81's one-second granularity. A file that
    moved, vanished or changed by one byte is named individually.

    ------------------------------------------------------------------------------
    WHAT IT DELIBERATELY DOES NOT COVER, so the guarantee is not overread
    ------------------------------------------------------------------------------
      * The SFPI toolchain is NOT hashed. It is pinned, it is not in this tree, and a check
        that needs it could only run on the card host -- this one runs anywhere, which is the
        point. `toolchain` in the recipe records the version string and nothing verifies it.
      * FIRMWARE images have no provenance records; they are built from `fw/` by the same
        builders, and rebuild-and-`cmp` is what pins their bytes.
      * It is a check on INPUTS, not on the output: it says the recorded inputs still match
        the tree, not that re-running the compiler would reproduce the ELF byte for byte.
        That stronger claim is what a rebuild-and-`cmp` gives, and the audit that installed
        this format did exactly that for all 21 programs.

    Status: added 2026-08-19 for the tt-ornith-optimized program-blob regeneration, after
            the mtime check passed 21 of 21 directories two of which were provably stale.
    Issues: None
    Todo:   None
"""

import hashlib
import os

FORMAT_NAME = "ft-tt-prov"
FORMAT_VERSION = 1

# 128 bits of SHA-256, hex. Long enough that a collision is not a thing that happens, short
# enough that a provenance block stays readable in a diff of the generated header -- which
# is where a human actually reads it.
DIGEST_HEX_LENGTH = 32

# Recipe keys, in the order they are written. Fixed order so two records of the same build
# are byte-identical and a diff of the generated header shows only what really moved.
RECIPE_KEYS = (
    "toolchain",
    "ctargs",
    "noc",
    "watcher",
    "lightweight_asserts",
    "relocate_kernel_text",
    "kernel_text_base",
    "cb_table",
    # `-D`s the build handed the compiler that no other field implies -- the DstSync mode
    # the matmul triad is rendered at, for one. Added 2026-08-26 after a variant built at
    # DstSync::SyncFull recorded a recipe IDENTICAL to the shipped SyncHalf one and rebuilt
    # itself into the wrong kernel: the pair WAS passed, and `format_recipe` dropped it
    # because it was not on this tuple. Records written before this exist without the field
    # and still verify -- it is only WRITING that this list governs.
    "defines",
)


class ProvenanceError(Exception):
    """A provenance record that cannot be parsed, or a check that cannot be run."""


def hash_bytes(payload):
    """The one digest function. Everything here is SHA-256 truncated to DIGEST_HEX_LENGTH."""
    return hashlib.sha256(payload).hexdigest()[:DIGEST_HEX_LENGTH]


def hash_file(path):
    with open(path, "rb") as handle:
        return hash_bytes(handle.read())


def format_recipe(recipe):
    """`key=value` pairs in RECIPE_KEYS order. An absent key is written EMPTY rather than
    omitted, so a build that stopped passing a variable reads as a change instead of
    silently matching a record that never mentioned it."""
    return " ".join("%s=%s" % (key, recipe.get(key, "")) for key in RECIPE_KEYS)


def build_record(image_name, source_name, recipe, inputs):
    """
    Assemble a provenance record.

    @param image_name   the ELF's file name, e.g. "brisc.elf"
    @param source_name  the kernel source's basename, matching the `.src` sidecar
    @param recipe       dict over RECIPE_KEYS; missing keys are recorded empty
    @param inputs       {tt-relative path: content hash} -- every in-tree file the compile
                        opened, plus any input the compiler never sees (the CB table)
    @return             the record text, ending in a newline
    """
    lines = [
        "%s %u" % (FORMAT_NAME, FORMAT_VERSION),
        "image %s" % image_name,
        "source %s" % source_name,
        "recipe %s" % format_recipe(recipe),
    ]
    for path in sorted(inputs):
        lines.append("input %s %s" % (path, inputs[path]))
    lines.append("build_id %s" % hash_bytes(("\n".join(lines) + "\n").encode("utf-8")))
    return "\n".join(lines) + "\n"


def parse_record(text, origin):
    """
    Read a record back.

    REFUSES a record whose `build_id` does not cover its own body: the id is what a reader
    quotes, so an edited record that still quotes its old id would be a record that lies
    about itself.

    @param text    the record's full text
    @param origin  where it came from, for error messages
    @return        {"image", "source", "recipe" (raw string), "inputs" {path: hash},
                    "build_id"}
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ProvenanceError("%s: empty provenance record" % origin)

    header = lines[0].split()
    if len(header) != 2 or header[0] != FORMAT_NAME:
        raise ProvenanceError("%s: not a %s record (first line %r)"
                              % (origin, FORMAT_NAME, lines[0]))
    if header[1] != str(FORMAT_VERSION):
        raise ProvenanceError("%s: provenance format version %s, this tool speaks %u"
                              % (origin, header[1], FORMAT_VERSION))

    record = {"image": None, "source": None, "recipe": None, "inputs": {}, "build_id": None}
    body = []
    for line in lines:
        key, _, rest = line.partition(" ")
        if key == "build_id":
            record["build_id"] = rest.strip()
            continue
        body.append(line)
        if key in ("image", "source", "recipe"):
            record[key] = rest.strip()
        elif key == "input":
            parts = rest.split()
            if len(parts) != 2:
                raise ProvenanceError("%s: malformed input line %r" % (origin, line))
            record["inputs"][parts[0]] = parts[1]
        elif key != FORMAT_NAME:
            raise ProvenanceError("%s: unknown provenance key %r" % (origin, key))

    for key in ("image", "source", "recipe", "build_id"):
        if not record[key]:
            raise ProvenanceError("%s: provenance record has no %s" % (origin, key))
    if not record["inputs"]:
        raise ProvenanceError("%s: provenance record lists no inputs -- a record that hashes "
                              "nothing cannot go stale, which makes it worse than none"
                              % origin)

    recomputed = hash_bytes(("\n".join(body) + "\n").encode("utf-8"))
    if recomputed != record["build_id"]:
        raise ProvenanceError("%s: build_id %s does not cover this record (recomputed %s) -- "
                              "the record has been edited since it was written"
                              % (origin, record["build_id"], recomputed))
    return record


def is_build_intermediate(path):
    """
    Is this recorded input a file the BUILD generated into its own output directory?

    `build_tensix_program.sh` writes `chlkc_*.cpp`, `defines_generated.h` and
    `gen_<proc>/kernel_includes.hpp` into `programs/<name>/` and the compiler opens them, so
    they land in the depfile and in the record. `tt/programs/` is gitignored: on a CLEAN
    CHECKOUT none of them exist, and reporting each one as STALE is what made this check
    unusable as a gate -- foundry main's blob reports 94 findings of which 66 are only this,
    and the 28 that matter (a gdnc kernel and the matmul reader/writer/compute, both moved
    since the blob was last regenerated) are buried in the noise. Nobody gates on a check
    that is red on a fresh clone for a reason that is not a defect.

    The exemption is narrow ON PURPOSE: it applies only when the file is ABSENT. A build
    intermediate that IS present is hashed like anything else, so a program directory on
    disk is checked exactly as strictly as before.
    """
    return path.startswith("programs/")


def check_record(record, tree_root, origin, unverifiable=None):
    """
    Re-hash every recorded input where it sits now.

    @param record        as returned by parse_record()
    @param tree_root     the directory the recorded paths are relative to (foundry/tt)
    @param origin        what to name in the findings
    @param unverifiable  optional list; absent BUILD INTERMEDIATES are appended to it as
                         `(origin, path)` instead of being reported as stale. Pass None (the
                         default) to keep the old, stricter behaviour -- a caller that
                         cannot report the count must not get the exemption, or "unknown"
                         would silently become "green".
    @return              a list of human-readable problems; EMPTY means current
    """
    problems = []
    for path in sorted(record["inputs"]):
        recorded = record["inputs"][path]
        full = os.path.join(tree_root, path)
        if not os.path.isfile(full):
            if unverifiable is not None and is_build_intermediate(path):
                unverifiable.append((origin, path))
                continue
            problems.append("%s: STALE -- %s was built from '%s', which is GONE from %s"
                            % (origin, record["image"], path, tree_root))
            continue
        current = hash_file(full)
        if current != recorded:
            problems.append("%s: STALE -- %s was built from '%s' at content %s, and that "
                            "file is %s now" % (origin, record["image"], path, recorded,
                                                current))
    return problems


def read_sidecar(image_path):
    """The `<image>.prov` beside an ELF, parsed. None when there is no sidecar -- which is
    NOT the same as a clean check and every caller must treat it as unverifiable."""
    sidecar = image_path + ".prov"
    if not os.path.isfile(sidecar):
        return None
    with open(sidecar) as handle:
        return parse_record(handle.read(), sidecar)
