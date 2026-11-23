#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
emit_tensix_program_header.py
    Turn whole Tensix PROGRAM DIRECTORIES -- the five kernel ELFs plus the five firmware
    ELFs that build_tensix_program.sh leaves together -- into foundry/tt/ft_tt_programs.h,
    the embedded blob ft_tt_program_blob.h defines.

    This is emit_tensix_kernel_header.py one level up. That emitter embeds single kernels
    so a Foundry BUILD needs no toolchain; this one embeds whole programs so a Foundry
    BINARY needs no program directory. Today Pantheon reads five loose ELFs out of
    one absolute programs directory at start-up, which is why the serve runs on exactly one
    machine -- and why the rtriad_bfp8 program, built into two per-agent roots and never
    into that one, silently cost 1.085x for a day.

    Usage:
      emit_tensix_program_header.py -o ../ft_tt_programs.h \\
          --kernels ../kernels --toolchain 7.68.0 \\
          --program ../programs/eltwise \\
          --program ../programs/reduce

      --program takes a DIRECTORY; the program's name is its basename. Repeat it.
      --toolchain and --recipe attach to the --program that follows them, or to all
      programs when given before the first one.

    ------------------------------------------------------------------------------
    WHAT IT REFUSES, AND WHY EACH REFUSAL IS HERE
    ------------------------------------------------------------------------------
    A stale artifact has shipped three times in this project (tools/
    check_tensix_program_sources.sh names all three). A stale artifact BAKED INTO A BINARY
    is worse than any of them, because it survives a rebuild -- so this emitter refuses,
    and writes nothing at all, on:

      * a kernel image with NO `.prov` provenance record, or one whose recorded input
        content no longer matches the tree. This is the CONTENT check
        (tools/tensix_image_provenance.py) and it is the one that matters: the `.src`
        sidecar's mtime comparison below is defeated by an `explode`, which rewrites every
        timestamp, and that is exactly how a reader with no cooperative multicast shipped;
      * a `.src` sidecar naming a kernel source that is not in --kernels;
      * a `.src` sidecar whose source is NEWER than the ELF beside it;
      * a `.src` sidecar with no ELF beside it;
      * a program directory with NO sidecars -- built by something that does not record
        its sources, so it cannot be checked at all and must not be trusted;
      * any of the five fw_*.elf missing -- firmware is all five or none, and a program
        whose firmware is incomplete boots a core into whatever L1 held;
      * an image that is not a loadable ELF32/LE/ET_EXEC/EM_RISCV Tensix binary, or whose
        text is under --min-text-bytes (the silently-empty --just-symbols build);
      * two programs with the same name;
      * zero programs, unless --allow-empty is passed. An empty blob is a real state -- it
        is what the tree carries before the first real regeneration -- but it must be ASKED
        for, because the way you get one by accident is a glob that matched nothing.

    The sidecar checks are DELEGATED to tools/check_tensix_program_sources.sh rather than
    reimplemented: one implementation of that rule, already proven, already the thing the
    test Makefiles call. Pass --no-source-check only to test THIS script's own behaviour.

    ------------------------------------------------------------------------------
    DEDUPLICATION
    ------------------------------------------------------------------------------
    Every program directory carries its own copy of the same five firmware images. They are
    emitted ONCE per distinct content and shared, so N programs from one build tree cost one
    firmware set rather than N. Identical kernel images are shared the same way. The
    sharing is by CONTENT, so it can never merge two things that differ.

    Status:  Written on the Mac. The gates are exercised by tests/Makefile.programblob
             against synthetic fixtures. NEVER RUN AGAINST REAL PROGRAM DIRECTORIES -- they
             and the SFPI toolchain live on the card host, so a real regeneration is untested.
    Issues:  None
    Todo:    None
"""

import hashlib
import io
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


def repo_relative(word):
    """An absolute path inside this checkout, recorded relative to the repo root so the
    header's command line carries no machine-specific prefix. Anything else is unchanged."""
    if not os.path.isabs(word):
        return word
    resolved = os.path.realpath(word)
    root = os.path.realpath(REPO_ROOT)
    if resolved == root or resolved.startswith(root + os.sep):
        return os.path.relpath(resolved, root)
    return word
sys.path.insert(0, HERE)

# ONE ELF reader, not a second one. emit_tensix_kernel_header.KernelElf already enforces
# every structural rule the launch path needs (ET_EXEC, EM_RISCV, aligned in-bounds
# PT_LOADs, e_entry == first segment) and has been run against real SFPI output.
from emit_tensix_kernel_header import (  # noqa: E402
    DEFAULT_MIN_TEXT_BYTES,
    PROCESSOR_MACROS,
    KernelElf,
    emit_blob,
)
from tensix_image_provenance import (  # noqa: E402
    ProvenanceError,
    check_record,
    parse_record,
)

# Indexed by FT_TT_PROC_*: DM0, DM1, MATH0, MATH1, MATH2.
PROCESSOR_ORDER = ["DM0", "DM1", "MATH0", "MATH1", "MATH2"]
KERNEL_FILE_NAMES = ["brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf"]
FIRMWARE_FILE_NAMES = [
    "fw_brisc.elf",
    "fw_ncrisc.elf",
    "fw_trisc0.elf",
    "fw_trisc1.elf",
    "fw_trisc2.elf",
]

SOURCE_CHECK_SCRIPT = os.path.join(HERE, "check_tensix_program_sources.sh")

FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
FNV64_MASK = 0xFFFFFFFFFFFFFFFF


def refuse(message):
    raise SystemExit("emit_tensix_program_header: REFUSED -- %s" % message)


def fnv1a_64(data):
    """The digest ft_tt_program_blob.c recomputes. Six lines there, six lines here, on
    purpose: the two are meant to be read against each other."""
    digest = FNV64_OFFSET
    for byte in data:
        digest ^= byte
        digest = (digest * FNV64_PRIME) & FNV64_MASK
    return digest


class ProgramDirectory:
    """One build_tensix_program.sh output directory, validated."""

    def __init__(self, path, toolchain, recipe):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path.rstrip("/"))
        self.toolchain = toolchain
        self.recipe = recipe
        self.set_id = None  # the shared firmware set, filled at emit time
        self.kernels = []   # (processor_name, file_name, source_name, bytes, KernelElf)
        self.firmware = []  # same, all five, in PROCESSOR_ORDER

    def load(self, minimum_text, tree_root):
        if not os.path.isdir(self.path):
            refuse("%s is not a directory" % self.path)
        if not self.name or self.name in (".", ".."):
            refuse("%s has no usable basename to name a program with" % self.path)
        self._load_firmware(minimum_text)
        self._load_kernels(minimum_text, tree_root)
        if not self.kernels:
            refuse("%s holds none of %s -- there is no program here"
                   % (self.path, ", ".join(KERNEL_FILE_NAMES)))

    def _load_firmware(self, minimum_text):
        for index, file_name in enumerate(FIRMWARE_FILE_NAMES):
            path = os.path.join(self.path, file_name)
            if not os.path.isfile(path):
                refuse("%s has no %s. Firmware is all five images or none: a program whose "
                       "firmware set is incomplete boots a core into whatever L1 held."
                       % (self.path, file_name))
            self.firmware.append(self._read_image(path, index, file_name, minimum_text))

    def _load_kernels(self, minimum_text, tree_root):
        for index, file_name in enumerate(KERNEL_FILE_NAMES):
            path = os.path.join(self.path, file_name)
            sidecar = path + ".src"
            if not os.path.isfile(path):
                if os.path.isfile(sidecar):
                    refuse("%s has %s.src but no %s -- a sidecar with no ELF beside it is a "
                           "half-written build" % (self.path, file_name, file_name))
                continue
            text = None
            if tree_root is not None:
                (text, record) = self._read_provenance(path)
                problems = check_record(record, tree_root, self.path)
                if problems:
                    refuse("\n        ".join(problems) + "\n        Nothing was written.")
            self.kernels.append(
                self._read_image(path, index, file_name, minimum_text, provenance=text))

    def _read_image(self, path, processor_index, file_name, minimum_text, provenance=None):
        with open(path, "rb") as handle:
            payload = handle.read()
        elf = KernelElf(path)
        if elf.text_bytes < minimum_text:
            refuse("%s text is only %u bytes -- this is the silently-empty --just-symbols "
                   "build (docs/tt_kernel_toolchain.md 4)" % (path, elf.text_bytes))
        return {
            "processor": PROCESSOR_ORDER[processor_index],
            "file_name": file_name,
            "source_name": read_source_name(path),
            "payload": payload,
            "digest": fnv1a_64(payload),
            "elf": elf,
            "provenance": provenance,
        }

    def _read_provenance(self, image_path):
        """The `.prov` record beside a kernel image, verified against the tree it names.

        FIRMWARE HAS NONE: the firmware build (tt/fw/) writes no `.prov` record. Only
        kernel images reach this."""
        sidecar = image_path + ".prov"
        if not os.path.isfile(sidecar):
            refuse("%s has no provenance record. An image nobody can vouch for is the "
                   "state that shipped an `rtriad_*` reader with no cooperative multicast "
                   "for a day -- rebuild it with build_tensix_program.sh, which writes "
                   "one." % image_path)
        with open(sidecar) as handle:
            text = handle.read()
        try:
            record = parse_record(text, sidecar)
        except ProvenanceError as error:
            refuse(str(error))
        return (text, record)

    def firmware_set_id(self):
        """A digest of all five firmware images, in order. Two programs whose firmware is
        byte-identical get the same id and are emitted once."""
        accumulator = hashlib.sha256()
        for image in self.firmware:
            accumulator.update(image["payload"])
            accumulator.update(b"\0")
        return accumulator.hexdigest()[:16]


def read_source_name(elf_path):
    """The kernel source an ELF was built from, per its `<elf>.src` sidecar. Firmware has
    no sidecar (it is not built from a kernel source), so the file name stands in."""
    sidecar = elf_path + ".src"
    if os.path.isfile(sidecar):
        with open(sidecar) as handle:
            name = handle.read().strip()
        if name:
            return os.path.basename(name)
    return os.path.basename(elf_path)


def run_source_check(kernels_directory, program_paths):
    """Delegate the stale-sidecar refusal to the script that already owns it."""
    if not os.path.isfile(SOURCE_CHECK_SCRIPT):
        refuse("%s is missing -- the stale-source check cannot run, and embedding a program "
               "directory nothing checked is exactly the failure this emitter exists to "
               "prevent" % SOURCE_CHECK_SCRIPT)
    environment = dict(os.environ)
    environment["KERNELS"] = kernels_directory
    completed = subprocess.run(
        ["bash", SOURCE_CHECK_SCRIPT] + list(program_paths),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout.decode("utf-8", "replace").rstrip()
    if output:
        print(output)
    if completed.returncode != 0:
        refuse("at least one program directory is STALE (above). Nothing was written. A "
               "stale directory on disk costs a re-run; a stale directory baked into the "
               "binary survives a rebuild.")


def parse_arguments(argv):
    output_path = None
    recorded_output = None
    kernels_directory = None
    tree_root = None
    minimum_text = DEFAULT_MIN_TEXT_BYTES
    allow_empty = False
    source_check = True
    toolchain = "unknown"
    recipe = "see the build log beside the directory"
    programs = []
    index = 0

    while index < len(argv):
        argument = argv[index]
        if argument in ("-o", "--output"):
            index += 1
            output_path = argv[index]
        elif argument == "--recorded-output":
            # The header records its own command line, and that line is the only record of
            # how the header was made -- so it has to be RUNNABLE. A driver that emits into
            # a staging file and installs it afterwards would otherwise record the staging
            # path, i.e. a recipe that writes to a temporary file nobody has.
            index += 1
            recorded_output = argv[index]
        elif argument == "--kernels":
            index += 1
            kernels_directory = argv[index]
        elif argument == "--tt-root":
            # What the provenance records' input paths are relative to. Defaults to the
            # parent of --kernels, which is what the tree looks like everywhere.
            index += 1
            tree_root = argv[index]
        elif argument == "--toolchain":
            index += 1
            toolchain = argv[index]
        elif argument == "--recipe":
            index += 1
            recipe = argv[index]
        elif argument == "--min-text-bytes":
            index += 1
            minimum_text = int(argv[index])
        elif argument == "--allow-empty":
            allow_empty = True
        elif argument == "--no-source-check":
            source_check = False
        elif argument == "--program":
            index += 1
            programs.append(ProgramDirectory(argv[index], toolchain, recipe))
        elif argument in ("-h", "--help"):
            print(__doc__)
            raise SystemExit(0)
        else:
            raise SystemExit("error: unrecognised argument %r" % argument)
        index += 1

    if not output_path:
        raise SystemExit("error: -o/--output is required")
    if not programs and not allow_empty:
        refuse("no --program was given. An EMPTY blob is a real state, but it must be asked "
               "for with --allow-empty: the way you get one by accident is a glob that "
               "matched nothing, and that blob refuses every program at run time.")
    if programs and source_check and not kernels_directory:
        raise SystemExit("error: --kernels is required (or --no-source-check, which is for "
                         "testing this script and nothing else)")

    names = {}
    for program in programs:
        if program.name in names:
            refuse("two program directories are both named %r (%s and %s). The name is the "
                   "lookup key, so one would shadow the other."
                   % (program.name, names[program.name], program.path))
        names[program.name] = program.path
    if tree_root is None and kernels_directory:
        tree_root = os.path.dirname(os.path.abspath(kernels_directory))
    return (output_path, recorded_output or output_path, kernels_directory, tree_root,
            programs, minimum_text, allow_empty, source_check)


class SymbolPool:
    """Emit each distinct payload once. Keyed by content, so it can never merge two
    images that differ."""

    def __init__(self):
        self.by_digest = {}
        self.order = []

    def symbol_for(self, payload, hint):
        key = hashlib.sha256(payload).hexdigest()
        if key not in self.by_digest:
            symbol = "ft_tt_program_image_%s_%s" % (hint, key[:12])
            self.by_digest[key] = symbol
            self.order.append((symbol, payload, hint))
        return self.by_digest[key]


def emit_image_row(handle, image, symbol):
    handle.write(
        '    {"%s", %s, %s, sizeof(%s), 0x%016xull,\n'
        % (image["source_name"], PROCESSOR_MACROS[image["processor"]], symbol, symbol,
           image["digest"])
    )
    handle.write(emit_provenance(image.get("provenance")))
    handle.write("    },\n")


def emit_provenance(text):
    """The `.prov` record as C string literals, ONE PER LINE.

    Line per line rather than one escaped blob, because this is the half of the header a
    human reads: a diff of a regenerated blob then shows which input moved, in words,
    instead of a single changed 900-character string."""
    if text is None:
        # Firmware. NULL is the honest value -- see ProgramDirectory._read_provenance.
        return "     NULL\n"
    lines = ['     "%s\\n"\n' % line.replace("\\", "\\\\").replace('"', '\\"')
             for line in text.splitlines()]
    return "".join(lines)


def emit_header(handle, programs, command_line, allow_empty):
    pool = SymbolPool()
    firmware_sets = {}

    for program in programs:
        set_id = program.firmware_set_id()
        if set_id not in firmware_sets:
            firmware_sets[set_id] = program
        program.set_id = set_id

    handle.write(HEADER_PREAMBLE % command_line)

    for image in _every_payload(programs):
        pool.symbol_for(image["payload"], image["hint"])
    for (symbol, payload, _hint) in pool.order:
        handle.write("/* %u bytes */\n" % len(payload))
        emit_blob(handle, symbol, payload)

    for (set_id, program) in sorted(firmware_sets.items()):
        handle.write(
            "/* firmware set %s -- first seen in %s */\n"
            "static const ft_tt_program_blob_firmware ft_tt_program_firmware_%s = {\n"
            '    "%s",\n    {\n' % (set_id, program.name, set_id, set_id)
        )
        for image in program.firmware:
            handle.write("    ")
            emit_image_row(handle, image, pool.symbol_for(image["payload"], "fw"))
        handle.write("    }\n};\n\n")

    for program in programs:
        handle.write(
            "static const ft_tt_program_blob_image ft_tt_program_%s_kernels[] = {\n"
            % program.name
        )
        for image in program.kernels:
            emit_image_row(handle, image, pool.symbol_for(image["payload"], "k"))
        handle.write("};\n\n")
        handle.write(
            "static const ft_tt_program_blob_entry ft_tt_program_%s = {\n"
            '    "%s", "%s", "%s",\n'
            "    &ft_tt_program_firmware_%s,\n"
            "    ft_tt_program_%s_kernels,\n"
            "    (unsigned)(sizeof(ft_tt_program_%s_kernels) / "
            "sizeof(ft_tt_program_%s_kernels[0]))\n};\n\n"
            % (program.name, program.name, program.toolchain, program.recipe,
               program.set_id, program.name, program.name, program.name)
        )

    if programs:
        handle.write("static const ft_tt_program_blob_entry *const "
                     "ft_tt_program_blob_table_storage[] = {\n")
        for program in programs:
            handle.write("    &ft_tt_program_%s,\n" % program.name)
        handle.write("};\n\n")
        handle.write("static const ft_tt_program_blob_entry *const *const "
                     "ft_tt_program_blob_table = ft_tt_program_blob_table_storage;\n")
        handle.write("#define FT_TT_PROGRAM_BLOB_TABLE_COUNT "
                     "((unsigned)(sizeof(ft_tt_program_blob_table_storage) / "
                     "sizeof(ft_tt_program_blob_table_storage[0])))\n\n")
    else:
        if not allow_empty:
            refuse("internal: reached the empty emission without --allow-empty")
        handle.write(EMPTY_TABLE)

    handle.write("#endif /* FT_TT_PROGRAMS_H */\n")


def _every_payload(programs):
    """Firmware first so the shared sets are emitted before the per-program kernels."""
    for program in programs:
        for image in program.firmware:
            yield dict(image, hint="fw")
    for program in programs:
        for image in program.kernels:
            yield dict(image, hint="k")


HEADER_PREAMBLE = '''/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_programs.h
 * @brief GENERATED -- do not edit. Embedded Tensix PROGRAM directories for the Foundry
 *   Blackhole backend, in the format ft_tt_program_blob.h defines.
 *
 * Produced by foundry/tt/tools/emit_tensix_program_header.py from directories built with
 * foundry/tt/build_tensix_program.sh. Regenerate rather than editing:
 *   make -f tools/Makefile regen-programs        (on the card host -- SFPI lives there)
 *
 * Each program carries BOTH its five kernel ELFs and the five fw_*.elf they were linked
 * against, because a kernel's text sits at its own firmware's __fw_export_text_end and the
 * two are one build output. Identical images are emitted once and shared.
 *
 * Command line:
 *   %s
 *
 * @status Generated artifact.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_PROGRAMS_H
#define FT_TT_PROGRAMS_H

#include "ft_tt_memmap.h"
#include "ft_tt_program_blob.h"

'''

EMPTY_TABLE = '''/* NO PROGRAMS. This blob was generated with --allow-empty, which is what the tree carries
 * until the first real regeneration on the card host. Every lookup refuses and says so; nothing
 * silently falls back. */
static const ft_tt_program_blob_entry *const *const ft_tt_program_blob_table = NULL;
#define FT_TT_PROGRAM_BLOB_TABLE_COUNT 0u

'''


def main(argv):
    (output_path, recorded_output, kernels_directory, tree_root, programs, minimum_text,
     allow_empty, source_check) = parse_arguments(argv)

    if programs and source_check:
        run_source_check(kernels_directory, [program.path for program in programs])
    else:
        # --no-source-check is the escape hatch for testing THIS script, and it turns off
        # the provenance check with it: both checks need the source tree, and half a check
        # is the state that reads as coverage.
        tree_root = None
    for program in programs:
        program.load(minimum_text, tree_root)

    recorded_argv = []
    skip_next = False
    for (position, word) in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if word == "--recorded-output":
            skip_next = True
            continue
        if position >= 1 and argv[position - 1] in ("-o", "--output"):
            recorded_argv.append(repo_relative(recorded_output))
            continue
        recorded_argv.append(repo_relative(word))
    command_line = " ".join(
        shlex.quote(word) for word in [os.path.basename(sys.argv[0])] + recorded_argv
    )
    # EMITTED IN FULL BEFORE THE OUTPUT IS TOUCHED. `open(path, "w")` truncates on the way
    # in, so emitting straight into the file would let a refusal half-way through destroy
    # the committed header -- turning a refusal into damage.
    buffer = io.StringIO()
    emit_header(buffer, programs, command_line, allow_empty)
    with open(output_path, "w") as handle:
        handle.write(buffer.getvalue())

    kernel_bytes = sum(len(i["payload"]) for p in programs for i in p.kernels)
    firmware_bytes = sum(len(i["payload"]) for p in programs for i in p.firmware)
    distinct = len({hashlib.sha256(i["payload"]).hexdigest()
                    for p in programs for i in (p.kernels + p.firmware)})
    print("wrote %s: %u program(s), %u distinct image(s), %u B kernels + %u B firmware "
          "before dedup" % (output_path, len(programs), distinct, kernel_bytes,
                            firmware_bytes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
