#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
make_program_blob_fixtures.py
    Build the synthetic Tensix program directories tests/test_program_blob.c gates the
    embedded-program blob against.

    WHY SYNTHETIC. Real program directories are built by build_tensix_program.sh, which
    needs SFPI, which exists only on the card host. So the blob mechanism is gated here against
    directories this script writes: minimal but genuinely valid ELF32/LE/ET_EXEC/EM_RISCV
    images that ft_tt_elf_parse accepts, laid out with the file names, the .src sidecars and
    the fw_*.elf set that build_tensix_program.sh produces.

    WHAT THAT DOES AND DOES NOT PROVE, stated so it is not overread: it proves the blob
    format, the generator's refusals, the lookup, the integrity checks and the override
    behaviour. It proves NOTHING about SFPI output. A real regeneration is untested and
    needs a bank session.

    Usage:  make_program_blob_fixtures.py <output-root>

    Writes:
      <root>/kernels/                   the "source tree" the .src sidecars name
      <root>/programs/prog_alpha/       five processors, firmware set A
      <root>/programs/prog_beta/        DM0+DM1 only, firmware set A   (dedup + partial mask)
      <root>/programs/prog_gamma/       five processors, firmware set B (a DIFFERENT set)
      <root>/bad/<case>/                one directory per refusal the generator must make

    Two of those cases are about PROVENANCE (tools/tensix_image_provenance.py):
    `no_provenance/prog_unvouched` records nothing, and `stale_provenance/prog_drifted`
    has sources edited after its records were written WITHOUT their mtimes moving — the
    case the mtime check is structurally unable to see.

    Status:  Runs on the Mac; no card, no toolchain.
    Issues:  None
    Todo:    None
"""

import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from tensix_image_provenance import build_record, hash_file  # noqa: E402

PROCESSOR_FILE_NAMES = ["brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf"]
FIRMWARE_FILE_NAMES = [
    "fw_brisc.elf",
    "fw_ncrisc.elf",
    "fw_trisc0.elf",
    "fw_trisc1.elf",
    "fw_trisc2.elf",
]

EHDR_SIZE = 52
PHDR_SIZE = 32
TEXT_VADDR = 0x00008000
DATA_VADDR = 0xFFB00820
DEFAULT_TEXT_BYTES = 512
DEFAULT_DATA_BYTES = 64


def deterministic_bytes(seed, count):
    """A distinct, reproducible payload per (program, processor), so a lookup that returns
    the WRONG image is visible as different bytes rather than as a passing test."""
    payload = bytearray(count)
    state = 0
    for character in seed.encode("utf-8"):
        state = (state * 131 + character) & 0xFFFFFFFF
    for index in range(count):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        payload[index] = (state >> 16) & 0xFF
    return bytes(payload)


def build_elf(seed, text_bytes=DEFAULT_TEXT_BYTES, data_bytes=DEFAULT_DATA_BYTES):
    """A two-PT_LOAD Tensix-shaped ELF32: text at its own vaddr with e_entry on its first
    byte, and a data segment whose VMA (local memory) differs from its LMA (staged in L1)
    exactly as a real kernel's does."""
    phdr_count = 2 if data_bytes else 1
    text_offset = EHDR_SIZE + PHDR_SIZE * phdr_count
    assert text_offset % 4 == 0
    data_offset = text_offset + text_bytes

    header = bytearray()
    header += b"\x7fELF" + bytes([1, 1, 1, 0]) + bytes(8)  # e_ident: ELF32, LE, v1
    header += struct.pack("<HH", 2, 243)                   # ET_EXEC, EM_RISCV
    header += struct.pack("<I", 1)                         # e_version
    header += struct.pack("<I", TEXT_VADDR)                # e_entry
    header += struct.pack("<I", EHDR_SIZE)                 # e_phoff
    header += struct.pack("<I", 0)                         # e_shoff — no sections
    header += struct.pack("<I", 0)                         # e_flags
    header += struct.pack("<HHHHHH", EHDR_SIZE, PHDR_SIZE, phdr_count, 40, 0, 0)
    assert len(header) == EHDR_SIZE

    program_headers = struct.pack(
        "<IIIIIIII", 1, text_offset, TEXT_VADDR, TEXT_VADDR, text_bytes, text_bytes, 5, 4
    )
    if data_bytes:
        program_headers += struct.pack(
            "<IIIIIIII", 1, data_offset, DATA_VADDR, TEXT_VADDR + text_bytes,
            data_bytes, data_bytes, 6, 4
        )

    body = deterministic_bytes(seed + ":text", text_bytes)
    if data_bytes:
        body += deterministic_bytes(seed + ":data", data_bytes)
    return bytes(header) + program_headers + body


def write_file(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)


def write_provenance(directory, elf_name, source_name, kernels_directory, tree_root):
    """The `.prov` record build_tensix_program.sh writes beside every kernel image.

    The fixture source is the only input, which is enough: the gate is whether the
    generator refuses a record that no longer describes the tree, and one input can be
    made to stop describing it exactly as well as twelve can."""
    source_path = os.path.join(kernels_directory, source_name)
    inputs = {os.path.relpath(source_path, tree_root): hash_file(source_path)}
    record = build_record(elf_name, source_name,
                          {"toolchain": "fixture-1.0", "ctargs": "0,1,2"}, inputs)
    write_file(os.path.join(directory, elf_name + ".prov"), record.encode("utf-8"))


def write_program(directory, program, processors, firmware_set, kernels_directory,
                  sidecars=True, text_bytes=DEFAULT_TEXT_BYTES, skip_firmware=(),
                  write_kernel_sources=True, provenance=True):
    """One build_tensix_program.sh-shaped directory. `processors` is a list of indices."""
    for index in range(len(FIRMWARE_FILE_NAMES)):
        if FIRMWARE_FILE_NAMES[index] in skip_firmware:
            continue
        write_file(os.path.join(directory, FIRMWARE_FILE_NAMES[index]),
                   build_elf("%s:%s" % (firmware_set, FIRMWARE_FILE_NAMES[index])))

    for index in processors:
        elf_name = PROCESSOR_FILE_NAMES[index]
        source_name = "%s_%s.c" % (program, elf_name.split(".")[0])
        write_file(os.path.join(directory, elf_name),
                   build_elf("%s:%s" % (program, elf_name), text_bytes=text_bytes))
        if sidecars:
            write_file(os.path.join(directory, elf_name + ".src"),
                       (source_name + "\n").encode("utf-8"))
        if write_kernel_sources:
            write_file(os.path.join(kernels_directory, source_name),
                       ("/* fixture source for %s */\n" % elf_name).encode("utf-8"))
        # The provenance record hashes the source, so it can only be written once that
        # source exists -- a directory built with write_kernel_sources=False is testing the
        # missing-source refusal and has nothing to hash.
        if provenance and write_kernel_sources:
            write_provenance(directory, elf_name, source_name, kernels_directory,
                             os.path.dirname(kernels_directory))


def age_tree(directory, seconds):
    """Push every file back in time, so that afterwards touching ONE source makes it
    unambiguously newer than every ELF.

    This matters more than it looks. `check_tensix_program_sources.sh` compares mtimes at
    one-second granularity, so a fixture whose source and ELF were written in the same
    second would pass the stale gate for the wrong reason — the same GNU make 3.81 trap one
    layer out. Aging the whole tree first makes the NEWER case a 600-second gap and the
    CURRENT case a 600-second margin, so neither answer can be an accident of timing."""
    stamp = time.time() - seconds
    for root, _dirs, files in os.walk(directory):
        for name in files:
            os.utime(os.path.join(root, name), (stamp, stamp))


def build_good_programs(root, kernels):
    programs = os.path.join(root, "programs")
    write_program(os.path.join(programs, "prog_alpha"), "prog_alpha", [0, 1, 2, 3, 4],
                  "fw_set_a", kernels)
    # DM0 + DM1 only: the memory family's shape, and the case where a caller's mask must
    # be refused for MATH0 rather than quietly dropped.
    write_program(os.path.join(programs, "prog_beta"), "prog_beta", [0, 1],
                  "fw_set_a", kernels)
    # A DIFFERENT firmware set, so the dedup check has something to NOT merge.
    write_program(os.path.join(programs, "prog_gamma"), "prog_gamma", [0, 1, 2, 3, 4],
                  "fw_set_b", kernels)


def build_bad_programs(root, kernels):
    """@return the source path main() must touch to create the "source newer than its ELF"
    case, once the tree has been aged."""
    bad = os.path.join(root, "bad")

    # 1. A sidecar naming a source that is not in the kernel tree — the exact shape of the
    #    incident where seven directories named .cpp files deleted weeks earlier. Written
    #    WITH its sources and provenance and then deleted, so this case is about the
    #    missing source and not about a directory that never recorded anything.
    directory = os.path.join(bad, "stale_missing_source", "prog_stale")
    write_program(directory, "prog_stale", [0, 1, 2, 3, 4], "fw_set_a", kernels)
    for elf_name in PROCESSOR_FILE_NAMES:
        os.remove(os.path.join(kernels, "prog_stale_%s.c" % elf_name.split(".")[0]))

    # 2. A source EDITED after its ELF was built.
    directory = os.path.join(bad, "stale_newer_source", "prog_newer")
    write_program(directory, "prog_newer", [0, 1, 2, 3, 4], "fw_set_a", kernels)
    # The touch itself is deferred to main(), AFTER the whole tree is aged: touching now
    # and aging afterwards would age the edit away.

    # 3. Built by something that records nothing.
    directory = os.path.join(bad, "no_sidecars", "prog_mute")
    write_program(directory, "prog_mute", [0, 1, 2, 3, 4], "fw_set_a", kernels,
                  sidecars=False)

    # 4. Four firmware images out of five.
    directory = os.path.join(bad, "missing_firmware", "prog_partial_fw")
    write_program(directory, "prog_partial_fw", [0, 1, 2, 3, 4], "fw_set_a", kernels,
                  skip_firmware=("fw_trisc2.elf",))

    # 5. A sidecar with no ELF beside it: a half-written build.
    directory = os.path.join(bad, "orphan_sidecar", "prog_orphan")
    write_program(directory, "prog_orphan", [0, 1, 2, 3], "fw_set_a", kernels)
    write_file(os.path.join(directory, "trisc2.elf.src"), b"prog_orphan_trisc2.c\n")
    write_file(os.path.join(kernels, "prog_orphan_trisc2.c"), b"/* fixture */\n")

    # 6. The silently-empty --just-symbols build: an ELF that links, runs and does nothing.
    directory = os.path.join(bad, "tiny_text", "prog_tiny")
    write_program(directory, "prog_tiny", [0, 1, 2, 3, 4], "fw_set_a", kernels,
                  text_bytes=128)

    # 7. Two directories with the same basename — the lookup key would be ambiguous.
    for side in ("dup_left", "dup_right"):
        directory = os.path.join(bad, side, "prog_dup")
        write_program(directory, "prog_dup", [0, 1, 2, 3, 4], "fw_set_a", kernels)

    # 8. Built by something that records no PROVENANCE. `.src` sidecars, current mtimes,
    #    and nothing that says what content produced the images — which is precisely what
    #    an exploded directory looks like, and why the mtime check passed 21 of 21
    #    directories two of which were stale.
    directory = os.path.join(bad, "no_provenance", "prog_unvouched")
    write_program(directory, "prog_unvouched", [0, 1, 2, 3, 4], "fw_set_a", kernels,
                  provenance=False)

    # 9. A source EDITED after the record was written, with the mtimes left ALONE. This is
    #    the case the mtime check cannot see and the whole reason the content record
    #    exists: age_tree() runs after this, so the edited source is not newer than its ELF
    #    and `check_tensix_program_sources.sh` calls this directory current.
    directory = os.path.join(bad, "stale_provenance", "prog_drifted")
    write_program(directory, "prog_drifted", [0, 1, 2, 3, 4], "fw_set_a", kernels)
    for elf_name in PROCESSOR_FILE_NAMES:
        source = os.path.join(kernels, "prog_drifted_%s.c" % elf_name.split(".")[0])
        write_file(source, b"/* fixture source, EDITED after the images were built */\n")

    return os.path.join(kernels, "prog_newer_trisc0.c")


def main(argv):
    if len(argv) != 1:
        raise SystemExit("usage: make_program_blob_fixtures.py <output-root>")
    root = os.path.abspath(argv[0])
    kernels = os.path.join(root, "kernels")
    os.makedirs(kernels, exist_ok=True)

    build_good_programs(root, kernels)
    newer_source = build_bad_programs(root, kernels)
    age_tree(root, 600)
    os.utime(newer_source, None)
    print("fixtures under %s" % root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
