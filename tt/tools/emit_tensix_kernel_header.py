#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
emit_tensix_kernel_header.py
    Turn the TRISC/dataflow ELFs that build_tensix_kernel.sh produces into
    foundry/tt/ft_tt_kernels.h -- the embedded-blob header the Foundry TT backend
    loads kernels from. This is the TT counterpart of nvcc -> fc_cuda_ptx.h on the
    CUDA side (plan Phase 2, first bullet; format in ../ft_tt_kernel_blob.h).

    Runs offline, on whatever machine has the ELFs. It needs no SFPI and no card --
    but the ELFs themselves can only be built where SFPI is installed, which is the
    bank (docs/tt_kernel_toolchain.md 1).

    Usage:
      emit_tensix_kernel_header.py -o ../ft_tt_kernels.h \\
          --program vector_add --toolchain 7.68.0 --ctargs 1,1,1 \\
              --image MATH0=/tmp/ttk/trisc0.elf \\
              --image MATH1=/tmp/ttk/trisc1.elf \\
              --image MATH2=/tmp/ttk/trisc2.elf

      Repeat --program to put several programs in one header. --toolchain, --ctargs
      and --image all attach to the --program that precedes them.

    Processor names are the FT_TT_PROC_* set from ../ft_tt_memmap.h:
      DM0 (BRISC)  DM1 (NCRISC)  MATH0 (TRISC0)  MATH1 (TRISC1)  MATH2 (TRISC2)

    The `source_name` recorded for each image comes from a `<elf>.src` sidecar that the
    build scripts write next to every ELF -- the ELF itself is named after the processor
    (brisc.elf), not after the kernel, and the header's contract wants the .cpp. Without
    a sidecar the ELF's own basename is used, which is honest but less useful in a log.

    Status:  Run on the p150a card host 2026-08-11 against a real SFPI build of the vector_add
             triad; the committed ../ft_tt_kernels.h is its output.
    Issues:  None
    Todo:    None
"""

import os
import shlex
import struct
import sys

PROCESSOR_INDICES = {"DM0": 0, "DM1": 1, "MATH0": 2, "MATH1": 3, "MATH2": 4}
PROCESSOR_MACROS = {
    "DM0": "FT_TT_PROC_DM0",
    "DM1": "FT_TT_PROC_DM1",
    "MATH0": "FT_TT_PROC_MATH0",
    "MATH1": "FT_TT_PROC_MATH1",
    "MATH2": "FT_TT_PROC_MATH2",
}

ELF_TYPE_EXEC = 2
ELF_MACHINE_RISCV = 243
ELF_PT_LOAD = 1

# A kernel ELF linked with --just-symbols against an UN-weakened firmware ELF links
# cleanly and is empty: LTO drops our _start because the firmware already defines one
# (docs/tt_kernel_toolchain.md 4). It runs, reports success, and does nothing. Refuse
# to embed one.
#
# THE FLOOR IS SET FROM MEASUREMENTS, 2026-10-08 (Foundry's own entries, clean-room Phase 2):
# such a build keeps only the C-library helpers, ~72 bytes on a TRISC and nothing on BRISC or
# NCRISC, while the SMALLEST real kernel (an empty kernel_main on TRISC2) is 200 bytes. It was
# 256 when the entry was the staged tree's; that refused the empty gdngate compute (212 bytes).
DEFAULT_MIN_TEXT_BYTES = 160


class KernelElf:
    """Just enough ELF32 to validate a blob and describe it in a comment."""

    def __init__(self, path):
        with open(path, "rb") as handle:
            self.data = handle.read()
        self.path = path
        self._parse()

    def _parse(self):
        blob = self.data
        if len(blob) < 52 or blob[:4] != b"\x7fELF":
            raise ValueError("%s: not an ELF file" % self.path)
        if blob[4] != 1 or blob[5] != 1:
            raise ValueError("%s: not 32-bit little-endian" % self.path)
        (elf_type, machine) = struct.unpack_from("<HH", blob, 16)
        if elf_type != ELF_TYPE_EXEC:
            raise ValueError("%s: ELF type %d is not ET_EXEC" % (self.path, elf_type))
        if machine != ELF_MACHINE_RISCV:
            raise ValueError("%s: machine %d is not EM_RISCV" % (self.path, machine))

        (self.entry, phoff) = struct.unpack_from("<II", blob, 24)
        (phentsize, phnum) = struct.unpack_from("<HH", blob, 42)
        if phentsize != 32 or phnum == 0:
            raise ValueError("%s: malformed program headers" % self.path)

        self.segments = []
        for index in range(phnum):
            base = phoff + index * phentsize
            fields = struct.unpack_from("<IIIIII", blob, base)
            (seg_type, offset, vaddr, paddr, filesz, memsz) = fields
            if seg_type != ELF_PT_LOAD:
                continue
            if (offset | vaddr | paddr | filesz) & 3:
                raise ValueError("%s: PT_LOAD %d is not 4-byte aligned" % (self.path, index))
            if offset + filesz > len(blob):
                raise ValueError("%s: PT_LOAD %d runs past end of file" % (self.path, index))
            self.segments.append(
                {"vaddr": vaddr, "paddr": paddr, "filesz": filesz, "memsz": memsz}
            )

        if not self.segments:
            raise ValueError("%s: no loadable segments" % self.path)
        if self.segments[0]["vaddr"] != self.entry:
            raise ValueError(
                "%s: entry 0x%x is not the start of the first segment (0x%x)"
                % (self.path, self.entry, self.segments[0]["vaddr"])
            )

    @property
    def text_bytes(self):
        return self.segments[0]["memsz"]

    def describe(self):
        parts = [
            "entry 0x%x" % self.entry,
            "text %u B @ 0x%x" % (self.text_bytes, self.segments[0]["vaddr"]),
        ]
        for segment in self.segments[1:]:
            parts.append(
                "data %u B vma 0x%x lma 0x%x"
                % (segment["memsz"], segment["vaddr"], segment["paddr"])
            )
        return ", ".join(parts)


class Program:
    def __init__(self, name):
        self.name = name
        self.toolchain = "unknown"
        self.ctargs = ""
        self.images = []  # (processor_name, path, KernelElf)


def parse_arguments(argv):
    """Ordered parse: --image and the metadata flags attach to the last --program."""
    output_path = None
    minimum_text = DEFAULT_MIN_TEXT_BYTES
    programs = []
    index = 0

    def current():
        if not programs:
            raise SystemExit("error: %s must follow a --program" % argv[index])
        return programs[-1]

    while index < len(argv):
        argument = argv[index]
        if argument in ("-o", "--output"):
            index += 1
            output_path = argv[index]
        elif argument == "--program":
            index += 1
            programs.append(Program(argv[index]))
        elif argument == "--toolchain":
            index += 1
            current().toolchain = argv[index]
        elif argument == "--ctargs":
            index += 1
            current().ctargs = argv[index]
        elif argument == "--min-text-bytes":
            index += 1
            minimum_text = int(argv[index])
        elif argument == "--image":
            index += 1
            if "=" not in argv[index]:
                raise SystemExit("error: --image wants PROCESSOR=path, got %r" % argv[index])
            (processor, path) = argv[index].split("=", 1)
            if processor not in PROCESSOR_INDICES:
                raise SystemExit(
                    "error: unknown processor %r, expected one of %s"
                    % (processor, ", ".join(sorted(PROCESSOR_INDICES)))
                )
            current().images.append([processor, path, None])
        elif argument in ("-h", "--help"):
            print(__doc__)
            raise SystemExit(0)
        else:
            raise SystemExit("error: unrecognised argument %r" % argument)
        index += 1

    if not output_path:
        raise SystemExit("error: -o/--output is required")
    if not programs:
        raise SystemExit("error: at least one --program is required")
    return (output_path, programs, minimum_text)


def load_and_validate(programs, minimum_text):
    for program in programs:
        if not program.images:
            raise SystemExit("error: program %r has no --image" % program.name)
        seen = set()
        for entry in program.images:
            (processor, path, _) = entry
            if processor in seen:
                raise SystemExit(
                    "error: program %r lists %s twice" % (program.name, processor)
                )
            seen.add(processor)
            elf = KernelElf(path)
            if elf.text_bytes < minimum_text:
                raise SystemExit(
                    "error: %s text is only %u bytes -- this is the silently-empty "
                    "--just-symbols build described in docs/tt_kernel_toolchain.md 4. "
                    "Weaken the firmware ELF and relink."
                    % (path, elf.text_bytes)
                )
            entry[2] = elf


def source_name_for(elf_path):
    """The kernel .cpp an ELF was built from, per its `<elf>.src` sidecar."""
    sidecar = elf_path + ".src"
    if os.path.exists(sidecar):
        with open(sidecar) as handle:
            name = handle.read().strip()
        if name:
            return os.path.basename(name)
    return os.path.basename(elf_path)


def emit_blob(handle, symbol, blob):
    handle.write("static const unsigned char %s[] = {\n" % symbol)
    for start in range(0, len(blob), 12):
        chunk = blob[start : start + 12]
        handle.write("    " + " ".join("0x%02x," % byte for byte in chunk) + "\n")
    handle.write("};\n\n")


def emit_header(handle, programs, command_line):
    handle.write(
        '/* SPDX-License-Identifier: AGPL-3.0-or-later */\n'
        '/* Copyright (C) 2026 David (Corey) Herrera */\n'
        '/**\n'
        ' * @file ft_tt_kernels.h\n'
        ' * @brief GENERATED -- do not edit. Embedded Tensix kernel ELFs for the Foundry\n'
        ' *   Blackhole backend, in the format ft_tt_kernel_blob.h defines.\n'
        ' *\n'
        ' * Produced by foundry/tt/tools/emit_tensix_kernel_header.py from ELFs built with\n'
        ' * foundry/tt/build_tensix_kernel.sh. Regenerate rather than editing; the sources\n'
        ' * are under foundry/tt/kernels/.\n'
        ' *\n'
        ' * Command line:\n'
        ' *   %s\n'
        ' *\n'
        ' * @status Generated artifact.\n'
        ' * @issues None\n'
        ' * @todo None\n'
        ' */\n\n'
        '#ifndef FT_TT_KERNELS_H\n'
        '#define FT_TT_KERNELS_H\n\n'
        '#include "ft_tt_kernel_blob.h"\n'
        '#include "ft_tt_memmap.h"\n\n' % command_line
    )

    for program in programs:
        for (processor, path, elf) in program.images:
            symbol = "ft_tt_kernel_%s_%s_elf" % (program.name, processor.lower())
            handle.write(
                "/* %s -> %s -- %s */\n"
                % (source_name_for(path), os.path.basename(path), elf.describe())
            )
            emit_blob(handle, symbol, elf.data)

        handle.write(
            "static const ft_tt_kernel_image ft_tt_kernel_%s_images[] = {\n" % program.name
        )
        for (processor, path, elf) in program.images:
            symbol = "ft_tt_kernel_%s_%s_elf" % (program.name, processor.lower())
            handle.write(
                '    {"%s", %s, %s, sizeof(%s)},\n'
                % (source_name_for(path), PROCESSOR_MACROS[processor], symbol, symbol)
            )
        handle.write("};\n\n")

        handle.write(
            'static const ft_tt_kernel_program ft_tt_kernel_%s = {\n'
            '    "%s", "%s", "%s",\n'
            '    ft_tt_kernel_%s_images,\n'
            '    (unsigned)(sizeof(ft_tt_kernel_%s_images) / sizeof(ft_tt_kernel_%s_images[0]))\n'
            "};\n\n"
            % (
                program.name,
                program.name,
                program.toolchain,
                program.ctargs,
                program.name,
                program.name,
                program.name,
            )
        )

    handle.write("static const ft_tt_kernel_program *const ft_tt_kernel_programs[] = {\n")
    for program in programs:
        handle.write("    &ft_tt_kernel_%s,\n" % program.name)
    handle.write("};\n\n")
    handle.write(
        "#define FT_TT_KERNEL_PROGRAM_COUNT "
        "((unsigned)(sizeof(ft_tt_kernel_programs) / sizeof(ft_tt_kernel_programs[0])))\n\n"
    )
    handle.write("#endif /* FT_TT_KERNELS_H */\n")


def main(argv):
    (output_path, programs, minimum_text) = parse_arguments(argv)
    load_and_validate(programs, minimum_text)

    # SHELL-QUOTED, because the line recorded in the header is the ONLY record of how the
    # header was made and it has to be runnable. `--ctargs` takes ONE string, so a value
    # like `DM0=0,1,2048 DM1=16,2048 MATH0..2=0,1,16` printed bare re-parses as three
    # arguments and argparse rejects it -- a recipe that looks authoritative and does not
    # run. shlex.quote() only adds quotes where they are needed, so the simple arguments
    # are unchanged.
    command_line = " ".join(
        shlex.quote(word) for word in [os.path.basename(sys.argv[0])] + argv
    )
    with open(output_path, "w") as handle:
        emit_header(handle, programs, command_line)

    total = sum(len(elf.data) for program in programs for (_, _, elf) in program.images)
    print(
        "wrote %s: %u program(s), %u image(s), %u bytes of ELF"
        % (
            output_path,
            len(programs),
            sum(len(program.images) for program in programs),
            total,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
