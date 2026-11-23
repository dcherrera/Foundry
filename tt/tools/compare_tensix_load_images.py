#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
@file compare_tensix_load_images.py
@brief Compare what a Tensix ELF puts on the card: entry address, every PT_LOAD (virtual and
  physical address, file and memory size, bytes) and the `.segments` limits record. Two images
  that agree on all of it load identically, whatever their symbol tables or section headers say.

Usage:  compare_tensix_load_images.py <a.elf> <b.elf>
        Prints LOAD-IDENTICAL, or each difference; exit 0 / 1.

The gate for a linker-script change (clean-room Phase 3): a layout that does not move must
leave every image LOAD-IDENTICAL.

@status Added 2026-10-08.
@issues None
@todo None
"""
import struct
import sys

PT_LOAD = 1


def load_view(path):
    """(entry, [(vaddr, paddr, filesz, memsz, bytes)], segments-record bytes or None)."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:4] != b"\x7fELF" or blob[4] != 1 or blob[5] != 1:
        sys.exit("%s: not a 32-bit little-endian ELF" % path)
    (entry, phoff, shoff) = struct.unpack_from("<III", blob, 24)
    (phentsize, phnum, shentsize, shnum, shstrndx) = struct.unpack_from("<HHHHH", blob, 42)
    loads = []
    for index in range(phnum):
        (kind, offset, vaddr, paddr, filesz, memsz) = struct.unpack_from(
            "<IIIIII", blob, phoff + index * phentsize)
        if kind == PT_LOAD:
            loads.append((vaddr, paddr, filesz, memsz, blob[offset:offset + filesz]))
    segments = None
    if shnum and shstrndx < shnum:
        names_offset = struct.unpack_from("<I", blob, shoff + shstrndx * shentsize + 16)[0]
        for index in range(shnum):
            header = struct.unpack_from("<IIIIIIIIII", blob, shoff + index * shentsize)
            name_end = blob.index(b"\0", names_offset + header[0])
            if blob[names_offset + header[0]:name_end] == b".segments":
                segments = blob[header[4]:header[4] + header[5]]
    return (entry, loads, segments)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    (entry_a, loads_a, segments_a) = load_view(sys.argv[1])
    (entry_b, loads_b, segments_b) = load_view(sys.argv[2])
    differences = []
    if entry_a != entry_b:
        differences.append("entry 0x%x != 0x%x" % (entry_a, entry_b))
    if len(loads_a) != len(loads_b):
        differences.append("PT_LOAD count %d != %d" % (len(loads_a), len(loads_b)))
    for (index, (a, b)) in enumerate(zip(loads_a, loads_b)):
        if a[:4] != b[:4]:
            differences.append("PT_LOAD %d vaddr/paddr/filesz/memsz %s != %s"
                               % (index, tuple(hex(v) for v in a[:4]), tuple(hex(v) for v in b[:4])))
        elif a[4] != b[4]:
            first = next(i for i in range(len(a[4])) if a[4][i] != b[4][i])
            differences.append("PT_LOAD %d bytes differ from offset 0x%x" % (index, first))
    if segments_a != segments_b:
        differences.append(".segments %s != %s" % (segments_a and segments_a.hex(),
                                                    segments_b and segments_b.hex()))
    if differences:
        print("LOAD-DIFFERS: " + "; ".join(differences))
        return 1
    print("LOAD-IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
