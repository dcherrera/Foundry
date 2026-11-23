#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
@file compare_tensix_relocated_images.py
@brief The gate for a kernel-text slab move: is a relinked Tensix ELF the SAME PROGRAM as the
  shipped one, only placed at another L1 address? LOAD-IDENTICAL cannot hold across a slab
  move, so this checks every byte that reaches the card and explains each difference.

Usage:  compare_tensix_relocated_images.py <shipped.elf> <relinked.elf>
        Prints RELOC-IDENTICAL (delta 0x..., N words relocated), LOAD-IDENTICAL (delta 0), or
        each difference it cannot explain; exit 0 / 1.

What it accepts, with delta = relinked .text address - shipped .text address:
  - entry and the slab-resident PT_LOADs (text, and the data image's load address) moved by
    delta; every other PT_LOAD address equal; every file and memory size equal;
  - the `.segments` record with its text words moved by delta and the text limit less delta;
  - every reference judged, whether or not its bytes changed:
      * an address pair (lui or auipc + addi/load/store/jalr on the same register) and every
        jal/branch: a value inside the shipped image's slab extent moves by exactly delta, a
        value outside it stays put (the code moved, the firmware or local memory did not);
      * any other word that changed, or whose value points inside the extent (a jump table,
        a pointer constant, every .data word): moved by delta if it pointed in, else equal.
  So a word that changed for another reason (another instruction, another register, a value
  off by anything but delta) is reported, and so is a reference into the image that did NOT
  move (a hard-coded slab address). A difference is never accepted by elimination.

@status Added 2026-11-19 (q38 decode frontier F0, asymmetric text slabs).
@issues Pairs are found by a forward scan of at most PAIR_WINDOW instructions; a pair the
  compiler split further apart is reported, not accepted (fail closed).
@todo None
"""
import struct
import sys

PT_LOAD = 1
PAIR_WINDOW = 32
OPCODE_LUI = 0x37
OPCODE_AUIPC = 0x17
OPCODE_JAL = 0x6F
OPCODE_JALR = 0x67
OPCODE_BRANCH = 0x63
OPCODE_LOAD = 0x03
OPCODE_STORE = 0x23
OPCODE_OP_IMM = 0x13
LO_I_OPCODES = (OPCODE_LOAD, OPCODE_OP_IMM, OPCODE_JALR)


def sign_extend(value, bits):
    """Two's-complement sign extension of the low `bits` of `value`."""
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def read_elf_view(path):
    """(entry, [PT_LOAD (vaddr, paddr, filesz, memsz, bytes)], .segments bytes or None)."""
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


def words_of(data):
    """The 32-bit little-endian words of `data` (length a multiple of 4, else None)."""
    if len(data) % 4:
        return None
    return list(struct.unpack_from("<%dI" % (len(data) // 4), data))


def lo_immediate(word):
    """The 12-bit low immediate of an I- or S-type word, or None for any other opcode."""
    opcode = word & 0x7F
    if opcode in LO_I_OPCODES:
        return sign_extend(word >> 20, 12)
    if opcode == OPCODE_STORE:
        return sign_extend(((word >> 25) << 5) | ((word >> 7) & 0x1F), 12)
    return None


def writes_register(word, register):
    """True when the instruction overwrites `register` (stores and branches write none)."""
    opcode = word & 0x7F
    if opcode in (OPCODE_STORE, OPCODE_BRANCH) or register == 0:
        return False
    return ((word >> 7) & 0x1F) == register


def same_but_immediate(old, new):
    """Same opcode and register fields; only the immediate bits may differ."""
    opcode = old & 0x7F
    if opcode != new & 0x7F:
        return False
    if opcode in (OPCODE_LUI, OPCODE_AUIPC, OPCODE_JAL):
        mask = 0x00000FFF
    elif opcode in LO_I_OPCODES:
        mask = 0x000FFFFF
    elif opcode in (OPCODE_STORE, OPCODE_BRANCH):
        mask = 0x01FFF07F
    else:
        return False
    return (old & mask) == (new & mask)


def jal_or_branch_target(word, pc):
    """Absolute target of a jal or conditional branch at `pc`."""
    if word & 0x7F == OPCODE_JAL:
        offset = (((word >> 31) & 1) << 20) | (((word >> 12) & 0xFF) << 12) | \
            (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3FF) << 1)
        return (pc + sign_extend(offset, 21)) & 0xFFFFFFFF
    offset = (((word >> 31) & 1) << 12) | (((word >> 7) & 1) << 11) | \
        (((word >> 25) & 0x3F) << 5) | (((word >> 8) & 0xF) << 1)
    return (pc + sign_extend(offset, 13)) & 0xFFFFFFFF


class RelocationJudge:
    """Checks every reference in one PT_LOAD against the move of the image by `delta`.

    References are checked whether or not their bytes changed: a reference into the image
    whose bytes stayed the same did NOT move with it (a hard-coded slab address), which is as
    wrong as a reference that moved by the wrong amount."""

    def __init__(self, old_words, new_words, old_base, delta, extent):
        self.old = old_words
        self.new = new_words
        self.old_base = old_base
        self.delta = delta
        (self.extent_low, self.extent_high) = extent
        self.explained = set()
        self.judged = set()
        self.problems = []

    def inside(self, address):
        """In the image's slab extent; one past the end counts (end symbols, an empty LMA)."""
        return self.extent_low <= address <= self.extent_high

    def moved_correctly(self, old_value, new_value):
        """A reference into the image moves by delta; any other reference stays put."""
        if self.inside(old_value):
            return new_value == (old_value + self.delta) & 0xFFFFFFFF
        return new_value == old_value

    def pair_uses(self, index):
        """Indices of the lo instructions that consume the U-type result at `index`."""
        register = (self.old[index] >> 7) & 0x1F
        uses = []
        for later in range(index + 1, min(index + 1 + PAIR_WINDOW, len(self.old))):
            word = self.old[later]
            if lo_immediate(word) is not None and ((word >> 15) & 0x1F) == register:
                uses.append(later)
            if writes_register(word, register):
                break
        return uses

    def judge_upper_pair(self, index):
        """A lui / auipc and each lo use: the pair's value must move as `moved_correctly` says."""
        old_hi = self.old[index] & 0xFFFFF000
        new_hi = self.new[index] & 0xFFFFF000
        is_auipc = self.old[index] & 0x7F == OPCODE_AUIPC
        old_pc = self.old_base + 4 * index
        for use in self.pair_uses(index):
            if not same_but_immediate(self.old[use], self.new[use]):
                continue
            old_value = (old_hi + lo_immediate(self.old[use])) & 0xFFFFFFFF
            new_value = (new_hi + lo_immediate(self.new[use])) & 0xFFFFFFFF
            if is_auipc:
                old_value = (old_value + old_pc) & 0xFFFFFFFF
                new_value = (new_value + old_pc + self.delta) & 0xFFFFFFFF
            self.judged.update((index, use))
            if self.moved_correctly(old_value, new_value):
                self.explained.update((index, use))
            else:
                self.problems.append("pair at +0x%x/+0x%x: 0x%08x -> 0x%08x is not the move"
                                     % (4 * index, 4 * use, old_value, new_value))

    def judge_pc_relative(self, index):
        """A jal / branch: a target outside the image must stay put as the code moves."""
        pc = self.old_base + 4 * index
        old_target = jal_or_branch_target(self.old[index], pc)
        new_target = jal_or_branch_target(self.new[index], pc + self.delta)
        self.judged.add(index)
        if self.moved_correctly(old_target, new_target):
            self.explained.add(index)
        else:
            self.problems.append("jump at +0x%x: target 0x%08x -> 0x%08x is not the move"
                                 % (4 * index, old_target, new_target))

    def judge_data_word(self, index):
        """A word read as a pointer: into the image it moves by delta, else it is unchanged.
        An UNCHANGED word that is not 4-byte aligned is a value, not a pointer: every text and
        load-image address is word aligned (no compressed code), and packed 16-bit tables
        such as 0x00020002 fall inside a slab's address range by chance."""
        (old, new) = (self.old[index], self.new[index])
        self.judged.add(index)
        if old == new and old % 4 != 0:
            return
        if (self.inside(old) or old != new) and not self.moved_correctly(old, new):
            self.problems.append("word at +0x%x: 0x%08x -> 0x%08x is not the move"
                                 % (4 * index, old, new))
        elif old != new:
            self.explained.add(index)

    def run_text(self):
        """Code: every address pair and jump, then any other word that changed or whose value
        points into the image, as a pointer (a jump table, a function-pointer constant)."""
        for index, (old, new) in enumerate(zip(self.old, self.new)):
            if old & 0x7F in (OPCODE_LUI, OPCODE_AUIPC) and same_but_immediate(old, new):
                self.judge_upper_pair(index)
        for index, (old, new) in enumerate(zip(self.old, self.new)):
            if index in self.judged or old & 0x3 != 0x3:
                continue
            if old & 0x7F in (OPCODE_JAL, OPCODE_BRANCH) and same_but_immediate(old, new):
                self.judge_pc_relative(index)
        for index, (old, new) in enumerate(zip(self.old, self.new)):
            if index not in self.judged and (old != new or self.inside(old)):
                self.judge_data_word(index)
        return self.relocated_count()

    def run_data(self):
        """Data: every word is a value or a pointer; pointers into the image must move."""
        for index in range(len(self.old)):
            self.judge_data_word(index)
        return self.relocated_count()

    def relocated_count(self):
        """Explained words whose bytes actually changed."""
        return sum(1 for index in self.explained if self.old[index] != self.new[index])


def compare_segments(old, new, delta):
    """`.segments`: text address twice (+delta), text limit (-delta), data words unchanged."""
    if old is None or new is None:
        return [] if old == new else [".segments present in only one image"]
    old_words, new_words = words_of(old), words_of(new)
    if old_words is None or new_words is None or len(old_words) != 6 or len(new_words) != 6:
        return [] if old == new else [".segments not the 6-word record"]
    expected = [(old_words[0] + delta) & 0xFFFFFFFF, (old_words[1] + delta) & 0xFFFFFFFF,
                (old_words[2] - delta) & 0xFFFFFFFF] + old_words[3:]
    return [] if new_words == expected else [".segments %s != expected %s"
                                             % (new.hex(), struct.pack("<6I", *expected).hex())]


def compare_relocated(old_path, new_path):
    """(verdict line, [problems]) for one shipped / relinked image pair."""
    (old_entry, old_loads, old_segments) = read_elf_view(old_path)
    (new_entry, new_loads, new_segments) = read_elf_view(new_path)
    if not old_loads or len(old_loads) != len(new_loads):
        return ("RELOC-DIFFERS", ["PT_LOAD count %d != %d" % (len(old_loads), len(new_loads))])
    delta = (new_loads[0][0] - old_loads[0][0]) & 0xFFFFFFFF
    delta = delta - (1 << 32) if delta & 0x80000000 else delta
    text_low = old_loads[0][0]
    extent_high = max(load[1] + load[2] for load in old_loads
                      if text_low <= load[1] < text_low + 0x100000)
    extent = (text_low, extent_high)
    problems = []
    if new_entry != (old_entry + delta) & 0xFFFFFFFF:
        problems.append("entry 0x%x != 0x%x + delta" % (new_entry, old_entry))
    relocated = 0
    for index, (old, new) in enumerate(zip(old_loads, new_loads)):
        moves = extent[0] <= old[1] <= extent[1]
        shift = delta if moves else 0
        want_vaddr = (old[0] + delta) & 0xFFFFFFFF if index == 0 else old[0]
        if (new[0], new[1], new[2], new[3]) != (want_vaddr, (old[1] + shift) & 0xFFFFFFFF,
                                               old[2], old[3]):
            problems.append("PT_LOAD %d %s != %s moved by %d" % (
                index, tuple(hex(v) for v in new[:4]), tuple(hex(v) for v in old[:4]), shift))
            continue
        if old[4] == new[4] and delta == 0:
            continue
        (old_words, new_words) = (words_of(old[4]), words_of(new[4]))
        if old_words is None:
            problems.append("PT_LOAD %d differs and is not word-sized" % index)
            continue
        judge = RelocationJudge(old_words, new_words, old[0], delta, extent)
        relocated += judge.run_text() if index == 0 else judge.run_data()
        problems += ["PT_LOAD %d %s" % (index, problem) for problem in judge.problems]
    problems += compare_segments(old_segments, new_segments, delta)
    if problems:
        return ("RELOC-DIFFERS", problems)
    if delta == 0 and relocated == 0:
        return ("LOAD-IDENTICAL", [])
    return ("RELOC-IDENTICAL (delta %+#x, %d words relocated)" % (delta, relocated), [])


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    (verdict, problems) = compare_relocated(sys.argv[1], sys.argv[2])
    if problems:
        print("RELOC-DIFFERS: " + "; ".join(problems[:20]) +
              ("; ... %d more" % (len(problems) - 20) if len(problems) > 20 else ""))
        return 1
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
