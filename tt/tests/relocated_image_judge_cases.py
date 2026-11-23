#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
@file relocated_image_judge_cases.py
@brief Host test of tools/compare_tensix_relocated_images.py on synthetic ELFs: a correct
  move passes, and every way a relinked image can be WRONG while looking moved is refused.

Usage:  python3 tests/relocated_image_judge_cases.py   (from tt/; no toolchain, no card)
        Exit 0 every check passed, 1 otherwise.

@status Added 2026-11-19 (q38 decode frontier F0).
@issues None
@todo None
"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import compare_tensix_relocated_images as judge  # noqa: E402

LOCAL_DATA = 0xFFB00810
FIRMWARE_CALL_TARGET = 0x00006800
checks_run = 0
checks_failed = 0


def lui(rd, value):
    return ((value & 0xFFFFF000) | (rd << 7) | 0x37) & 0xFFFFFFFF


def addi(rd, rs1, immediate):
    return (((immediate & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13) & 0xFFFFFFFF


def jal(rd, offset):
    offset &= 0x1FFFFF
    return ((((offset >> 20) & 1) << 31) | (((offset >> 1) & 0x3FF) << 21) |
            (((offset >> 11) & 1) << 20) | (((offset >> 12) & 0xFF) << 12) | (rd << 7) | 0x6F)


def hi_lo(value):
    """(hi, lo) such that hi + sign_extend(lo) == value, as lui/addi need."""
    lo = value & 0xFFF
    lo = lo - 0x1000 if lo & 0x800 else lo
    return ((value - lo) & 0xFFFFFFFF, lo)


def program_words(base, text_words, data_lma):
    """A tiny program at `base`: an absolute address pair to the data load image, a jal to a
    fixed firmware address, an untouched register move, and a pointer word to its own text."""
    (hi, lo) = hi_lo(data_lma)
    call_offset = FIRMWARE_CALL_TARGET - (base + 12)
    words = [lui(11, hi), addi(0, 0, 0), addi(11, 11, lo), jal(1, call_offset),
             addi(12, 13, 5), base + 4]
    return words + [addi(0, 0, 0)] * (text_words - len(words))


def build_elf(base, words, data_words, segments_limit=0x1000):
    """ELF32 RISC-V: PT_LOAD text at `base`, PT_LOAD data at LOCAL_DATA loaded from just past
    the text, and a 6-word `.segments` section."""
    text = struct.pack("<%dI" % len(words), *words)
    data = struct.pack("<%dI" % len(data_words), *data_words)
    data_lma = base + len(text)
    segments = struct.pack("<6I", base, base, segments_limit, LOCAL_DATA, LOCAL_DATA, 0x400)
    phoff, text_off = 52, 52 + 2 * 32
    data_off = text_off + len(text)
    seg_off = data_off + len(data)
    names = b"\0.segments\0.shstrtab\0"
    names_off = seg_off + len(segments)
    shoff = (names_off + len(names) + 3) & ~3
    header = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9)
    header += struct.pack("<HHIIIIIHHHHHH", 2, 243, 1, base, phoff, shoff, 0, 52, 32, 2, 40, 3, 2)
    phdrs = struct.pack("<8I", 1, text_off, base, base, len(text), len(text), 5, 16)
    phdrs += struct.pack("<8I", 1, data_off, LOCAL_DATA, data_lma, len(data), len(data), 6, 16)
    body = header + phdrs + text + data + segments + names
    body += bytes(shoff - len(body))
    sections = bytes(40)
    sections += struct.pack("<10I", 1, 1, 0, 0, seg_off, len(segments), 0, 0, 1, 0)
    sections += struct.pack("<10I", 11, 3, 0, 0, names_off, len(names), 0, 0, 1, 0)
    return body + sections


def image(base, mutate=None, data_words=None, segments_limit=None):
    """Bytes of the tiny program linked at `base`, after `mutate(words)` if given."""
    words = program_words(base, 8, base + 8 * 4)
    if mutate:
        mutate(words, base)
    data = data_words if data_words is not None else [0x11111111, base + 8]
    limit = segments_limit if segments_limit is not None else 0x4000 - (base - 0x1B000)
    return build_elf(base, words, data, limit)


def verdict_of(old_bytes, new_bytes):
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(__file__))) as work:
        (old_path, new_path) = (os.path.join(work, "a.elf"), os.path.join(work, "b.elf"))
        with open(old_path, "wb") as handle:
            handle.write(old_bytes)
        with open(new_path, "wb") as handle:
            handle.write(new_bytes)
        return judge.compare_relocated(old_path, new_path)


def check(condition, message):
    global checks_run, checks_failed
    checks_run += 1
    if not condition:
        checks_failed += 1
    print("  %s  %s" % ("ok  " if condition else "FAIL", message))


def expect_pass(old_bytes, new_bytes, message):
    (verdict, problems) = verdict_of(old_bytes, new_bytes)
    check(not problems, "%s -> %s" % (message, verdict if not problems else problems[:2]))


def expect_refused(old_bytes, new_bytes, message):
    (verdict, problems) = verdict_of(old_bytes, new_bytes)
    check(bool(problems), "%s -> %s" % (message, problems[:1] if problems else verdict))


def main():
    old_base, new_base = 0x1B000, 0x1BC00
    shipped = image(old_base)
    print("relocated-image judge: a move passes")
    expect_pass(shipped, shipped, "the same image is LOAD-IDENTICAL")
    expect_pass(shipped, image(new_base), "the program relinked 0xC00 higher")
    expect_pass(shipped, image(old_base + 0x1400), "and 0x1400 higher (lui and addi both move)")

    packed = old_base + 0x2
    expect_pass(image(old_base, data_words=[packed, old_base + 8]),
                image(new_base, data_words=[packed, new_base + 8]),
                "an unaligned value inside the old slab's range stays (a packed table)")

    print("relocated-image judge: wrong relinks are refused")

    def off_by_four(words, base):
        (hi, lo) = hi_lo(base + 8 * 4 + 4)
        words[0], words[2] = lui(11, hi), addi(11, 11, lo)
    expect_refused(shipped, image(new_base, off_by_four), "an address pair off by 4 from the move")

    def kept_old_address(words, base):
        (hi, lo) = hi_lo(old_base + 8 * 4)
        words[0], words[2] = lui(11, hi), addi(11, 11, lo)
    expect_refused(shipped, image(new_base, kept_old_address),
                   "an address pair that still points at the OLD slab")

    def call_moved_with_code(words, base):
        words[3] = jal(1, FIRMWARE_CALL_TARGET - (old_base + 12))
    expect_refused(shipped, image(new_base, call_moved_with_code),
                   "a call to fixed firmware whose target moved with the code")

    def other_register(words, base):
        words[4] = addi(12, 14, 5)
    expect_refused(shipped, image(new_base, other_register), "an unrelated instruction changed")

    def pointer_not_moved(words, base):
        words[5] = old_base + 4
    expect_refused(shipped, image(new_base, pointer_not_moved),
                   "a pointer word into the image that did not move")
    expect_refused(shipped, image(new_base, data_words=[0x11111112, new_base + 8]),
                   "a data word that is not a pointer changed")
    expect_refused(shipped, image(new_base, segments_limit=0x4000),
                   "a .segments text limit that did not shrink by the move")
    old_header = bytearray(image(new_base))
    struct.pack_into("<I", old_header, 24, new_base + 4)
    expect_refused(shipped, bytes(old_header), "an entry point not moved with the text")

    print("\n%d/%d checks failed" % (checks_failed, checks_run))
    return 0 if checks_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
