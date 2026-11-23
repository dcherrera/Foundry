#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Diagnose the PCIe bridge-window state of card 8 (holdout) vs a working card.
import os, sys

def rd(bdf, off, size):
    with open(f"/sys/bus/pci/devices/{bdf}/config", "rb") as f:
        f.seek(off)
        return int.from_bytes(f.read(size), "little")

def chain(die):
    rp = os.path.realpath(f"/sys/bus/pci/devices/{die}")
    parts = [p for p in rp.split("/") if len(p) == 12 and p[4] == ":" and p[7] == ":" and p[10] == "."]
    return parts[:-1]  # ancestors (bridges), excluding the die itself

def win(base16, lim16):
    base = (base16 & 0xfff0) << 16
    lim = ((lim16 & 0xfff0) << 16) | 0xfffff
    return base, lim

def dump(br):
    cmd = rd(br, 0x04, 2)
    mb = rd(br, 0x20, 2); ml = rd(br, 0x22, 2)
    pb = rd(br, 0x24, 2); pl = rd(br, 0x26, 2)
    pbu = rd(br, 0x28, 4); plu = rd(br, 0x2c, 4)
    nb, nl = win(mb, ml)
    prb, prl = win(pb, pl); prb |= pbu << 32; prl |= plu << 32
    mem = "ON " if (cmd & 2) else "OFF"
    pref_dis = (pb & 0xfff0) > (pl & 0xfff0)
    dis = "  <PREF WINDOW DISABLED>" if pref_dis else ""
    sec = rd(br, 0x19, 1); sub = rd(br, 0x1a, 1)
    print(f"  {br} [bus {sec:02x}->{sub:02x}] cmd=0x{cmd:04x} mem={mem}")
    print(f"      nonpref [{nb:#011x} - {nl:#011x}]")
    print(f"      pref    [{prb:#x} - {prl:#x}]{dis}")
    print(f"      raw: 0x24={pb:#06x} 0x26={pl:#06x} 0x28={pbu:#010x} 0x2c={plu:#010x}")

for lbl, die in [("WORKING card1", "0000:03:00.0"), ("CARD 8 holdout", "0000:20:00.0")]:
    print(f"=== {lbl}  chain to {die} ===")
    for br in chain(die):
        dump(br)
    print()
