#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Post-boot check of the HEADER-fixup kernel: did all 16 dies get a 256 MB BAR1
# assigned, and is card 8's bridge window now enabled (not [disabled])?
import glob, os

def rd(bdf, off, size):
    with open(f"/sys/bus/pci/devices/{bdf}/config", "rb") as f:
        f.seek(off); return int.from_bytes(f.read(size), "little")

def chain(die):
    rp = os.path.realpath(f"/sys/bus/pci/devices/{die}")
    return [p for p in rp.split("/") if len(p) == 12 and p[4] == ":" and p[7] == ":" and p[10] == "."][:-1]

# 1) all 16 dies' BAR1 (kernel-assigned size)
n256 = 0; tot = 0; rows = []
for d in sorted(glob.glob("/sys/bus/pci/devices/*")):
    try:
        if open(d+"/vendor").read().strip() != "0x10de" or open(d+"/device").read().strip() != "0x102d":
            continue
    except Exception:
        continue
    tot += 1
    L = open(d+"/resource").read().splitlines()
    s = int(L[1].split()[0], 16); e = int(L[1].split()[1], 16)
    sz = (e - s + 1) if e > s else 0
    mb = sz // (1024*1024)
    n256 += (mb == 256)
    rows.append((os.path.basename(d), mb))
print("=== all 16 GK210 dies — kernel-assigned BAR1 ===")
for bdf, mb in rows:
    print(f"  {bdf} BAR1={mb} MB" if mb else f"  {bdf} BAR1=unassigned")
print(f">>> {n256}/{tot} dies with a 256 MB BAR1 assigned <<<")

# 2) card 8's bridge prefetchable window
print("\n=== card 8 bridge chain (was [disabled]) ===")
for br in chain("0000:20:00.0"):
    cmd = rd(br, 0x04, 2); pb = rd(br, 0x24, 2); pl = rd(br, 0x26, 2)
    pbu = rd(br, 0x28, 4); plu = rd(br, 0x2c, 4)
    base = ((pb & 0xfff0) << 16) | (pbu << 32)
    lim = (((pl & 0xfff0) << 16) | 0xfffff) | (plu << 32)
    disabled = (pb & 0xfff0) > (pl & 0xfff0)
    state = "<DISABLED>" if disabled else f"[{base:#x}-{lim:#x}]"
    print(f"  {br} mem={'ON' if cmd&2 else 'off'} pref={state}")
