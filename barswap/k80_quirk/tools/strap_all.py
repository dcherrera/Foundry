#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Strap all 16 GK210 dies' silicon BAR1 to 256 MB (PSTRAPS field [17:14]=2 +
# override bit 31), matching the kernel's 256 MB assignment from the HEADER
# fixup. No remove/rescan needed — the kernel already placed 256 MB BARs; this
# just makes the silicon decode match so each die owns only its own window.
import glob, os, mmap, struct, subprocess

dies = []
for d in sorted(glob.glob("/sys/bus/pci/devices/*")):
    try:
        if open(d+"/vendor").read().strip() != "0x10de" or open(d+"/device").read().strip() != "0x102d":
            continue
    except Exception:
        continue
    dies.append(os.path.basename(d))

ok = 0
for bdf in dies:
    short = bdf[5:]
    subprocess.run(["setpci", "-s", short, "COMMAND=0x02:0x02"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bar0 = int(open(f"/sys/bus/pci/devices/{bdf}/resource").read().splitlines()[0].split()[0], 16)
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    pg = (bar0 + 0x101000) & ~0xfff
    off = (bar0 + 0x101000) - pg
    m = mmap.mmap(fd, 0x1000, offset=pg)
    v = struct.unpack("<I", m[off:off+4])[0]
    if v == 0xffffffff:
        print(f"{bdf}  UNREACHABLE — skip")
        m.close(); os.close(fd); continue
    nv = (v & ~(0xf << 14)) | (2 << 14) | (1 << 31)   # field=2 (256MB) + override
    m[off:off+4] = struct.pack("<I", nv)
    rb = struct.unpack("<I", m[off:off+4])[0]
    field = (rb >> 14) & 0xf
    ok += (field == 2)
    print(f"{bdf}  PSTRAPS {v:#010x} -> {rb:#010x}  field={field}{'  OK' if field==2 else '  !!'}")
    m.close(); os.close(fd)

print(f">>> strapped silicon to 256 MB (field=2): {ok}/{len(dies)} <<<")
