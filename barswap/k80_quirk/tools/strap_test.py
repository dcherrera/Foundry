#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# On a FRESH boot, test whether card 8's dies (20/21) are reachable for a PSTRAPS
# read — before any strap/rescan of other dies disturbs the bridges.
import os, mmap, struct, subprocess

def bar0_of(bdf):
    line = open(f"/sys/bus/pci/devices/{bdf}/resource").read().splitlines()[0]
    return int(line.split()[0], 16)

def read_pstraps(bar0):
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    pg = (bar0 + 0x101000) & ~0xfff
    off = (bar0 + 0x101000) - pg
    m = mmap.mmap(fd, 0x1000, offset=pg)
    v = struct.unpack("<I", m[off:off+4])[0]
    m.close(); os.close(fd)
    return v

for bdf in ["0000:20:00.0", "0000:21:00.0"]:
    short = bdf[5:]
    subprocess.run(["setpci", "-s", short, "COMMAND=0x02:0x02"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bar0 = bar0_of(bdf)
    v = read_pstraps(bar0)
    routes = (v != 0xffffffff)
    field = (v >> 14) & 0xf
    print(f"{bdf}  BAR0={bar0:#x}  PSTRAPS={v:#010x}  "
          f"{'ROUTES — reachable!' if routes else 'does NOT route (all-ones)'}  "
          f"field[17:14]={field} (2=256MB,8=16GB)")
