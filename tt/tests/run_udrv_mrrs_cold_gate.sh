#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# run_udrv_mrrs_cold_gate.sh — prove the userspace bring-up performs BOTH cold-path config
# steps: MEMORY DECODE and MRRS.
#
# MEMORY DECODE WAS ADDED 2026-08-18 AFTER IT BIT ON A REAL COLD HOST. the TT host has never had
# tt-kmd installed, and there PCI_COMMAND reads 0x0100 — Memory Space Enable CLEAR. The BAR
# mmap still succeeds and every word of it reads 0xFFFFFFFF, so the card looks dead rather
# than unconfigured. The p150a could never show it: pci_enable_device() sets the bit and
# pci_disable_device() does not clear it, so even `rmmod` left 0x0102 there. It is the same
# gate shape as MRRS and it fails harder — MRRS is probe parity, this is whether the BARs
# answer at all.
#
# THE THING BEING PROVED, and why it needs a gate at all: on this bank MRRS already reads
# 4096, because tt-kmd's probe once set it and `pci_disable_device()` does not clear DEVCTL.
# So "MRRS is 4096 after our bring-up" is true whether or not our bring-up does anything —
# that is a vacuous gate, and it is the exact shape this project has been caught by eleven
# times. The gate therefore SEEDS A WRONG VALUE FIRST, which is the only way to simulate a
# machine where tt-kmd never ran, and it seeds a DIFFERENT wrong value on every run (from the
# pid and the clock) so a pass cannot be inherited from the previous run's state.
#
# NO RESET ANYWHERE IN THIS SCRIPT. The existing config-restore path already rewrites DEVCTL,
# so proving it on a reset path proves nothing about a cold box. This is a plain open.
#
# The arms:
#   GREEN  udrv open on a card seeded wrong  -> MRRS 4096 and memory decode ON, with the tool
#          saying "WRITTEN by us" for each. `already correct` is a pass that proves nothing.
#   DISCRIMINATOR  kmd open on a card seeded wrong -> MRRS UNCHANGED. This is what makes the
#          green mean something: it shows 4096 is not something that happens anyway. It is
#          SKIPPED, loudly, on a host with no /dev/tenstorrent — see below.
#   RESTORE  put 4096 and memory decode back whatever happened.
#
# The RED arm (a build with the MRRS call deleted) is a separate mutation build; this script
# is what both builds are run through.
#
# usage: sudo ./run_udrv_mrrs_cold_gate.sh <path-to-ft_tt_core> [pci-slot]

set -u

CORE_BINARY="${1:?usage: run_udrv_mrrs_cold_gate.sh <ft_tt_core> [slot]}"
# Default to whatever Blackhole this host actually has. It was 12:00.0 on the first
# host and is 03:00.0 on the current one; a hardcoded default silently gates the wrong device.
SLOT="${2:-$(lspci -Dn | awk '/1e52:b140/ {sub(/^0000:/, "", $1); print $1; exit}')}"
[ -n "$SLOT" ] || { echo "no Tenstorrent Blackhole (1e52:b140) on this host"; exit 2; }
FAILURES=0

# DEVCTL bits 14:12. Read as a whole word and mask, because setpci writes whole words.
read_devctl() { setpci -s "$SLOT" CAP_EXP+8.w; }
read_mrrs_code() { echo $(( (0x$(read_devctl) >> 12) & 7 )); }
mrrs_bytes() { echo $(( 128 << $1 )); }

write_mrrs_code() {
    local code="$1" devctl
    devctl=$(( 0x$(read_devctl) ))
    devctl=$(( (devctl & ~0x7000) | (code << 12) ))
    setpci -s "$SLOT" "CAP_EXP+8.w=$(printf '%04x' $devctl)"
}

check() {
    local name="$1" ok="$2" detail="$3"
    if [ "$ok" = "1" ]; then
        printf '  ok    %s  [%s]\n' "$name" "$detail"
    else
        printf '  FAIL  %s  [%s]\n' "$name" "$detail"
        FAILURES=$((FAILURES + 1))
    fi
}

# A wrong code that is NOT 5, seeded from pid and clock so two runs disagree and a pass
# cannot be a leftover.
SEED=$(( ($$ + $(date +%N)) % 5 ))
echo "=== MRRS cold-path gate on $SLOT; this run seeds code $SEED ($(mrrs_bytes $SEED) bytes) ==="

ORIGINAL_CODE=$(read_mrrs_code)
echo "as found: code $ORIGINAL_CODE ($(mrrs_bytes "$ORIGINAL_CODE") bytes), DEVCTL 0x$(read_devctl)"

read_command() { setpci -s "$SLOT" COMMAND; }
memory_decode_on() { echo $(( (0x$(read_command) >> 1) & 1 )); }
write_memory_decode() { setpci -s "$SLOT" "COMMAND=$1:0002"; }

restore() {
    write_mrrs_code 5
    write_memory_decode 0002
    echo "restored: code $(read_mrrs_code) ($(mrrs_bytes "$(read_mrrs_code)") bytes), \
PCI_COMMAND 0x$(read_command)"
}
trap restore EXIT

echo "as found: PCI_COMMAND 0x$(read_command) (memory decode $(memory_decode_on))"

# ---------------------------------------------------------------- GREEN
write_mrrs_code "$SEED"
write_memory_decode 0000
SEEDED=$(read_mrrs_code)
check "the MRRS seed lands — a card that looks like one tt-kmd never touched" \
      "$([ "$SEEDED" = "$SEED" ] && echo 1 || echo 0)" "code $SEEDED"
check "memory decode seeded OFF — the state a cold host actually powers on in" \
      "$([ "$(memory_decode_on)" = "0" ] && echo 1 || echo 0)" \
      "PCI_COMMAND 0x$(read_command)"

UDRV_LOG=$(FT_TT_DRIVER=udrv "$CORE_BINARY" status 0 --driver udrv 2>&1)
UDRV_RC=$?
AFTER_UDRV=$(read_mrrs_code)
check "the udrv open succeeded" "$([ $UDRV_RC -eq 0 ] && echo 1 || echo 0)" "rc=$UDRV_RC"
check "memory decode is ON after a udrv open with NO reset in the path" \
      "$([ "$(memory_decode_on)" = "1" ] && echo 1 || echo 0)" \
      "PCI_COMMAND 0x$(read_command)"
if echo "$UDRV_LOG" | grep -q 'memory decode PCI_COMMAND .* (WRITTEN by us)'; then
    check "the bring-up reports WRITING memory decode, not finding it enabled" 1 \
          "$(echo "$UDRV_LOG" | grep -o 'memory decode [^)]*)' | head -1)"
else
    check "the bring-up reports WRITING memory decode, not finding it enabled" 0 \
          "$(echo "$UDRV_LOG" | grep -i 'memory decode' | head -1)"
fi
check "MRRS is 4096 after a udrv open with NO reset in the path" \
      "$([ "$AFTER_UDRV" = "5" ] && echo 1 || echo 0)" \
      "code $AFTER_UDRV ($(mrrs_bytes "$AFTER_UDRV") bytes)"
# `already correct` would mean the card arrived at 4096 and nothing was written — a pass that
# proves nothing. The bring-up must say it WROTE it.
if echo "$UDRV_LOG" | grep -q 'MRRS .* -> 4096 bytes (WRITTEN by us)'; then
    check "the bring-up reports WRITING it, not finding it correct" 1 \
          "$(echo "$UDRV_LOG" | grep -o 'MRRS [^)]*)' | head -1)"
else
    check "the bring-up reports WRITING it, not finding it correct" 0 \
          "$(echo "$UDRV_LOG" | grep -i mrrs | head -1)"
fi

# ---------------------------------------------------- DISCRIMINATOR (kmd)
#
# The point of this arm is that 4096 is NOT something that happens anyway: the same command
# over tt-kmd leaves the seeded value alone. IT NEEDS tt-kmd TO EXIST. On the board this whole
# phase is aimed at there is no vendor module, and a kmd open there refuses before it reaches
# config space — so "MRRS was left wrong" becomes trivially true and the discriminator is
# VACUOUS. Say that out loud rather than counting a vacuous pass, and rather than counting the
# refusal as a failure of the mechanism under test.
if [ -e /dev/tenstorrent/0 ]; then
    write_mrrs_code "$SEED"
    KMD_LOG=$("$CORE_BINARY" status 0 2>&1)
    KMD_RC=$?
    AFTER_KMD=$(read_mrrs_code)
    check "the kmd open succeeded" "$([ $KMD_RC -eq 0 ] && echo 1 || echo 0)" "rc=$KMD_RC"
    check "the SAME command over tt-kmd leaves MRRS wrong — so 4096 is not automatic" \
          "$([ "$AFTER_KMD" = "$SEED" ] && echo 1 || echo 0)" \
          "code $AFTER_KMD ($(mrrs_bytes "$AFTER_KMD") bytes)"
else
    echo "  N/A   the kmd discriminator CANNOT BE TAKEN on this host — no /dev/tenstorrent."
    echo "        A kmd open refuses before touching config space, so 'MRRS stayed wrong'"
    echo "        would pass for the wrong reason. It was taken on the p150a bank, where the"
    echo "        module existed; nothing on this box can re-take it."
fi

echo "=== $FAILURES failed ==="
exit $(( FAILURES == 0 ? 0 : 1 ))
