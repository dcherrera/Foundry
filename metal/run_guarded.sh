#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_guarded.sh — run a model-loading test without risking the machine.
#
# On 2026-08-08 an unguarded run of tests/test_qwen35moe_metal_forward against the
# 15 GB Q3_K_M GGUF kernel-panicked this Mac: 38 GB of page-ins on a 24 GB box whose
# disk was 96% full, so the compressor could not spill to swap, reclaim returned zero
# pages, watchdogd starved for 90 s and AppleARMWatchdogTimer fired. The whole system
# went down and took five agents' uncommitted work with it.
#
# The failure is not gradual. Once free RAM and swap headroom are both gone the box is
# already unrecoverable, so this script watches from OUTSIDE the process and kills the
# child while the machine can still recover.
#
# Usage:  metal/run_guarded.sh ./tests/test_qwen35moe_metal_forward [args...]
#
# Env:
#   GUARD_MIN_DISK_GB   refuse to start below this much free disk   (default 25)
#   GUARD_MIN_FREE_GB   refuse to start below free+inactive+purgeable RAM (default 12)
#   GUARD_KILL_FREE_MB  kill the child below this much reclaimable RAM (default 900)
#   GUARD_KILL_SWAP_MB  kill the child once swap USED exceeds this    (default 4096)
#   GUARD_POLL_SEC      sample interval                              (default 2)
#   GUARD_TIMEOUT_SEC   hard wall-clock cap on the child             (default 1800)
#
set -uo pipefail

MIN_DISK_GB=${GUARD_MIN_DISK_GB:-25}
MIN_FREE_GB=${GUARD_MIN_FREE_GB:-12}
KILL_FREE_MB=${GUARD_KILL_FREE_MB:-900}
KILL_SWAP_MB=${GUARD_KILL_SWAP_MB:-4096}
POLL_SEC=${GUARD_POLL_SEC:-2}
TIMEOUT_SEC=${GUARD_TIMEOUT_SEC:-1800}

if [ $# -lt 1 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

page_size=$(sysctl -n hw.pagesize)

# Reclaimable RAM in MiB: free + inactive + purgeable + speculative. Inactive pages are
# mostly clean file cache, which the kernel can drop for free; excluding them would make
# a healthy machine look starved.
reclaimable_mb() {
    vm_stat | awk -v ps="$page_size" '
        /Pages free/          { f = $3 }
        /Pages inactive/      { i = $3 }
        /Pages purgeable/     { p = $3 }
        /Pages speculative/   { s = $3 }
        END { gsub(/\./, "", f); gsub(/\./, "", i); gsub(/\./, "", p); gsub(/\./, "", s);
              printf "%d", (f + i + p + s) * ps / 1048576 }'
}

# "total = 0.00M  used = 1.50G  free = ..." -> MiB. The unit suffix must be read
# BEFORE stripping it; getting this backwards silently under-reports GiB by 1024x
# and the swap cap then never fires.
swap_used_mb() {
    sysctl -n vm.swapusage | awk '{
        for (i = 1; i <= NF; i++) if ($i == "used") {
            v = $(i + 2)
            unit = substr(v, length(v), 1)
            gsub(/[A-Za-z]/, "", v)
            if      (unit == "G") printf "%d", v * 1024
            else if (unit == "K") printf "%d", v / 1024
            else                  printf "%d", v
            exit
        }
    }'
}

free_disk_gb() {
    df -g /System/Volumes/Data | awk 'NR == 2 { print $4 }'
}

# ---------------------------------------------------------------- preflight
disk_gb=$(free_disk_gb)
free_mb=$(reclaimable_mb)
free_gb=$((free_mb / 1024))

echo "guard: preflight — ${free_gb} GiB reclaimable RAM, ${disk_gb} GiB free disk"

if [ "$disk_gb" -lt "$MIN_DISK_GB" ]; then
    cat >&2 <<EOF
guard: REFUSING — ${disk_gb} GiB free disk is below the ${MIN_DISK_GB} GiB minimum.

  Free disk IS swap headroom on macOS. With none, memory pressure stops being a slow
  machine and becomes a kernel panic: the compressor cannot spill, reclaim returns zero
  pages, and watchdogd starves. That is exactly the 2026-08-08 panic.

  Free space, or set GUARD_MIN_DISK_GB deliberately if you accept the risk.
EOF
    exit 1
fi

if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    cat >&2 <<EOF
guard: REFUSING — ${free_gb} GiB reclaimable RAM is below the ${MIN_FREE_GB} GiB minimum.

  A 15 GB model load needs its working set resident. Close browsers / other agents, or
  use a smaller GGUF via PN_QWEN35_MODEL.
EOF
    exit 1
fi

# ---------------------------------------------------------------- run
echo "guard: starting: $*"
"$@" &
child=$!

cleanup() { kill -TERM "$child" 2>/dev/null; }
trap cleanup INT TERM

killed=""
elapsed=0
peak_rss_mb=0

while kill -0 "$child" 2>/dev/null; do
    sleep "$POLL_SEC"
    elapsed=$((elapsed + POLL_SEC))

    free_mb=$(reclaimable_mb)
    swap_mb=$(swap_used_mb)
    swap_mb=${swap_mb:-0}
    rss_mb=$(ps -o rss= -p "$child" 2>/dev/null | awk '{ printf "%d", $1 / 1024 }')
    rss_mb=${rss_mb:-0}
    [ "$rss_mb" -gt "$peak_rss_mb" ] && peak_rss_mb=$rss_mb

    if [ "$free_mb" -lt "$KILL_FREE_MB" ]; then
        killed="reclaimable RAM fell to ${free_mb} MiB (floor ${KILL_FREE_MB} MiB)"
    elif [ "$swap_mb" -gt "$KILL_SWAP_MB" ]; then
        killed="swap in use hit ${swap_mb} MiB (cap ${KILL_SWAP_MB} MiB)"
    elif [ "$elapsed" -ge "$TIMEOUT_SEC" ]; then
        killed="wall-clock timeout after ${elapsed}s"
    fi

    if [ -n "$killed" ]; then
        echo "" >&2
        echo "guard: KILLING pid $child — $killed" >&2
        case "$killed" in
        wall-clock*) ;;
        *) echo "guard: the machine was heading for the 2026-08-08 panic; the test died instead." >&2 ;;
        esac
        kill -KILL "$child" 2>/dev/null
        wait "$child" 2>/dev/null
        echo "guard: peak child RSS ${peak_rss_mb} MiB over ${elapsed}s" >&2
        exit 137
    fi
done

wait "$child"
status=$?
echo "guard: child exited ${status}; peak RSS ${peak_rss_mb} MiB over ${elapsed}s"
exit "$status"
