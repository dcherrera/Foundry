#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Cross-build the K80 BAR1-strap kernel .deb via Docker (see Dockerfile).
# Output: ./out/linux-image-6.8.0-k80strap_*_amd64.deb
#
# Runs anywhere Docker runs, including Apple Silicon — the kernel is
# cross-compiled for x86-64, so the arm64 host builds it at native speed.
set -euo pipefail
cd "$(dirname "$0")"

IMG=k80-strap-kernel:6.8.0-106

# The kernel config is the target's own installed one, fetched at build time
# (it is Ubuntu's file, so it is not committed). Pass a local copy as $1, or set
# K80_HOST to the ssh destination (user@host) whose /boot holds it.
KCONFIG_NAME=config-6.8.0-106-generic
if [ -n "${1:-}" ]; then cp "$1" target.config
else ssh "${K80_HOST:?set K80_HOST=user@host (the K80 machine), or pass a kernel config as \$1}" \
        cat "/boot/$KCONFIG_NAME" > target.config; fi
grep -q '^CONFIG_X86_64=y' target.config || { echo "target.config is missing or not an x86-64 kernel config" >&2; exit 1; }

echo ">> building (cross-compile x86-64 kernel; first run pulls source, ~minutes)..."
docker build -t "$IMG" -f Dockerfile .

echo ">> extracting .deb..."
mkdir -p out
cid=$(docker create "$IMG")
docker cp "$cid:/out/." out/
docker rm "$cid" >/dev/null
ls -la out/

cat <<'EOF'

============================================================================
Built. Install on the K80 host — SAFE one-shot test (auto-reverts on any failure):

  scp out/linux-image-*.deb "$K80_HOST":~/
  ssh "$K80_HOST" 'bash -s' <<'REMOTE'
    set -e
    sudo dpkg -i ~/linux-image-6.8.0-k80strap_*_amd64.deb
    sudo update-grub
    # boot the new kernel exactly ONCE; a power-cycle returns to 6.8.0-106-generic
    entry=$(awk -F\' '/menuentry / && /6.8.0-k80strap/ {print $2; exit}' /boot/grub/grub.cfg)
    sub=$(awk -F\' '/submenu /{print $2; exit}' /boot/grub/grub.cfg)
    sudo grub-reboot "${sub}>${entry}"
    echo "one-shot set to: ${sub}>${entry}"
    sudo reboot
REMOTE

After it reboots: check `uname -r` == 6.8.0-k80strap and all 16 dies at 256 MB.
If it does NOT come back / ssh is lost: power-cycle -> boots known-good
6.8.0-106-generic automatically (one-shot only applied to the next boot).

To make it permanent once verified:
  ssh "$K80_HOST" 'sudo sed -i "s/^GRUB_DEFAULT=.*/GRUB_DEFAULT=\"${sub}>${entry}\"/" /etc/default/grub && sudo update-grub'
============================================================================
EOF
