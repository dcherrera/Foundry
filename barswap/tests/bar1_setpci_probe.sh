#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# bar1_setpci_probe.sh — Read GK210 BAR config via setpci while die is acquired
# Run as root on server with bar_swap loaded

set -e

BDF="03:00.0"  # Die 0

echo "=== BAR1 Size Probe via setpci ==="
echo ""

# Step 1: Acquire die 0 using the test binary (we'll hold it with a backgrounded process)
# Instead, use a simple C program that acquires, sleeps, then releases

cat > /tmp/hold_die.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdint.h>

/* Inline the ioctl definitions */
#define BAR_SWAP_MAGIC 'B'
struct bs_acquire_req {
    uint32_t die_id;
    uint32_t flags;
    uint64_t bar0_addr;
    uint64_t bar1_addr;
    uint64_t bar1_size;
    uint64_t bar3_addr;
};
struct bs_release_req {
    uint32_t die_id;
    uint32_t flags;
};
#define BAR_SWAP_ACQUIRE _IOWR(BAR_SWAP_MAGIC, 4, struct bs_acquire_req)
#define BAR_SWAP_RELEASE _IOW(BAR_SWAP_MAGIC, 5, struct bs_release_req)

int main(int argc, char **argv) {
    int fd = open("/dev/bar_swap", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    struct bs_acquire_req aq = { .die_id = 0, .flags = 0 };
    if (ioctl(fd, BAR_SWAP_ACQUIRE, &aq) < 0) { perror("acquire"); close(fd); return 1; }
    printf("ACQUIRED bar0=0x%lx bar1=0x%lx\n", (unsigned long)aq.bar0_addr, (unsigned long)aq.bar1_addr);
    fflush(stdout);

    /* Wait for signal */
    pause();

    struct bs_release_req rr = { .die_id = 0, .flags = 0 };
    ioctl(fd, BAR_SWAP_RELEASE, &rr);
    printf("RELEASED\n");
    close(fd);
    return 0;
}
CEOF

gcc -o /tmp/hold_die /tmp/hold_die.c
echo "Compiled hold_die"

# Acquire die 0 in background
/tmp/hold_die &
HOLD_PID=$!
sleep 1
echo "Die 0 acquired (PID $HOLD_PID)"
echo ""

# Step 2: Read current state via setpci
echo "--- Current Config Space ---"
echo "Vendor/Device: $(setpci -s $BDF 00.L)"
echo "CMD register:  $(setpci -s $BDF 04.W)"
echo ""
for i in 0 1 2 3 4 5; do
    OFF=$(printf "%02x" $((0x10 + i * 4)))
    echo "BAR$i (0x$OFF): $(setpci -s $BDF $OFF.L)"
done
echo ""

# Step 3: Disable memory decode
ORIG_CMD=$(setpci -s $BDF 04.W)
setpci -s $BDF 04.W=0000
echo "Decode disabled (CMD was 0x$ORIG_CMD, now 0x$(setpci -s $BDF 04.W))"
echo ""

# Step 4: Save original BAR1 values (BAR1 = offset 0x14, BAR2 = offset 0x18 for 64-bit)
# First figure out which BAR is the 64-bit prefetchable one
echo "--- Probing BAR sizes ---"
echo "Writing 0xFFFFFFFF to each BAR, reading back size mask..."
echo ""

for i in 0 1 2 3 4 5; do
    OFF=$(printf "%02x" $((0x10 + i * 4)))
    ORIG=$(setpci -s $BDF $OFF.L)

    # Write all-ones
    setpci -s $BDF $OFF.L=FFFFFFFF

    # Read back mask
    MASK=$(setpci -s $BDF $OFF.L)

    # Restore
    setpci -s $BDF $OFF.L=$ORIG

    echo "BAR$i: orig=0x$ORIG mask=0x$MASK"
done
echo ""

# Step 5: Restore CMD
setpci -s $BDF 04.W=$ORIG_CMD
echo "Decode restored (CMD=0x$(setpci -s $BDF 04.W))"

# Step 6: Release die
kill $HOLD_PID 2>/dev/null
wait $HOLD_PID 2>/dev/null
echo ""
echo "Die released."
echo ""
echo "=== Done ==="
