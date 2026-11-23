#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# nic_relocate.sh — Move NIC BAR4 from 0x61F400000 to 0x21F600000
#
# Frees 0x400000000-0x7FFFFFFFF for a second GPU BAR1 slot.
#
# SAFETY: Starts a background revert timer. If networking breaks and we
# lose SSH, the timer restores original values after 60 seconds.
#
# Run as root on the K80 server.

set -e

NIC="09:00.0"
BRIDGE="00:1c.2"

# New location for NIC BAR4: 0x21F600000 (below 0x400, inside PCI space)
# Bridge pref window: 0x21F600000 - 0x21F6FFFFF (1 MB)
NEW_BAR4_LO="1F60000C"   # 0x1F600000 | 0xC (memory, 64-bit, pref)
NEW_BAR4_HI="00000002"   # upper 32 bits: 0x2 -> full addr = 0x21F600000
NEW_PREF_BASE="1F61"     # bits [15:4] = 0x1F6, bit 0 = 1 (64-bit)
NEW_PREF_LIMIT="1F61"    # same 1 MB block
NEW_PREF_BASE_UPPER="00000002"
NEW_PREF_LIMIT_UPPER="00000002"

# Save original values
echo "=== Saving original config ==="
ORIG_BAR4_LO=$(setpci -s $NIC 20.L)
ORIG_BAR4_HI=$(setpci -s $NIC 24.L)
ORIG_PREF_BASE=$(setpci -s $BRIDGE 24.W)
ORIG_PREF_LIMIT=$(setpci -s $BRIDGE 26.W)
ORIG_PREF_BASE_UPPER=$(setpci -s $BRIDGE 28.L)
ORIG_PREF_LIMIT_UPPER=$(setpci -s $BRIDGE 2C.L)

echo "  NIC BAR4: $ORIG_BAR4_LO $ORIG_BAR4_HI"
echo "  Bridge pref: base=$ORIG_PREF_BASE upper=$ORIG_PREF_BASE_UPPER limit=$ORIG_PREF_LIMIT upper=$ORIG_PREF_LIMIT_UPPER"

# Write revert script
cat > /tmp/nic_revert.sh << EOF
#!/bin/bash
echo "REVERTING NIC config..."
echo $NIC > /sys/bus/pci/drivers/r8169/unbind 2>/dev/null || true
sleep 1
setpci -s $BRIDGE 24.W=$ORIG_PREF_BASE
setpci -s $BRIDGE 26.W=$ORIG_PREF_LIMIT
setpci -s $BRIDGE 28.L=$ORIG_PREF_BASE_UPPER
setpci -s $BRIDGE 2C.L=$ORIG_PREF_LIMIT_UPPER
setpci -s $NIC 20.L=$ORIG_BAR4_LO
setpci -s $NIC 24.L=$ORIG_BAR4_HI
echo $NIC > /sys/bus/pci/drivers/r8169/bind 2>/dev/null || true
echo "Reverted."
EOF
chmod +x /tmp/nic_revert.sh

# Start safety revert timer (60 seconds)
echo ""
echo "=== Starting 60s safety revert timer ==="
(sleep 60 && /tmp/nic_revert.sh) &
REVERT_PID=$!
echo "  Revert PID: $REVERT_PID (will auto-revert in 60s)"

# Step 1: Unbind r8169
echo ""
echo "=== Step 1: Unbind r8169 ==="
echo $NIC > /sys/bus/pci/drivers/r8169/unbind 2>/dev/null || true
sleep 1
echo "  r8169 unbound"

# Step 2: Disable NIC decode
echo "=== Step 2: Disable NIC decode ==="
ORIG_CMD=$(setpci -s $NIC 04.W)
setpci -s $NIC 04.W=0000
echo "  NIC CMD: $ORIG_CMD -> 0000"

# Step 3: Reprogram bridge prefetchable window
echo "=== Step 3: Reprogram bridge pref window ==="
echo "  Old: base=${ORIG_PREF_BASE}:${ORIG_PREF_BASE_UPPER} limit=${ORIG_PREF_LIMIT}:${ORIG_PREF_LIMIT_UPPER}"

setpci -s $BRIDGE 24.W=$NEW_PREF_BASE
setpci -s $BRIDGE 26.W=$NEW_PREF_LIMIT
setpci -s $BRIDGE 28.L=$NEW_PREF_BASE_UPPER
setpci -s $BRIDGE 2C.L=$NEW_PREF_LIMIT_UPPER

echo "  New: base=${NEW_PREF_BASE}:${NEW_PREF_BASE_UPPER} limit=${NEW_PREF_LIMIT}:${NEW_PREF_LIMIT_UPPER}"
echo "  Window: 0x21F600000 - 0x21F6FFFFF"

# Step 4: Reprogram NIC BAR4
echo "=== Step 4: Reprogram NIC BAR4 ==="
setpci -s $NIC 20.L=$NEW_BAR4_LO
setpci -s $NIC 24.L=$NEW_BAR4_HI
echo "  BAR4: 0x${NEW_BAR4_HI}${NEW_BAR4_LO} -> address 0x21F600000"

# Step 5: Re-enable NIC decode
echo "=== Step 5: Re-enable NIC ==="
setpci -s $NIC 04.W=$ORIG_CMD
echo "  NIC CMD restored to $ORIG_CMD"

# Step 6: Rebind r8169
echo "=== Step 6: Rebind r8169 ==="
echo $NIC > /sys/bus/pci/drivers/r8169/bind 2>/dev/null || true
sleep 2
echo "  r8169 rebound"

# Step 7: Test network
echo ""
echo "=== Step 7: Network test ==="
if ping -c 3 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "  PING OK — network works!"
    echo ""
    echo "  Cancelling revert timer..."
    kill $REVERT_PID 2>/dev/null
    wait $REVERT_PID 2>/dev/null || true
    echo ""
    echo "=== SUCCESS ==="
    echo "NIC BAR4 relocated from 0x61F400000 to 0x21F600000"
    echo "0x400000000 - 0x7FFFFFFFF is now FREE for GPU BAR1"
    echo ""
    echo "Verify:"
    echo "  NIC BAR4: $(setpci -s $NIC 20.L) $(setpci -s $NIC 24.L)"
    echo "  Bridge pref: $(setpci -s $BRIDGE 24.W):$(setpci -s $BRIDGE 28.L) - $(setpci -s $BRIDGE 26.W):$(setpci -s $BRIDGE 2C.L)"

    # Check iomem
    echo ""
    echo "iomem check (NIC should be at new address):"
    cat /proc/iomem | grep -E "(61f|21f)" || echo "  (no matches)"
else
    echo "  PING FAILED — network broken!"
    echo "  Reverting immediately..."
    kill $REVERT_PID 2>/dev/null
    wait $REVERT_PID 2>/dev/null || true
    /tmp/nic_revert.sh
    echo ""
    echo "=== FAILED — reverted to original config ==="
fi
