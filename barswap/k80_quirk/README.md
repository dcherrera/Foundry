# K80 16/16 — map all 16 GK210 dies at boot on the BTC-S37

**What this does:** makes all **16 GK210 dies** of an 8× Tesla K80 machine (192 GB VRAM)
map into MMIO **simultaneously, at boot, automatically** — on a cheap BTC-S37 mining
board whose firmware + 36-bit CPU otherwise let only one 16 GB GPU BAR map at a time.

**How:** a single built-in Linux kernel quirk (`k80-strap-quirk.c`). No barswap, no
runtime tool, no VBIOS flash, no userspace. Drop the kernel in, reboot, done.

**Status:** ✅ **WORKING — 16/16 verified on the reference 8× K80 machine, card 8 included.**

---

## The problem

- Each K80 die advertises a **16 GB BAR1** (a CPU window into its VRAM). 16 dies ×
  16 GB = **256 GB** of MMIO.
- The BTC-S37's soldered Pentium 2117U (Ivy Bridge) has **36-bit physical addressing
  → a 64 GB ceiling.** 256 GB can't be placed. The kernel's PCI allocator fails to
  place the big BARs and **disables the prefetchable window of the bridge above the
  last card** (`Prefetchable memory behind bridge: [disabled]`).
- Once that bridge window is disabled, **that card's dies stop decoding entirely** —
  they read `0xffffffff` and can't be reached by any runtime tool. This stranded the
  8th card ("card 8": dies `20`/`21` on root port `00:1c.7`).

The fix is to make every BAR1 **256 MB** instead of 16 GB: 16 × 256 MB = 4 GB, fits
trivially. BAR1 size is a **strap** (`PSTRAPS`, BAR0 + `0x101000`, field `[17:14]`,
value 2 = 256 MB, override bit 31), not silicon — so it's settable. But the strap
must be applied *before the kernel sizes the bridge windows*, or card 8 is already lost.

## The solution — one kernel quirk, two parts

`k80-strap-quirk.c` is built into the kernel (`obj-y`) and does:

1. **`quirk_gk210_shrink_bar1`** — a `DECLARE_PCI_FIXUP_HEADER` that rewrites the
   kernel's *view* of each die's BAR1 to 256 MB **before resource assignment**
   (modeled on the in-tree `quirk_s3_64M`). The allocator never attempts the 16 GB
   placement → **never disables any bridge** → all 16 dies get a 256 MB BAR1 and every
   bridge stays alive, **card 8 included**.
   - *Must be HEADER, not EARLY:* `pci_fixup_early` runs before `pci_read_bases()`
     (BAR0/resources unpopulated — an EARLY version reports "BAR0 unassigned"); HEADER
     runs after sizing, before assignment.
2. **`k80_strap_silicon`** — a `late_initcall` that, after assignment (when BAR0 is
   reachable), writes the real `PSTRAPS` strap on each die so the **silicon** decode is
   256 MB too. Required because the two dies on a card share a bridge window — an
   unstrapped 16 GB die would alias over its sibling. Part 1 keeps the bridges alive so
   this reaches all 16.

## What was ruled out (so you don't repeat it)

- **barswap** time-multiplexing — the original approach; works but fragile and only
  ever one die in the slot. The strap shrink makes all 16 map at once → barswap unneeded.
- **VBIOS strap edit + nvflash** — flashes fine (Kepler is unsigned), but this board
  **does not run GPU option ROMs at POST**, so the VBIOS devinit never executes before
  BAR sizing. (See `../vbios_*` / the dump tooling.)
- **`pci_fixup_early` quirk** — too early; BAR0 unassigned at that point.
- **Runtime `setpci`/`k80_bar_resize`** — works for 14/16, but can't reach card 8
  post-boot (its bridge is already disabled). Kept as `tools/strap_all.py` (manual
  fallback / diagnostics).

## Build

Reproducible: pinned Ubuntu kernel source **6.8.0-106.106** (from Launchpad) + the
target machine's exact config + this quirk → kernel **6.8.12-k80strap** (6.8.12 = the
stable level Ubuntu's source rebases to).

**Cross-compile, anywhere with Docker** (incl. Apple Silicon — builds x86-64 at native
speed):

```sh
./build.sh /path/to/config-6.8.0-106-generic    # a local copy of the target's /boot config
K80_HOST=user@k80-host ./build.sh               # or fetch it from the target over ssh
# output: out/linux-image-*.deb
```

To build natively on an x86-64 Ubuntu box instead, run the steps in `Dockerfile`
(steps 1-5) directly; the result is the *same* quirk from the *same* pinned source.

## Install (the appliance path)

The kernel is **fully self-contained** — drop the `.deb` into an identical box
(BTC-S37 + K80 + Ubuntu 24.04, Secure Boot off) and reboot. The `.deb` is NOT in
the repo (it is a Linux kernel binary): build it first with either recipe above
and copy it into `prebuilt/` (gitignored).

```sh
scp prebuilt/linux-image-6.8.12-k80strap_*.deb <box>:~/
ssh <box> 'sudo dpkg -i ~/linux-image-6.8.12-k80strap_*.deb && sudo update-grub'
# make it the default boot (after first verifying with grub-reboot one-shot):
ssh <box> 'sudo sed -i "s/^GRUB_DEFAULT=.*/GRUB_DEFAULT=\"Advanced options for Ubuntu>Ubuntu, with Linux 6.8.12-k80strap\"/" /etc/default/grub && sudo update-grub'
sudo reboot
```

Prereqs: Secure Boot **off**; grub cmdline keeps
`pci=nommconf`; BIOS iGPU aperture 128 MB (frees 32-bit MMIO for the BAR0s);
**`nvidiafb` blacklisted** — see the gotcha below, this one is mandatory.

> **MANDATORY: blacklist `nvidiafb`.** The legacy in-kernel NVIDIA *framebuffer*
> driver (`nvidiafb`, nothing to do with `nvidia.ko`/compute) auto-binds to any
> NVIDIA display chip it finds. On the *stock* kernel the K80 BARs are unassigned so
> it bails harmlessly — but **once this quirk maps the dies, `nvidiafb` probes one as
> a CRT framebuffer, faults reading its clocks (`nvGetClocks`), and hard-hangs the
> box mid-boot** (network comes up, then a silent wedge — no oops reaches disk). The
> console runs on the Intel iGPU (`i915`), so `nvidiafb` has no legitimate role here.
> Ubuntu ships a `blacklist-framebuffer.conf` that covers it; make sure it's active
> and belt-and-suspenders the cmdline:
> ```sh
> ssh <box> 'sudo mv -n /etc/modprobe.d/blacklist-framebuffer.conf.OFF /etc/modprobe.d/blacklist-framebuffer.conf 2>/dev/null; \
>   grep -q modprobe.blacklist=nvidiafb /etc/default/grub || sudo sed -i "s/\(GRUB_CMDLINE_LINUX_DEFAULT=\"[^\"]*\)\"/\1 modprobe.blacklist=nvidiafb\"/" /etc/default/grub; \
>   sudo update-initramfs -u -k all && sudo update-grub'
> ```
> Symptom if you skip it: quirk kernel hangs at boot while the *stock* kernel boots
> fine, and `journalctl -b -1 -k | grep nvidiafb` shows `nvGetClocks` in a call trace.

**A clone box needs the `.deb` + the `nvidiafb` blacklist above.** Rebuild only for
*different* hardware (a NIC / storage controller not in this config) — the recipe
above handles that.

> **MANDATORY for CUDA compute: disable MSI (`NVreg_EnableMSI=0`).** Mapping the
> dies (this quirk) and NVML/`nvidia-smi` work fine, but the nvidia driver
> **`request_irq() fails (-22)`** for the PLX-switched x1 slots on this board — the
> IRQ routing can't do MSI. CUDA needs interrupts for kernel completion, so
> **`cuInit` returns 101** and every CUDA app silently falls back to CPU. Force
> legacy INTx and it works:
> ```sh
> echo "options nvidia NVreg_EnableMSI=0" | sudo tee /etc/modprobe.d/nvidia-msi.conf
> sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia; sudo modprobe --ignore-install nvidia
> ```
> Tell if you skip it: `dmesg | grep request_irq` shows `-22`, and a driver-API
> `cuInit(0)` returns `101` (`CUDA_ERROR_INVALID_DEVICE`).

> **MANDATORY for *sustained* CUDA: disable runtime D3 (`NVreg_DynamicPowerManagement=0`
> + persistence mode).** With MSI fixed, a brief CUDA run works — but under a
> multi-minute load the die goes idle between ops, drops to **D3cold**, and this
> board **can't wake it**: `Unable to change power state from D3cold to D0, Device
> inaccessible` → **Xid 79 "GPU has fallen off the bus"** → the job hangs. It looks
> thermal but isn't (temps stay ~60–70 °C). Keep the GPU in D0:
> ```sh
> echo "options nvidia NVreg_EnableMSI=0 NVreg_DynamicPowerManagement=0" | sudo tee /etc/modprobe.d/nvidia-msi.conf
> sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia; sudo modprobe --ignore-install nvidia
> sudo nvidia-smi -pm 1   # persistence mode, before running any CUDA
> ```
> Tell if you skip it: brief CUDA works, sustained decode dies with `dmesg | grep
> -i "fallen off\|D3cold"`. Recovery needs a reboot (the die is off the bus).

## Verify

```sh
sudo python3 tools/check_fixup.py     # 16/16 dies BAR1=256MB + card-8 bridge alive
sudo dmesg | grep k80_strap           # 16x "BAR1 ->256MB" + 16x "silicon BAR1 ->256MB"
sudo python3 tools/bridge_diag.py     # bridge-window state, working card vs card 8
```
Expected: `16/16 dies with a 256 MB BAR1 assigned`, card-8 bridge `pref=[0x3.._-0x3.._]`
(not `<DISABLED>`), and every die's PSTRAPS `field=2`.

## Files

```
k80-strap-quirk.c          the kernel quirk (HEADER fixup + late_initcall strap)  <- the whole solution
Dockerfile + build.sh      reproducible cross-compile (any Docker host); fetches the
                           target's /boot kernel config at build time (never committed)
prebuilt/*.deb             build output drop location (gitignored; build it, see Build)
tools/check_fixup.py       verify 16/16 + card-8 bridge
tools/bridge_diag.py       dump bridge windows (working card vs holdout)
tools/strap_all.py         manual silicon strap (the late_initcall does this now)
tools/strap_test.py        per-die PSTRAPS reachability probe
archive/                   the loadable-module attempt that did NOT work (history)
```

## Hardware (fixed — the point is the cheap hardware)

BTC-S37 mining board · Intel Pentium 2117U (Ivy Bridge, 36-bit) · 8× Tesla K80 =
16 GK210 dies, 192 GB · PCIe x1 links · AMI BIOS · Ubuntu 24.04, kernel 6.8.
