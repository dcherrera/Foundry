# Compatible GPU Cards — BTC-S37 / 8× K80 rig

## Rig Constraints

- **Board:** BTC-S37 (8 PCIe slots, x1/x4 electrical)
- **CPU:** Intel Pentium 2117U (Ivy Bridge, 36-bit physical address = 64 GB max)
- **RAM:** 8 GB DDR3
- **GPUs installed:** 8x K80 (16 dies, 192 GB VRAM) — all 16 dies tested and passing
- **PSU:** 3x 750W (2,250W total)
- **MMIO budget:** ~56 GB (64 GB address space minus 8 GB RAM)
- **Working BAR1 slot:** 0xC00000000 (16 GB, the only slot that routes correctly)
- **Driver:** nvidia 470 (supports Kepler through Turing)
- **CUDA:** 11.4 (sm_35 through sm_86)

## Card Compatibility Table

| Card | Arch | sm | VRAM | Mem BW | FP16 TFLOPS | INT8 TOPS | BAR1 | TDP | Power Conn | Slots | Bar Swap? | Max on Board | Status |
|------|------|----|------|--------|-------------|-----------|------|-----|------------|-------|-----------|--------------|--------|
| **K80** | Kepler | 37 | 2x12 GB | 240 GB/s/die | 0.37/die | — | 16 GB | 300W | 1x8-pin | 2 (dual) | Required | 8 (16 dies) | **16/16 passing** |
| K40 | Kepler | 35 | 12 GB | 288 GB/s | 0.43 | — | 16 GB | 235W | 1x8-pin | 2 (full-length) | Required | 3-4 | Compatible |
| K20X | Kepler | 35 | 6 GB | 250 GB/s | 0.39 | — | 8 GB | 235W | 1x8-pin | 2 (full-length) | Required | 3-4 | Compatible |
| M40 24GB | Maxwell | 52 | 24 GB | 288 GB/s | 0.21 | — | 16 GB | 250W | 1x8-pin | 2 (full-length) | Required | 3-4 | Likely works |
| M60 | Maxwell | 52 | 2x8 GB | 160 GB/s/die | 0.15/die | — | 16 GB | 300W | 1x8-pin | 2 (dual-die) | Required | 3 (6 dies) | Likely works |
| **P100 PCIe 16GB** | Pascal | 60 | 16 GB HBM2 | 732 GB/s | 21.2 | — | 16 GB | 250W | 1x8-pin | 2 (full-length) | Required | 3-4 | Likely works |
| P4 | Pascal | 61 | 8 GB | 192 GB/s | 5.5 | 22 | 256 MB | 75W | None | 1 (low-profile) | Not needed | **8** | Likely works |
| **V100 PCIe 16GB** | Volta | 70 | 16 GB HBM2 | 900 GB/s | 28.3 | — | 16 GB | 250W | 1x8+6-pin | 2 (full-length) | Required | 3-4 | Likely works |
| **T4** | Turing | 75 | 16 GB GDDR6 | 320 GB/s | 65 | 130 | 256 MB | 75W | None | 1 (low-profile) | Not needed | **8** | Likely works |

**INT8 TOPS note:** Cards marked `—` have no hardware INT8 dot product. Kepler and Maxwell lack INT8 math entirely. Pascal sm_60 (P100) has FP16 Tensor-like throughput but no INT8 path. Volta Tensor Cores only accelerate FP16→FP32 mixed precision, not INT8. The first INT8 dot product arrived with Pascal sm_61 (P4) via the `dp4a` instruction, and Turing (T4) added INT8 Tensor Core mode for a massive throughput jump. On cards without INT8 hardware, INT8 data must be upcast to FP32 for computation — no speed benefit over FP32 math.

### Cards That Will NOT Fit

| Card | Reason |
|------|--------|
| V100 32GB | BAR1 = 32 GB, exceeds single working 16 GB slot |
| P40 24GB | BAR1 = 32 GB |
| A30 24GB | BAR1 = 32 GB, also needs driver 495+ |
| A100 40/80GB | BAR1 = 64/128 GB, needs driver 495+ |
| H100 | BAR1 = 128 GB, needs driver 525+ |
| Any card with BAR1 > 16 GB | Only one 16 GB MMIO slot available at 0xC00000000 |

## Notes

### Bar Swap vs No Bar Swap

Cards with BAR1 <= 16 GB need bar_swap because only one 16 GB MMIO slot works on this board. All such cards share the slot via time-multiplexing. Only one die is mapped at a time.

Cards with BAR1 = 256 MB (P4, T4) have a small enough MMIO footprint that all 8 fit simultaneously in the address space (8 x 256 MB = 2 GB, well within the 56 GB MMIO budget). The nvidia driver manages VRAM internally through page tables — the 256 MB window is enough for the driver to function. Bar_swap is not needed because all cards are visible to nvidia at once. Standard multi-GPU CUDA works out of the box.

### Current Rig: 8x K80 (16 Dies, 192 GB VRAM)

The board currently has 8 K80 cards installed. 3 cards (6 dies) are tested and working with bar_swap. The remaining 5 cards (10 dies) are installed but untested.

All 16 dies tested and passing (2026-03-25):
- 192 GB total VRAM
- All 16 dies time-multiplexed through the single 0xC00000000 BAR1 slot
- Enough VRAM to hold a 70B+ model in FP16 or a 140B+ model in INT8
- ~1.2ms acquire time per die, ~2.5s full cycle (with nvidia load/unload)
- Power: 8x 300W = 2,400W (needs 30A breaker for sustained load)

### T4 — Best Upgrade Option

8x T4 is the most interesting configuration for this board:
- 128 GB total VRAM (vs 192 GB with 8x K80)
- 520 TFLOPS FP16 / 1,040 TOPS INT8 (vs ~17 TFLOPS FP16 with K80s)
- 600W total power, no power cables (vs 2,400W + cables)
- Single-slot passive cooling — designed for dense server deployment
- All 8 visible to nvidia simultaneously, standard multi-GPU CUDA
- Used price: ~$150-200 each (~$1,200-1,600 for 8)

### P4 — Budget Small-BAR Option

8x P4 is the budget version of the same idea:
- 64 GB total VRAM (8x 8 GB)
- Pascal sm_61 (same as GTX 1080)
- 75W, no power cables, single-slot, passive
- Cheaper than T4 (~$50-80 used)
- Less compute than T4 but still faster than K80

### P100 / V100 16GB — Maximum Single-Card Performance

These are the fastest cards that fit the 16 GB BAR1 slot:
- P100: ~21 TFLOPS FP16, 16 GB HBM2 (732 GB/s bandwidth)
- V100 16GB: ~125 TFLOPS FP16, 16 GB HBM2 (900 GB/s bandwidth)
- Both require bar_swap (16 GB BAR1)
- A single V100-16GB through bar_swap outperforms all 16 K80 dies in compute

### PCIe Bandwidth

All BTC-S37 slots are x1 or x4 electrical:
- PCIe 2.0 x1: 500 MB/s
- PCIe 3.0 x1: 1 GB/s
- PCIe 3.0 x4: 4 GB/s

This limits host-to-device transfers (model weight uploads) but does not affect on-device compute. With VRAM persistence (bar_swap) or standard multi-GPU (T4/P4), weights are uploaded once and stay resident.

### Driver Compatibility

| Driver | Architectures Supported |
|--------|------------------------|
| 470 (installed) | Kepler, Maxwell, Pascal, Volta, Turing |
| 495+ | Adds Ampere (sm_80/86) |
| 525+ | Adds Hopper (sm_90) |

Driver 470 covers all cards in the compatibility table. No driver change needed.

### Dual-Die Cards

K80 and M60 are dual-die cards (2 GPUs on one PCB) connected via an on-card PLX PEX 8747 switch. The BTC-S37 already handles this topology — the PLX downstream ports appear as separate PCI devices. Bar_swap treats each die independently.

### What Has NOT Been Tested

- All 8 K80 cards (16 dies) tested and passing with bar_swap as of 2026-03-25
- No card other than K80 has been tested on this board
- BAR1 sizes for non-K80 cards are based on published specs and typical configurations
- T4/P4 small-BAR behavior assumes the card does not request resizable BAR expansion on a board that doesn't support it (expected behavior, but unverified)
- BIOS compatibility with non-Kepler cards is assumed but not confirmed
- 8x K80 power draw (2,400W) may exceed PSU capacity — verify PSU wiring before testing all 8 cards under load
