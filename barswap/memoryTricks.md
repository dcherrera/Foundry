# Memory & Topology Tricks — 8× K80 rig (no NVLink)

The 8× K80 rig has **no NVLink**. A model split across dies is bottlenecked not by memory *capacity* (12 GB/die is plenty) but by inter-die **bandwidth and latency**. You have exactly **one fast link** and **one slow link** — every trick here is some version of *"stay on the fast link, dodge the slow one."*

## The asymmetry that drives everything

| Path | Bandwidth | Notes |
|------|-----------|-------|
| Same-card die↔die (PLX PEX 8747 switch) | **77.88 GB/s** | the one fast link |
| Cross-card (x1 riser, host-bounced) | **~0.4 GB/s** | the slow link |
| Per-die VRAM weight read | ~240 GB/s | the compute floor (unavoidable) |

Same-card is **~190× faster** than cross-card. That ratio is the whole game.

> A die dedicated purely as "extra memory" does **not** help — you'd reach it over the same slow link. The bottleneck is bandwidth/latency, not capacity. The win is *placement*, not a memory tier.

---

## Trick 1 — Co-locate a model on one card's two dies *(biggest win, free)*

Each K80 pairs 2 dies behind the fast PLX switch. A model split *within* a card moves activations at 77.88 GB/s and **never touches the 0.4 GB/s cross-card path**.

- The mini at **Q3_K_M (16 GB) fits on a single card's 2 dies (24 GB)** with ~8 GB left for KV → zero slow hops, bound only by the 240 GB/s per-die weight read.
- Deployment consequence: **8 single-card model instances (one per card)** is faster *and* simpler than fewer-bigger models that span cards. The 8-card topology is begging for it.

## Trick 2 — P2P DMA over the PLX *(don't host-stage same-card copies)*

The 77.88 GB/s figure **is** the PLX peer-to-peer path. For same-card die pairs, use `cuMemcpyPeer` / GPUDirect P2P so activations go die→die **through the switch, never through host RAM**. Host-staging a same-card copy needlessly drops you toward the slow path. (Cross-card P2P does **not** work on this board — those must bounce through host.)

## Trick 3 — Sibling-die VRAM as a fast spill tier *(the "memory die" idea, done right)*

A same-card sibling die's VRAM is reachable at **77.88 GB/s — ~190× faster than host RAM (0.4 GB/s)**. So if a model on die A needs to spill beyond 12 GB, **spilling to its same-card sibling die B beats spilling to host**. This is the legit version of "use a die as specialized memory" — a fast scratch/overflow tier.

- **Only holds for same-card siblings.** Cross-card, host RAM and a donor die are both gated by the slow link, so it stops helping.

## Trick 4 — Overlap compute and comms *(for the unavoidable cross-card case)*

When a model *must* span cards (e.g. a 4-die model for 256K context = one cross-card boundary), hide the slow hop: while die N computes its layers, **prefetch the prior activation to die N+1** so the 0.4 GB/s transfer happens *under* compute, not in series. Classic pipeline parallelism (estimated ~0.8 ms overhead/token). Activations are tiny (KB/token), so the cost is mostly latency + BAR-swap, which overlap hides well — especially during **prefill**.

## Trick 5 — Pin the host relay buffer *(small, free)*

For cross-card hops that *do* bounce through host, use **page-locked (pinned) host memory** for the staging buffer. DMA to/from pinned memory is meaningfully faster than pageable. Free win on the one path you can't avoid.

---

## The meta-trick: don't create a cross-card boundary you don't need

The best optimization is topology-aware placement:

1. **Default: co-locate a model on one card (2 dies, 24 GB).** No cross-card boundary exists, so there's nothing to optimize — you're back to the clean 240 GB/s-per-die story.
2. **Span cards only when context genuinely demands it** (e.g. 4-die for 256K).
3. **When spanning, minimize boundaries** — place the layer split so just one layer-junction crosses cards.
4. **barswap should be card-aware** — it must know which dies share a card and prefer same-card layouts.

The placement *is* the performance work. For the mini, the answer to "how do we go fast across cards" is **"don't span cards at all."**

---

## What does NOT help (honest)

- **A die as a pure memory/cache tier** — bottleneck is bandwidth/latency, not capacity; you'd reach it over the same slow link (unless it's a same-card sibling → Trick 3).
- **Quantizing activations for transfer** — they're already tiny (KB/token); the cross-card cost is latency-bound, not size-bound.
- **Nothing here beats real NVLink** — these are mitigations that exploit the one fast link (on-card PLX) and hide the slow one. The per-die 240 GB/s weight read remains the floor regardless.

---

## Related

- `CARDS.md` — card compatibility (BAR size = whether barswap is needed at all).
- `MULTI_SLOT.md` — the 3-slot ceiling from 36-bit addressing.
- `README.md` — barswap design and usage.
