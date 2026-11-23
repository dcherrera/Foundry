/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file matvec_nx_staging.cuh
 * @brief The ONE activation-staging loop shared by every batched (_nx*) K-quant
 *   matvec, in two shapes: the shipped FLAT form (a single flat index divided by
 *   the runtime `cend`) and the NESTED form that removes the division entirely.
 *   Selected by FC_NX_STAGE_NESTED, default 0 = the shipped flat form.
 *
 * WHY THIS FILE EXISTS. `fc_matvec_*_nx_body` stages NC * cend activation floats
 * per column-chunk. The shipped loop recovers the column from a flat index:
 *
 *     for (int i = threadIdx.x; i < NC * cend; i += 256) {
 *         int c = i / cend, e = i - c * cend;         // cend is a RUNTIME value
 *         sh_x[c * FC_NX_CHUNK + e] = (c < nc) ? __ldg(...) : 0.0f;
 *     }
 *
 * `cend = min(cols - cbase, FC_NX_CHUNK)` is a kernel argument, so nvcc cannot
 * fold the divide: ptxas emits the full signed-division expansion per staged
 * element — I2F.F32.S32.RP / MUFU.RCP / F2I.FTZ.U32.F32.TRUNC to build the
 * reciprocal, then IMUL.U32.U32.HI + IMAD for the quotient/remainder and three
 * ISETP + two conditional IADD32I of signed fixups. COUNTED in SASS on sm_37 /
 * CUDA 11.4: D = 14.25
 * instructions per staged element, 16 staged elements per thread per chunk at
 * the shipped prefill shape (NC=8, cend=512, 256 threads), i.e. 228 executed
 * instructions per thread per chunk against a 710-instruction dot.
 *
 * The nested form never forms a quotient: the column is the outer loop, so it is
 * loop-invariant, and `e` walks the chunk with a stride-256 copy. Whole-kernel
 * executed cut 1.42-1.49x on fc_matvec_q3k_nx8, and it drops registers 95 -> 80,
 * which BUYS a resident block (5 -> 6 blocks/SMX) instead of costing one.
 *
 * BIT-EXACT BY CONSTRUCTION, and this is the whole argument:
 *   - the two loops write the SAME SET of shared addresses. Flat covers
 *     i in [0, NC*cend) <-> (c, e) = (i / cend, i mod cend), a bijection onto
 *     [0, NC) x [0, cend); nested iterates that product set directly.
 *   - each address is written exactly ONCE by exactly ONE thread in both forms,
 *     so the change of *order* is unobservable.
 *   - the value written is the same expression: __ldg of the same address, or
 *     the same 0.0f for a column past nc.
 *   - both sit between the same two __syncthreads() of the caller, which are
 *     NOT part of this macro and are not moved.
 *   - ZERO floating-point operations exist in either form. There is no
 *     arithmetic to reassociate and no contraction to break — nothing is
 *     hoisted out of a loop, and no value moves from shared into a register
 *     where ptxas could contract it differently (the two traps this project has
 *     already hit: the attention online-softmax rescale, and the SSM kernel's
 *     need for __fmul_rn/__fadd_rn as contraction barriers).
 *
 * NO SHAPE GUARD IS NEEDED. The nested form is valid for every (NC, cend, nc,
 * cols) the flat form is valid for — it makes no divisibility assumption, unlike
 * the `i >> 9` shift variant, which is only correct when cend == FC_NX_CHUNK.
 * The shared-memory addressing (stride FC_NX_CHUNK per column) is byte-for-byte
 * the shipped one, so the two forms share the same domain of validity exactly,
 * including its one pre-existing limitation: the launcher allocates
 * NC * min(cols, FC_NX_CHUNK) floats while the kernels address column c at
 * c * FC_NX_CHUNK, so cols < FC_NX_CHUNK is out of bounds in BOTH forms. That is
 * not introduced here and is not fixed here; every shape Pantheon launches has
 * cols in {512, 2048, 4096}.
 *
 * XF is a one-argument transform macro applied to each loaded value. It exists
 * so the microbenchmark's poison arm can be the SHIPPED staging code plus
 * exactly one perturbation, rather than a hand-copied lookalike that could drift
 * from what ships. Production always passes FC_NX_STAGE_KEEP, which is the
 * identity.
 *
 * @status FC_NX_STAGE_NESTED defaults to 0 — the shipped flat form. With the
 *   flag at 0 the generated PTX is byte-identical to the checked-in
 *   cuda/fc_cuda_ptx.h; that identity is the gate on this refactor.
 * @issues None
 * @todo None
 */

#ifndef FC_MATVEC_NX_STAGING_CUH
#define FC_MATVEC_NX_STAGING_CUH

/* Activation floats staged per column per chunk. Also the COMPILE-TIME shared
 * stride between columns, which is why it is not the same thing as `cend`. */
#define FC_NX_CHUNK 512

/*
 * WHICH STAGING FORM THE KERNELS GET. Three values, ordered by MEASURED speed on
 * `fc_matvec_q3k_nx8` at the prefill shape (die 6, pinned 666 MHz, noise floor
 * 0.19 %, bit-EXACT on 1,048,576 outputs in every case):
 *
 *   0  FLAT     the shipped loop, one runtime-divisor division per element  1.000x
 *   1  NESTED   column outer, element inner — fewest registers (80)         1.151x
 *   2  EOUTER   element outer, all NC columns unrolled inside               1.426x
 *
 * ★ COUNTED INSTRUCTIONS PICK THE WRONG WINNER HERE, which is why this comment is
 * long. NESTED executes FEWER instructions than the `i >> 9` shift variant
 * (968 vs 1032 per thread per chunk) and has BETTER occupancy (6 vs 5 blocks/SMX),
 * and it still MEASURED SLOWER (1.151x vs 1.263x). This loop is LATENCY-bound, and
 * what tracks wall clock is the max `TEXDEPBAR` argument in the staging region —
 * the number of global loads outstanding before the first wait:
 *
 *   FLAT 4 in flight · shift 4 · NESTED 1 · EOUTER 8
 *
 * NESTED's rolled element loop issues one load and waits on it before storing.
 * EOUTER puts the runtime-bounded element loop OUTSIDE and the compile-time column
 * loop inside, so one iteration issues NC independent loads and then NC stores:
 * fewest instructions AND the most memory-level parallelism. It costs 96 registers
 * against NESTED's 80 — it gives the 6th resident block back and sits at the flat
 * form's own 5 blocks/SMX — and it still wins by 1.24x over NESTED. That is how
 * firmly latency beats occupancy here.
 *
 * Compile-time because the kernels are pre-compiled into cuda/fc_cuda_ptx.h: a
 * runtime switch would mean a second copy of all fifteen batched matvecs in the
 * embedded PTX, paying module size and JIT time on every model load for a flag
 * meant to end up permanently on. Build a non-default arm with
 *   cd cuda_kernels && make NVCCFLAGS_EXTRA=-DFC_NX_STAGE_NESTED=2
 * and diff the regenerated header.
 */
#define FC_NX_STAGE_FLAT_FORM   0
#define FC_NX_STAGE_NESTED_FORM 1
#define FC_NX_STAGE_EOUTER_FORM 2

/* ★ SHIPPED DEFAULT since 2026-08-10: EOUTER. Gated on one K80 board's two dies at a pinned
 * 666 MHz — kernel 1.426x, in-model 1.0929x/1.0839x at 8,192/14,080 tokens
 * (70.3 -> 76.2 tok/s), Amdahl closing to 0.027 %, full-logit FNV-1a identical on
 * 512 positions cross-build while a 1-ulp poisoned sibling differs on 512/512.
 * results/prefill_nx_staging_port.md §0. Set to 0 to restore the flat loop. */
#ifndef FC_NX_STAGE_NESTED
#define FC_NX_STAGE_NESTED FC_NX_STAGE_EOUTER_FORM
#endif

/*
 * FALSIFICATION ARM ONLY — never define this outside a gate build.
 * -DFC_NX_STAGE_POISON=1 scales every staged activation by (1 + 2^-23), one ulp,
 * so the model's logits land ~1.2e-07 relative from correct: inside every
 * tolerance in this tree, and large enough to move the last bit of essentially
 * every output. It exists so the IN-MODEL digest gate can be shown to fail, the
 * same way `bench_matvec_nx_staging --poison` shows the kernel gate can. A digest
 * comparison that cannot catch this is not comparing bits.
 * Calibrated host-side in tests/calib_nx_stage_poison.c: 84.85 % of a matvec's
 * output floats change bit pattern. Do NOT weaken it to make something pass.
 */
#ifndef FC_NX_STAGE_POISON
#define FC_NX_STAGE_POISON 0
#endif

/* Identity transform — what production stages. */
#if FC_NX_STAGE_POISON
#define FC_NX_STAGE_KEEP(v) __fmul_rn((v), 1.00000011920928955078125f)
#else
#define FC_NX_STAGE_KEEP(v) (v)
#endif

/*
 * The nested form's inner loop MUST NOT be unrolled, and this is a correctness-
 * of-cost argument, not a taste one. `cend <= FC_NX_CHUNK` (512) and the block is
 * always 256 threads, so `for (e = threadIdx.x; e < cend; e += 256)` trips at
 * most TWICE — for every shape any of these kernels can ever be launched with.
 * Left to itself nvcc emits a `trip mod 4` peel plus x16, x8 and x4 unrolled
 * bodies; COUNTED in SASS, none of those bodies ever executes, the staging runs
 * entirely out of the peel, and the ladder's guards cost 459 executed
 * instructions per thread per chunk against 264 with the ladder suppressed.
 * (This is exactly why the audit's first reading of this arm was wrong: it priced
 * the x16 body as "executed once" when the body is unreachable.)
 *
 * Guarded on __CUDACC__ so tests/test_nx_staging_equivalence.c can compile this
 * header with a host compiler that has no such pragma.
 */
#ifdef __CUDACC__
#define FC_NX_STAGE_ROLLED _Pragma("unroll 1")
#else
#define FC_NX_STAGE_ROLLED
#endif

/*
 * SHIPPED FLAT FORM. Kept verbatim (including the `c`/`e` names and the
 * conditional-expression shape) so that with FC_NX_STAGE_NESTED == 0 the
 * regenerated PTX is byte-identical to the checked-in artifact.
 */
#define FC_NX_STAGE_X_FLAT(SH_X, X, COL0, COLS, CBASE, CEND, NC, NC_LIVE, XF)  \
    do {                                                                       \
        for (int i = threadIdx.x; i < (NC) * (CEND); i += 256) {                \
            int c = i / (CEND), e = i - c * (CEND);                             \
            (SH_X)[c * FC_NX_CHUNK + e] = (c < (NC_LIVE))                       \
                ? XF(__ldg((X) + (size_t)((COL0) + c) * (COLS) + (CBASE) + e))  \
                : 0.0f;                                                         \
        }                                                                      \
    } while (0)

/*
 * NESTED FORM — the division-free rewrite. The column is the outer loop, so no
 * quotient is ever formed and the `c < nc` test is hoisted out of the copy (it
 * is block-uniform, one predicate per column instead of one per element).
 *
 * The COLUMN loop is deliberately left for the compiler to unroll (it is a
 * compile-time NC and nvcc does unroll it), while the ELEMENT loop is pinned
 * rolled by FC_NX_STAGE_ROLLED — see the argument above. Counted on
 * fc_matvec_q3k_nx8 at the prefill shape, executed SASS per thread per chunk:
 * shipped flat 587, nested 264, and registers 95 -> 80, which lifts residency
 * 5 -> 6 blocks/SMX with a 0 B stack frame.
 */
#define FC_NX_STAGE_X_NESTED(SH_X, X, COL0, COLS, CBASE, CEND, NC, NC_LIVE, XF) \
    do {                                                                        \
        for (int nxc = 0; nxc < (NC); nxc++) {                                  \
            float *nxdst = (SH_X) + nxc * FC_NX_CHUNK;                           \
            if (nxc < (NC_LIVE)) {                                              \
                const float *nxsrc =                                            \
                    (X) + (size_t)((COL0) + nxc) * (COLS) + (CBASE);            \
                FC_NX_STAGE_ROLLED                                              \
                for (int e = (int)threadIdx.x; e < (CEND); e += 256)            \
                    nxdst[e] = XF(__ldg(nxsrc + e));                            \
            } else {                                                            \
                FC_NX_STAGE_ROLLED                                              \
                for (int e = (int)threadIdx.x; e < (CEND); e += 256)            \
                    nxdst[e] = 0.0f;                                            \
            }                                                                   \
        }                                                                       \
    } while (0)

/*
 * EOUTER FORM — division-free AND memory-parallel, the measured winner.
 *
 * The runtime-bounded ELEMENT loop is outermost and pinned rolled (it trips at
 * most twice, see above); the compile-time COLUMN loop is innermost and fully
 * unrolled, so one iteration issues NC independent `__ldg`s and then NC stores.
 * Measured 8 global loads in flight at NC = 8, against 1 for NESTED and 4 for the
 * shipped flat loop.
 *
 * The `nxc < nc` test is a per-column predicate here rather than a hoisted branch
 * — exactly as in the shipped flat loop, whose ternary it reproduces — so a column
 * past `nc` stores the same `0.0f` and its load is predicated off.
 *
 * Same set of shared addresses, same values, same single write per address as both
 * other forms: `tests/test_nx_staging_equivalence.c` proves it on 342 shapes, and
 * `tests/probe_nx_stage_mlp.cu` measured it bit-EXACT against the flat kernel on
 * all 1,048,576 outputs of the prefill shape.
 */
#define FC_NX_STAGE_X_EOUTER(SH_X, X, COL0, COLS, CBASE, CEND, NC, NC_LIVE, XF) \
    do {                                                                        \
        FC_NX_STAGE_ROLLED                                                      \
        for (int nxe = (int)threadIdx.x; nxe < (CEND); nxe += 256) {             \
            _Pragma("unroll")                                                   \
            for (int nxc = 0; nxc < (NC); nxc++)                                \
                (SH_X)[nxc * FC_NX_CHUNK + nxe] = (nxc < (NC_LIVE))              \
                    ? XF(__ldg((X) + (size_t)((COL0) + nxc) * (COLS)             \
                               + (CBASE) + nxe))                                \
                    : 0.0f;                                                     \
        }                                                                       \
    } while (0)

#if FC_NX_STAGE_NESTED == FC_NX_STAGE_EOUTER_FORM
#define FC_NX_STAGE_X(...) FC_NX_STAGE_X_EOUTER(__VA_ARGS__)
#elif FC_NX_STAGE_NESTED == FC_NX_STAGE_NESTED_FORM
#define FC_NX_STAGE_X(...) FC_NX_STAGE_X_NESTED(__VA_ARGS__)
#else
#define FC_NX_STAGE_X(...) FC_NX_STAGE_X_FLAT(__VA_ARGS__)
#endif

#endif /* FC_MATVEC_NX_STAGING_CUH */
