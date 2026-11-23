/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file moe_dedup_shx_staging.cuh
 * @brief The ONE x-staging loop inside fc_moe_gateup_q3k_dedup_shx's body, in
 *   four shapes: the shipped FLAT form (a runtime-bounded request loop, one
 *   global load per request per thread) and three forms that raise the number of
 *   global loads in flight. Selected by FC_MOE_SHX_STAGE, default 0 = FLAT.
 *
 * WHY THIS FILE EXISTS. `fc_moe_gateup_q3k_dedup_shx` is the gate+up kernel production
 * actually runs (QWEN35_MOE_DEDUP=1 FC_MOE_DEDUP_SHX=1) and ~16 % of prefill.
 * Its staging loop is
 *
 *     for (int j = 0; j < rc; j++) {                      // rc RUNTIME, 1..RT
 *         const float *src = x + sh_tok[j] * cols + sb * 256;
 *         sh_x[j * 256 + threadIdx.x] = __ldg(src + threadIdx.x);
 *     }
 *
 * `rc` is a runtime value, so nvcc emits a x4-unrolled body plus a `rc & 3`
 * remainder ladder, and ptxas schedules that body with only **2 global loads in
 * flight** (COUNTED in SASS as the longest run of LDG before its TEXDEPBAR:
 * results/prefill_moe_grid_compaction.md §4.3). The four LDG of the x4 body are
 * mutually independent — nothing in the loop is dependency-bound — so 2 is a
 * scheduling artifact, not a forced serialisation. The matvec staging gate
 * measured **1.426x** on the kernel for taking that same count from 1 -> 8,
 * and established that on this
 * silicon loads-in-flight, NOT executed instructions and NOT occupancy, is the
 * predictor that tracks wall clock.
 *
 * THE FORMS.
 *   FLAT (0)     the shipped loop, verbatim. 2 loads in flight.
 *   UNROLL8 (1)  the eouter analogue: the request loop becomes the COMPILE-TIME
 *                [0, RT) loop with a `j < rc` predicate, so all RT loads sit in
 *                one straight-line body with no loop-carried dependency and
 *                ptxas may issue them all before the first wait. Costs registers
 *                for the in-flight values.
 *   VEC4 (2)     the same total traffic as one `ld.global.nc.v4` per 64-thread
 *                group per request: RT*64 float4 units over 256 threads is
 *                exactly RT/4 compile-time rounds, each an independent 16 B
 *                load. Fewer load INSTRUCTIONS (RT -> RT/4 per thread) and 4x
 *                the BYTES in flight per load. Requires cols % 4 == 0 and a
 *                16 B-aligned sh_x — both hold (see @issues).
 *   HOIST8 (3)   UNROLL8 plus the block-uniform token ids lifted out of the
 *                super-block loop into registers ONCE, which removes the
 *                ld.shared(sh_tok[j]) -> LDG address chain (and any shared
 *                aliasing ptxas must assume between that load and the
 *                st.shared into sh_x) from every staged element. Costs RT more
 *                live registers than UNROLL8.
 *
 * BIT-EXACT BY CONSTRUCTION, for every form, and this is the whole argument:
 *   - the four forms read the SAME SET of global addresses and write the SAME
 *     SET of shared addresses: {x + tok_j*cols + sb*256 + e} -> {sh_x[j*256+e]}
 *     for j in [0, rc), e in [0, 256). FLAT/UNROLL8/HOIST8 assign (j, e) to
 *     threads identically (e = threadIdx.x); VEC4 re-partitions (j, e) across
 *     threads, which is unobservable because
 *   - each shared address is written exactly ONCE by exactly ONE thread in every
 *     form, so the change of write ORDER (and of which thread writes it) cannot
 *     be observed: there is no address where two writes could race and none left
 *     stale;
 *   - the value written is the same expression — __ldg of the same address;
 *   - all four sit between the SAME two __syncthreads() of the caller, which are
 *     not part of these macros and are not moved. The second barrier is what
 *     makes the thread re-partition invisible to the dot;
 *   - `rc` gates only the loop/predicate; it never enters an arithmetic
 *     expression, so which requests a tile holds cannot change a value;
 *   - ZERO floating-point operations exist in any form. There is nothing to
 *     reassociate and no contraction to break: nothing is hoisted out of a loop
 *     into an expression ptxas could contract, and no float moves from shared
 *     into a register where nvcc could newly fuse it (the two traps this project
 *     has paid for: the attention online-softmax rescale that ptxas had been
 *     contracting to an fma, and the SSM kernel's need for __fmul_rn/__fadd_rn
 *     as contraction barriers). HOIST8 moves an INT out of shared into a
 *     register; integers have no rounding and no contraction.
 *
 * XF is a one-argument transform applied to each staged value. It exists so the
 * microbenchmark's poison arm is the SHIPPED staging code plus exactly one
 * perturbation rather than a hand-copied lookalike that could drift from what
 * ships. Production always passes FC_MOE_SHX_KEEP, the identity.
 *
 * @status FC_MOE_SHX_STAGE defaults to 0 — the shipped FLAT form. With the flag
 *   at 0 the regenerated cuda/fc_cuda_ptx.h module FC_PTX_MOE_EXPERT_GROUP is
 *   byte-identical to the checked-in artifact (179,655 B, md5
 *   a32ae4d144b4cb9111f88e3096e36468); that identity is the gate on this
 *   refactor. NO form has been timed on a GPU.
 * @issues VEC4 assumes cols % 4 == 0 (so every staged 16 B group is aligned) and
 *   a 16 B-aligned sh_x. The first is guaranteed by the kernel's existing
 *   contract — cols is a multiple of 256, since nsb = cols >> 8. The second is
 *   why FC_MOE_SHX_SHX_ALIGN exists: sh_x is declared __align__(16) in a VEC4
 *   build only, so the FLAT build's `.shared .align 4` declaration (and hence
 *   its PTX) is untouched. VEC4 also requires FC_MOE_DEDUP_RT % 4 == 0.
 * @todo None
 */

#ifndef FC_MOE_DEDUP_SHX_STAGING_CUH
#define FC_MOE_DEDUP_SHX_STAGING_CUH

/* Form ids for FC_MOE_SHX_STAGE. */
#define FC_MOE_SHX_STAGE_FLAT    0
#define FC_MOE_SHX_STAGE_UNROLL8 1
#define FC_MOE_SHX_STAGE_VEC4    2
#define FC_MOE_SHX_STAGE_HOIST8  3
#define FC_MOE_SHX_STAGE_SPLIT8  4
#define FC_MOE_SHX_STAGE_SPLITV4 5

/*
 * Compile-time and not a runtime env switch because these kernels ship
 * pre-compiled in cuda/fc_cuda_ptx.h: a runtime switch would mean a second copy
 * of the gate+up kernel in the embedded PTX — module size and JIT time on every
 * model load, forever, for a flag meant to end up permanently on or permanently
 * off. Build a non-default arm with
 *   cd cuda_kernels && make NVCCFLAGS_EXTRA=-DFC_MOE_SHX_STAGE=0
 * and note that the two arms of an in-model A/B are then two binaries, which is
 * what a cross-build bit-exactness digest wants anyway.
 */

/* ★ SHIPPED DEFAULT since 2026-08-11: VEC4. Gated on die 6 at a pinned 666 MHz —
 * kernel 1.1252x (beating SPLITV4's 1.1211x by 9x the 0.040 % floor, sign stable
 * over three router draws), bit-EXACT on 0 of 524 288 outputs at every rep, and
 * 63 registers, one below the 64-register 8-blocks/SMX cliff.
 * results/prefill_moe_shx_staging_gate.md §VERDICT.2. Set to 0 for the flat loop.
 *
 * The default lives HERE, in the source, and NOT as a -D in cuda_kernels/Makefile
 * — exactly as FC_NX_STAGE_NESTED does (matvec_nx_staging.cuh). This kernel spent
 * every header revision from 563c701 to 8364afc silently FLAT because the win
 * existed only inside an A/B `-D` arm while `NVCCFLAGS_EXTRA ?=` was empty, so a
 * plain `make` regenerated the shipped PTX without it. A default that holds only
 * when someone remembers a flag is not a default. Do not move it to the Makefile.
 *
 * Verify a regenerated header MODULE-SCOPED, never header-wide: extract
 * FC_PTX_MOE_EXPERT_GROUP and md5 it. VEC4 = a0c5e2da07a4af74501ce7c64c72c31d,
 * FLAT = a32ae4d144b4cb9111f88e3096e36468. Header-wide token counts cannot see
 * this kernel at all — every ld.global.nc.v4.f32 in the header belongs to
 * MATVEC_Q3K. */
#ifndef FC_MOE_SHX_STAGE
#define FC_MOE_SHX_STAGE FC_MOE_SHX_STAGE_VEC4
#endif

/* Identity transform — what production passes. See XF above. */
#define FC_MOE_SHX_KEEP(v) (v)

/*
 * 16 B alignment for sh_x, needed by VEC4's st.shared.v4 and by nothing else.
 * Empty in every other build so the FLAT build's shared declaration — and
 * therefore its PTX — is byte-for-byte unchanged.
 */
#if FC_MOE_SHX_STAGE == FC_MOE_SHX_STAGE_VEC4 ||                               \
    FC_MOE_SHX_STAGE == FC_MOE_SHX_STAGE_SPLITV4
#define FC_MOE_SHX_ALIGN __align__(16)
#else
#define FC_MOE_SHX_ALIGN
#endif

/* ------------------------------------------------------------------------- *
 * The four staging forms. Each expands inside FC_MOE_DEDUP_SHX_BODY with
 * gate_all/up_all/x/cols in scope plus the shared arrays sh_x/sh_tok. RC is the
 * request count in the tile (1..FC_MOE_DEDUP_RT), SB the super-block index.
 * ------------------------------------------------------------------------- */

/* FLAT — the shipped loop. Runtime trip count; 2 global loads in flight. */
#define FC_MOE_SHX_STAGE_FLAT_FORM(RC, SB, XF)                                 \
            for (int j = 0; j < (RC); j++) {                                    \
                const float *src =                                              \
                    x + (size_t)sh_tok[j] * cols + (size_t)(SB) * 256;          \
                sh_x[j * 256 + threadIdx.x] = XF(__ldg(src + threadIdx.x));     \
            }

/* UNROLL8 — compile-time request loop, predicated. The eouter analogue: RT
 * independent loads in one straight-line body, no loop-carried dependency. */
#define FC_MOE_SHX_STAGE_UNROLL8_FORM(RC, SB, XF)                              \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {                         \
                if (j < (RC)) {                                                 \
                    const float *src =                                          \
                        x + (size_t)sh_tok[j] * cols + (size_t)(SB) * 256;      \
                    sh_x[j * 256 + threadIdx.x] =                               \
                        XF(__ldg(src + threadIdx.x));                           \
                }                                                               \
            }

/* VEC4 — one 16 B load per 64-thread group per request. RT*64 float4 units over
 * 256 threads = RT/4 compile-time rounds, each an independent ld.global.nc.v4.
 * Re-partitions (request, element) across threads; the trailing barrier makes
 * that unobservable (see the bit-exactness argument above). */
#define FC_MOE_SHX_VEC4_ROUNDS (FC_MOE_DEDUP_RT / 4)
#define FC_MOE_SHX_STAGE_VEC4_FORM(RC, SB, XF)                                 \
            _Pragma("unroll")                                                   \
            for (int m = 0; m < FC_MOE_SHX_VEC4_ROUNDS; m++) {                  \
                int u = (int)threadIdx.x + m * 256;                             \
                int j = u >> 6;                                                 \
                int q = (u & 63) << 2;                                          \
                if (j < (RC)) {                                                 \
                    float4 v = __ldg((const float4 *)(x +                       \
                        (size_t)sh_tok[j] * cols + (size_t)(SB) * 256 + q));    \
                    v.x = XF(v.x);                                              \
                    v.y = XF(v.y);                                              \
                    v.z = XF(v.z);                                              \
                    v.w = XF(v.w);                                              \
                    *(float4 *)(sh_x + j * 256 + q) = v;                        \
                }                                                               \
            }

/* HOIST8 — UNROLL8 with the block-uniform token ids lifted into registers once,
 * outside the super-block loop, so no ld.shared sits on the address path of a
 * global load and ptxas has no shared-store/shared-load ordering to respect
 * inside the staging body. */
#define FC_MOE_SHX_STAGE_HOIST8_PRE()                                          \
        int fc_shx_tok[FC_MOE_DEDUP_RT];                                        \
        _Pragma("unroll")                                                       \
        for (int j = 0; j < FC_MOE_DEDUP_RT; j++)                               \
            fc_shx_tok[j] = sh_tok[j];
#define FC_MOE_SHX_STAGE_HOIST8_FORM(RC, SB, XF)                               \
            _Pragma("unroll")                                                   \
            for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {                         \
                if (j < (RC)) {                                                 \
                    const float *src = x + (size_t)fc_shx_tok[j] * cols +       \
                                       (size_t)(SB) * 256;                      \
                    sh_x[j * 256 + threadIdx.x] =                               \
                        XF(__ldg(src + threadIdx.x));                           \
                }                                                               \
            }

/* SPLIT8 — UNROLL8 in two phases: read ALL RT block-uniform token ids out of
 * shared into locals FIRST, then issue the RT global loads, then the RT shared
 * stores. This is the form that actually raises loads in flight, and the reason
 * is the one thing the counting turned up that nothing predicted: in FLAT,
 * UNROLL8 and VEC4 the address of every global load comes from an
 * ld.shared(sh_tok[j]) that sits AFTER the previous st.shared(sh_x), and ptxas
 * will not hoist a shared load above a shared store it cannot disambiguate — so
 * each LDG waits on the one before it. Draining the shared reads first breaks
 * that chain without lifting anything out of the super-block loop, so the RT
 * token registers are live only across the staging region (where the dot's
 * registers are dead) rather than across the whole kernel as in HOIST8. */
#define FC_MOE_SHX_STAGE_SPLIT8_FORM(RC, SB, XF)                               \
            {                                                                   \
                int tk[FC_MOE_DEDUP_RT];                                         \
                _Pragma("unroll")                                               \
                for (int j = 0; j < FC_MOE_DEDUP_RT; j++)                       \
                    tk[j] = sh_tok[j];                                          \
                _Pragma("unroll")                                               \
                for (int j = 0; j < FC_MOE_DEDUP_RT; j++) {                     \
                    if (j < (RC)) {                                             \
                        const float *src = x + (size_t)tk[j] * cols +           \
                                           (size_t)(SB) * 256;                  \
                        sh_x[j * 256 + threadIdx.x] =                           \
                            XF(__ldg(src + threadIdx.x));                       \
                    }                                                           \
                }                                                               \
            }

/* SPLITV4 — VEC4 with the same two-phase split: the RT/4 token ids read out of
 * shared before the RT/4 independent 16 B loads. RT/4 loads in flight at 16 B
 * each — the same bytes in flight as SPLIT8, at a quarter of the load and store
 * instructions. */
#define FC_MOE_SHX_STAGE_SPLITV4_FORM(RC, SB, XF)                              \
            {                                                                   \
                int tk[FC_MOE_SHX_VEC4_ROUNDS];                                 \
                _Pragma("unroll")                                               \
                for (int m = 0; m < FC_MOE_SHX_VEC4_ROUNDS; m++)                \
                    tk[m] = sh_tok[((int)threadIdx.x + m * 256) >> 6];           \
                _Pragma("unroll")                                               \
                for (int m = 0; m < FC_MOE_SHX_VEC4_ROUNDS; m++) {              \
                    int u = (int)threadIdx.x + m * 256;                         \
                    int j = u >> 6;                                             \
                    int q = (u & 63) << 2;                                      \
                    if (j < (RC)) {                                             \
                        float4 v = __ldg((const float4 *)(x +                   \
                            (size_t)tk[m] * cols + (size_t)(SB) * 256 + q));    \
                        v.x = XF(v.x);                                          \
                        v.y = XF(v.y);                                          \
                        v.z = XF(v.z);                                          \
                        v.w = XF(v.w);                                          \
                        *(float4 *)(sh_x + j * 256 + q) = v;                    \
                    }                                                           \
                }                                                               \
            }

/* Forms that need nothing hoisted. */
#define FC_MOE_SHX_STAGE_NO_PRE()

/* ------------------------------------------------------------------------- *
 * Selection. FC_MOE_SHX_STAGE_PRE / _X are what the kernel body expands.
 * ------------------------------------------------------------------------- */
#if FC_MOE_SHX_STAGE == FC_MOE_SHX_STAGE_FLAT
#define FC_MOE_SHX_STAGE_PRE()     FC_MOE_SHX_STAGE_NO_PRE()
#define FC_MOE_SHX_STAGE_X(RC, SB) FC_MOE_SHX_STAGE_FLAT_FORM(RC, SB, FC_MOE_SHX_KEEP)
#elif FC_MOE_SHX_STAGE == FC_MOE_SHX_STAGE_UNROLL8
#define FC_MOE_SHX_STAGE_PRE()     FC_MOE_SHX_STAGE_NO_PRE()
#define FC_MOE_SHX_STAGE_X(RC, SB) FC_MOE_SHX_STAGE_UNROLL8_FORM(RC, SB, FC_MOE_SHX_KEEP)
#elif FC_MOE_SHX_STAGE == FC_MOE_SHX_STAGE_VEC4
#define FC_MOE_SHX_STAGE_PRE()     FC_MOE_SHX_STAGE_NO_PRE()
#define FC_MOE_SHX_STAGE_X(RC, SB) FC_MOE_SHX_STAGE_VEC4_FORM(RC, SB, FC_MOE_SHX_KEEP)
#elif FC_MOE_SHX_STAGE == FC_MOE_SHX_STAGE_HOIST8
#define FC_MOE_SHX_STAGE_PRE()     FC_MOE_SHX_STAGE_HOIST8_PRE()
#define FC_MOE_SHX_STAGE_X(RC, SB) FC_MOE_SHX_STAGE_HOIST8_FORM(RC, SB, FC_MOE_SHX_KEEP)
#elif FC_MOE_SHX_STAGE == FC_MOE_SHX_STAGE_SPLIT8
#define FC_MOE_SHX_STAGE_PRE()     FC_MOE_SHX_STAGE_NO_PRE()
#define FC_MOE_SHX_STAGE_X(RC, SB) FC_MOE_SHX_STAGE_SPLIT8_FORM(RC, SB, FC_MOE_SHX_KEEP)
#elif FC_MOE_SHX_STAGE == FC_MOE_SHX_STAGE_SPLITV4
#define FC_MOE_SHX_STAGE_PRE()     FC_MOE_SHX_STAGE_NO_PRE()
#define FC_MOE_SHX_STAGE_X(RC, SB) FC_MOE_SHX_STAGE_SPLITV4_FORM(RC, SB, FC_MOE_SHX_KEEP)
#else
#error "FC_MOE_SHX_STAGE must be 0 (FLAT), 1 (UNROLL8), 2 (VEC4), 3 (HOIST8), 4 (SPLIT8) or 5 (SPLITV4)"
#endif

#endif /* FC_MOE_DEDUP_SHX_STAGING_CUH */
