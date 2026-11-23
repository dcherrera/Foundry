/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_matmul_dst_sync.h
 * @brief The resident matmul's MATH<->PACK destination-register handshake, chosen at build time
 *   by FT_TT_MATMUL_DST_SYNC (resident_matmul_block_protocol.h):
 *
 *     HALF         DST split in two sections that MATH and PACK alternate between. The three
 *                  entry points forward to the shared compute-library functions unchanged.
 *     FULL         DST is one section of twice the size; MATH and PACK strictly alternate.
 *     HALF_SERIAL  HALF's section size with the overlap removed (semaphore max count 1).
 *
 *   FULL needs only these three replacements: the section id ft_tensix_dest_offset_id is reset
 *   to 0 at init and never flipped, so every shared "id != 0 ? half : 0" DST address resolves to
 *   0 and the packer's DST select always picks the low base.
 *
 *   The header lives apart from the shared compute headers so those (included by every compute
 *   kernel) do not change. It relies on the including kernel having already included
 *   ft_tensix_compute.h and, per thread, ft_tensix_compute_math.h / ft_tensix_compute_pack.h.
 *
 * @status Clean-room rewrite (plan Phase 4.7). Only includer: resident_matmul_compute.c; the
 *   shipped rtriad_bfp8 program builds FULL, every other build HALF.
 * @issues None known.
 * @todo None
 */
#ifndef RESIDENT_MATMUL_DST_SYNC_H
#define RESIDENT_MATMUL_DST_SYNC_H

#include <stdint.h>

#include "resident_matmul_block_protocol.h"

#if FT_TT_MATMUL_DST_SYNC != FT_TT_MATMUL_DST_SYNC_HALF &&                                     \
    FT_TT_MATMUL_DST_SYNC != FT_TT_MATMUL_DST_SYNC_FULL &&                                     \
    FT_TT_MATMUL_DST_SYNC != FT_TT_MATMUL_DST_SYNC_HALF_SERIAL
#error "FT_TT_MATMUL_DST_SYNC must be 0 (HALF), 1 (FULL) or 2 (HALF_SERIAL)"
#endif

/* Max count of the MATH<->PACK semaphore: two sections in flight for HALF, one otherwise. A
 * plain literal because it feeds an immediate instruction field. */
#if FT_TT_MATMUL_DST_SYNC == FT_TT_MATMUL_DST_SYNC_HALF
#define FT_TT_MATMUL_DST_SYNC_MAX_COUNT 2
#else
#define FT_TT_MATMUL_DST_SYNC_MAX_COUNT 1
#endif

#ifdef FT_TRISC_MATH

/** MATH-side startup: reset the semaphore, the section id and the DST base offset. */
static FT_BH_INLINE void ftmm_math_pack_sync_init(void)
{
#if FT_TT_MATMUL_DST_SYNC == FT_TT_MATMUL_DST_SYNC_HALF
    ft_tensix_math_pack_sync_init();
#else
    ft_tensix_sync();
    while (ft_tensix_semaphore_read(FT_TENSIX_SEM_MATH_PACK) > 0) {
    }
    FT_BH_TTINSN_SEMINIT(FT_TT_MATMUL_DST_SYNC_MAX_COUNT, 0, FT_BH_SEMAPHORE_1);
    ft_tensix_dest_offset_id = 0;
    FT_BH_TTINSN_SETC16(FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD, 0);
#endif
}

/** MATH-side commit of one DST section to PACK. */
static FT_BH_INLINE void ftmm_math_dest_section_done(void)
{
#if FT_TT_MATMUL_DST_SYNC == FT_TT_MATMUL_DST_SYNC_FULL
    /* Wait for MATH and SFPU to drain, then hand the whole DST to PACK; no section flip. */
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_SYNC,
                           FT_BH_STALLWAIT_COND_MATH | FT_BH_STALLWAIT_COND_SFPU);
    FT_BH_TTINSN_SEMPOST(FT_TENSIX_T6_SEM(FT_TENSIX_SEM_MATH_PACK));
#else
    ft_tensix_math_dest_section_done();
#endif
}

#endif /* FT_TRISC_MATH */

#ifdef FT_TRISC_PACK

/** PACK-side release of one DST section back to MATH. */
static FT_BH_INLINE void ftmm_pack_dest_section_done(uint32_t is_fp32_dest_acc_en)
{
#if FT_TT_MATMUL_DST_SYNC == FT_TT_MATMUL_DST_SYNC_FULL
    /* Wait for the packs to finish, zero ALL of DST (the runtime 32-bit flag forces the store
     * issue form), then free the semaphore slot; no section flip, no packer DST reselect. */
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_MATH, FT_BH_STALLWAIT_COND_PACK);
    FT_BH_PUSH_ZEROACC(FT_BH_ZEROACC_CLR_ALL, is_fp32_dest_acc_en, 0, FT_BH_ADDR_MOD_1, 0);
    FT_BH_TTINSN_SEMGET(FT_TENSIX_T6_SEM(FT_TENSIX_SEM_MATH_PACK));
#else
    ft_tensix_pack_dest_section_done(is_fp32_dest_acc_en);
#endif
}

#endif /* FT_TRISC_PACK */

#endif /* RESIDENT_MATMUL_DST_SYNC_H */
