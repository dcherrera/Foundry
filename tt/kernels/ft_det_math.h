/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_det_math.h
 * @brief Deterministic, INTEGER-ONLY fp32 math for the Gated-DeltaNet gates, on fp32 BIT
 *        PATTERNS (uint32_t). Two layers:
 *          - FAST (the card path): ft_det_gdn_gate (decay, fused) and ft_det_gdn_beta.
 *          - REFERENCE: add, mul, exp, softplus, sigmoid and the fp32-step composites
 *            ft_det_gdn_g / ft_det_gdn_decay / ft_det_gdn_beta_reference.
 *
 * WHY: the card's RISC-V data-movement core (rv32im, no FPU) and the x86 host (-ffast-math,
 * FTZ+DAZ) must produce the SAME BITS for beta/g/decay. libm cannot be matched, so both run
 * this header. No float type or float operation appears below, so the result is independent
 * of compiler float flags; no 64-bit division, no recursion, no libc, no <math.h>.
 *
 * SEMANTICS (every function):
 *   - DAZ: a denormal input reads as a zero of the same sign.
 *   - FTZ after rounding: a result whose 24-bit-rounded value (unbounded exponent) is below
 *     2^-126 is returned as a signed zero (x86 SSE with MXCSR FTZ|DAZ).
 *   - add/mul: IEEE RNE, bit-identical to x86 addss/mulss under FTZ|DAZ, including x86's NaN
 *     rules (first NaN operand quieted; invalid -> default NaN 0xFFC00000). ft_det_mul equals
 *     ftgc_compact.h's ftgc_mul on every finite pair (tests/test_det_math.c checks it).
 *   - exp/softplus/sigmoid: an internal fixed-point value good to ~2^-29 relative, rounded
 *     ONCE to nearest-even. NaN in -> the same NaN quieted. exp(+Inf)=+Inf, exp(-Inf)=+0,
 *     sigmoid(+-Inf)=1/+0, softplus(+Inf)=+Inf, softplus(-Inf)=+0.
 *   - softplus(x) returns x unchanged when x > 20.0f (the host's `x > 20.0f` branch).
 *   Measured accuracy vs the correctly rounded result: tests/test_det_math.c.
 *
 * Constant data: four 16-word tables (256 B; the fast path uses three). Code avoids clz/ctz
 * builtins (libgcc calls on rv32im without Zbb); right shifts of negative int32/int64 are
 * assumed arithmetic (gcc/clang). The REFERENCE exp core has a variable 64-bit shift
 * (__lshrdi3 under gcc -Os on rv32); the fast path has none.
 *
 * Size (clang 22 -Os): fast path (gate + beta + helpers + pack) 2048 B rv32im / 1560 B
 * rv32imc; everything 4248 / 3206 B. No libgcc calls. Not measured with riscv-tt-elf-gcc.
 *
 * @status Off-hardware verified (tests/test_det_math.c); fast path not yet run on the card.
 * @issues The tables are .rodata: the kernel linker script must place .rodata in local memory.
 * @todo None
 */
#ifndef FT_DET_MATH_H
#define FT_DET_MATH_H

#include <stdint.h>

#define FT_DET_SIGN 0x80000000u
#define FT_DET_INF 0x7F800000u
#define FT_DET_QUIET 0x00400000u
#define FT_DET_DEFAULT_NAN 0xFFC00000u /* x86 "real indefinite" */
#define FT_DET_ONE 0x3F800000u
#define FT_DET_HALF 0x3F000000u
#define FT_DET_SOFTPLUS_THRESHOLD 0x41A00000u /* 20.0f */
#define FT_DET_ABS_128 0x43000000u            /* |x| >= 128: exp saturates */

/* Overridable ONLY so tests/test_det_math.c can build mutated (red) arms. */
#ifndef FT_DET_ROUND_HALF
#define FT_DET_ROUND_HALF 0x80u /* half an ulp of ft_det_pack's 8 round bits */
#endif
#ifndef FT_DET_EXP_C6
#define FT_DET_EXP_C6 0x2AAAAAABu /* 1/6, Q32 */
#endif

/* Every non-trivial function is kept out of line: the cores have several callers, and one
 * copy of each is what holds the library to its text budget on the reader core (clang -Os
 * otherwise inlines exp_core three times). `inline` keeps unused ones warning-free. */
#define FT_DET_SHARED static inline __attribute__((noinline))

/** (2^(i/16) - 1) in Q32, i = 0..15. */
static const uint32_t ft_det_exp2_table[16] = {
    0x00000000u, 0x0B5586D0u, 0x172B83C8u, 0x2387A6E7u, 0x306FE0A3u, 0x3DEA64C1u,
    0x4BFDAD53u, 0x5AB07DD5u, 0x6A09E668u, 0x7A11473Fu, 0x8ACE5423u, 0x9C49182Au,
    0xAE89F996u, 0xC199BDD8u, 0xD5818DD0u, 0xEA4AFA2Au};

/** r_i = 1 / (1 + (i + 0.5)/16) in Q32, rounded. */
static const uint32_t ft_det_log_recip_table[16] = {
    0xF83E0F84u, 0xEA0EA0EAu, 0xDD67C8A6u, 0xD20D20D2u, 0xC7CE0C7Du, 0xBE82FA0Cu,
    0xB60B60B6u, 0xAE4C415Du, 0xA72F0539u, 0xA0A0A0A1u, 0x9A90E7D9u, 0x94F2094Fu,
    0x8FB823EEu, 0x8AD8F2FCu, 0x864B8A7Eu, 0x82082082u};

/** -ln(r_i) in Q32, for the ROUNDED r_i above. */
static const uint32_t ft_det_log_neg_table[16] = {
    0x07E0A6C3u, 0x16F0D28Bu, 0x252AA5F0u, 0x32A4B53Au, 0x3F7230DBu, 0x4BA38AEBu,
    0x5746F6FDu, 0x6268CE1Au, 0x6D13DDF0u, 0x7751A812u, 0x812A952Eu, 0x8AA61E98u,
    0x93CAF094u, 0x9C9F069Au, 0xA527C2EDu, 0xAD6A0262u};

static inline uint32_t ft_det_mulhu(uint32_t a, uint32_t b)
{
    return (uint32_t)(((uint64_t)a * b) >> 32);
}

/** a * b >> 35 on signed values (Q35 x QN -> QN). */
static inline int32_t ft_det_mul_q35(int32_t a, int32_t b)
{
    return (int32_t)(((int64_t)a * b) >> 35);
}

static inline int ft_det_is_nan(uint32_t a)
{
    return (a & 0x7FFFFFFFu) > FT_DET_INF;
}

/** x86 two-operand NaN rule: the first NaN operand, quieted. */
static inline uint32_t ft_det_nan_pick(uint32_t a, uint32_t b)
{
    return (ft_det_is_nan(a) ? a : b) | FT_DET_QUIET;
}

/* An unrounded intermediate: low word the mantissa M (bit 31 set), high word the exponent k
 * (int32), value M * 2^(k - 31). A uint64_t so it comes back in two registers on rv32. */
static inline uint64_t ft_det_unpacked(int32_t exponent, uint32_t mantissa)
{
    return ((uint64_t)(uint32_t)exponent << 32) | mantissa;
}

static inline int32_t ft_det_unpacked_exponent(uint64_t value)
{
    return (int32_t)(uint32_t)(value >> 32);
}

/**
 * Round an unpacked value to fp32 (mantissa bit 31 set, any sticky information already OR-ed
 * into bit 0). RNE on the low 8 bits; FTZ after rounding; overflow -> Inf.
 */
FT_DET_SHARED uint32_t ft_det_pack(uint32_t sign, uint64_t value)
{
    int32_t exponent = ft_det_unpacked_exponent(value);
    uint32_t mant = (uint32_t)value >> 8;
    const uint32_t round = (uint32_t)value & 0xFFu;

    if (round > FT_DET_ROUND_HALF || (round == FT_DET_ROUND_HALF && (mant & 1u)))
        mant++;
    if (mant == 0x01000000u) {
        mant >>= 1;
        exponent++;
    }
    exponent += 127;
    if (exponent <= 0)
        return sign;
    if (exponent >= 255)
        return sign | FT_DET_INF;
    return sign | ((uint32_t)exponent << 23) | (mant & 0x007FFFFFu);
}

/* ---------------------------------------------------------------- add / mul (x86 SSE) */

static inline uint32_t ft_det_add_special(uint32_t a, uint32_t b)
{
    const uint32_t a_inf = (a & 0x7FFFFFFFu) == FT_DET_INF;
    const uint32_t b_inf = (b & 0x7FFFFFFFu) == FT_DET_INF;

    if (ft_det_is_nan(a) || ft_det_is_nan(b))
        return ft_det_nan_pick(a, b);
    if (a_inf && b_inf && ((a ^ b) & FT_DET_SIGN))
        return FT_DET_DEFAULT_NAN;
    return a_inf ? a : b;
}

/** a + b, x86 addss under FTZ|DAZ, round to nearest even. */
FT_DET_SHARED uint32_t ft_det_add(uint32_t a, uint32_t b)
{
    const uint32_t ea = (a >> 23) & 0xFFu, eb = (b >> 23) & 0xFFu;
    uint32_t big, small, shift, ma, mb, m;
    int32_t e;

    if (ea == 0xFFu || eb == 0xFFu)
        return ft_det_add_special(a, b);
    if (ea == 0u || eb == 0u) { /* zeros, denormals read as zero (DAZ) */
        if (ea != 0u)
            return a;
        if (eb != 0u)
            return b;
        return a & b & FT_DET_SIGN; /* -0 only if both are negative */
    }
    big = ((a & 0x7FFFFFFFu) >= (b & 0x7FFFFFFFu)) ? a : b;
    small = (big == a) ? b : a;
    shift = ((big >> 23) & 0xFFu) - ((small >> 23) & 0xFFu);
    e = (int32_t)((big >> 23) & 0xFFu) - 126; /* exponent of a bit-31-led mantissa */
    ma = ((big << 8) | FT_DET_SIGN) >> 1;     /* 1.m at bits [30:7]: one bit of carry room */
    mb = ((small << 8) | FT_DET_SIGN) >> 1;
    if (shift >= 31u)
        mb = 1u; /* only a sticky bit survives */
    else if (shift != 0u)
        mb = (mb >> shift) | (uint32_t)((mb << (32u - shift)) != 0u);
    if ((big ^ small) & FT_DET_SIGN) {
        m = ma - mb;
        if (m == 0u)
            return 0u; /* exact cancellation is +0 under RNE */
    } else {
        m = ma + mb;
    }
    while (!(m & FT_DET_SIGN)) {
        m <<= 1;
        e--;
    }
    return ft_det_pack(big & FT_DET_SIGN, ft_det_unpacked(e, m));
}

/** a * b, x86 mulss under FTZ|DAZ (the same function as ftgc_mul, plus Inf/NaN). */
FT_DET_SHARED uint32_t ft_det_mul(uint32_t a, uint32_t b)
{
    const uint32_t ea = (a >> 23) & 0xFFu, eb = (b >> 23) & 0xFFu;
    const uint32_t sign = (a ^ b) & FT_DET_SIGN;
    uint64_t product;
    uint32_t hi, lo;
    int32_t e;

    if (ea == 0xFFu || eb == 0xFFu) {
        if (ft_det_is_nan(a) || ft_det_is_nan(b))
            return ft_det_nan_pick(a, b);
        if (ea == 0u || eb == 0u) /* Inf * 0 (or * denormal, DAZ) */
            return FT_DET_DEFAULT_NAN;
        return sign | FT_DET_INF;
    }
    if (ea == 0u || eb == 0u) /* zero, or a denormal read as zero (DAZ) */
        return sign;
    product = (uint64_t)((a << 8) | FT_DET_SIGN) * ((b << 8) | FT_DET_SIGN); /* Q62, [1, 4) */
    hi = (uint32_t)(product >> 32);
    lo = (uint32_t)product;
    e = (int32_t)(ea + eb) - 253;
    if (!(hi & FT_DET_SIGN)) { /* product in [1, 2) */
        hi = (hi << 1) | (lo >> 31);
        lo <<= 1;
        e--;
    }
    return ft_det_pack(sign, ft_det_unpacked(e, hi | (uint32_t)(lo != 0u)));
}

/* ---------------------------------------------------------------- transcendental cores */

/**
 * e^x for finite x with |x| < 128 (zero/denormal allowed: reads as 0), unpacked, good to
 * ~2^-29 relative.
 *
 * x*log2(e) = k + f in Q32 (log2(e) to 64 bits), 2^f = 2^(i/16) * e^(r*ln2) with i the top
 * 4 bits of f and e^y - 1 a degree-5 Taylor polynomial on y < ln2/16 (error < 2^-36).
 */
FT_DET_SHARED uint64_t ft_det_exp_core(uint32_t x)
{
    const uint32_t ex = (x >> 23) & 0xFFu;
    uint32_t whole = 0u, frac = 0u, y, h, em1, tf;
    int32_t k = 0;
    uint64_t f;

    if (ex >= 86u) { /* below 2^-41, |x*log2 e| < 2^-32: e^x is 1 to this precision */
        const uint32_t m = (x & 0x007FFFFFu) | 0x00800000u;
        uint64_t t = (uint64_t)m * 0xB8AA3B29u + (((uint64_t)m * 0x5C17F0BBu) >> 32);

        t >>= (149u - ex); /* |x| * log2(e) in Q32 */
        whole = (uint32_t)(t >> 32);
        frac = (uint32_t)t;
    }
    if (x & FT_DET_SIGN) {
        k = -(int32_t)whole - (int32_t)(frac != 0u);
        frac = 0u - frac;
    } else {
        k = (int32_t)whole;
    }
    y = ft_det_mulhu(frac & 0x0FFFFFFFu, 0xB17217F8u); /* r * ln2, Q32 */
    h = 0x0AAAAAABu + ft_det_mulhu(y, 0x02222222u);    /* 1/24 + y/120 */
    h = FT_DET_EXP_C6 + ft_det_mulhu(y, h);            /* 1/6 + ... */
    h = 0x80000000u + ft_det_mulhu(y, h);              /* 1/2 + ... */
    em1 = y + ft_det_mulhu(ft_det_mulhu(y, y), h);     /* e^y - 1 */
    tf = ft_det_exp2_table[frac >> 28];
    f = (uint64_t)tf + em1 + ft_det_mulhu(tf, em1); /* (1+tf)(1+em1) - 1, Q32 */
    if (f > 0xFFFFFFFFu)
        f = 0xFFFFFFFFu;
    return ft_det_unpacked(k, FT_DET_SIGN | (uint32_t)(f >> 1) | ((uint32_t)f & 1u));
}

/** h(v) = 1/2 - v/3 + v^2/4 - v^3/5 + v^4/6, v in Q35, result Q31: log1p(v) = v - v^2 h(v). */
FT_DET_SHARED int32_t ft_det_log1p_horner(int32_t v)
{
    int32_t h = 0x1999999A - ft_det_mul_q35(v, 0x15555555);

    h = 0x20000000 - ft_det_mul_q35(v, h);
    h = 0x2AAAAAAB - ft_det_mul_q35(v, h);
    return 0x40000000 - ft_det_mul_q35(v, h);
}

/**
 * log(1 + u) for u = mant_u * 2^(ku - 31) in (0, 1] (ft_det_exp_core of a value <= 0).
 * Returns an unpacked value like ft_det_exp_core.
 *   u >= 2^-5: w = 1+u, w*r_i = 1+v with |v| < 1/32, log w = -ln r_i + log1p(v).
 *   u <  2^-5: log1p(u) = u (1 - u h(u)), relative, so tiny u keeps full precision.
 */
FT_DET_SHARED uint64_t ft_det_log1p_core(uint32_t mant_u, int32_t ku)
{
    uint32_t m;
    int32_t e;

    if (ku == 0) /* u == 1 */
        return ft_det_unpacked(-1, 0xB17217F8u); /* ln 2 */
    if (ku >= -5) {
        const uint32_t u = mant_u >> (uint32_t)(-1 - ku); /* Q32, [2^-5, 1) */
        const uint32_t recip = ft_det_log_recip_table[u >> 28];
        const uint64_t prod = (((uint64_t)u * recip) >> 1) + ((uint64_t)recip << 31); /* Q63 */
        const int32_t v = (int32_t)((int64_t)(prod >> 28) - ((int64_t)1 << 35));    /* Q35 */
        const int32_t v2h =
            (int32_t)(((int64_t)ft_det_mul_q35(v, v) * ft_det_log1p_horner(v)) >> 31);

        m = ft_det_log_neg_table[u >> 28] + (uint32_t)((v - v2h) >> 3); /* Q32 */
        e = -1;
        while (!(m & FT_DET_SIGN)) {
            m <<= 1;
            e--;
        }
    } else {
        const uint32_t shift = (uint32_t)(-4 - ku);
        const uint32_t u35 = (shift < 32u) ? (mant_u >> shift) : 0u; /* Q35, < 2^30 */
        const uint32_t uh = (uint32_t)(
            ((uint64_t)u35 * (uint32_t)ft_det_log1p_horner((int32_t)u35)) >> 34); /* Q32 */

        m = mant_u - ft_det_mulhu(mant_u, uh);
        e = ku;
        if (!(m & FT_DET_SIGN)) {
            m <<= 1;
            e--;
        }
    }
    return ft_det_unpacked(e, m);
}

/**
 * floor(2^63 / w) for w in (2^31, 2^32), i.e. 1/w in Q32, with a non-zero remainder OR-ed
 * into bit 0 (sticky). Restoring division, 32-bit only.
 */
FT_DET_SHARED uint32_t ft_det_reciprocal_q32(uint32_t w)
{
    uint32_t remainder = 0x80000000u, quotient = 0u, carry, bit;

    for (bit = 0u; bit < 32u; bit++) {
        carry = remainder >> 31;
        remainder <<= 1;
        quotient <<= 1;
        if (carry || remainder >= w) {
            remainder -= w;
            quotient |= 1u;
        }
    }
    return quotient | (uint32_t)(remainder != 0u);
}

/* ---------------------------------------------------------------- public functions */

/** e^x. FTZ below 2^-126, +Inf on overflow. */
FT_DET_SHARED uint32_t ft_det_exp(uint32_t x)
{
    if ((x & 0x7FFFFFFFu) > FT_DET_INF)
        return x | FT_DET_QUIET;
    if ((x & 0x7FFFFFFFu) >= FT_DET_ABS_128) /* includes +-Inf */
        return (x & FT_DET_SIGN) ? 0u : FT_DET_INF;
    return ft_det_pack(0u, ft_det_exp_core(x));
}

/**
 * x + L for 0 < x <= 20, with L = log(1 + e^-x) unpacked.
 * 32-bit only: both terms in units of 2^-q, q chosen so x's mantissa sits at [29:6] (x >= 1)
 * or q = 30 (x < 1): six guard bits below the result's ulp and no overflow (x + L < 2^31).
 */
static inline uint32_t ft_det_softplus_positive(uint32_t x, uint64_t log_value)
{
    const uint32_t log_mant = (uint32_t)log_value;
    const int32_t log_exp = ft_det_unpacked_exponent(log_value);
    const int32_t ex = (int32_t)((x >> 23) & 0xFFu);
    const int32_t q = (ex >= 127) ? 156 - ex : 30;
    const int32_t x_shift = q + ex - 150; /* x = M * 2^(ex-150) = (M << x_shift) * 2^-q */
    const uint32_t log_shift = (uint32_t)(31 - q - log_exp); /* >= 2 */
    const uint32_t mant = (x & 0x007FFFFFu) | 0x00800000u;
    uint32_t sum = 0u;
    int32_t e = 31 - q;

    if (log_shift < 32u)
        sum = (log_mant >> log_shift) | (uint32_t)((log_mant << (32u - log_shift)) != 0u);
    if (ex != 0 && x_shift >= 0)
        sum += mant << x_shift;
    else if (ex != 0 && x_shift > -24)
        sum += mant >> -x_shift;
    while (!(sum & FT_DET_SIGN)) { /* sum >= 2^29: at most two steps */
        sum <<= 1;
        e--;
    }
    return ft_det_pack(0u, ft_det_unpacked(e, sum));
}

/** log(1 + e^x); x itself when x > 20.0f (bit-identical to the host's branch). */
FT_DET_SHARED uint32_t ft_det_softplus(uint32_t x)
{
    const uint32_t magnitude = x & 0x7FFFFFFFu;
    uint64_t u, log_value;

    if (magnitude > FT_DET_INF)
        return x | FT_DET_QUIET;
    if (!(x & FT_DET_SIGN) && magnitude > FT_DET_SOFTPLUS_THRESHOLD)
        return x; /* includes +Inf */
    if ((x & FT_DET_SIGN) && magnitude >= FT_DET_ABS_128)
        return 0u; /* e^x flushes; includes -Inf */
    u = ft_det_exp_core(x | FT_DET_SIGN); /* u = e^-|x| */
    log_value = ft_det_log1p_core((uint32_t)u, ft_det_unpacked_exponent(u));
    if (x & FT_DET_SIGN)
        return ft_det_pack(0u, log_value);
    return ft_det_softplus_positive(x, log_value);
}

/** 1 / (1 + e^-x); for x < 0 computed as u / (1 + u), u = e^x. */
FT_DET_SHARED uint32_t ft_det_sigmoid(uint32_t x)
{
    const uint32_t negative = x & FT_DET_SIGN;
    uint32_t mant_u, w, q;
    uint64_t u, prod;
    int32_t ku;

    if ((x & 0x7FFFFFFFu) > FT_DET_INF)
        return x | FT_DET_QUIET;
    if ((x & 0x7FFFFFFFu) >= FT_DET_ABS_128) /* includes +-Inf */
        return negative ? 0u : FT_DET_ONE;
    u = ft_det_exp_core(x | FT_DET_SIGN); /* u = e^-|x| */
    mant_u = (uint32_t)u;
    ku = ft_det_unpacked_exponent(u);
    if (ku == 0)
        return FT_DET_HALF; /* u == 1 */
    w = FT_DET_SIGN + ((-ku < 32) ? (mant_u >> (uint32_t)(-ku)) : 0u); /* 1 + u, Q31 */
    if (w == FT_DET_SIGN)
        return negative ? ft_det_pack(0u, u) : FT_DET_ONE;
    q = ft_det_reciprocal_q32(w); /* 1 / (1 + u), Q32, in (1/2, 1), sticky in bit 0 */
    if (!negative)
        return ft_det_pack(0u, ft_det_unpacked(-1, q));
    prod = (uint64_t)mant_u * q; /* u / (1 + u) = prod * 2^(ku - 63), prod in (2^62, 2^64) */
    if (!(prod >> 63)) {
        prod <<= 1;
        ku--;
    }
    return ft_det_pack(0u, ft_det_unpacked(ku, (uint32_t)(prod >> 32) |
                                                   (uint32_t)((uint32_t)prod != 0u)));
}

/* ---------------------------------------------------------------- fast gate path
 *
 * The card path. 32-bit arithmetic plus 32x32->64 products only: no variable 64-bit shift
 * (gcc -Os on rv32 turns those into __lshrdi3 calls), no division, short fixed loops.
 * Works in base 2: e^x = 2^t with t = x*log2(e), so
 *   decay = exp(a * softplus(x)) = 2^(a * S(t)),  S(t) = max(t, 0) + log2(1 + 2^-|t|)
 *   beta  = 1 / (1 + 2^-t)                        (Newton reciprocal, 3 steps)
 * Accuracy is defined against an all-double reference: tests/test_det_math.c.
 */

#ifndef FT_DET_FAST_LOG1P_C3
#define FT_DET_FAST_LOG1P_C3 0x2AAAAAABu /* 1/3, Q31 (overridable for a red arm) */
#endif
#ifndef FT_DET_FAST_NEWTON_STEPS
#define FT_DET_FAST_NEWTON_STEPS 3 /* 1/17 seed -> 2^-8 -> 2^-16 -> 2^-32 */
#endif

/** -log2(r_i) in Q31, for the rounded r_i of ft_det_log_recip_table. */
static const uint32_t ft_det_log2_neg_table[16] = {
    0x05AEB4DDu, 0x108C588Du, 0x1ACF5E2Eu, 0x24880F56u, 0x2DC4439Bu, 0x368FD7EEu,
    0x3EF50AD2u, 0x46FCC47Au, 0x4EAECFEBu, 0x5612089Au, 0x5D2C7F5Au, 0x64039858u,
    0x6A9C23D6u, 0x70FA728Bu, 0x772266ADu, 0x7D17822Fu};

/** a * b / 2^32 rounded to nearest (unbiased products keep the gate's error from adding up). */
static inline uint32_t ft_det_mulhu_round(uint32_t a, uint32_t b)
{
    return (uint32_t)(((uint64_t)a * b + 0x80000000u) >> 32);
}

/** p / 2^shift rounded to nearest, shift in [1, 31]. */
static inline uint32_t ft_det_shift_round(uint32_t p, uint32_t shift)
{
    return ((p >> (shift - 1u)) + 1u) >> 1;
}

static inline int32_t ft_det_mulh(int32_t a, int32_t b)
{
    return (int32_t)(((int64_t)a * b) >> 32);
}

/** 2^F for F in [0, 1) (Q32): mantissa in [2^31, 2^32), ~2^-29 relative. */
FT_DET_SHARED uint32_t ft_det_fast_exp2_frac(uint32_t frac)
{
    const uint32_t tf = ft_det_exp2_table[frac >> 28];
    const uint32_t y = ft_det_mulhu(frac & 0x0FFFFFFFu, 0xB17217F8u); /* r ln2 < 0.0434, Q32 */
    uint32_t h = FT_DET_EXP_C6 + ft_det_mulhu(y, 0x0AAAAAABu);       /* 1/6 + y/24 */
    uint32_t em1, cross, sum, carry;

    h = 0x80000000u + ft_det_mulhu(y, h);          /* 1/2 + y/6 + y^2/24 */
    em1 = y + ft_det_mulhu(ft_det_mulhu(y, y), h); /* e^y - 1, error < 2^-29 */
    cross = ft_det_mulhu(tf, em1);
    sum = tf + em1; /* (1 + tf)(1 + em1) - 1 */
    carry = sum < tf;
    sum += cross;
    carry |= sum < cross;
    return carry ? 0xFFFFFFFFu : (FT_DET_SIGN | (sum >> 1));
}

/**
 * |x| * log2(e) for |x| = m * 2^(e - 31) (m bit 31 set, e <= 7): high word the integer
 * part k, low word the fraction (Q32). The 64-bit product is funnel-shifted in 32-bit pieces.
 */
FT_DET_SHARED uint64_t ft_det_fast_log2e_split(uint32_t m, int32_t e)
{
    const uint32_t correction = ft_det_mulhu(m, 0x5C17F0BBu); /* log2(e) bits 32..63 */
    const uint32_t shift = (uint32_t)(30 - e);                /* >= 23 */
    uint32_t lo = m * 0xB8AA3B29u, hi = ft_det_mulhu(m, 0xB8AA3B29u);

    lo += correction;
    hi += (uint32_t)(lo < correction); /* P = |x| log2(e) 2^(62-e) */
    if (shift < 32u)
        return ((uint64_t)(hi >> shift) << 32) | (hi << (32u - shift)) | (lo >> shift);
    return (shift < 64u) ? (hi >> (shift - 32u)) : 0u;
}

/** log2(1 + u), u in [0, 1) Q32 -> Q31 (clamped at 0), absolute error ~2^-31. */
FT_DET_SHARED uint32_t ft_det_fast_log2_1p(uint32_t u)
{
    const uint32_t recip = ft_det_log_recip_table[u >> 28];
    const int32_t v = (int32_t)(recip + ft_det_mulhu(u, recip)); /* (1+u) r - 1, Q32, |v|<1/32 */
    int32_t h = 0x20000000 - ft_det_mulh(v, 0x1999999A);         /* Q31: 1/4 - v/5 */
    int32_t natural, result;

    h = (int32_t)FT_DET_FAST_LOG1P_C3 - ft_det_mulh(v, h);
    h = 0x40000000 - ft_det_mulh(v, h);                      /* 1/2 - v/3 + v^2/4 - v^3/5 */
    natural = v - ft_det_mulh(ft_det_mulh(v, v), h) * 2;    /* log1p(v), Q32 */
    result = (int32_t)ft_det_log2_neg_table[u >> 28] +
             ((natural + ft_det_mulh(natural, 0x71547653)) >> 1); /* * log2(e) -> Q31 */
    return result < 0 ? 0u : (uint32_t)result;
}

/** p * 2^shift (p < 0xFFFFFFFF), rounded to nearest, saturating at 0xFFFFFFFF. */
static inline uint32_t ft_det_fast_scale(uint32_t p, int32_t shift)
{
    if (shift == 0)
        return p;
    if (shift < 0)
        return (shift > -32) ? ft_det_shift_round(p, (uint32_t)-shift) : 0u;
    if (shift >= 32 || (p >> (32u - (uint32_t)shift)) != 0u)
        return 0xFFFFFFFFu;
    return p << (uint32_t)shift;
}

/** Normalizes m (non-zero) to bit 31 in five fixed steps; returns the shift in the high word. */
static inline uint64_t ft_det_fast_normalize(uint32_t m)
{
    uint32_t n = 0u;

    if (!(m >> 16)) { m <<= 16; n += 16u; }
    if (!(m >> 24)) { m <<= 8; n += 8u; }
    if (!(m >> 28)) { m <<= 4; n += 4u; }
    if (!(m >> 30)) { m <<= 2; n += 2u; }
    if (!(m >> 31)) { m <<= 1; n += 1u; }
    return ((uint64_t)n << 32) | m;
}

/** Inf/NaN anywhere in the gate's inputs (alpha + dt = x; decay = exp(a * softplus(x))). */
FT_DET_SHARED uint32_t ft_det_fast_gate_special(uint32_t alpha, uint32_t dt, uint32_t a)
{
    const uint32_t alpha_inf = (alpha & 0x7FFFFFFFu) == FT_DET_INF;
    const uint32_t dt_inf = (dt & 0x7FFFFFFFu) == FT_DET_INF;
    const uint32_t a_zero = ((a >> 23) & 0xFFu) == 0u;
    uint32_t x_sign;

    if (ft_det_is_nan(alpha) || ft_det_is_nan(dt))
        return ft_det_nan_pick(alpha, dt);
    if (ft_det_is_nan(a))
        return a | FT_DET_QUIET;
    if (alpha_inf && dt_inf && ((alpha ^ dt) & FT_DET_SIGN))
        return FT_DET_DEFAULT_NAN;
    if (!alpha_inf && !dt_inf) /* only a is infinite; softplus(finite) > 0 */
        return (a & FT_DET_SIGN) ? 0u : FT_DET_INF;
    x_sign = (alpha_inf ? alpha : dt) & FT_DET_SIGN;
    if (x_sign) /* softplus(-Inf) = 0: a * 0 */
        return ((a & 0x7FFFFFFFu) == FT_DET_INF) ? FT_DET_DEFAULT_NAN : FT_DET_ONE;
    if (a_zero) /* 0 * Inf */
        return FT_DET_DEFAULT_NAN;
    return (a & FT_DET_SIGN) ? 0u : FT_DET_INF;
}

/**
 * x = alpha + dt as an unrounded 32-bit mantissa: returns (E << 32) | m with |x| =
 * m * 2^(E - 31), m bit 31 set, and the sign in *negative; 0 when x == 0. Exact whenever the
 * operands overlap (cancellation), else truncated ~2^-30 relative. Finite inputs only.
 */
static inline uint64_t ft_det_fast_sum(uint32_t alpha, uint32_t dt, uint32_t *negative)
{
    const uint32_t big = ((alpha & 0x7FFFFFFFu) >= (dt & 0x7FFFFFFFu)) ? alpha : dt;
    const uint32_t small = (big == alpha) ? dt : alpha;
    const uint32_t e_big = (big >> 23) & 0xFFu, e_small = (small >> 23) & 0xFFu;
    const uint32_t shift = e_big - e_small;
    uint32_t m_big = ((big << 8) | FT_DET_SIGN) >> 1, m_small = 0u, m;
    uint64_t normalized;

    if (e_big == 0u) /* both zero (DAZ) */
        return 0u;
    if (e_small != 0u && shift < 32u) /* exact when the operands overlap; else rounded */
        m_small = shift ? ft_det_shift_round(((small << 8) | FT_DET_SIGN) >> 1, shift)
                        : ((small << 8) | FT_DET_SIGN) >> 1;
    m = ((big ^ small) & FT_DET_SIGN) ? m_big - m_small : m_big + m_small;
    if (m == 0u)
        return 0u;
    *negative = big & FT_DET_SIGN;
    normalized = ft_det_fast_normalize(m);
    return ft_det_unpacked((int32_t)e_big - 126 - (int32_t)(normalized >> 32),
                           (uint32_t)normalized);
}

/** 2^y for y = +-Y, Y in Q25 (sign of `a_bits`), rounded to fp32 with FTZ. */
static inline uint32_t ft_det_fast_exp2_q25(uint32_t y_q25, uint32_t a_bits)
{
    const int32_t k = (int32_t)(y_q25 >> 25);
    const uint32_t frac = y_q25 << 7;

    if (!(a_bits & FT_DET_SIGN))
        return ft_det_pack(0u, ft_det_unpacked(k, ft_det_fast_exp2_frac(frac)));
    return ft_det_pack(0u, ft_det_unpacked(-k - (int32_t)(frac != 0u),
                                           ft_det_fast_exp2_frac(0u - frac)));
}

/**
 * decay = exp(ssm_a * softplus(alpha + dt)) as ONE fixed-point pipeline (no intermediate
 * fp32 rounding): 2^(a * S(t)), |a S| accumulated in Q25 (saturating; flushes to +0 when the
 * result is below 2^-126). alpha may be any fp32 (bf16-widened on the card).
 */
FT_DET_SHARED uint32_t ft_det_gdn_gate(uint32_t alpha, uint32_t dt, uint32_t ssm_a)
{
    const uint32_t ea = (ssm_a >> 23) & 0xFFu;
    const uint32_t a_mant = (ssm_a << 8) | FT_DET_SIGN;
    uint32_t negative = 0u, log_q31 = 0u, y_q25 = 0u, term;
    uint64_t x, split;
    int32_t ex;

    if (((alpha >> 23) & 0xFFu) == 0xFFu || ((dt >> 23) & 0xFFu) == 0xFFu || ea == 0xFFu)
        return ft_det_fast_gate_special(alpha, dt, ssm_a);
    if (ea == 0u) /* a == 0 (DAZ) */
        return FT_DET_ONE;
    x = ft_det_fast_sum(alpha, dt, &negative);
    ex = ft_det_unpacked_exponent(x);
    if (x == 0u) {
        log_q31 = 0x7FFFFFFFu; /* S(0) = 1 */
    } else if (ex <= 5) {      /* |t| < 93: 2^-|t| still matters */
        split = ft_det_fast_log2e_split((uint32_t)x, ex);
        if ((split >> 32) < 32u) {
            const uint32_t k = (uint32_t)(split >> 32), frac = (uint32_t)split;
            const uint32_t u = (frac == 0u) ? (k == 0u ? 0xFFFFFFFFu : (FT_DET_SIGN >> (k - 1u)))
                                            : (ft_det_fast_exp2_frac(0u - frac) >> k);

            log_q31 = ft_det_fast_log2_1p(u); /* u = 2^-|t|, Q32 */
        }
    }
    if (x != 0u && !negative) { /* + |a| t, from t's 32-bit mantissa */
        const uint32_t t_mant = ft_det_mulhu_round((uint32_t)x, 0xB8AA3B29u); /* 2^(ex-30) */

        y_q25 = ft_det_fast_scale(ft_det_mulhu_round(a_mant, t_mant), (int32_t)ea + ex - 131);
    }
    term = ft_det_fast_scale(ft_det_mulhu_round(a_mant, log_q31), (int32_t)ea - 132);
    y_q25 = (y_q25 + term < y_q25) ? 0xFFFFFFFFu : y_q25 + term;
    return ft_det_fast_exp2_q25(y_q25, ssm_a);
}

/** 1 / (1 + d) for d in (0, 1) Q32 -> Q32: linear seed, FT_DET_FAST_NEWTON_STEPS steps. */
static inline uint32_t ft_det_fast_reciprocal_1p(uint32_t d)
{
    uint32_t q = 0xF0F0F0F1u - ft_det_mulhu(d, 0x78787878u); /* 16/17 - 8/17 d */
    int step;

    for (step = 0; step < FT_DET_FAST_NEWTON_STEPS; step++) {
        const int32_t error = (int32_t)(0u - q - ft_det_mulhu(d, q)); /* 1 - (1+d) q */

        q += (uint32_t)(int32_t)(((int64_t)error * (int64_t)q) >> 32);
    }
    return q;
}

/** beta = sigmoid(b), the card path: u = 2^-|t|, 1/(1+u) by Newton; u/(1+u) for b < 0. */
FT_DET_SHARED uint32_t ft_det_gdn_beta(uint32_t b)
{
    const uint32_t magnitude = b & 0x7FFFFFFFu, negative = b & FT_DET_SIGN;
    uint32_t k, frac, mant_u, d, q;
    int32_t eu;
    uint64_t split;

    if (magnitude > FT_DET_INF)
        return b | FT_DET_QUIET;
    if (magnitude >= FT_DET_ABS_128) /* includes +-Inf */
        return negative ? 0u : FT_DET_ONE;
    if (magnitude < 0x00800000u) /* +-0, denormal (DAZ) */
        return FT_DET_HALF;
    split = ft_det_fast_log2e_split((b << 8) | FT_DET_SIGN, (int32_t)(magnitude >> 23) - 127);
    k = (uint32_t)(split >> 32);
    frac = (uint32_t)split;
    if (frac == 0u && k == 0u)
        return FT_DET_HALF;
    mant_u = (frac == 0u) ? FT_DET_SIGN : ft_det_fast_exp2_frac(0u - frac);
    eu = -(int32_t)k - (int32_t)(frac != 0u); /* u = mant_u 2^(eu-31) */
    d = (eu > -32) ? (mant_u >> (uint32_t)(-1 - eu)) : 0u; /* u, Q32 */
    if (d == 0u)
        return negative ? ft_det_pack(0u, ft_det_unpacked(eu, mant_u)) : FT_DET_ONE;
    q = ft_det_fast_reciprocal_1p(d);
    if (!negative)
        return ft_det_pack(0u, ft_det_unpacked(-1, q));
    q = ft_det_mulhu(mant_u, q); /* u / (1 + u) = q 2^(eu-31), q in (2^30, 2^32) */
    if (!(q & FT_DET_SIGN)) {
        q <<= 1;
        eu--;
    }
    return ft_det_pack(0u, ft_det_unpacked(eu, q));
}

/* ---------------------------------------------------------------- Gated-DeltaNet gates
 * The reference composites: each step rounded to fp32 exactly as the host spells it. */

/** beta = sigmoid(b) through the reference sigmoid. */
static inline uint32_t ft_det_gdn_beta_reference(uint32_t b)
{
    return ft_det_sigmoid(b);
}

/** g = ssm_a * softplus(alpha + dt), each step one fp32 rounding as the host does it. */
static inline uint32_t ft_det_gdn_g(uint32_t alpha, uint32_t dt, uint32_t ssm_a)
{
    return ft_det_mul(ssm_a, ft_det_softplus(ft_det_add(alpha, dt)));
}

/** decay = exp(g). */
static inline uint32_t ft_det_gdn_decay(uint32_t g)
{
    return ft_det_exp(g);
}

#endif /* FT_DET_MATH_H */
