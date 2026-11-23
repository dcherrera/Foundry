/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_softfloat.c
 * @brief IEEE-754 binary32 arithmetic for data-movement cores, which have no F extension.
 *   GCC lowers every `float` operation on such a core to a call to a fixed runtime name
 *   (__addsf3, __mulsf3, __ltsf2, ...); this file IS those names, so no toolchain runtime
 *   library links into a Tensix image (every link is -nostdlib).
 *
 * SEMANTICS (what must stay bit-exact against the host reference sampler):
 *   - round to nearest, ties to even; subnormals produced and consumed, never flushed;
 *   - every NaN result is the canonical quiet NaN 0x7FC00000 (the RISC-V F-extension rule);
 *   - comparisons: ordered results -1/0/1; an unordered compare returns the value that
 *     makes the predicate GCC tests false (eq/ne/lt/le -> 1, gt/ge -> -1);
 *   - float -> int32 truncates toward zero; out of range, inf or NaN saturate by sign
 *     (positive -> INT32_MAX, negative -> INT32_MIN).
 *   Proven against the host's hardware binary32 by tests/test_softfloat.c.
 *
 * Self-contained by construction: no builtin that could lower to another runtime call
 * (no __builtin_clz, no 64-bit division). Compiled WITHOUT LTO into the support archive,
 * so a member links only into an image that calls it.
 *
 * @status Added 2026-10-08 (clean-room plan, libgcc/newlib removal).
 * @issues None
 * @todo None
 */
#include <stdint.h>

#ifdef FT_SOFTFLOAT_HOST_TEST
/* Host test build: own names, so the host's runtime is not replaced. */
#define FT_SF(name) ft_test_##name
#else
#define FT_SF(name) name
#endif

#define SIGN_BIT 0x80000000u
#define ABS_MASK 0x7FFFFFFFu
#define INF_BITS 0x7F800000u
#define CANONICAL_NAN 0x7FC00000u
#define HIDDEN_BIT 0x00800000u
#define FRACTION_MASK 0x007FFFFFu
/* Working significands carry 3 extra low bits (guard, round, sticky): leading 1 at bit 26. */
#define WORK_LEAD (1u << 26)

typedef union {
    float value;
    uint32_t bits;
} ft_sf_word;

static uint32_t bits_of(float value)
{
    ft_sf_word word;
    word.value = value;
    return word.bits;
}

static float float_of(uint32_t bits)
{
    ft_sf_word word;
    word.bits = bits;
    return word.value;
}

static int is_nan(uint32_t bits)
{
    return (bits & ABS_MASK) > INF_BITS;
}

static int is_inf(uint32_t bits)
{
    return (bits & ABS_MASK) == INF_BITS;
}

/** Shift right by `count`, OR-ing every bit shifted out into bit 0 (sticky). */
static uint32_t shift_right_sticky(uint32_t value, uint32_t count)
{
    if (count == 0u)
        return value;
    if (count >= 32u)
        return value != 0u;
    return (value >> count) | ((value & ((1u << count) - 1u)) != 0u);
}

/**
 * Round a working significand to 24 bits and assemble the result. `exponent` is the biased
 * exponent of the leading bit position 26; a significand below WORK_LEAD with exponent 1 is
 * subnormal. Overflow becomes infinity.
 */
static uint32_t round_and_pack(uint32_t sign, int32_t exponent, uint32_t significand)
{
    uint32_t low = significand & 7u;
    uint32_t field;

    significand >>= 3;
#ifdef FT_SOFTFLOAT_RED_TIES_AWAY /* red for tests/test_softfloat.c: ties round away */
    if (low >= 4u)
#else
    if (low > 4u || (low == 4u && (significand & 1u)))
#endif
        significand++;
    /* The hidden bit (bit 23, or bit 24 after a carry) adds itself into the exponent. */
    if (exponent >= 255 || (int32_t)(exponent - 1 + (int32_t)(significand >> 23)) >= 255)
        return sign | INF_BITS;
    field = ((uint32_t)(exponent - 1) << 23) + significand;
    return sign | field;
}

/** Unpack a finite non-zero value: significand with its leading 1 at bit 23. */
static uint32_t unpack_normalized(uint32_t bits, int32_t *out_exponent)
{
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu);
    uint32_t significand = bits & FRACTION_MASK;

    if (exponent == 0) {
        exponent = 1;
        while (!(significand & HIDDEN_BIT)) {
            significand <<= 1;
            exponent--;
        }
    } else {
        significand |= HIDDEN_BIT;
    }
    *out_exponent = exponent;
    return significand;
}

/** Bring a working significand whose exponent fell below 1 into the subnormal range. */
static uint32_t denormalize(int32_t *exponent, uint32_t significand)
{
    if (*exponent >= 1)
        return significand;
    significand = shift_right_sticky(significand, (uint32_t)(1 - *exponent));
    *exponent = 1;
    return significand;
}

static uint32_t add_bits(uint32_t a, uint32_t b)
{
    uint32_t sign_a = a & SIGN_BIT;
    uint32_t sign_b = b & SIGN_BIT;
    int32_t exp_a = (int32_t)((a >> 23) & 0xFFu);
    int32_t exp_b = (int32_t)((b >> 23) & 0xFFu);
    uint32_t sig_a = a & FRACTION_MASK;
    uint32_t sig_b = b & FRACTION_MASK;
    uint32_t significand;

    if (is_nan(a) || is_nan(b))
        return CANONICAL_NAN;
    if (is_inf(a))
        return (is_inf(b) && sign_a != sign_b) ? CANONICAL_NAN : a;
    if (is_inf(b))
        return b;
    if (((a | b) & ABS_MASK) == 0u)
        return sign_a & sign_b; /* +0 unless both are -0 */

    if (exp_a == 0)
        exp_a = 1;
    else
        sig_a |= HIDDEN_BIT;
    if (exp_b == 0)
        exp_b = 1;
    else
        sig_b |= HIDDEN_BIT;
    sig_a <<= 3;
    sig_b <<= 3;
    /* Make `a` the larger magnitude. */
    if (exp_b > exp_a || (exp_b == exp_a && sig_b > sig_a)) {
        uint32_t t;
        int32_t e;
        t = sig_a; sig_a = sig_b; sig_b = t;
        e = exp_a; exp_a = exp_b; exp_b = e;
        t = sign_a; sign_a = sign_b; sign_b = t;
    }
    sig_b = shift_right_sticky(sig_b, (uint32_t)(exp_a - exp_b));
    if (sign_a == sign_b) {
        significand = sig_a + sig_b;
        if (significand & (WORK_LEAD << 1)) {
            significand = (significand >> 1) | (significand & 1u);
            exp_a++;
        }
    } else {
        significand = sig_a - sig_b;
        if (significand == 0u)
            return 0u; /* an exact zero difference is +0 */
        while (significand < WORK_LEAD && exp_a > 1) {
            significand <<= 1;
            exp_a--;
        }
    }
    return round_and_pack(sign_a, exp_a, significand);
}

static uint32_t mul_bits(uint32_t a, uint32_t b)
{
    uint32_t sign = (a ^ b) & SIGN_BIT;
    int32_t exp_a;
    int32_t exp_b;
    int32_t exponent;
    uint64_t product;
    uint32_t significand;

    if (is_nan(a) || is_nan(b))
        return CANONICAL_NAN;
    if (is_inf(a) || is_inf(b)) {
        if ((a & ABS_MASK) == 0u || (b & ABS_MASK) == 0u)
            return CANONICAL_NAN;
        return sign | INF_BITS;
    }
    if ((a & ABS_MASK) == 0u || (b & ABS_MASK) == 0u)
        return sign;
    product = (uint64_t)unpack_normalized(a, &exp_a) * unpack_normalized(b, &exp_b);
    exponent = exp_a + exp_b - 127;
    /* product is in [2^46, 2^48): put its leading 1 at bit 47. */
    if (product & ((uint64_t)1 << 47))
        exponent++;
    else
        product <<= 1;
    significand = (uint32_t)(product >> 21) | ((product & ((1u << 21) - 1u)) != 0u);
    significand = denormalize(&exponent, significand);
    return round_and_pack(sign, exponent, significand);
}

static uint32_t div_bits(uint32_t a, uint32_t b)
{
    uint32_t sign = (a ^ b) & SIGN_BIT;
    int32_t exp_a;
    int32_t exp_b;
    int32_t exponent;
    uint32_t sig_a;
    uint32_t sig_b;
    uint32_t quotient = 0u;

    if (is_nan(a) || is_nan(b))
        return CANONICAL_NAN;
    if (is_inf(a))
        return is_inf(b) ? CANONICAL_NAN : (sign | INF_BITS);
    if (is_inf(b))
        return sign;
    if ((b & ABS_MASK) == 0u)
        return (a & ABS_MASK) == 0u ? CANONICAL_NAN : (sign | INF_BITS);
    if ((a & ABS_MASK) == 0u)
        return sign;
    sig_a = unpack_normalized(a, &exp_a);
    sig_b = unpack_normalized(b, &exp_b);
    exponent = exp_a - exp_b + 127;
    if (sig_a < sig_b) {
        sig_a <<= 1;
        exponent--;
    }
    /* sig_a / sig_b is in [1, 2): 27 quotient bits by restoring division, 32-bit only. */
    for (int bit = 0; bit < 27; bit++) {
        quotient <<= 1;
        if (sig_a >= sig_b) {
            sig_a -= sig_b;
            quotient |= 1u;
        }
        sig_a <<= 1;
    }
    quotient |= (sig_a != 0u);
    quotient = denormalize(&exponent, quotient);
    return round_and_pack(sign, exponent, quotient);
}

/** -1 / 0 / 1 for ordered operands, `unordered` when either is NaN. */
static int compare_bits(uint32_t a, uint32_t b, int unordered)
{
#ifdef FT_SOFTFLOAT_RED_COMPARE_INVERTED /* card red: every ordered compare answers backwards */
    uint32_t t = a;
    a = b;
    b = t;
#endif
    if (is_nan(a) || is_nan(b))
        return unordered;
    if (((a | b) & ABS_MASK) == 0u)
        return 0;
    if ((a & SIGN_BIT) != (b & SIGN_BIT))
        return (a & SIGN_BIT) ? -1 : 1;
    if (a == b)
        return 0;
    if (a & SIGN_BIT)
        return (a > b) ? -1 : 1;
    return (a < b) ? -1 : 1;
}

/** Leading-bit position of a non-zero value, without a builtin. */
static int32_t leading_bit(uint32_t value)
{
    int32_t position = 31;

    while (!(value & (1u << position)))
        position--;
    return position;
}

static uint32_t from_magnitude(uint32_t sign, uint32_t magnitude)
{
    int32_t position;
    uint32_t significand;

    if (magnitude == 0u)
        return 0u;
    position = leading_bit(magnitude);
    if (position > 26)
        significand = shift_right_sticky(magnitude, (uint32_t)(position - 26));
    else
        significand = magnitude << (26 - position);
    return round_and_pack(sign, 127 + position, significand);
}

float FT_SF(__addsf3)(float a, float b);
float FT_SF(__subsf3)(float a, float b);
float FT_SF(__mulsf3)(float a, float b);
float FT_SF(__divsf3)(float a, float b);
int FT_SF(__eqsf2)(float a, float b);
int FT_SF(__nesf2)(float a, float b);
int FT_SF(__ltsf2)(float a, float b);
int FT_SF(__lesf2)(float a, float b);
int FT_SF(__gtsf2)(float a, float b);
int FT_SF(__gesf2)(float a, float b);
int FT_SF(__unordsf2)(float a, float b);
int32_t FT_SF(__fixsfsi)(float a);
float FT_SF(__floatsisf)(int32_t a);
float FT_SF(__floatunsisf)(uint32_t a);

float FT_SF(__addsf3)(float a, float b)
{
    return float_of(add_bits(bits_of(a), bits_of(b)));
}

float FT_SF(__subsf3)(float a, float b)
{
    return float_of(add_bits(bits_of(a), bits_of(b) ^ SIGN_BIT));
}

float FT_SF(__mulsf3)(float a, float b)
{
    return float_of(mul_bits(bits_of(a), bits_of(b)));
}

float FT_SF(__divsf3)(float a, float b)
{
    return float_of(div_bits(bits_of(a), bits_of(b)));
}

int FT_SF(__eqsf2)(float a, float b)
{
    return compare_bits(bits_of(a), bits_of(b), 1);
}

int FT_SF(__nesf2)(float a, float b)
{
    return compare_bits(bits_of(a), bits_of(b), 1);
}

int FT_SF(__ltsf2)(float a, float b)
{
    return compare_bits(bits_of(a), bits_of(b), 1);
}

int FT_SF(__lesf2)(float a, float b)
{
    return compare_bits(bits_of(a), bits_of(b), 1);
}

int FT_SF(__gtsf2)(float a, float b)
{
    return compare_bits(bits_of(a), bits_of(b), -1);
}

int FT_SF(__gesf2)(float a, float b)
{
    return compare_bits(bits_of(a), bits_of(b), -1);
}

int FT_SF(__unordsf2)(float a, float b)
{
    return is_nan(bits_of(a)) || is_nan(bits_of(b));
}

int32_t FT_SF(__fixsfsi)(float a)
{
    uint32_t bits = bits_of(a);
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127;
    uint32_t magnitude;

    if (exponent < 0)
        return 0;
    if (exponent >= 31) {
        /* |a| >= 2^31, inf or NaN. -2^31 itself is representable and is INT32_MIN too. */
        return (bits & SIGN_BIT) ? INT32_MIN : INT32_MAX;
    }
    magnitude = (bits & FRACTION_MASK) | HIDDEN_BIT;
    magnitude = exponent >= 23 ? magnitude << (exponent - 23) : magnitude >> (23 - exponent);
    return (bits & SIGN_BIT) ? -(int32_t)magnitude : (int32_t)magnitude;
}

float FT_SF(__floatsisf)(int32_t a)
{
    uint32_t sign = a < 0 ? SIGN_BIT : 0u;
    uint32_t magnitude = a < 0 ? 0u - (uint32_t)a : (uint32_t)a;

    return float_of(from_magnitude(sign, magnitude));
}

float FT_SF(__floatunsisf)(uint32_t a)
{
    return float_of(from_magnitude(0u, a));
}
