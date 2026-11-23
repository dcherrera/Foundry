/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_softfloat.c
 * @brief Proves fw/ft_softfloat.c against the host's hardware binary32 (round to nearest
 *   even, no flush-to-zero): every operation on edge-value pairs, many millions of random
 *   and near-equal-exponent pairs, and int -> float over a stride of all 2^32 inputs
 *   (FULL=1 in the environment: every input). NaN results must be exactly 0x7FC00000.
 *
 *   make -f Makefile.softfloat check     (red: make -f Makefile.softfloat red)
 *
 * @status Added 2026-10-08.
 * @issues None
 * @todo None
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float ft_test___addsf3(float a, float b);
float ft_test___subsf3(float a, float b);
float ft_test___mulsf3(float a, float b);
float ft_test___divsf3(float a, float b);
int ft_test___eqsf2(float a, float b);
int ft_test___nesf2(float a, float b);
int ft_test___ltsf2(float a, float b);
int ft_test___lesf2(float a, float b);
int ft_test___gtsf2(float a, float b);
int ft_test___gesf2(float a, float b);
int32_t ft_test___fixsfsi(float a);
float ft_test___floatsisf(int32_t a);
float ft_test___floatunsisf(uint32_t a);

static uint64_t checks;
static uint64_t failures;

static uint32_t bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static float flt(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

static void report(const char *what, uint32_t a, uint32_t b, uint32_t got, uint32_t want)
{
    if (failures++ < 20)
        printf("  FAIL  %s(%08x, %08x) = %08x, hardware %08x\n", what, a, b, got, want);
}

/** Same bits, or both NaN with ours canonical. */
static void same(const char *what, uint32_t a, uint32_t b, float got, float want)
{
    uint32_t g = bits(got), w = bits(want);
    checks++;
    if (want != want) {
        if (g != 0x7FC00000u)
            report(what, a, b, g, w);
    } else if (g != w) {
        report(what, a, b, g, w);
    }
}

static volatile float sink_a, sink_b; /* keep the hardware op at run time */

static void pair(uint32_t a_bits, uint32_t b_bits)
{
    float a, b;
    sink_a = flt(a_bits);
    sink_b = flt(b_bits);
    a = sink_a;
    b = sink_b;
    same("add", a_bits, b_bits, ft_test___addsf3(a, b), a + b);
    same("sub", a_bits, b_bits, ft_test___subsf3(a, b), a - b);
    same("mul", a_bits, b_bits, ft_test___mulsf3(a, b), a * b);
    same("div", a_bits, b_bits, ft_test___divsf3(a, b), a / b);
    checks += 6;
    if ((ft_test___eqsf2(a, b) == 0) != (a == b)) report("eq", a_bits, b_bits, 0, 0);
    if ((ft_test___nesf2(a, b) != 0) != (a != b)) report("ne", a_bits, b_bits, 0, 0);
    if ((ft_test___ltsf2(a, b) < 0) != (a < b)) report("lt", a_bits, b_bits, 0, 0);
    if ((ft_test___lesf2(a, b) <= 0) != (a <= b)) report("le", a_bits, b_bits, 0, 0);
    if ((ft_test___gtsf2(a, b) > 0) != (a > b)) report("gt", a_bits, b_bits, 0, 0);
    if ((ft_test___gesf2(a, b) >= 0) != (a >= b)) report("ge", a_bits, b_bits, 0, 0);
}

static void to_int(uint32_t a_bits)
{
    float a = flt(a_bits);
    int32_t got = ft_test___fixsfsi(a);
    int32_t want;
    checks++;
    if (a != a)
        want = (a_bits & 0x80000000u) ? INT32_MIN : INT32_MAX;
    else if (a >= 2147483648.0f)
        want = INT32_MAX;
    else if (a < -2147483648.0f)
        want = INT32_MIN;
    else
        want = (int32_t)a;
    if (got != want)
        report("fix", a_bits, 0, (uint32_t)got, (uint32_t)want);
}

static uint64_t state = 0x243F6A8885A308D3ull;
static uint32_t next(void)
{
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(state >> 32);
}

int main(void)
{
    static const uint32_t edges[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x00000002u, 0x007FFFFFu,
        0x807FFFFFu, 0x00800000u, 0x80800000u, 0x00800001u, 0x00FFFFFFu, 0x3F800000u,
        0xBF800000u, 0x3F800001u, 0x3F7FFFFFu, 0x40000000u, 0x3F000000u, 0x33800000u,
        0x34000000u, 0x4B000000u, 0x4B7FFFFFu, 0x4F000000u, 0xCF000000u, 0x4EFFFFFFu,
        0x7F000000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
        0xFFC00000u, 0x7F800001u, 0x7FBFFFFFu, 0x1F800000u, 0x5F800000u, 0x20000000u,
    };
    const size_t edge_count = sizeof(edges) / sizeof(edges[0]);
    const char *full = getenv("FULL");
    uint64_t stride = (full && *full == '1') ? 1u : 4099u;

    for (size_t i = 0; i < edge_count; i++) {
        to_int(edges[i]);
        for (size_t j = 0; j < edge_count; j++)
            pair(edges[i], edges[j]);
    }
    for (uint32_t n = 0; n < 20000000u; n++) {
        uint32_t a = next(), b = next();
        switch (n % 4u) {
        case 0: pair(a, b); break;
        case 1: pair(a, (b & 0x80FFFFFFu) | (a & 0x7F000000u)); break; /* close exponents */
        case 2: pair(a & 0x80FFFFFFu, b & 0x80FFFFFFu); break;          /* subnormal-heavy */
        default: pair(a, a ^ (b & 0x8000001Fu)); break;                  /* near-cancellation */
        }
        to_int(a);
    }
    for (uint64_t v = 0; v <= 0xFFFFFFFFull; v += stride) {
        uint32_t u = (uint32_t)v;
        same("floatsi", u, 0, ft_test___floatsisf((int32_t)u), (float)(int32_t)u);
        same("floatunsi", u, 0, ft_test___floatunsisf(u), (float)u);
        to_int(u);
    }
    printf("%llu checks, %llu failed\n", (unsigned long long)checks,
           (unsigned long long)failures);
    return failures != 0;
}
