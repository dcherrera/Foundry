/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Minimal test framework for Foundry
 * Inspired by JACLibc's testing.h but standalone
 */

#ifndef FOUNDRY_TEST_H
#define FOUNDRY_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Test state */
static int __ft_total = 0;
static int __ft_passed = 0;
static int __ft_failed = 0;
static int __ft_current_failed = 0;

/* Colors */
#define FT_GREEN  "\x1b[92m"
#define FT_RED    "\x1b[31m"
#define FT_YELLOW "\x1b[93m"
#define FT_RESET  "\x1b[0m"

/* Test runner macro */
#define RUN_TEST(name) do { \
    __ft_total++; \
    __ft_current_failed = 0; \
    test_##name(); \
    if (__ft_current_failed == 0) { \
        __ft_passed++; \
        printf("  %s%-50s%s%s[PASS]%s\n", FT_YELLOW, #name, FT_RESET, FT_GREEN, FT_RESET); \
    } \
} while(0)

/* Assertions */
#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected true: %s\n", __FILE__, __LINE__, #x); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))

#define ASSERT_NOT_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected non-NULL: %s\n", __FILE__, __LINE__, #ptr); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_NULL(ptr) do { \
    if ((ptr) != NULL) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected NULL: %s\n", __FILE__, __LINE__, #ptr); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_INT_EQ(a, b) do { \
    long __a = (long)(a); \
    long __b = (long)(b); \
    if (__a != __b) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected %ld == %ld\n", __FILE__, __LINE__, __a, __b); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_INT_NE(a, b) do { \
    long __a = (long)(a); \
    long __b = (long)(b); \
    if (__a == __b) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected %ld != %ld\n", __FILE__, __LINE__, __a, __b); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_FLT_NEAR(a, b, tol) do { \
    float __a = (float)(a); \
    float __b = (float)(b); \
    float __diff = __a > __b ? __a - __b : __b - __a; \
    if (__diff > (float)(tol)) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected %f ≈ %f (diff: %f, tol: %f)\n", __FILE__, __LINE__, __a, __b, __diff, (float)(tol)); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_FLT_EQ(a, b) ASSERT_FLT_NEAR(a, b, 0.0001f)

#define ASSERT_DBL_EQ(a, b) do { \
    double __a = (double)(a); \
    double __b = (double)(b); \
    if (__a != __b) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected %f == %f\n", __FILE__, __LINE__, __a, __b); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_DBL_GT(a, b) do { \
    double __a = (double)(a); \
    double __b = (double)(b); \
    if (!(__a > __b)) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected %f > %f\n", __FILE__, __LINE__, __a, __b); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_DBL_LT(a, b) do { \
    double __a = (double)(a); \
    double __b = (double)(b); \
    if (!(__a < __b)) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected %f < %f\n", __FILE__, __LINE__, __a, __b); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
    const char* __a = (a); \
    const char* __b = (b); \
    if (__a == NULL || __b == NULL || strcmp(__a, __b) != 0) { \
        if (__ft_current_failed == 0) { \
            printf("  %s%-50s%s%s[FAIL]%s\n", FT_YELLOW, __func__ + 5, FT_RESET, FT_RED, FT_RESET); \
            __ft_failed++; \
        } \
        printf("    %s:%d - Expected \"%s\" == \"%s\"\n", __FILE__, __LINE__, __a ? __a : "(null)", __b ? __b : "(null)"); \
        __ft_current_failed = 1; \
        return; \
    } \
} while(0)

/* Test report */
#define TEST_REPORT() do { \
    printf("\n%s\n", __ft_failed == 0 \
        ? FT_GREEN "  ALL TESTS PASSED" FT_RESET \
        : FT_RED "  SOME TESTS FAILED" FT_RESET); \
    printf("\n  %d total   %s%d passed%s   %s%d failed%s\n\n", \
        __ft_total, \
        FT_GREEN, __ft_passed, FT_RESET, \
        FT_RED, __ft_failed, FT_RESET); \
} while(0)

#endif /* FOUNDRY_TEST_H */
