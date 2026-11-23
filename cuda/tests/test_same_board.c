/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_same_board.c
 * @brief fc_cuda_devices_same_board over every pair of visible dies.
 * @status complete
 * @issues none
 * @todo none
 *
 * Prints the same-board matrix of the visible devices and gates that it is
 * reflexive and symmetric. FC_SAME_BOARD_PAIRS="01,23" (device indices as the
 * process sees them) additionally gates the exact matrix: listed pairs on one
 * board, every other pair not. On a two-board K80 host with CUDA_DEVICE_ORDER=PCI_BUS_ID
 * CUDA_VISIBLE_DEVICES=8,9,10,11 the two K80 boards are "01,23".
 *
 * Build: make USE_CUDA=1 test-same-board
 */

#include "lib/test.h"
#include "cuda/fc_cuda.h"

#include <string.h>

#define MAXD 16

static int expected(const char *pairs, int i, int j)
{
    if (i == j)
        return 1;
    for (const char *p = pairs; p[0] && p[1]; p += 3) {
        int a = p[0] - '0', b = p[1] - '0';
        if ((a == i && b == j) || (a == j && b == i))
            return 1;
        if (!p[2])
            break;
    }
    return 0;
}

static void test_matrix(void)
{
    const char *pairs = getenv("FC_SAME_BOARD_PAIRS");
    int n = 1;

    while (n < MAXD && fc_cuda_devices_same_board(n, 0) >= 0)   /* -1: no such device */
        n++;
    printf("    %d dies\n", n);
    for (int i = 0; i < n; i++) {
        printf("    d%d:", i);
        for (int j = 0; j < n; j++) {
            int s = fc_cuda_devices_same_board(i, j);
            printf(" %2d", s);
            ASSERT_TRUE(s == fc_cuda_devices_same_board(j, i));
            if (i == j)
                ASSERT_TRUE(s == 1);
            if (pairs)
                ASSERT_TRUE(s == expected(pairs, i, j));
        }
        printf("\n");
    }
}

int main(void)
{
    printf("Same-board Tests\n");
    printf("================\n");
    if (fc_init() != 0) {
        printf("  [SKIP] no CUDA device\n");
        TEST_REPORT();
        return 0;
    }
    RUN_TEST(matrix);
    fc_free();
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
