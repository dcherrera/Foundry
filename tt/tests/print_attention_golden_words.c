/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * print_attention_golden_words.c
 *   Print the sixteen words of every attention_golden_requests.inc request as a C table,
 *   encoded by whichever ft_tt_op_attention.c this is linked against. Built and run by
 *   make_attention_golden_words.sh against another tree's encoder.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_op_attention.h"

#include "attention_golden_requests.inc"

int main(void)
{
    for (uint32_t i = 0; i < GOLDEN_REQUEST_COUNT; i++) {
        ft_tt_op_attention_request request;
        uint32_t words[FT_TT_OP_ATTENTION_WORDS];

        golden_request(i, &request);
        if (ft_tt_op_attention_parameters(&request, words) != FT_TT_OK) {
            fprintf(stderr, "request %u does not encode\n", i);
            return 1;
        }
        printf("    {");
        for (uint32_t w = 0; w < FT_TT_OP_ATTENTION_WORDS; w++)
            printf("%s0x%08Xu", w == 0u ? " " : (w % 6u == 0u ? ",\n      " : ", "), words[w]);
        printf(" },\n");
    }
    return 0;
}
