/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Wrapper: include real cdefs.h then fix glibc 2.39 vs nvcc 11.4 */
#include_next <sys/cdefs.h>

/* Override the extended __malloc__(f,i) form that nvcc 11.4 cannot parse */
#undef __attr_dealloc
#define __attr_dealloc(dealloc, argno)
#undef __attr_dealloc_free
#define __attr_dealloc_free
