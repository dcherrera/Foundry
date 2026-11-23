/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_dataflow.h
 * @brief The one header a data-movement kernel (BRISC or NCRISC) includes.
 *
 * It defines nothing; it pulls in Foundry's C99 data-movement library:
 *   ../fw/ft_bh_regs.h           Blackhole register map and access primitives
 *   ft_tensix_circular_buffer.h  CB records and data-movement CB flow control
 *   ft_tensix_kernel_args.h      runtime and compile-time arguments
 *   ft_tensix_noc.h              unicast NoC transfers, atomics, counter barriers
 *   ft_tensix_transfer_guard.h   pre-issue refusal predicates
 * A kernel that includes it includes nothing from any vendor tree. Acceptance of the library
 * is an instruction-stream diff of the built images against Foundry's reference set.
 *
 * The include order is fixed: it sets the order in which the plain `static inline` functions
 * are defined in every kernel, which can decide the order of their out-of-line copies.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_DATAFLOW_H
#define FT_TENSIX_DATAFLOW_H

#include "../fw/ft_bh_regs.h"
#include "ft_tensix_circular_buffer.h"
#include "ft_tensix_kernel_args.h"
#include "ft_tensix_noc.h"
#include "ft_tensix_transfer_guard.h"

#endif /* FT_TENSIX_DATAFLOW_H */
