/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_error.c
 * @brief  Error code names.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_error.h"

const char *ingot_error_name(int err) {
    switch (err < 0 ? -err : err) {
    case INGOT_OK: return "ok";
    case INGOT_EINVAL: return "invalid argument";
    case INGOT_ENOMEM: return "out of memory";
    case INGOT_EIO: return "i/o error";
    case INGOT_EFORMAT: return "invalid ingot format";
    case INGOT_ECORRUPT: return "corrupt data (decode check failed)";
    case INGOT_ENOTFOUND: return "tensor not found";
    case INGOT_EBASE: return "delta base missing or mismatched";
    case INGOT_EUNSUPPORTED: return "unsupported";
    case INGOT_ETRUNCATED: return "input ended early (truncated)";
    default: return "unknown error";
    }
}
