/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_error.h
 * @brief  Error codes shared by the Ingot codec. Functions return 0 or a negated code.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#ifndef FOUNDRY_INGOT_ERROR_H
#define FOUNDRY_INGOT_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INGOT_OK = 0,
    INGOT_EINVAL = 1,      /* bad argument */
    INGOT_ENOMEM = 2,      /* allocation failed */
    INGOT_EIO = 3,         /* read, write, mmap or open failed */
    INGOT_EFORMAT = 4,     /* file or payload structure is invalid */
    INGOT_ECORRUPT = 5,    /* decode finished but a check failed (rANS state, hash) */
    INGOT_ENOTFOUND = 6,   /* tensor not present */
    INGOT_EBASE = 7,       /* delta base missing or not the one the file was coded against */
    INGOT_EUNSUPPORTED = 8, /* valid input this version cannot handle */
    INGOT_ETRUNCATED = 9    /* input ended before the data did (e.g. a cut download) */
} ingot_error;

/**
 * Human-readable name of an error code (positive or negated).
 *
 * @param err  Error code
 * @return     Static string, never NULL
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
const char *ingot_error_name(int err);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_ERROR_H */
