/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Simple configuration file parser.
 * Supports a subset of TOML: sections, key=value pairs.
 */

#ifndef FOUNDRY_SWARM_CONFIG_H
#define FOUNDRY_SWARM_CONFIG_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration handle */
typedef struct sw_config sw_config;

/* Load configuration from file */
sw_config* sw_config_load(const char* path);

/* Free configuration */
void sw_config_free(sw_config* cfg);

/* Get string value (returns default if not found) */
const char* sw_config_get_str(sw_config* cfg, const char* section,
                               const char* key, const char* default_val);

/* Get integer value (returns default if not found) */
int sw_config_get_int(sw_config* cfg, const char* section,
                       const char* key, int default_val);

/* Get float value (returns default if not found) */
float sw_config_get_float(sw_config* cfg, const char* section,
                           const char* key, float default_val);

/* Get boolean value (returns default if not found) */
bool sw_config_get_bool(sw_config* cfg, const char* section,
                         const char* key, bool default_val);

/* Check if key exists */
bool sw_config_has_key(sw_config* cfg, const char* section, const char* key);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_SWARM_CONFIG_H */
