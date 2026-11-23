/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Simple configuration file parser.
 * Supports a subset of TOML: [sections], key = value
 */

#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 1024
#define MAX_ENTRIES 256
#define MAX_KEY_LEN 128
#define MAX_VAL_LEN 512

typedef struct {
    char section[MAX_KEY_LEN];
    char key[MAX_KEY_LEN];
    char value[MAX_VAL_LEN];
} config_entry;

struct sw_config {
    config_entry entries[MAX_ENTRIES];
    int count;
};

/* Trim whitespace from both ends */
static char* trim(char* str) {
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;

    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';

    return str;
}

/* Remove quotes from string value */
static void unquote(char* str) {
    size_t len = strlen(str);
    if (len >= 2) {
        if ((str[0] == '"' && str[len-1] == '"') ||
            (str[0] == '\'' && str[len-1] == '\'')) {
            memmove(str, str + 1, len - 2);
            str[len - 2] = '\0';
        }
    }
}

sw_config* sw_config_load(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;

    sw_config* cfg = calloc(1, sizeof(sw_config));
    if (!cfg) {
        fclose(f);
        return NULL;
    }

    char line[MAX_LINE];
    char current_section[MAX_KEY_LEN] = "";

    while (fgets(line, sizeof(line), f)) {
        char* trimmed = trim(line);

        /* Skip empty lines and comments */
        if (*trimmed == '\0' || *trimmed == '#') {
            continue;
        }

        /* Section header: [section] */
        if (*trimmed == '[') {
            char* end = strchr(trimmed, ']');
            if (end) {
                *end = '\0';
                strncpy(current_section, trimmed + 1, MAX_KEY_LEN - 1);
                current_section[MAX_KEY_LEN - 1] = '\0';
            }
            continue;
        }

        /* Key = value */
        char* eq = strchr(trimmed, '=');
        if (eq && cfg->count < MAX_ENTRIES) {
            *eq = '\0';
            char* key = trim(trimmed);
            char* value = trim(eq + 1);

            /* Remove quotes from value */
            unquote(value);

            config_entry* e = &cfg->entries[cfg->count++];
            strncpy(e->section, current_section, MAX_KEY_LEN - 1);
            strncpy(e->key, key, MAX_KEY_LEN - 1);
            strncpy(e->value, value, MAX_VAL_LEN - 1);
        }
    }

    fclose(f);
    return cfg;
}

void sw_config_free(sw_config* cfg) {
    free(cfg);
}

static const char* find_value(sw_config* cfg, const char* section, const char* key) {
    if (!cfg) return NULL;

    for (int i = 0; i < cfg->count; i++) {
        if (strcmp(cfg->entries[i].section, section) == 0 &&
            strcmp(cfg->entries[i].key, key) == 0) {
            return cfg->entries[i].value;
        }
    }
    return NULL;
}

const char* sw_config_get_str(sw_config* cfg, const char* section,
                               const char* key, const char* default_val) {
    const char* val = find_value(cfg, section, key);
    return val ? val : default_val;
}

int sw_config_get_int(sw_config* cfg, const char* section,
                       const char* key, int default_val) {
    const char* val = find_value(cfg, section, key);
    if (!val) return default_val;
    return atoi(val);
}

float sw_config_get_float(sw_config* cfg, const char* section,
                           const char* key, float default_val) {
    const char* val = find_value(cfg, section, key);
    if (!val) return default_val;
    return (float)atof(val);
}

bool sw_config_get_bool(sw_config* cfg, const char* section,
                         const char* key, bool default_val) {
    const char* val = find_value(cfg, section, key);
    if (!val) return default_val;

    /* Check for true values */
    if (strcmp(val, "true") == 0 ||
        strcmp(val, "True") == 0 ||
        strcmp(val, "TRUE") == 0 ||
        strcmp(val, "yes") == 0 ||
        strcmp(val, "1") == 0) {
        return true;
    }

    /* Check for false values */
    if (strcmp(val, "false") == 0 ||
        strcmp(val, "False") == 0 ||
        strcmp(val, "FALSE") == 0 ||
        strcmp(val, "no") == 0 ||
        strcmp(val, "0") == 0) {
        return false;
    }

    return default_val;
}

bool sw_config_has_key(sw_config* cfg, const char* section, const char* key) {
    return find_value(cfg, section, key) != NULL;
}
