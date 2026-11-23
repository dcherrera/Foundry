/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "progress.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define FP_DEFAULT_WIDTH 40
#define FP_POSTFIX_MAX 256

/* Unicode block characters for smooth progress bar */
static const char* BLOCKS[] = {" ", "\xe2\x96\x8f", "\xe2\x96\x8e", "\xe2\x96\x8d",
                                "\xe2\x96\x8c", "\xe2\x96\x8b", "\xe2\x96\x8a",
                                "\xe2\x96\x89", "\xe2\x96\x88"};

struct fp_progress {
    int64_t total;
    int64_t current;
    char* desc;
    char postfix[FP_POSTFIX_MAX];
    char* unit;

    double start_time;
    double last_update_time;
    int64_t last_update_n;

    int width;
    bool disabled;
    bool closed;

    fp_progress* parent;
};

/* Get current time in seconds */
static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/* Get terminal width */
static int get_terminal_width(void) {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
        return w.ws_col;
    }
    return 80;
}

/* Format time as HH:MM:SS or MM:SS */
static void format_time(double seconds, char* buf, size_t size) {
    if (seconds < 0 || seconds > 86400 * 365) {
        snprintf(buf, size, "?");
        return;
    }

    int s = (int)seconds;
    int h = s / 3600;
    int m = (s % 3600) / 60;
    s = s % 60;

    if (h > 0) {
        snprintf(buf, size, "%d:%02d:%02d", h, m, s);
    } else {
        snprintf(buf, size, "%02d:%02d", m, s);
    }
}

/* Format rate with appropriate unit */
static void format_rate(double rate, const char* unit, char* buf, size_t size) {
    if (rate < 0.01) {
        snprintf(buf, size, "%.2e %s/s", rate, unit ? unit : "it");
    } else if (rate < 1.0) {
        snprintf(buf, size, "%.2f %s/s", rate, unit ? unit : "it");
    } else if (rate < 1000.0) {
        snprintf(buf, size, "%.1f %s/s", rate, unit ? unit : "it");
    } else {
        snprintf(buf, size, "%.2fk %s/s", rate / 1000.0, unit ? unit : "it");
    }
}

fp_progress* fp_new(int64_t total, const char* desc) {
    fp_progress* p = calloc(1, sizeof(fp_progress));
    if (!p) return NULL;

    p->total = total;
    p->current = 0;
    p->width = FP_DEFAULT_WIDTH;
    p->start_time = get_time();
    p->last_update_time = p->start_time;
    p->last_update_n = 0;
    p->disabled = false;
    p->closed = false;
    p->parent = NULL;
    p->postfix[0] = '\0';

    if (desc) {
        p->desc = strdup(desc);
    }

    p->unit = strdup("it");

    return p;
}

fp_progress* fp_new_nested(fp_progress* parent, int64_t total, const char* desc) {
    fp_progress* p = fp_new(total, desc);
    if (p) {
        p->parent = parent;
    }
    return p;
}

void fp_free(fp_progress* p) {
    if (!p) return;
    if (p->desc) free(p->desc);
    if (p->unit) free(p->unit);
    free(p);
}

void fp_update(fp_progress* p, int64_t n) {
    if (!p || p->closed) return;

    p->current += n;
    if (p->current > p->total && p->total > 0) {
        p->current = p->total;
    }

    double now = get_time();

    /* Throttle updates to avoid excessive terminal writes */
    if (now - p->last_update_time >= 0.1 || p->current >= p->total) {
        p->last_update_time = now;
        p->last_update_n = p->current;

        if (!p->disabled) {
            fp_refresh(p);
        }
    }
}

void fp_set(fp_progress* p, int64_t n) {
    if (!p || p->closed) return;

    int64_t delta = n - p->current;
    if (delta > 0) {
        fp_update(p, delta);
    } else {
        p->current = n;
        if (!p->disabled) {
            fp_refresh(p);
        }
    }
}

void fp_set_postfix(fp_progress* p, const char* fmt, ...) {
    if (!p) return;

    va_list args;
    va_start(args, fmt);
    vsnprintf(p->postfix, FP_POSTFIX_MAX, fmt, args);
    va_end(args);
}

void fp_close(fp_progress* p) {
    if (!p || p->closed) return;

    p->closed = true;

    if (!p->disabled) {
        fp_refresh(p);
        printf("\n");
        fflush(stdout);
    }
}

int64_t fp_total(fp_progress* p) {
    return p ? p->total : 0;
}

int64_t fp_current(fp_progress* p) {
    return p ? p->current : 0;
}

double fp_elapsed(fp_progress* p) {
    if (!p) return 0.0;
    return get_time() - p->start_time;
}

double fp_rate(fp_progress* p) {
    if (!p) return 0.0;
    double elapsed = fp_elapsed(p);
    if (elapsed < 0.001) return 0.0;
    return p->current / elapsed;
}

double fp_eta(fp_progress* p) {
    if (!p || p->total <= 0) return 0.0;
    double rate = fp_rate(p);
    if (rate < 0.001) return 0.0;
    int64_t remaining = p->total - p->current;
    return remaining / rate;
}

float fp_percent(fp_progress* p) {
    if (!p || p->total <= 0) return 0.0f;
    return (float)p->current / (float)p->total * 100.0f;
}

void fp_set_width(fp_progress* p, int width) {
    if (p && width > 0) {
        p->width = width;
    }
}

void fp_set_unit(fp_progress* p, const char* unit) {
    if (!p) return;
    if (p->unit) free(p->unit);
    p->unit = unit ? strdup(unit) : strdup("it");
}

void fp_disable(fp_progress* p) {
    if (p) p->disabled = true;
}

void fp_refresh(fp_progress* p) {
    if (!p || p->disabled) return;

    char line[1024];
    char* ptr = line;
    int remaining = sizeof(line);
    int written;

    /* Clear line and return to start */
    printf("\r\033[K");

    /* Description */
    if (p->desc && p->desc[0]) {
        written = snprintf(ptr, remaining, "%s: ", p->desc);
        ptr += written;
        remaining -= written;
    }

    /* Percentage */
    float pct = fp_percent(p);
    written = snprintf(ptr, remaining, "%3.0f%%|", pct);
    ptr += written;
    remaining -= written;

    /* Progress bar */
    int bar_width = p->width;
    int term_width = get_terminal_width();

    /* Adjust bar width based on available space */
    int overhead = (int)(ptr - line) + 50; /* space for stats */
    if (p->postfix[0]) overhead += (int)strlen(p->postfix) + 2;

    if (bar_width > term_width - overhead) {
        bar_width = term_width - overhead;
    }
    if (bar_width < 10) bar_width = 10;

    /* Calculate filled portion */
    float progress = (p->total > 0) ? (float)p->current / (float)p->total : 0.0f;
    if (progress > 1.0f) progress = 1.0f;

    float filled = progress * bar_width;
    int full_blocks = (int)filled;
    int partial = (int)((filled - full_blocks) * 8);

    for (int i = 0; i < bar_width && remaining > 4; i++) {
        const char* block;
        if (i < full_blocks) {
            block = BLOCKS[8]; /* Full block */
        } else if (i == full_blocks && partial > 0) {
            block = BLOCKS[partial];
        } else {
            block = BLOCKS[0]; /* Space */
        }
        written = snprintf(ptr, remaining, "%s", block);
        ptr += written;
        remaining -= written;
    }

    written = snprintf(ptr, remaining, "| ");
    ptr += written;
    remaining -= written;

    /* Current/Total */
    written = snprintf(ptr, remaining, "%lld/%lld ",
                       (long long)p->current, (long long)p->total);
    ptr += written;
    remaining -= written;

    /* Time elapsed/remaining */
    char elapsed_str[16], eta_str[16];
    format_time(fp_elapsed(p), elapsed_str, sizeof(elapsed_str));
    format_time(fp_eta(p), eta_str, sizeof(eta_str));

    written = snprintf(ptr, remaining, "[%s<%s, ", elapsed_str, eta_str);
    ptr += written;
    remaining -= written;

    /* Rate */
    char rate_str[32];
    format_rate(fp_rate(p), p->unit, rate_str, sizeof(rate_str));
    written = snprintf(ptr, remaining, "%s", rate_str);
    ptr += written;
    remaining -= written;

    /* Postfix */
    if (p->postfix[0]) {
        written = snprintf(ptr, remaining, ", %s", p->postfix);
        ptr += written;
        remaining -= written;
    }

    written = snprintf(ptr, remaining, "]");
    ptr += written;

    printf("%s", line);
    fflush(stdout);
}

void fp_log_step(fp_train_state* state) {
    if (!state) return;
    printf("step %d | loss: %.4f | lr: %.2e | %.1f samples/s\n",
           state->step, state->loss, state->lr, state->throughput);
    fflush(stdout);
}

void fp_log_epoch(fp_train_state* state) {
    if (!state) return;
    printf("epoch %d | loss: %.4f | lr: %.2e | %.1f samples/s\n",
           state->epoch, state->loss, state->lr, state->throughput);
    fflush(stdout);
}
