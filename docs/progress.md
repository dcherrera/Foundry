# progress

Progress bar and training metrics reporting. Python counterpart: tqdm.

**Header:** `progress/progress.h`
**Source:** `progress/progress.c`
**Status:** API Complete - Pending Verification

---

## Overview

The progress module provides:
- Terminal progress bars with ETA
- Training state logging
- Nested progress bars
- Rate calculation

---

## Quick Start

```c
#include "progress/progress.h"

// Create progress bar
fp_progress* pbar = fp_new(1000, "Training");

// Update progress
for (int i = 0; i < 1000; i++) {
    // ... do work ...
    fp_update(pbar, 1);
    fp_set_postfix(pbar, "loss=%.4f", loss);
}

// Finish
fp_close(pbar);
fp_free(pbar);
```

---

## API Reference

### Creation

```c
fp_progress* fp_new(int64_t total, const char* desc);
fp_progress* fp_new_nested(fp_progress* parent, int64_t total, const char* desc);
void fp_free(fp_progress* p);
```

### Updates

```c
void fp_update(fp_progress* p, int64_t n);          // Increment by n
void fp_set(fp_progress* p, int64_t n);             // Set absolute position
void fp_set_postfix(fp_progress* p, const char* fmt, ...);  // Set postfix text
void fp_close(fp_progress* p);                       // Finish and print final
```

### Accessors

```c
int64_t fp_total(fp_progress* p);
int64_t fp_current(fp_progress* p);
double fp_elapsed(fp_progress* p);   // Seconds elapsed
double fp_rate(fp_progress* p);      // Items per second
double fp_eta(fp_progress* p);       // Estimated time remaining
float fp_percent(fp_progress* p);    // Completion percentage
```

### Training Logging

```c
typedef struct {
    int epoch;
    int step;
    float loss;
    float lr;
    float throughput;  // samples/sec or tokens/sec
} fp_train_state;

void fp_log_step(fp_train_state* state);
void fp_log_epoch(fp_train_state* state);
```

### Configuration

```c
void fp_set_width(fp_progress* p, int width);
void fp_set_unit(fp_progress* p, const char* unit);
void fp_disable(fp_progress* p);
void fp_refresh(fp_progress* p);
```

---

## See Also

- [optim](optim.md) - Optimizer for training loops
- [swarm](swarm.md) - Distributed training
