# data

Dataset loading and batching. Python counterpart: HuggingFace Datasets.

**Header:** `data/data.h`
**Source:** `data/data.c`
**Status:** API Complete - Pending Verification

---

## Overview

The data module provides:
- JSONL file loading
- Sample manipulation
- Dataset shuffling and splitting
- DataLoader with batching

---

## Quick Start

```c
#include "data/data.h"

// Load dataset from JSONL
fd_dataset* ds = fd_load_jsonl("train.jsonl");
printf("Loaded %d samples\n", fd_len(ds));

// Create data loader
fd_loader* loader = fd_loader_new(ds, 32, true);  // batch_size=32, shuffle=true

// Iterate batches
fd_batch* batch;
while ((batch = fd_loader_next(loader)) != NULL) {
    for (int i = 0; i < batch->size; i++) {
        const char* text = fd_sample_get(batch->samples[i], "text");
        // ... process sample ...
    }
    fd_batch_free(batch);
}

// Reset for next epoch
fd_loader_reset(loader);

fd_loader_free(loader);
fd_free(ds);
```

---

## API Reference

### Dataset Loading

```c
fd_dataset* fd_load_jsonl(const char* path);
fd_dataset* fd_new(void);
void fd_free(fd_dataset* ds);
```

### Dataset Operations

```c
void fd_add_sample(fd_dataset* ds, fd_sample* sample);
int fd_len(fd_dataset* ds);
fd_sample* fd_get(fd_dataset* ds, int idx);
void fd_shuffle(fd_dataset* ds);
void fd_split(fd_dataset* ds, float ratio, fd_dataset** train, fd_dataset** test);
```

### Sample Operations

```c
fd_sample* fd_sample_new(void);
void fd_sample_free(fd_sample* s);
void fd_sample_set(fd_sample* s, const char* key, const char* value);
const char* fd_sample_get(fd_sample* s, const char* key);
bool fd_sample_has(fd_sample* s, const char* key);
fd_sample* fd_sample_clone(fd_sample* s);
```

### DataLoader

```c
fd_loader* fd_loader_new(fd_dataset* ds, int batch_size, bool shuffle);
void fd_loader_free(fd_loader* loader);
void fd_loader_reset(fd_loader* loader);
fd_batch* fd_loader_next(fd_loader* loader);  // NULL when exhausted
void fd_batch_free(fd_batch* batch);
int fd_loader_num_batches(fd_loader* loader);
```

---

## See Also

- [tokenizer](tokenizer.md) - Text tokenization
- [eval](eval.md) - Evaluation tasks
