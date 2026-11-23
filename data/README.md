# data/

C implementation of dataset loading and processing.

## Python counterpart

**datasets** - https://github.com/huggingface/datasets

## Why datasets?

HuggingFace datasets provides:

- Unified dataset loading (local files, HuggingFace Hub)
- Memory-mapped files for large datasets
- Streaming for datasets that don't fit in memory
- Data transforms and filtering

Built on Apache Arrow for efficient columnar storage.

## Scope

This module implements:

1. **JSONL loading** - Line-delimited JSON (most common format)
2. **Memory mapping** - Handle datasets larger than RAM
3. **Batching** - Create training batches
4. **Shuffling** - Randomize data order
5. **binidx format** - Pre-tokenized binary format (fast loading)

## Getting Started

### Dependencies

- tokenizer/ module (for on-the-fly tokenization)
- Optional: mmap support (POSIX)

### Key Files to Study

From datasets:
- `src/datasets/arrow_dataset.py` - Core dataset class
- `src/datasets/iterable_dataset.py` - Streaming datasets

From PyTorch:
- `torch/utils/data/dataloader.py` - DataLoader pattern

### Core API

> Original design sketch, not the implemented API. The real API is `data/data.h`; see
> [`docs/data.md`](../docs/data.md).

```c
// Load dataset
fd_dataset* fd_load_jsonl(const char* path);
fd_dataset* fd_load_binidx(const char* path);  // Pre-tokenized

// Dataset info
int64_t fd_len(fd_dataset* ds);
tm_tensor* fd_get(fd_dataset* ds, int64_t index);

// DataLoader
fd_dataloader* fd_dataloader_new(fd_dataset* ds, int batch_size, bool shuffle);
tm_tensor* fd_next_batch(fd_dataloader* dl);  // Returns NULL at end
void fd_reset(fd_dataloader* dl);

// Transforms
fd_dataset* fd_map(fd_dataset* ds, tm_tensor* (*fn)(tm_tensor*));
fd_dataset* fd_filter(fd_dataset* ds, bool (*fn)(tm_tensor*));
fd_dataset* fd_shuffle(fd_dataset* ds, int seed);

// Memory-mapped loading
fd_dataset* fd_load_mmap(const char* path);
```

## Implementation Order

1. JSONL parser
2. In-memory dataset
3. Batching and iteration
4. Shuffling
5. Memory-mapped loading
6. binidx format (pre-tokenized)
7. Streaming for very large datasets

## References

- [HuggingFace datasets](https://huggingface.co/docs/datasets)
- [Apache Arrow](https://arrow.apache.org/)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)
