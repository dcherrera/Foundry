# safetensors/

C implementation of the safetensors model format.

## Python counterpart

**safetensors** - https://github.com/huggingface/safetensors

## Why safetensors?

safetensors is the modern standard for storing model weights:

- **Safe** - No arbitrary code execution (unlike pickle)
- **Fast** - Memory-mapped loading, zero-copy when possible
- **Simple** - JSON header + raw tensor data
- **Portable** - Works across frameworks

Alternatives (pickle/torch.save) are unsafe and Python-specific.

## Scope

This module implements:

1. **Reading** - Load tensors from .safetensors files
2. **Writing** - Save tensors to .safetensors files
3. **Memory mapping** - Zero-copy loading for large models
4. **Metadata** - Read/write custom metadata
5. **PyTorch .pth loading** - For compatibility with existing models

## Getting Started

### Dependencies

- tensor/ module (tensor data structure)
- JSON parser (cJSON or similar)

### Key Files to Study

From safetensors:
- `safetensors/src/lib.rs` - Core Rust implementation
- Format spec in README

Format:
```
[8 bytes: header_size (little-endian u64)]
[header_size bytes: JSON header]
[remainder: raw tensor data]
```

### Core API

> Original design sketch, not the implemented API. The real API is `safetensors/safetensors.h`; see
> [`docs/safetensors.md`](../docs/safetensors.md).

```c
// Load all tensors from file
fs_file* fs_open(const char* path);
int fs_num_tensors(fs_file* f);
const char** fs_tensor_names(fs_file* f);
tm_tensor* fs_get_tensor(fs_file* f, const char* name);
void fs_close(fs_file* f);

// Save tensors to file
fs_writer* fs_writer_new(const char* path);
void fs_add_tensor(fs_writer* w, const char* name, tm_tensor* t);
void fs_set_metadata(fs_writer* w, const char* key, const char* value);
void fs_write(fs_writer* w);

// Memory-mapped loading (zero-copy)
fs_file* fs_mmap(const char* path);

// PyTorch compatibility
fs_file* fs_load_pytorch(const char* path);  // .pth/.pt files
```

## File Format

```
┌─────────────────────────────────────────┐
│ Header Size (8 bytes, u64 LE)           │
├─────────────────────────────────────────┤
│ JSON Header                             │
│ {                                       │
│   "tensor_name": {                      │
│     "dtype": "F32",                     │
│     "shape": [4096, 4096],              │
│     "data_offsets": [0, 67108864]       │
│   },                                    │
│   ...                                   │
│   "__metadata__": {"format": "pt"}      │
│ }                                       │
├─────────────────────────────────────────┤
│ Raw tensor data (concatenated)          │
│ [tensor1 bytes][tensor2 bytes]...       │
└─────────────────────────────────────────┘
```

## Implementation Order

1. JSON header parsing
2. Read tensor data
3. Write tensors
4. Memory-mapped reading
5. Metadata support
6. PyTorch .pth loading (pickle parsing)

## References

- [safetensors format](https://github.com/huggingface/safetensors)
- [safetensors spec](https://huggingface.co/docs/safetensors/index)
