# safetensors

Safe tensor serialization format. Python counterpart: HuggingFace Safetensors.

**Header:** `safetensors/safetensors.h`
**Source:** `safetensors/safetensors.c`
**Status:** API Complete - Pending Verification

---

## Overview

The safetensors module provides:
- Memory-mapped file reading (zero-copy)
- All standard data types
- Metadata support
- Writing new safetensors files

---

## Quick Start

```c
#include "safetensors/safetensors.h"

// Open file (memory-mapped)
fs_file* f = fs_open("model.safetensors");

// List tensors
int count;
const char** names = fs_tensor_names(f, &count);
for (int i = 0; i < count; i++) {
    printf("Tensor: %s\n", names[i]);
}
free(names);

// Get tensor data (zero-copy)
const float* weights = fs_get_f32(f, "model.embed.weight");

// Get tensor info
fs_tensor_info* info = fs_get_info(f, "model.embed.weight");
printf("Shape: [%lld, %lld]\n", info->shape[0], info->shape[1]);
fs_free_info(info);

// Close file
fs_close(f);
```

---

## API Reference

### Reading

```c
fs_file* fs_open(const char* path);                    // Open file (mmap)
fs_file* fs_open_buffer(const void* data, size_t size); // Open from memory
void fs_close(fs_file* f);

int fs_num_tensors(fs_file* f);
const char** fs_tensor_names(fs_file* f, int* count);
bool fs_has_tensor(fs_file* f, const char* name);

fs_tensor_info* fs_get_info(fs_file* f, const char* name);
void fs_free_info(fs_tensor_info* info);

const void* fs_get_data(fs_file* f, const char* name);  // Raw data
const float* fs_get_f32(fs_file* f, const char* name);  // Type-checked
const double* fs_get_f64(fs_file* f, const char* name);
```

### Writing

```c
fs_writer* fs_writer_new(void);
void fs_writer_free(fs_writer* w);

void fs_writer_add_metadata(fs_writer* w, const char* key, const char* value);
void fs_writer_add_tensor(fs_writer* w, const char* name, fs_dtype dtype,
                          const int64_t* shape, int ndim, const void* data);
void fs_writer_add_f32(fs_writer* w, const char* name,
                       const int64_t* shape, int ndim, const float* data);

bool fs_writer_save(fs_writer* w, const char* path);
void* fs_writer_to_buffer(fs_writer* w, size_t* size);
```

### Data Types

| Type | Constant | Description |
|------|----------|-------------|
| F32 | `FS_DTYPE_F32` | 32-bit float |
| F64 | `FS_DTYPE_F64` | 64-bit float |
| F16 | `FS_DTYPE_F16` | 16-bit float |
| BF16 | `FS_DTYPE_BF16` | bfloat16 |
| I8-I64 | `FS_DTYPE_I8/16/32/64` | Signed integers |
| U8-U64 | `FS_DTYPE_U8/16/32/64` | Unsigned integers |
| BOOL | `FS_DTYPE_BOOL` | Boolean |

---

## See Also

- [tensor](tensor.md) - Tensor operations
- [lora](lora.md) - LoRA adapter save/load
