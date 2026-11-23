# Architecture

Overview of Foundry's design and architecture.

---

## Design Goals

1. **Zero Python Dependencies** - Pure C, no Python runtime
2. **Minimal External Dependencies** - POSIX APIs, plus opt-in vendor SDKs (CUDA, Metal, BLAS); see NOTICE
3. **Memory Efficient** - No unnecessary copies, mmap support
4. **Embeddable** - Plain C sources; compile in the modules you need (no library target)
5. **Portable** - x86_64 and ARM (Raspberry Pi)

---

## Module Dependency Graph

```
                        ┌─────────┐
                        │ tensor  │ ◄── Foundation
                        └────┬────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │  optim  │        │attention│        │quantize │
    └────┬────┘        └────┬────┘        └─────────┘
         │                   │
         │              ┌────▼────┐
         │              │  lora   │
         │              └─────────┘
         │
    ┌────▼────┐
    │ swarm   │ ◄── Distributed training
    └─────────┘

         ┌─────────┐   ┌─────────┐   ┌─────────┐
         │tokenizer│   │safetens │   │  data   │ ◄── I/O modules
         └─────────┘   └─────────┘   └─────────┘

         ┌─────────┐   ┌─────────┐
         │progress │   │  eval   │ ◄── Utilities
         └─────────┘   └─────────┘
```

---

## Backends

CPU is the baseline; every op works there first.

| Backend | Directory | Reaches Foundry ops via | Notes |
|---------|-----------|-------------------------|-------|
| CPU | (core modules) | — | the reference path |
| CUDA | `cuda/`, `cuda_kernels/` | `tensor.c` / `rwkvx.c` dispatch | `dlopen`, embedded PTX, `sm_37`+ |
| Metal | `metal/` | `tensor.c` / `rwkvx.c` dispatch | Apple Silicon, embedded MSL |
| Tenstorrent Blackhole | `tt/` | **driven directly, no tensor dispatch** | own userspace driver (tt-kmd optional), SFPI to rebuild kernels; kernels are reader/compute/writer triads on Tensix cores, embedded as ELF |
| Vulkan | — | — | planned |

`tt/` is the odd one out on purpose: there is no vendor runtime to hand a buffer to, so the
backend owns the whole stack from device open to op dispatch. It is not yet wired into
`tensor/`. See [tenstorrent.md](tenstorrent.md).

---

## Memory Model

### Tensor Memory

- Tensors own their data by default
- Reference counting for safe memory management
- Views share underlying data (no copy)
- Explicit `tm_contiguous()` forces copy when needed

```c
tm_tensor* a = tm_randn(shape, 2, TM_F32);  // Owns data
tm_tensor* b = tm_transpose(a, 0, 1);       // View, shares data
tm_tensor* c = tm_contiguous(b);            // Copy, owns new data
tm_free(c);
tm_free(b);  // Only frees view struct, not data
tm_free(a);  // Frees data
```

### Autograd Graph

- Operations create graph nodes
- Nodes hold references to inputs
- `tm_backward()` traverses graph in reverse
- Graph freed after backward (or via `tm_detach()`)

---

## Autograd Implementation

```
Forward:
    a ──┐
        ├── add_backward ── c
    b ──┘

c.grad_fn = {
    .backward = add_backward,
    .inputs = {a, b},
    .num_inputs = 2
}

Backward:
    tm_backward(c)
    ├── c.grad = 1.0
    └── c.grad_fn.backward(c)
        ├── a.grad += c.grad
        └── b.grad += c.grad
```

---

## File Format: Safetensors

```
┌──────────────────────────────────────────┐
│ Header Length (8 bytes, little-endian)   │
├──────────────────────────────────────────┤
│ JSON Header (variable length)            │
│ {                                        │
│   "__metadata__": {...},                 │
│   "tensor_name": {                       │
│     "dtype": "F32",                      │
│     "shape": [1024, 1024],               │
│     "data_offsets": [0, 4194304]         │
│   },                                     │
│   ...                                    │
│ }                                        │
├──────────────────────────────────────────┤
│ Tensor Data (contiguous, aligned)        │
│ [tensor 1 data][tensor 2 data][...]      │
└──────────────────────────────────────────┘
```

---

## Swarm Protocol

### Wire Format

```
┌──────────────────────────────────────────┐
│ Magic (4 bytes): "SWRM"                  │
├──────────────────────────────────────────┤
│ Version (2 bytes)                        │
├──────────────────────────────────────────┤
│ Message Type (1 byte)                    │
├──────────────────────────────────────────┤
│ Flags (1 byte)                           │
├──────────────────────────────────────────┤
│ Payload Length (4 bytes)                 │
├──────────────────────────────────────────┤
│ Checksum (4 bytes, CRC32)                │
├──────────────────────────────────────────┤
│ Payload (variable length)                │
└──────────────────────────────────────────┘
```

### Compression: TopK

Only send the K largest gradients by magnitude:

```
Original: [0.01, 5.0, 0.02, -3.0, 0.03, 2.0, 0.0, -1.0]
TopK (k=3):
  Indices: [1, 3, 5]
  Values:  [5.0, -3.0, 2.0]

Compression ratio: 3/8 = 37.5%
With k=10%: 90% bandwidth reduction
```

---

## Threading Model

- Main thread: Coordinator event loop
- Worker threads: Forward/backward computation
- I/O threads: Network send/receive

Thread safety:
- Tensors are NOT thread-safe
- Coordinator uses mutex for worker list
- Worker uses mutex for state transitions

---

## Build System

```makefile
# Top-level Makefile structure
all: tensor optim tokenizer safetensors data progress \
     quantization attention lora distributed eval swarm

tensor:
    $(CC) -c tensor/tensor.c -o obj/tensor.o

# ... other modules ...

lib/libfoundry.a: $(ALL_OBJS)
    ar rcs $@ $^
```

---

## Error Handling

- Functions return NULL or false on error
- No exceptions (C doesn't have them)
- Errors logged to stderr
- Assertions for programmer errors

```c
tm_tensor* t = tm_zeros(shape, 2, TM_F32);
if (t == NULL) {
    fprintf(stderr, "Failed to allocate tensor\n");
    return 1;
}
```

---

## Future Considerations

### GPU Support

Planned abstraction:

```c
typedef enum {
    TM_DEVICE_CPU,
    TM_DEVICE_CUDA,
    TM_DEVICE_METAL,
} tm_device;

tm_tensor* tm_zeros_on(shape, ndim, dtype, device);
tm_tensor* tm_to(tensor, device);
```

### Python Bindings

If needed, via ctypes or cffi:

```python
import ctypes

lib = ctypes.CDLL("libfoundry.so")
lib.tm_zeros.restype = ctypes.c_void_p
lib.tm_zeros.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int]
```
