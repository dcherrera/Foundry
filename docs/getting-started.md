# Getting Started

A quick guide to building and using Foundry.

---

## Prerequisites

- GCC or Clang with C11 support
- Make
- POSIX-compatible system (Linux, macOS)
- Optional: BLAS (`USE_BLAS=1` — Accelerate on macOS, OpenBLAS on Linux)
- `make test` downloads the RWKV vocabulary once (network, `curl`, `sha256sum` or `shasum`)

The default flags include `-march=native`: build on the machine that will run the code. `PORTABLE=1`
builds for a baseline instead (`-march=x86-64-v2` on x86-64, the compiler default elsewhere) so the
binaries can be copied between machines.

---

## Building

```bash
# Clone repository
git clone https://github.com/dcherrera/Foundry.git
cd Foundry

# Build all modules
make all

# Run tests (CPU only: 504 tests in 30 suites)
make test
```

### Build options

| Flag | Effect |
|------|--------|
| `USE_BLAS=1` | BLAS matmul (Accelerate on macOS, OpenBLAS on Linux) |
| `USE_THREADS=1` | pthread thread pool |
| `USE_OMP=1` | OpenMP (takes precedence over `USE_THREADS`) |
| `USE_CUDA=1` | CUDA backend (`cuda/`), Linux. Loads the driver with `dlopen` and embeds PTX, so no nvcc is needed to build |
| `USE_METAL=1` | Metal backend (`metal/`), macOS |
| `USE_TT=1` | Tenstorrent Blackhole host backend (`tt/`), Linux. Device images are embedded, so no SFPI is needed to build |
| `DEBUG=1` | `-O0 -g` |

```bash
make USE_CUDA=1 cuda test-cuda
make USE_METAL=1 metal test-metal
make USE_TT=1 tt
```

### What runs without hardware

| Target | Runs on |
|--------|---------|
| `make all`, `make test`, `make test-<module>`, `make train` | any Linux or macOS machine |
| `make USE_TT=1 tt`; off-hardware `tt/tests/` suites (`Makefile.coord`, `.noc`, `.queue`, ...) | any machine |
| `make USE_CUDA=1 test-cuda`, `test-cuda-multi`, `test-moe-expert-quants` | Linux with an NVIDIA GPU and driver |
| `make USE_METAL=1 test-metal` and the other `test-metal-*` targets | macOS with a Metal GPU |
| on-card `tt/tests/` suites (`PHASE1B=1`) | Linux with a Blackhole p150a |
| rebuilding Tensix kernel images | needs the SFPI compiler and `python3` |

Without the flag, the GPU targets print "... disabled. Build with USE_...=1 to enable." and do
nothing. `make test-tt` never runs suites; they run from `tt/tests/` (see
[`tt/README.md`](../tt/README.md)).

---

## First Steps

### 1. Create and Manipulate Tensors

```c
#include "tensor/tensor.h"

int main() {
    // Create tensors
    int64_t shape[] = {32, 64};
    tm_tensor* a = tm_randn(shape, 2, TM_F32);
    tm_tensor* b = tm_randn(shape, 2, TM_F32);

    // Operations
    tm_tensor* c = tm_add(a, b);
    tm_tensor* d = tm_relu(c);

    // Print info
    tm_print(d);

    // Cleanup
    tm_free(d);
    tm_free(c);
    tm_free(b);
    tm_free(a);

    return 0;
}
```

### 2. Train a Simple Model

```c
#include <stdio.h>
#include "tensor/tensor.h"
#include "optim/optim.h"

int main() {
    // Create weights
    int64_t w_shape[] = {2, 4};
    int64_t b_shape[] = {4};
    tm_tensor* W = tm_randn(w_shape, 2, TM_F32);
    tm_tensor* b = tm_zeros(b_shape, 1, TM_F32);

    tm_set_requires_grad(W, true);
    tm_set_requires_grad(b, true);

    // Create optimizer
    tm_tensor* params[] = {W, b};
    fo_adam_config cfg = fo_adam_default(1e-3);
    fo_optimizer* opt = fo_adam_new(params, 2, &cfg);

    // Training loop
    for (int step = 0; step < 1000; step++) {
        fo_zero_grad(opt);

        // Forward pass
        int64_t x_shape[] = {8, 2};
        tm_tensor* x = tm_randn(x_shape, 2, TM_F32);
        tm_tensor* h = tm_matmul(x, W);
        tm_tensor* h_bias = tm_add(h, b);
        tm_tensor* out = tm_relu(h_bias);
        tm_tensor* loss = tm_mean(out, -1, false);

        // Backward pass
        tm_backward(loss);

        // Update weights
        fo_step(opt);

        if (step % 100 == 0) {
            printf("Step %d, loss = %.4f\n", step, tm_get_f32(loss, 0));
        }

        // Cleanup
        tm_free(loss);
        tm_free(out);
        tm_free(h_bias);
        tm_free(h);
        tm_free(x);
    }

    fo_free(opt);
    tm_free(b);
    tm_free(W);

    return 0;
}
```

### 3. Load and Use a Tokenizer

```c
#include <stdio.h>
#include <stdlib.h>
#include "tokenizer/tokenizer.h"

int main() {
    // Load tokenizer
    ft_tokenizer* tok = ft_load("tokenizer.json");

    // Encode text
    int len;
    int* ids = ft_encode(tok, "Hello, world!", &len);

    printf("Tokens: ");
    for (int i = 0; i < len; i++) {
        printf("%d ", ids[i]);
    }
    printf("\n");

    // Decode
    char* text = ft_decode(tok, ids, len);
    printf("Decoded: %s\n", text);

    free(text);
    free(ids);
    ft_free(tok);

    return 0;
}
```

---

## Next Steps

- [tensor](tensor.md) - Full tensor API reference
- [optim](optim.md) - Optimizers and schedulers
- [examples](examples.md) - Complete examples
- [swarm](swarm.md) - Distributed training

---

## Compiling Your Code

There is no prebuilt library; compile the module sources you use with your program:

```bash
cc -std=c11 -O2 -I/path/to/foundry -o myprogram myprogram.c \
    /path/to/foundry/tensor/tensor.c /path/to/foundry/optim/optim.c -lm
```
