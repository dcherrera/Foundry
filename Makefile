# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Foundry Makefile

CC ?= clang
# CPU target. Default -march=native: fastest, but the binary only runs on a CPU
# with the build machine's instruction set, so build on the machine that runs it.
# PORTABLE=1 builds a binary you can copy between machines: -march=x86-64-v2 on
# x86-64 (SSE4.2, no AVX; runs on any x86-64 CPU since ~2009), the compiler's
# generic default elsewhere.
ifeq ($(PORTABLE),1)
  ifneq ($(filter x86_64 amd64,$(shell uname -m)),)
    ARCH_FLAGS = -march=x86-64-v2
  else
    ARCH_FLAGS =
  endif
else
  ARCH_FLAGS = -march=native
endif
CFLAGS = -Wall -Wextra -std=c11 -O3 $(ARCH_FLAGS) -ffast-math
CFLAGS += -I.
# On Linux, -std=c11 hides POSIX/GNU symbols (clock_gettime, M_PI, strdup, etc.)
ifneq ($(shell uname -s),Darwin)
  CFLAGS += -D_GNU_SOURCE
endif

# Debug build (overrides optimization flags)
ifdef DEBUG
CFLAGS = -Wall -Wextra -std=c11 -O0 -g -DDEBUG
CFLAGS += -I.
ifneq ($(shell uname -s),Darwin)
  CFLAGS += -D_GNU_SOURCE
endif
endif

# OpenMP (opt-in: make USE_OMP=1)
ifdef USE_OMP
  CFLAGS += -fopenmp
  LDFLAGS += -fopenmp
endif

# pthread thread pool (opt-in: make USE_THREADS=1)
# Alternative to OpenMP — same parallelism, no compiler dependency.
# OpenMP takes precedence if both are set.
ifdef USE_THREADS
  ifndef USE_OMP
    CFLAGS += -DUSE_THREADS -pthread
    LDFLAGS += -pthread
    THREADPOOL_SRC = threadpool/threadpool.c
  endif
else
  THREADPOOL_SRC =
endif

# BLAS acceleration (opt-in: make USE_BLAS=1)
ifdef USE_BLAS
  CFLAGS += -DUSE_BLAS
  ifeq ($(shell uname -s),Darwin)
    CFLAGS += -DACCELERATE_NEW_LAPACK
    LDFLAGS_BLAS = -framework Accelerate
  else
    CFLAGS += -DUSE_OPENBLAS
    LDFLAGS_BLAS = -lopenblas
  endif
else
  LDFLAGS_BLAS =
endif

# Metal GPU backend (opt-in: make USE_METAL=1, macOS only)
ifdef USE_METAL
  CFLAGS += -DUSE_METAL
  LDFLAGS_METAL = -framework Metal -framework MetalPerformanceShaders -framework Foundation
  # Core (device, pipeline cache, buffers, regions, dispatch) plus one unit per
  # kernel family. A family unit carries its host dispatch AND its MSL source
  # fragment, so adding a kernel touches one file. Add new families here.
  METAL_SRC = metal/fm_metal.m \
              metal/kernels/fm_kernel_registry.m \
              metal/kernels/fm_kernel_ssm.m \
              metal/kernels/fm_kernel_ssm_chunked.m \
              metal/kernels/fm_kernel_moe.m \
              metal/kernels/fm_kernel_attn.m \
              metal/kernels/fm_kernel_probe.m \
              metal/kernels/fm_kernel_gemm_f32.m \
              metal/kernels/fm_kernel_wkv7.m \
              metal/kernels/fm_kernel_norm.m \
              metal/kernels/fm_kernel_elementwise.m \
              metal/kernels/fm_kernel_rope.m \
              metal/kernels/fm_kernel_convert.m \
              metal/kernels/fm_kernel_matvec_q8.m \
              metal/kernels/fm_kernel_matvec_q4_k.m \
              metal/kernels/fm_kernel_matvec_q3_k.m \
              metal/kernels/fm_kernel_matvec_q5_k.m \
              metal/kernels/fm_kernel_matvec_q6_k.m \
              metal/kernels/fm_kernel_matvec_q8_0.m
  METAL_HDR = metal/fm_metal.h metal/fm_metal_internal.h $(wildcard metal/kernels/*.h)
  # Objective-C needs its own flags: no -std=c11, add -fobjc-arc
  OBJCFLAGS = -Wall -Wextra -O3 -ffast-math -fobjc-arc -DUSE_METAL -I.
  ifdef DEBUG
    OBJCFLAGS = -Wall -Wextra -O0 -g -DDEBUG -fobjc-arc -DUSE_METAL -I.
  endif
else
  LDFLAGS_METAL =
  METAL_SRC =
  METAL_HDR =
endif

# CUDA GPU backend (opt-in: make USE_CUDA=1, Linux only)
# Zero compile-time CUDA dependency — uses dlopen() at runtime.
# Mutually exclusive with Metal (Linux vs macOS).
ifdef USE_CUDA
  CFLAGS += -DUSE_CUDA
  LDFLAGS_CUDA = -ldl
  CUDA_SRC = cuda/fc_cuda.c
else
  LDFLAGS_CUDA =
  CUDA_SRC =
endif

# Tenstorrent Blackhole GPU backend (opt-in: make USE_TT=1, Linux only)
# Host backend is C99 against tt-kmd ioctls and builds anywhere. Device kernels
# need SFPI (/opt/tenstorrent/sfpi), but their ELFs are pre-baked into
# tt/ft_tt_kernels.h — the same embedding trick cuda/ uses for PTX — so USE_TT=1
# needs no SFPI toolchain present. SFPI is needed only to REGENERATE that header
# via tt/build_tensix_*.sh, never to build against it.
ifdef USE_TT
  CFLAGS += -DUSE_TT
  LDFLAGS_TT =
  # Host backend: device open, TLBs, NoC, launch, queue, per-op drivers, FTTW.
  TT_SRC = tt/ft_tt_kmd.c \
           tt/ft_tt_arch.c \
           tt/ft_tt_arch_blackhole.c \
           tt/ft_tt_device.c \
           tt/ft_tt_tlb.c \
           tt/ft_tt_arc_telemetry.c \
           tt/ft_tt_coord.c \
           tt/ft_tt_noc.c \
           tt/ft_tt_buffer.c \
           tt/ft_tt_elf.c \
           tt/ft_tt_kernel_blob.c \
           tt/ft_tt_launch.c \
           tt/ft_tt_tilize.c \
           tt/ft_tt_watchdog.c \
           tt/ft_tt_queue.c \
           tt/ft_tt_group.c \
           tt/ft_tt_fttw.c \
           tt/ft_tt_op_dispatch.c \
           tt/ft_tt_op_matmul.c \
           tt/ft_tt_op_memory.c \
           tt/ft_tt_op_rms_norm.c \
           tt/ft_tt_op_rope.c \
           tt/ft_tt_op_sample.c \
           tt/ft_tt_op_vector.c \
           tt/ft_tt_op_attention.c \
           tt/ft_tt_op_gdn.c \
           tt/ft_tt_op_gdnc.c
  # FTTW weight converter (GGUF in, tiled on-card artifact out) — part of this backend.
  TT_CONVERT_SRC = tt/convert/ft_bfp.c \
                   tt/convert/ft_float_convert.c \
                   tt/convert/ft_gguf_dequant.c \
                   tt/convert/ft_gguf_map.c \
                   tt/convert/ft_tt_precision_ledger.c \
                   tt/convert/ft_tt_weight_convert.c
  TT_HDR = $(wildcard tt/*.h) $(wildcard tt/convert/*.h)
else
  LDFLAGS_TT =
  TT_SRC =
  TT_CONVERT_SRC =
  TT_HDR =
endif

# Modules
MODULES = progress tensor tokenizer data safetensors lora quantization attention distributed eval optim hdc hybridmem threadpool rwkvx metal cuda tt barswap pth

# Default target
all: $(MODULES)

# Progress module
progress: progress/progress.o

progress/progress.o: progress/progress.c progress/progress.h
	$(CC) $(CFLAGS) -c $< -o $@

# Tensor module
tensor: tensor/tensor.o

tensor/tensor.o: tensor/tensor.c tensor/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

# Tokenizer module
tokenizer: tokenizer/tokenizer.o

tokenizer/tokenizer.o: tokenizer/tokenizer.c tokenizer/tokenizer.h
	$(CC) $(CFLAGS) -c $< -o $@

# Safetensors module
safetensors: safetensors/safetensors.o

safetensors/safetensors.o: safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -c $< -o $@

# Test targets
test: test-progress test-tensor test-tokenizer test-safetensors test-lora test-rfa-train test-rfa-parallel test-data test-quantization test-export test-attention test-turboquant test-distributed test-eval test-optim test-hdc test-hybridmem test-threadpool test-rwkvx test-barswap test-pth test-q2-0-half-round test-ingot

# Ingot: exact BF16 weight compression (own Makefile, strict warnings)
test-ingot:
	$(MAKE) -C ingot test

# Q2_0 activation-scale binary16 rounding (cuda_kernels/matvec_kquants.cu), exhaustive.
# Own flags: -ffast-math would fold the (v + 0.5f) - 0.5f subnormal rounding.
test-q2-0-half-round: cuda_kernels/tests/test_q2_0_half_round
	./cuda_kernels/tests/test_q2_0_half_round

cuda_kernels/tests/test_q2_0_half_round: cuda_kernels/tests/test_q2_0_half_round.c
	$(CC) -Wall -Wextra -std=c11 -O2 -o $@ $<

test-progress: progress/tests/test_progress
	./progress/tests/test_progress

progress/tests/test_progress: progress/tests/test_progress.c progress/progress.c progress/progress.h
	$(CC) $(CFLAGS) -o $@ progress/tests/test_progress.c progress/progress.c

test-tensor: tensor/tests/test_tensor
	./tensor/tests/test_tensor

tensor/tests/test_tensor: tensor/tests/test_tensor.c tensor/tensor.c tensor/tensor.h $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ tensor/tests/test_tensor.c tensor/tensor.c $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

test-tokenizer: tokenizer/tests/test_tokenizer
	./tokenizer/tests/test_tokenizer

tokenizer/tests/test_tokenizer: tokenizer/tests/test_tokenizer.c tokenizer/tokenizer.c tokenizer/tokenizer.h
	$(CC) $(CFLAGS) -o $@ tokenizer/tests/test_tokenizer.c tokenizer/tokenizer.c

test-safetensors: safetensors/tests/test_safetensors
	./safetensors/tests/test_safetensors

safetensors/tests/test_safetensors: safetensors/tests/test_safetensors.c safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -o $@ safetensors/tests/test_safetensors.c safetensors/safetensors.c

# LoRA module
lora: lora/lora.o

lora/lora.o: lora/lora.c lora/lora.h tensor/tensor.h safetensors/safetensors.h
	$(CC) $(CFLAGS) -c $< -o $@

test-lora: lora/tests/test_lora
	./lora/tests/test_lora

lora/tests/test_lora: lora/tests/test_lora.c lora/lora.c lora/lora.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -o $@ lora/tests/test_lora.c lora/lora.c tensor/tensor.c safetensors/safetensors.c -lm $(LDFLAGS_BLAS)

# Potluck batch module
test-potluck-batch: potluck/tests/test_pl_batch
	./potluck/tests/test_pl_batch

potluck/tests/test_pl_batch: potluck/tests/test_pl_batch.c potluck/pl_batch.c potluck/pl_batch.h
	$(CC) $(CFLAGS) -o $@ potluck/tests/test_pl_batch.c potluck/pl_batch.c -lm

# Data module
data: data/data.o

data/data.o: data/data.c data/data.h
	$(CC) $(CFLAGS) -c $< -o $@

test-data: data/tests/test_data
	./data/tests/test_data

data/tests/test_data: data/tests/test_data.c data/data.c data/data.h
	$(CC) $(CFLAGS) -o $@ data/tests/test_data.c data/data.c

# Byte-level enwik8 loader (RWKV-x scale test). Deliberately NOT wired into the
# default slow `test` target — it is a fast standalone check.
test-byte-data: data/tests/test_byte_dataset
	./data/tests/test_byte_dataset

data/tests/test_byte_dataset: data/tests/test_byte_dataset.c data/fd_byte_dataset.c data/fd_byte_dataset.h
	$(CC) $(CFLAGS) -o $@ data/tests/test_byte_dataset.c data/fd_byte_dataset.c

# potluck pl_data tests
test-potluck-data: potluck/tests/test_pl_data
	./potluck/tests/test_pl_data

potluck/tests/test_pl_data: potluck/tests/test_pl_data.c potluck/pl_data.c potluck/pl_data.h data/data.c data/data.h rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC)
	$(CC) $(CFLAGS) -o $@ potluck/tests/test_pl_data.c potluck/pl_data.c data/data.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS)

# Quantization module
quantization: quantization/quantization.o

quantization/quantization.o: quantization/quantization.c quantization/quantization.h
	$(CC) $(CFLAGS) -c $< -o $@

test-quantization: quantization/tests/test_quantization
	./quantization/tests/test_quantization

quantization/tests/test_quantization: quantization/tests/test_quantization.c quantization/quantization.c quantization/quantization.h
	$(CC) $(CFLAGS) -o $@ quantization/tests/test_quantization.c quantization/quantization.c -lm $(LDFLAGS_BLAS)

# Potluck: Q8_0 frozen-base linear (dequant-tile GEMM)
test-potluck-q8: potluck/tests/test_pl_q8
	./potluck/tests/test_pl_q8

potluck/tests/test_pl_q8: potluck/tests/test_pl_q8.c potluck/pl_q8.c potluck/pl_q8.h tt/convert/ft_float_convert.c
	$(CC) $(CFLAGS) -o $@ potluck/tests/test_pl_q8.c potluck/pl_q8.c tt/convert/ft_float_convert.c -lm $(LDFLAGS_BLAS) $(LDFLAGS)

test-export: quantization/tests/test_export
	./quantization/tests/test_export

quantization/tests/test_export: quantization/tests/test_export.c quantization/export.c quantization/export.h quantization/quantization.c quantization/quantization.h safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -o $@ quantization/tests/test_export.c quantization/export.c quantization/quantization.c safetensors/safetensors.c -lm $(LDFLAGS_BLAS)

examples/quantize_model: examples/quantize_model.c quantization/export.c quantization/export.h quantization/quantization.c quantization/quantization.h safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -o $@ examples/quantize_model.c quantization/export.c quantization/quantization.c safetensors/safetensors.c -lm $(LDFLAGS_BLAS)

quantize-model: examples/quantize_model

# HDC module
hdc: hdc/hdc.o

hdc/hdc.o: hdc/hdc.c hdc/hdc.h tensor/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

test-hdc: hdc/tests/test_hdc
	./hdc/tests/test_hdc

hdc/tests/test_hdc: hdc/tests/test_hdc.c hdc/hdc.c hdc/hdc.h tensor/tensor.c tensor/tensor.h
	$(CC) $(CFLAGS) -o $@ hdc/tests/test_hdc.c hdc/hdc.c tensor/tensor.c -lm $(LDFLAGS_BLAS)

# HybridMem module
hybridmem: hybridmem/hybridmem.o

hybridmem/hybridmem.o: hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.h tensor/tensor.h safetensors/safetensors.h
	$(CC) $(CFLAGS) -c $< -o $@

test-hybridmem: hybridmem/tests/test_hybridmem
	./hybridmem/tests/test_hybridmem

hybridmem/tests/test_hybridmem: hybridmem/tests/test_hybridmem.c hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -o $@ hybridmem/tests/test_hybridmem.c hybridmem/hybridmem.c hdc/hdc.c tensor/tensor.c safetensors/safetensors.c -lm $(LDFLAGS_BLAS)

# BAR Swap userspace library (kernel module builds separately on server)
barswap: barswap/barswap.o

barswap/barswap.o: barswap/barswap.c barswap/barswap.h barswap/bar_swap_ko/bar_swap.h
	$(CC) $(CFLAGS) -c $< -o $@

test-barswap: barswap/tests/test_barswap
	./barswap/tests/test_barswap

barswap/tests/test_barswap: barswap/tests/test_barswap.c barswap/barswap.c barswap/barswap.h barswap/bar_swap_ko/bar_swap.h
	$(CC) $(CFLAGS) -o $@ barswap/tests/test_barswap.c barswap/barswap.c

barswap/tests/vram_persist: barswap/tests/vram_persist.c
	$(CC) $(CFLAGS) -o $@ barswap/tests/vram_persist.c -ldl

vram-persist: barswap/tests/vram_persist

barswap/tests/compute_persist: barswap/tests/compute_persist.c barswap/barswap.c barswap/barswap.h barswap/bar_swap_ko/bar_swap.h
	$(CC) $(CFLAGS) -o $@ barswap/tests/compute_persist.c barswap/barswap.c -ldl

compute-persist: barswap/tests/compute_persist

barswap/tests/xcard_persist: barswap/tests/xcard_persist.c barswap/barswap.c barswap/barswap.h barswap/bar_swap_ko/bar_swap.h
	$(CC) $(CFLAGS) -o $@ barswap/tests/xcard_persist.c barswap/barswap.c -ldl

xcard-persist: barswap/tests/xcard_persist

barswap/tests/roundrobin_persist: barswap/tests/roundrobin_persist.c barswap/barswap.c barswap/barswap.h barswap/bar_swap_ko/bar_swap.h
	$(CC) $(CFLAGS) -o $@ barswap/tests/roundrobin_persist.c barswap/barswap.c -ldl

roundrobin-persist: barswap/tests/roundrobin_persist

barswap/tests/gpu_switch_probe: barswap/tests/gpu_switch_probe.c barswap/barswap.c barswap/barswap.h barswap/bar_swap_ko/bar_swap.h
	$(CC) $(CFLAGS) -o $@ barswap/tests/gpu_switch_probe.c barswap/barswap.c -ldl

gpu-switch-probe: barswap/tests/gpu_switch_probe

barswap/tests/worker_switch: barswap/tests/worker_switch.c barswap/barswap.c barswap/barswap.h barswap/bar_swap_ko/bar_swap.h
	$(CC) $(CFLAGS) -o $@ barswap/tests/worker_switch.c barswap/barswap.c -ldl

worker-switch: barswap/tests/worker_switch

barswap/tests/gpu_owner: barswap/tests/gpu_owner.c barswap/gpu_owner.c barswap/gpu_owner_util.c barswap/barswap.c barswap/gpu_owner.h barswap/gpu_owner_util.h barswap/barswap.h barswap/bar_swap_ko/bar_swap.h
	$(CC) $(CFLAGS) -o $@ barswap/tests/gpu_owner.c barswap/gpu_owner.c barswap/gpu_owner_util.c barswap/barswap.c -ldl

gpu-owner: barswap/tests/gpu_owner

# Threadpool module
threadpool: threadpool/threadpool.o

threadpool/threadpool.o: threadpool/threadpool.c threadpool/threadpool.h
	$(CC) $(CFLAGS) -pthread -c $< -o $@

test-threadpool: threadpool/tests/test_threadpool
	./threadpool/tests/test_threadpool

threadpool/tests/test_threadpool: threadpool/tests/test_threadpool.c threadpool/threadpool.c threadpool/threadpool.h
	$(CC) $(CFLAGS) -pthread -o $@ threadpool/tests/test_threadpool.c threadpool/threadpool.c

# Metal GPU backend module
ifdef USE_METAL
METAL_OBJ = $(METAL_SRC:.m=.o)

metal: $(METAL_OBJ)

# Every unit sees every kernel header (fm_metal.h includes them all), so they
# all depend on $(METAL_HDR) rather than on a hand-maintained subset.
%.o: %.m $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -c $< -o $@

test-metal: metal/tests/test_metal
	./metal/tests/test_metal

metal/tests/test_metal: metal/tests/test_metal.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-fc: metal/tests/test_metal_fc
	./metal/tests/test_metal_fc

metal/tests/test_metal_fc: metal/tests/test_metal_fc.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_fc.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-region: metal/tests/test_metal_region
	./metal/tests/test_metal_region

metal/tests/test_metal_region: metal/tests/test_metal_region.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_region.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-q3-k: metal/tests/test_metal_q3_k
	./metal/tests/test_metal_q3_k

metal/tests/test_metal_q3_k: metal/tests/test_metal_q3_k.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_q3_k.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-matvec-q5-k: metal/tests/test_metal_matvec_q5_k
	./metal/tests/test_metal_matvec_q5_k

metal/tests/test_metal_matvec_q5_k: metal/tests/test_metal_matvec_q5_k.c metal/tests/gguf_test_reader.h $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_matvec_q5_k.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-matvec-q6-k: metal/tests/test_metal_matvec_q6_k
	./metal/tests/test_metal_matvec_q6_k

metal/tests/test_metal_matvec_q6_k: metal/tests/test_metal_matvec_q6_k.c metal/tests/gguf_test_reader.h $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_matvec_q6_k.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-rope: metal/tests/test_metal_rope
	./metal/tests/test_metal_rope

metal/tests/test_metal_rope: metal/tests/test_metal_rope.c metal/tests/fm_golden_vectors.h $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_rope.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-norm: metal/tests/test_metal_norm
	./metal/tests/test_metal_norm

metal/tests/test_metal_norm: metal/tests/test_metal_norm.c metal/tests/fm_golden_vectors.h $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_norm.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-elementwise: metal/tests/test_metal_elementwise metal/tests/test_matvec_q8_0 metal/tests/test_matvec_q4_k metal/tests/bench_matvec_bandwidth metal/tests/test_metal_ssm metal/tests/test_metal_attn metal/tests/test_metal_moe metal/tests/test_metal_moe_experts metal/tests/bench_attn_split
	./metal/tests/test_metal_elementwise

metal/tests/test_metal_elementwise: metal/tests/test_metal_elementwise.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_elementwise.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-matvec-q8-0: metal/tests/test_matvec_q8_0
	./metal/tests/test_matvec_q8_0

metal/tests/test_matvec_q8_0: metal/tests/test_matvec_q8_0.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_matvec_q8_0.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-matvec-q4-k: metal/tests/test_matvec_q4_k
	./metal/tests/test_matvec_q4_k

metal/tests/test_matvec_q4_k: metal/tests/test_matvec_q4_k.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_matvec_q4_k.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

bench-matvec-bandwidth: metal/tests/bench_matvec_bandwidth
	./metal/tests/bench_matvec_bandwidth

metal/tests/bench_matvec_bandwidth: metal/tests/bench_matvec_bandwidth.m metal/tests/gguf_test_reader.h $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/bench_matvec_bandwidth.m $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-ssm: metal/tests/test_metal_ssm
	./metal/tests/test_metal_ssm

metal/tests/test_metal_ssm: metal/tests/test_metal_ssm.c metal/tests/fm_golden_vectors.h $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_ssm.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-attn: metal/tests/test_metal_attn metal/tests/bench_attn_split
	./metal/tests/test_metal_attn

metal/tests/test_metal_attn: metal/tests/test_metal_attn.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_attn.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

metal/tests/bench_attn_split: metal/tests/bench_attn_split.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/bench_attn_split.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-moe: metal/tests/test_metal_moe
	./metal/tests/test_metal_moe

metal/tests/test_metal_moe: metal/tests/test_metal_moe.c metal/tests/fm_golden_vectors.h $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_moe.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-moe-experts: metal/tests/test_metal_moe_experts
	./metal/tests/test_metal_moe_experts

metal/tests/test_metal_moe_experts: metal/tests/test_metal_moe_experts.c metal/tests/gguf_test_reader.h $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_moe_experts.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-metal-gemm-batched: metal/tests/test_metal_gemm_batched
	./metal/tests/test_metal_gemm_batched

metal/tests/test_metal_gemm_batched: metal/tests/test_metal_gemm_batched.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_metal_gemm_batched.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-gemm-q4-k: metal/tests/test_gemm_q4_k
	./metal/tests/test_gemm_q4_k

metal/tests/test_gemm_q4_k: metal/tests/test_gemm_q4_k.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_gemm_q4_k.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

test-gemm-q4-k-f16: metal/tests/test_gemm_q4_k_f16
	./metal/tests/test_gemm_q4_k_f16

metal/tests/test_gemm_q4_k_f16: metal/tests/test_gemm_q4_k_f16.c $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/test_gemm_q4_k_f16.c $(METAL_SRC) -lm $(LDFLAGS_METAL)

bench-dispatch: metal/tests/bench_dispatch
	./metal/tests/bench_dispatch

metal/tests/bench_dispatch: metal/tests/bench_dispatch.m $(METAL_SRC) $(METAL_HDR)
	$(CC) $(OBJCFLAGS) -o $@ metal/tests/bench_dispatch.m $(METAL_SRC) -lm $(LDFLAGS_METAL)
else
metal:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-fc:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-gemm-batched:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
bench-dispatch:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-region:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-q3-k:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-matvec-q5-k:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-matvec-q6-k:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-rope:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-norm:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-elementwise:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-matvec-q8-0:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-matvec-q4-k:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
bench-matvec-bandwidth:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-ssm:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-attn:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-moe:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
test-metal-moe-experts:
	@echo "Metal disabled. Build with USE_METAL=1 to enable."
endif

# CUDA GPU backend module
ifdef USE_CUDA
cuda: cuda/fc_cuda.o

cuda/fc_cuda.o: cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -c $< -o $@

test-cuda: cuda/tests/test_cuda
	./cuda/tests/test_cuda

cuda/tests/test_cuda: cuda/tests/test_cuda.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_cuda.c cuda/fc_cuda.c -lm -ldl

# Multi-die context isolation (run on a K80 host with >=2 dies)
test-cuda-multi: cuda/tests/test_multi_ctx
	./cuda/tests/test_multi_ctx

cuda/tests/test_multi_ctx: cuda/tests/test_multi_ctx.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_multi_ctx.c cuda/fc_cuda.c -lm -ldl

# Grouped-MoE gate+up quant dispatch + CPU-replica parity (needs ONE K80 die).
# NOT in the top-level `test` list: that list runs on the Mac, where there is
# no CUDA driver. Same convention as test-cuda / test-cuda-multi above.
test-moe-expert-quants: cuda/tests/test_moe_expert_quants
	./cuda/tests/test_moe_expert_quants

cuda/tests/test_moe_expert_quants: cuda/tests/test_moe_expert_quants.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_moe_expert_quants.c cuda/fc_cuda.c -lm -ldl

# Device top-k: exact vs a host reference, no-winner rows (clean -1, no stray
# write), LM-head shape under a time budget (ONE K80 die).
test-topk-logits: cuda/tests/test_topk_logits
	./cuda/tests/test_topk_logits

cuda/tests/test_topk_logits: cuda/tests/test_topk_logits.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_topk_logits.c cuda/fc_cuda.c -lm -ldl

# Rank-select MoE router bit-exact to the block router, and under a time budget (ONE K80 die).
test-moe-router-rank: cuda/tests/test_moe_router_rank
	./cuda/tests/test_moe_router_rank

cuda/tests/test_moe_router_rank: cuda/tests/test_moe_router_rank.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_moe_router_rank.c cuda/fc_cuda.c -lm -ldl

# fc_fused_multi Q8_0 segments vs fc_matvec_q80: bit-exact and as fast (ONE K80 die).
test-fused-q80: cuda/tests/test_fused_q80
	./cuda/tests/test_fused_q80

cuda/tests/test_fused_q80: cuda/tests/test_fused_q80.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_fused_q80.c cuda/fc_cuda.c -lm -ldl

# fc_cuda_same_board matrix over the visible dies (FC_SAME_BOARD_PAIRS gates it exactly).
test-same-board: cuda/tests/test_same_board
	./cuda/tests/test_same_board

cuda/tests/test_same_board: cuda/tests/test_same_board.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_same_board.c cuda/fc_cuda.c -lm -ldl

# Q8_0 G4 repack layout + dense Q8_0 matvec bandwidth (ONE K80 die).
test-matvec-q80-g4: cuda/tests/test_matvec_q80_g4
	./cuda/tests/test_matvec_q80_g4

cuda/tests/test_matvec_q80_g4: cuda/tests/test_matvec_q80_g4.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_matvec_q80_g4.c cuda/fc_cuda.c -lm -ldl

# Q8_0 parity K-split (tensor-parallel FFN down) bit-exact vs the full kernel; 2-D peer copy.
test-matvec-q80-ksplit: cuda/tests/test_matvec_q80_ksplit
	./cuda/tests/test_matvec_q80_ksplit

cuda/tests/test_matvec_q80_ksplit: cuda/tests/test_matvec_q80_ksplit.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_matvec_q80_ksplit.c cuda/fc_cuda.c -lm -ldl

# Q8_0 parity K-split tiles vs the full kernel at half the columns (ONE K80 die; timing only).
cuda/tests/bench_q80_ksplit: cuda/tests/bench_q80_ksplit.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/bench_q80_ksplit.c cuda/fc_cuda.c -lm -ldl

# Capturable 2-D copy, direct and inside a graph captured across two dies (TP graphs).
test-copy-2d-graph: cuda/tests/test_copy_2d_graph
	./cuda/tests/test_copy_2d_graph

cuda/tests/test_copy_2d_graph: cuda/tests/test_copy_2d_graph.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_copy_2d_graph.c cuda/fc_cuda.c -lm -ldl

# Two-die exchange ordered by device counters, direct and in a two-die graph (TP gathers).
test-xchg-2d: cuda/tests/test_xchg_2d
	./cuda/tests/test_xchg_2d

cuda/tests/test_xchg_2d: cuda/tests/test_xchg_2d.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_xchg_2d.c cuda/fc_cuda.c -lm -ldl
# Grouped MoE Q8_0 gate+up / down split by output rows bit-exact vs the full launch (tensor-parallel MoE).
test-moe-q80-rowsplit: cuda/tests/test_moe_q80_rowsplit
	./cuda/tests/test_moe_q80_rowsplit

cuda/tests/test_moe_q80_rowsplit: cuda/tests/test_moe_q80_rowsplit.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_moe_q80_rowsplit.c cuda/fc_cuda.c -lm -ldl

test-moe-q80-dedup: cuda/tests/test_moe_q80_dedup
	./cuda/tests/test_moe_q80_dedup

cuda/tests/test_moe_q80_dedup: cuda/tests/test_moe_q80_dedup.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_moe_q80_dedup.c cuda/fc_cuda.c -lm -ldl

# Grouped MoE Q8_0 gate+up / down bandwidth at the Qwen3.6 expert shape (ONE K80 die).
test-moe-q80-bw: cuda/tests/test_moe_q80_bw
	./cuda/tests/test_moe_q80_bw

cuda/tests/test_moe_q80_bw: cuda/tests/test_moe_q80_bw.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_moe_q80_bw.c cuda/fc_cuda.c -lm -ldl

# Q4_0 KV append kernels (embedded PTX) vs a host quantizer, byte for byte (ONE K80 die).
test-kvq4-scale: cuda/tests/test_kvq4_scale
	./cuda/tests/test_kvq4_scale

cuda/tests/test_kvq4_scale: cuda/tests/test_kvq4_scale.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/test_kvq4_scale.c cuda/fc_cuda.c -lm -ldl

# The micro-bench / bit-exact verifier the parity method above is taken from.
cuda/tests/bench_matvec: cuda/tests/bench_matvec.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/bench_matvec.c cuda/fc_cuda.c -lm -ldl

# Tensor-parallelism pricing. bench_split_scaling needs ONE die; bench_cross_die
# needs the TWO dies of one board (CUDA_VISIBLE_DEVICES=6,7) and must run alone
# on them — a busy sibling drops the other die's clock and corrupts the result.
cuda/tests/bench_split_scaling: cuda/tests/bench_split_scaling.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/bench_split_scaling.c cuda/fc_cuda.c -lm -ldl

cuda/tests/bench_cross_die: cuda/tests/bench_cross_die.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/bench_cross_die.c cuda/fc_cuda.c -lm -ldl

cuda/tests/bench_peer_stream: cuda/tests/bench_peer_stream.c cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/bench_peer_stream.c cuda/fc_cuda.c -lm -ldl

# Qwen3.8-27B Q8_0 TP pricing on 4 dies (peer | mv | mvc | xchg | step).
cuda/tests/bench_tp_q38: cuda/tests/bench_tp_q38.c cuda/tests/tp_xboard_copy.c cuda/tests/tp_xboard_copy.h cuda/fc_cuda.c cuda/fc_cuda.h cuda/fc_cuda_ptx.h
	$(CC) $(CFLAGS) -o $@ cuda/tests/bench_tp_q38.c cuda/tests/tp_xboard_copy.c cuda/fc_cuda.c -lm -ldl

bench-tp: cuda/tests/bench_split_scaling cuda/tests/bench_cross_die cuda/tests/bench_peer_stream cuda/tests/bench_tp_q38
else
cuda:
	@echo "CUDA disabled. Build with USE_CUDA=1 to enable."
test-cuda:
	@echo "CUDA disabled. Build with USE_CUDA=1 to enable."
test-cuda-multi:
	@echo "CUDA disabled. Build with USE_CUDA=1 to enable."
test-moe-expert-quants:
	@echo "CUDA disabled. Build with USE_CUDA=1 to enable."
test-kvq4-scale:
	@echo "CUDA disabled. Build with USE_CUDA=1 to enable."
endif

# Tenstorrent Blackhole GPU backend module
ifdef USE_TT
TT_OBJ = $(TT_SRC:.c=.o) $(TT_CONVERT_SRC:.c=.o)

tt: $(TT_OBJ)

# Every unit sees every backend header (there is no single umbrella header, like
# METAL's fm_metal.h), so objects depend on the full header set rather than a
# hand-maintained per-file subset.
tt/%.o: tt/%.c $(TT_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

tt/convert/%.o: tt/convert/%.c $(TT_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

# On-card acceptance and off-hardware suites live in tt/tests/ and tt/convert/tests/,
# each with its own Makefile (49 total) — see tt/README.md. They are not aggregated
# here: several need PHASE1B=1 (opens the NoC fence) and the card host specifically,
# which is not true of anything else `make test` runs.
test-tt:
	@echo "tt/ suites run from tt/tests/ and tt/convert/tests/ (see tt/README.md), not from the root Makefile."
else
tt:
	@echo "Tenstorrent disabled. Build with USE_TT=1 to enable."
test-tt:
	@echo "Tenstorrent disabled. Build with USE_TT=1 to enable."
endif

# Clean
clean:
	rm -f cuda_kernels/tests/test_q2_0_half_round
	$(MAKE) -C ingot clean
	rm -f progress/*.o
	rm -f progress/tests/test_progress
	rm -f tensor/*.o
	rm -f tensor/tests/test_tensor
	rm -f tokenizer/*.o
	rm -f tokenizer/tests/test_tokenizer
	rm -f data/*.o
	rm -f data/tests/test_data
	rm -f data/tests/test_byte_dataset
	rm -f potluck/tests/test_pl_data
	rm -f safetensors/*.o
	rm -f safetensors/tests/test_safetensors
	rm -f lora/*.o
	rm -f lora/tests/test_lora
	rm -f quantization/*.o
	rm -f quantization/tests/test_quantization
	rm -f quantization/tests/test_export
	rm -f examples/quantize_model
	rm -f attention/*.o
	rm -f attention/tests/test_attention
	rm -f attention/tests/test_turboquant
	rm -f distributed/*.o
	rm -f distributed/tests/test_distributed
	rm -f eval/*.o
	rm -f eval/tests/test_eval
	rm -f optim/*.o
	rm -f optim/tests/test_optim
	rm -f hdc/*.o
	rm -f hdc/tests/test_hdc
	rm -f hybridmem/*.o
	rm -f hybridmem/tests/test_hybridmem
	rm -f swarm/*.o
	rm -f swarm/tests/test_swarm
	rm -f bin/swarm_coordinator bin/swarm_worker
	rm -f bin/compare_helper
	rm -rf tensor/testing_suite/compare/__pycache__
	rm -f threadpool/*.o
	rm -f threadpool/tests/test_threadpool
	rm -f rwkvx/*.o
	rm -f rwkvx/tests/test_rwkvx
	rm -f examples/train_mlp
	rm -f examples/bench_hybridmem
	rm -f examples/rwkvx_generate
	rm -f examples/test_hybridmem_retrieval
	rm -f examples/test_attn_weights
	rm -f examples/test_hdc_correlation
	rm -f examples/test_pooling_loss
	rm -f examples/test_precision_at_scale
	rm -f examples/test_eviction_boundary
	rm -f examples/test_multi_fact
	rm -f examples/test_filler_sensitivity
	rm -f examples/test_context_scaling
	rm -f examples/test_false_positive
	rm -f examples/test_contiguous_chunk
	rm -f examples/test_query_vector
	rm -f rwkvx/tests/test_rfa_train
	rm -f examples/rfa_train
	rm -f examples/rfa_lora_merge
	rm -f rwkvx/tests/test_rfa_parallel
	rm -f barswap/*.o
	rm -f barswap/tests/test_barswap
	rm -f barswap/tests/vram_persist
	rm -f barswap/tests/compute_persist
	rm -f barswap/tests/xcard_persist
	rm -f barswap/tests/roundrobin_persist
	rm -f barswap/tests/gpu_switch_probe
	rm -f barswap/tests/worker_switch
	rm -f barswap/tests/gpu_owner
	rm -f pth/*.o
	rm -f pth/tests/test_pth
	rm -f examples/pth_convert
	rm -f metal/*.o metal/kernels/*.o
	rm -f metal/tests/test_metal metal/tests/test_metal_fc metal/tests/test_metal_gemm_batched metal/tests/test_metal_region metal/tests/test_metal_q3_k metal/tests/test_metal_matvec_q5_k metal/tests/test_metal_matvec_q6_k metal/tests/test_metal_rope metal/tests/test_metal_norm metal/tests/test_metal_elementwise
	rm -f metal/tests/test_gemm_q4_k metal/tests/test_gemm_q4_k_f16 metal/tests/bench_dispatch
	rm -f cuda/*.o
	rm -f cuda/tests/test_cuda cuda/tests/test_multi_ctx
	rm -f cuda/tests/test_moe_expert_quants cuda/tests/bench_matvec
	rm -f tt/*.o
	rm -f tt/convert/*.o

# PTH module (PyTorch .pth reader)
pth: pth/pth.o

pth/pth.o: pth/pth.c pth/pth.h
	$(CC) $(CFLAGS) -c $< -o $@

test-pth: pth/tests/test_pth
	./pth/tests/test_pth

pth/tests/test_pth: pth/tests/test_pth.c pth/pth.c pth/pth.h safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -o $@ pth/tests/test_pth.c pth/pth.c safetensors/safetensors.c -lm

examples/pth_convert: examples/pth_convert.c pth/pth.c pth/pth.h safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -o $@ examples/pth_convert.c pth/pth.c safetensors/safetensors.c -lm

pth-convert: examples/pth_convert

# Attention module
attention: attention/attention.o attention/turboquant.o

attention/attention.o: attention/attention.c attention/attention.h tensor/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

# TurboQuant KV compression (tensor-free; raw float*/uint8_t* API).
attention/turboquant.o: attention/turboquant.c attention/turboquant.h attention/tq_codebooks.h
	$(CC) $(CFLAGS) -c $< -o $@

test-attention: attention/tests/test_attention
	./attention/tests/test_attention

attention/tests/test_attention: attention/tests/test_attention.c attention/attention.c attention/attention.h tensor/tensor.c tensor/tensor.h
	$(CC) $(CFLAGS) -o $@ attention/tests/test_attention.c attention/attention.c tensor/tensor.c -lm $(LDFLAGS_BLAS)

test-turboquant: attention/tests/test_turboquant
	./attention/tests/test_turboquant

attention/tests/test_turboquant: attention/tests/test_turboquant.c attention/turboquant.c attention/turboquant.h attention/tq_codebooks.h
	$(CC) $(CFLAGS) -o $@ attention/tests/test_turboquant.c attention/turboquant.c -lm

# KV-quality report: cosine of attention output, full-precision vs compressed,
# across bit widths and synthetic K/V distributions. Not a pass/fail gate (it
# prints distortion figures), so it is run manually rather than from `test`.
validate-turboquant: attention/tests/validate_tq_attention
	./attention/tests/validate_tq_attention

attention/tests/validate_tq_attention: attention/tests/validate_tq_attention.c attention/turboquant.c attention/turboquant.h attention/tq_codebooks.h
	$(CC) $(CFLAGS) -o $@ attention/tests/validate_tq_attention.c attention/turboquant.c -lm

# Distributed module
distributed: distributed/distributed.o

distributed/distributed.o: distributed/distributed.c distributed/distributed.h tensor/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

test-distributed: distributed/tests/test_distributed
	./distributed/tests/test_distributed

distributed/tests/test_distributed: distributed/tests/test_distributed.c distributed/distributed.c distributed/distributed.h tensor/tensor.c tensor/tensor.h
	$(CC) $(CFLAGS) -o $@ distributed/tests/test_distributed.c distributed/distributed.c tensor/tensor.c -lm $(LDFLAGS_BLAS)

# Eval module
eval: eval/eval.o

eval/eval.o: eval/eval.c eval/eval.h data/data.h tokenizer/tokenizer.h
	$(CC) $(CFLAGS) -c $< -o $@

test-eval: eval/tests/test_eval
	./eval/tests/test_eval

eval/tests/test_eval: eval/tests/test_eval.c eval/eval.c eval/eval.h data/data.c data/data.h tokenizer/tokenizer.c tokenizer/tokenizer.h
	$(CC) $(CFLAGS) -o $@ eval/tests/test_eval.c eval/eval.c data/data.c tokenizer/tokenizer.c -lm $(LDFLAGS_BLAS)

# Optim module
optim: optim/optim.o

optim/optim.o: optim/optim.c optim/optim.h tensor/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

test-optim: optim/tests/test_optim
	./optim/tests/test_optim

optim/tests/test_optim: optim/tests/test_optim.c optim/optim.c optim/optim.h tensor/tensor.c tensor/tensor.h
	$(CC) $(CFLAGS) -o $@ optim/tests/test_optim.c optim/optim.c tensor/tensor.c -lm $(LDFLAGS_BLAS)

# Potluck RNG module
test-potluck-rng: potluck/tests/test_pl_rng
	mkdir -p build/potluck_test/rng
	./potluck/tests/test_pl_rng

potluck/tests/test_pl_rng: potluck/tests/test_pl_rng.c potluck/pl_rng.c potluck/pl_rng.h
	$(CC) $(CFLAGS) -pthread -o $@ potluck/tests/test_pl_rng.c potluck/pl_rng.c -lm -pthread

# Potluck LoRA module
test-potluck-lora: potluck/tests/test_pl_lora
	./potluck/tests/test_pl_lora

potluck/tests/test_pl_lora: potluck/tests/test_pl_lora.c potluck/pl_lora.c potluck/pl_lora.h potluck/pl_rng.c potluck/pl_rng.h
	$(CC) $(CFLAGS) -o $@ potluck/tests/test_pl_lora.c potluck/pl_lora.c potluck/pl_rng.c -lm

# Examples
examples: rwkv-vocab examples/train_mlp examples/bench_hybridmem examples/rwkvx_generate examples/test_hybridmem_retrieval examples/test_attn_weights examples/test_hdc_correlation examples/test_pooling_loss examples/test_precision_at_scale examples/test_eviction_boundary examples/test_multi_fact examples/test_filler_sensitivity examples/test_context_scaling examples/test_false_positive examples/test_contiguous_chunk examples/test_query_vector

examples/train_mlp: examples/train_mlp.c tensor/tensor.c tensor/tensor.h optim/optim.c optim/optim.h progress/progress.c progress/progress.h
	$(CC) $(CFLAGS) -o $@ examples/train_mlp.c tensor/tensor.c optim/optim.c progress/progress.c -lm $(LDFLAGS_BLAS)

train: examples/train_mlp
	./examples/train_mlp

examples/bench_hybridmem: examples/bench_hybridmem.c hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h
	$(CC) $(CFLAGS) -o $@ examples/bench_hybridmem.c hybridmem/hybridmem.c hdc/hdc.c tensor/tensor.c safetensors/safetensors.c -lm $(LDFLAGS_BLAS)

bench-hybridmem: examples/bench_hybridmem
	./examples/bench_hybridmem

# RWKV-X inference module
rwkvx: rwkvx/rwkvx.o

rwkvx/rwkvx.o: rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.h safetensors/safetensors.h hybridmem/hybridmem.h hdc/hdc.h
	$(CC) $(CFLAGS) -c $< -o $@

# The RWKV World vocab is third-party data, fetched (pinned URL + sha256) rather
# than committed. Fails loudly if it cannot be fetched or does not verify.
.PHONY: rwkv-vocab
rwkv-vocab:
	sh rwkvx/data/fetch_rwkv_vocab.sh

test-rwkvx: rwkv-vocab rwkvx/tests/test_rwkvx
	./rwkvx/tests/test_rwkvx

rwkvx/tests/test_rwkvx: rwkvx/tests/test_rwkvx.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ rwkvx/tests/test_rwkvx.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/rwkvx_generate: examples/rwkvx_generate.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/rwkvx_generate.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_hybridmem_retrieval: examples/test_hybridmem_retrieval.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_hybridmem_retrieval.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_attn_weights: examples/test_attn_weights.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_attn_weights.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_hdc_correlation: examples/test_hdc_correlation.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_hdc_correlation.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_pooling_loss: examples/test_pooling_loss.c hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_pooling_loss.c hybridmem/hybridmem.c hdc/hdc.c tensor/tensor.c safetensors/safetensors.c $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_precision_at_scale: examples/test_precision_at_scale.c hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_precision_at_scale.c hybridmem/hybridmem.c hdc/hdc.c tensor/tensor.c safetensors/safetensors.c $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_eviction_boundary: examples/test_eviction_boundary.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_eviction_boundary.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_multi_fact: examples/test_multi_fact.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_multi_fact.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_filler_sensitivity: examples/test_filler_sensitivity.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_filler_sensitivity.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_context_scaling: examples/test_context_scaling.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_context_scaling.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_false_positive: examples/test_false_positive.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_false_positive.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

examples/test_contiguous_chunk: examples/test_contiguous_chunk.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_contiguous_chunk.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

test-contiguous-chunk: examples/test_contiguous_chunk

examples/test_query_vector: examples/test_query_vector.c rwkvx/rwkvx.c rwkvx/rwkvx.h tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC)
	$(CC) $(CFLAGS) -o $@ examples/test_query_vector.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC) $(METAL_SRC) $(CUDA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) $(LDFLAGS_METAL) $(LDFLAGS_CUDA)

# Model format converter (safetensors → .btw/.btv)
examples/model_convert: examples/model_convert.c convert/convert.c convert/convert.h safetensors/safetensors.c safetensors/safetensors.h quantization/quantization.c quantization/quantization.h tokenizer/tokenizer.c tokenizer/tokenizer.h
	$(CC) $(CFLAGS) -o $@ examples/model_convert.c convert/convert.c safetensors/safetensors.c quantization/quantization.c tokenizer/tokenizer.c -lm $(LDFLAGS_BLAS)

model-convert: examples/model_convert

# RFA (RWKV-7 + Full Attention hybrid) engine
RFA_DEPS = rwkvx/rfa.c rwkvx/rfa.h rwkvx/rwkvx.c rwkvx/rwkvx.h \
	tensor/tensor.c tensor/tensor.h safetensors/safetensors.c safetensors/safetensors.h \
	hybridmem/hybridmem.c hybridmem/hybridmem.h hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC)

RFA_SRC = rwkvx/rfa.c rwkvx/rwkvx.c tensor/tensor.c safetensors/safetensors.c \
	hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC)

examples/rfa_generate: examples/rfa_generate.c $(RFA_DEPS)
	$(CC) $(CFLAGS) -o $@ examples/rfa_generate.c $(RFA_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS)

rfa-generate: examples/rfa_generate

# RFA training module
RFA_TRAIN_DEPS = rwkvx/rfa_train.c rwkvx/rfa_train.h rwkvx/rfa.c rwkvx/rfa.h \
	rwkvx/rwkvx.c rwkvx/rwkvx.h \
	tensor/tensor.c tensor/tensor.h lora/lora.c lora/lora.h \
	optim/optim.c optim/optim.h \
	safetensors/safetensors.c safetensors/safetensors.h \
	hybridmem/hybridmem.c hybridmem/hybridmem.h \
	hdc/hdc.c hdc/hdc.h $(THREADPOOL_SRC)

RFA_TRAIN_SRC = rwkvx/rfa_train.c rwkvx/rfa.c rwkvx/rwkvx.c tensor/tensor.c lora/lora.c \
	optim/optim.c safetensors/safetensors.c hybridmem/hybridmem.c hdc/hdc.c $(THREADPOOL_SRC)

examples/rfa_train: examples/rfa_train.c $(RFA_TRAIN_DEPS)
	$(CC) $(CFLAGS) -o $@ examples/rfa_train.c $(RFA_TRAIN_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS)

rfa-train: examples/rfa_train

# RFA LoRA merge tool
examples/rfa_lora_merge: examples/rfa_lora_merge.c $(RFA_TRAIN_DEPS)
	$(CC) $(CFLAGS) -o $@ examples/rfa_lora_merge.c $(RFA_TRAIN_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS)

rfa-lora-merge: examples/rfa_lora_merge

test-rfa-train: rwkvx/tests/test_rfa_train
	./rwkvx/tests/test_rfa_train

rwkvx/tests/test_rfa_train: rwkvx/tests/test_rfa_train.c $(RFA_TRAIN_DEPS)
	$(CC) $(CFLAGS) -o $@ rwkvx/tests/test_rfa_train.c $(RFA_TRAIN_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS)

# RFA parallel training tests
test-rfa-parallel: rwkvx/tests/test_rfa_parallel
	./rwkvx/tests/test_rfa_parallel

rwkvx/tests/test_rfa_parallel: rwkvx/tests/test_rfa_parallel.c $(RFA_TRAIN_DEPS)
	$(CC) $(CFLAGS) -pthread -o $@ rwkvx/tests/test_rfa_parallel.c $(RFA_TRAIN_SRC) -lm $(LDFLAGS_BLAS) $(LDFLAGS) -pthread

rwkvx-generate: examples/rwkvx_generate

# SQLite support (opt-in: make HAS_SQLITE=1)
ifdef HAS_SQLITE
  CFLAGS += -DHAS_SQLITE
  LDFLAGS_SQLITE = -lsqlite3
else
  LDFLAGS_SQLITE =
endif

# ============================================================
# Swarm module (custom extension)
# ============================================================

SWARM_SRC = swarm/protocol.c swarm/network.c swarm/coordinator.c swarm/worker.c swarm/config.c
SWARM_HDR = swarm/protocol.h swarm/network.h swarm/coordinator.h swarm/worker.h swarm/config.h
SWARM_OBJ = swarm/protocol.o swarm/network.o swarm/coordinator.o swarm/worker.o swarm/config.o

# Swarm object files
swarm: $(SWARM_OBJ)

swarm/protocol.o: swarm/protocol.c swarm/protocol.h
	$(CC) $(CFLAGS) -c $< -o $@

swarm/network.o: swarm/network.c swarm/network.h swarm/protocol.h
	$(CC) $(CFLAGS) -c $< -o $@

swarm/config.o: swarm/config.c swarm/config.h
	$(CC) $(CFLAGS) -c $< -o $@

swarm/coordinator.o: swarm/coordinator.c swarm/coordinator.h swarm/protocol.h swarm/network.h tensor/tensor.h optim/optim.h
	$(CC) $(CFLAGS) -c $< -o $@

swarm/worker.o: swarm/worker.c swarm/worker.h swarm/protocol.h swarm/network.h tensor/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@

# Swarm binaries
SWARM_DEPS = $(SWARM_SRC) $(SWARM_HDR) tensor/tensor.c tensor/tensor.h optim/optim.c optim/optim.h

bin/swarm_coordinator: swarm/coordinator_main.c $(SWARM_DEPS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ swarm/coordinator_main.c $(SWARM_SRC) tensor/tensor.c optim/optim.c -lm $(LDFLAGS_BLAS)

bin/swarm_worker: swarm/worker_main.c $(SWARM_DEPS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ swarm/worker_main.c $(SWARM_SRC) tensor/tensor.c optim/optim.c -lm $(LDFLAGS_BLAS)

swarm-bin: bin/swarm_coordinator bin/swarm_worker

# Swarm tests
test-swarm: swarm/tests/test_swarm
	./swarm/tests/test_swarm

swarm/tests/test_swarm: swarm/tests/test_swarm.c $(SWARM_DEPS)
	@mkdir -p swarm/tests
	$(CC) $(CFLAGS) -o $@ swarm/tests/test_swarm.c $(SWARM_SRC) tensor/tensor.c optim/optim.c -lm $(LDFLAGS_BLAS)

# ============================================================
# Tensor Comparison Helper (for PyTorch parity testing)
# ============================================================

bin/compare_helper: tensor/testing_suite/compare_helper.c tensor/tensor.c tensor/tensor.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ tensor/testing_suite/compare_helper.c tensor/tensor.c -lm $(LDFLAGS_BLAS)

compare-helper: bin/compare_helper

# Run comparison tests against PyTorch. Needs python3 with torch and numpy.
compare-test: bin/compare_helper
	cd tensor/testing_suite && python3 run_all.py

compare-test-quick: bin/compare_helper
	cd tensor/testing_suite && python3 run_all.py --quick

.PHONY: bench-dispatch all clean test test-progress test-tensor test-tokenizer test-safetensors test-lora test-data test-byte-data test-quantization test-export test-attention test-turboquant validate-turboquant test-distributed test-eval test-optim test-hdc test-hybridmem test-threadpool test-rwkvx test-rfa-train test-rfa-parallel test-barswap test-pth test-swarm test-metal test-cuda test-cuda-multi test-tt test-ingot vram-persist compute-persist xcard-persist roundrobin-persist gpu-switch-probe worker-switch gpu-owner examples train bench-hybridmem rwkvx-generate rfa-generate rfa-train rfa-lora-merge quantize-model pth-convert swarm swarm-bin compare-helper compare-test compare-test-quick $(MODULES)
