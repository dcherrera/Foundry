# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# cuda_kernels/Makefile — Compile CUDA kernels to PTX and embed as C strings
#
# Usage (on GPU server with CUDA 11.4):
#   make                  # Compile all .cu → .ptx → ../cuda/fc_cuda_ptx.h
#   make clean            # Remove .ptx files
#
# The generated fc_cuda_ptx.h is checked into git so foundry can build
# without nvcc on the development machine.

NVCC ?= /usr/local/cuda-11.4/bin/nvcc
ARCH = sm_37

# Phase 2 qwen35moe kernels added: matvec_q3k, rope, moe_routing.
# QK-RMSNorm lives inside normalization.cu (kernel fc_qk_rms_norm).
# SSM Gated-Delta-Net kernels live in ssm_gated_delta.cu (kernels
# fc_ssm_recurrence, fc_ssm_conv1d_silu, fc_ssm_gated_norm).
# On-die causal GQA flash attention lives in attention.cu (kernels
# fc_attn_qz_split, fc_attn_kv_append, fc_attn_flash_prefill).
KERNELS = elementwise normalization matvec_q8 matvec_q3k matvec_kquants rope moe_routing moe_expert_group ssm_gated_delta fused_layer topk_logits attention
PTX_FILES = $(KERNELS:=.ptx)
PTX_HEADER = ../cuda/fc_cuda_ptx.h

.PHONY: all clean

all: $(PTX_HEADER)

# Compile .cu → .ptx
HOST_CC ?= gcc-11
# NVCCFLAGS_EXTRA: extra flags for an A/B build of a compile-time kernel gate,
# e.g. the division-free batched staging loop (matvec_nx_staging.cuh):
#   make NVCCFLAGS_EXTRA=-DFC_NX_STAGE_NESTED=1
# With no extra flags the regenerated header must come out byte-identical to the
# checked-in ../cuda/fc_cuda_ptx.h.
NVCCFLAGS_EXTRA ?=
# -isystem nvcc_shim: wrapper for sys/cdefs.h to fix glibc 2.39 vs nvcc 11.4
%.ptx: %.cu
	$(NVCC) -ptx -arch=$(ARCH) -ccbin $(HOST_CC) -Wno-deprecated-gpu-targets -isystem nvcc_shim $(NVCCFLAGS_EXTRA) -o $@ $<

# Embed all .ptx files into a single C header
$(PTX_HEADER): $(PTX_FILES)
	@echo "Generating $(PTX_HEADER)..."
	@echo '/* SPDX-License-Identifier: AGPL-3.0-or-later */' > $(PTX_HEADER)
	@echo '/* Copyright (C) 2026 David (Corey) Herrera */' >> $(PTX_HEADER)
	@echo '/*' >> $(PTX_HEADER)
	@echo ' * fc_cuda_ptx.h — Auto-generated embedded PTX kernels' >> $(PTX_HEADER)
	@echo ' *' >> $(PTX_HEADER)
	@echo ' * DO NOT EDIT — regenerate with: cd cuda_kernels && make' >> $(PTX_HEADER)
	@echo ' * Compiled for $(ARCH) with CUDA 11.4' >> $(PTX_HEADER)
	@echo ' */' >> $(PTX_HEADER)
	@echo '' >> $(PTX_HEADER)
	@echo '#ifndef FC_CUDA_PTX_H' >> $(PTX_HEADER)
	@echo '#define FC_CUDA_PTX_H' >> $(PTX_HEADER)
	@echo '' >> $(PTX_HEADER)
	@for f in $(PTX_FILES); do \
		name=$$(echo $$f | sed 's/\.ptx//' | tr 'a-z' 'A-Z'); \
		echo "static const char FC_PTX_$$name[] =" >> $(PTX_HEADER); \
		sed 's/\\/\\\\/g; s/"/\\"/g; s/^/    "/; s/$$/\\n"/' $$f >> $(PTX_HEADER); \
		echo '    ;' >> $(PTX_HEADER); \
		echo '' >> $(PTX_HEADER); \
	done
	@echo '#endif /* FC_CUDA_PTX_H */' >> $(PTX_HEADER)
	@echo "Generated $(PTX_HEADER) from $(PTX_FILES)"

clean:
	rm -f $(PTX_FILES)
