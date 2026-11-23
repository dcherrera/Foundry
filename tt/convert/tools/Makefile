# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Foundry Blackhole backend — Phase 3 conversion tools.
#
# RUN THE CONVERSION ON THE CARD HOST, not the Mac. Spec §8.4 said the Mac, on the
# reasoning that this is bulk scalar work and the card host is 2 no-AVX cores at
# 1.8 GHz — but the binding constraint is MEMORY, not CPU: peak RSS is 76-215 MB
# on the card host against 641 MB on macOS, where MADV_DONTNEED is advisory-weak and
# the whole GGUF mapping is held. The checkpoints are on the card host anyway.
# MiniCPM5-1B is ~45 s there; the full 50.9 GiB Qwen3.8 is ~23 minutes.
#
#   make
#   ./build/ft_tt_gguf_plan    model.gguf                 # sizes and fit, no conversion
#   ./build/ft_tt_convert_gguf model.gguf model.fttw --ledger production
#   ./build/ft_tt_pcc_gate     model.gguf --format all
#   ./build/ft_tt_fttw_list    model.fttw                 # what an artifact ACTUALLY holds
#   ./build/ft_tt_gguf_slim    model.gguf model.meta.gguf # keys + non-matmul tensors only
#   ./build/ft_tt_fttw_splice  base.fttw --tile-file t.mxtl --out tier.fttw  # splice tiles in
#
# The peak-RSS acceptance is measured, not assumed:
#   /usr/bin/time -l ./build/ft_tt_convert_gguf ...   (macOS)
#   /usr/bin/time -v ./build/ft_tt_convert_gguf ...   (Linux)

CC      ?= cc
CONVERT := ..
TT      := ../..
BUILD   := build

CFLAGS  := -std=c99 -O2 -g \
           -Wall -Wextra -Werror -pedantic \
           -Wshadow -Wconversion -Wstrict-prototypes -Wformat=2 \
           -fstack-protector-strong \
           -I$(CONVERT) -I$(TT)

LDLIBS  := -lm

SOURCES := $(CONVERT)/ft_bfp.c \
           $(CONVERT)/ft_float_convert.c \
           $(CONVERT)/ft_gguf_map.c \
           $(CONVERT)/ft_gguf_dequant.c \
           $(CONVERT)/ft_gguf_write.c \
           $(CONVERT)/ft_tt_precision_ledger.c \
           $(CONVERT)/ft_tt_weight_convert.c \
           $(CONVERT)/ft_tt_format_map.c \
           $(CONVERT)/ft_tt_capacity_plan.c \
           $(CONVERT)/ft_tt_artifact_provenance.c \
           $(CONVERT)/ft_tt_artifact_cache.c \
           $(CONVERT)/ft_sha256.c \
           $(CONVERT)/ft_tt_tile_file.c \
           $(CONVERT)/ft_tt_artifact_splice.c \
           $(TT)/ft_tt_tilize.c

PROGRAMS := ft_tt_convert_gguf ft_tt_pcc_gate ft_tt_ppl_arm_gguf ft_tt_fttw_list ft_tt_gguf_plan ft_tt_gguf_slim \
            ft_tt_fttw_splice
TARGETS  := $(addprefix $(BUILD)/,$(PROGRAMS))

.PHONY: all clean

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

# Whole-program builds on purpose: GNU make 3.81 compares mtimes at one-second
# granularity, so an incremental build can silently serve a stale object
# (coding-best-practices.md §9).
$(BUILD)/%: %.c $(SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(SOURCES) $< -o $@ $(LDLIBS)

clean:
	rm -rf $(BUILD)
