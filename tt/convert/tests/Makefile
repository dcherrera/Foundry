# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Foundry Blackhole backend — Phase 3 host-side conversion tests.
#
# Nothing here touches the card: the block-float comparison runs against the
# committed reference vectors in ../reference/ (regenerate: `make bfp-reference`). Everything builds and runs on the
# Mac, which is also where the real conversion runs (spec §8.4).
#
# `make red` exists because a green test nobody has watched fail is not
# evidence (coding-best-practices.md §9). It rebuilds from scratch: Apple ships
# GNU make 3.81, whose one-second mtime granularity silently skips a rebuild of
# a source edited in the same second as its object, so a mutation loop that
# relies on incremental make can be testing the PREVIOUS binary.

CC      ?= cc
CONVERT := ..
TT      := ../..
BUILD   := build

CFLAGS  := -std=c99 -O1 -g \
           -Wall -Wextra -Werror -pedantic \
           -Wshadow -Wconversion -Wstrict-prototypes -Wformat=2 \
           -fstack-protector-strong \
           -I$(CONVERT) -I$(TT) -I$(CONVERT)/reference

# macOS puts libm inside libSystem, so this suite linked on the Mac without it
# and failed to link on the card host the first time anyone tried — the same shape of
# gap as the MADV_DONTNEED one in ft_gguf_map.c. The converter uses sqrt() for
# every error figure it reports, so this is not optional anywhere.
LDLIBS  := -lm

BFP_SOURCES := $(CONVERT)/ft_bfp.c $(TT)/ft_tt_tilize.c test_bfp.c
BFP_TARGET  := $(BUILD)/test_bfp

BFP6_SOURCES := $(CONVERT)/ft_bfp.c test_bfp6.c
BFP6_TARGET  := $(BUILD)/test_bfp6

SPLIT_SOURCES := $(CONVERT)/ft_bfp.c $(CONVERT)/ft_bfp_split.c test_bfp_split.c
SPLIT_TARGET  := $(BUILD)/test_bfp_split

CONVERT_SOURCES := $(CONVERT)/ft_bfp.c \
                   $(CONVERT)/ft_float_convert.c \
                   $(CONVERT)/ft_gguf_map.c \
                   $(CONVERT)/ft_gguf_dequant.c \
                   $(CONVERT)/ft_tt_precision_ledger.c \
                   $(CONVERT)/ft_tt_weight_convert.c \
                   $(TT)/ft_tt_tilize.c \
                   test_convert.c
CONVERT_TARGET  := $(BUILD)/test_convert

MOE_SOURCES := $(CONVERT)/ft_bfp.c \
               $(CONVERT)/ft_float_convert.c \
               $(CONVERT)/ft_gguf_map.c \
               $(CONVERT)/ft_gguf_dequant.c \
               $(CONVERT)/ft_gguf_write.c \
               $(CONVERT)/ft_tt_precision_ledger.c \
               $(CONVERT)/ft_tt_weight_convert.c \
               $(CONVERT)/ft_tt_capacity_plan.c \
               $(TT)/ft_tt_tilize.c \
               test_convert_moe.c
MOE_TARGET  := $(BUILD)/test_convert_moe

CACHE_SOURCES := $(CONVERT)/ft_bfp.c \
                 $(CONVERT)/ft_float_convert.c \
                 $(CONVERT)/ft_gguf_map.c \
                 $(CONVERT)/ft_gguf_dequant.c \
                 $(CONVERT)/ft_tt_precision_ledger.c \
                 $(CONVERT)/ft_tt_weight_convert.c \
                 $(CONVERT)/ft_tt_capacity_plan.c \
                 $(CONVERT)/ft_tt_artifact_provenance.c \
                 $(CONVERT)/ft_tt_artifact_cache.c \
                 $(TT)/ft_tt_tilize.c \
                 test_artifact_cache.c
CACHE_TARGET  := $(BUILD)/test_artifact_cache

# The per-tensor format map (--format-map), on a synthetic Qwen3.8-shaped Q8_0 checkpoint.
FORMAT_MAP_SOURCES := $(CONVERT)/ft_bfp.c \
                      $(CONVERT)/ft_float_convert.c \
                      $(CONVERT)/ft_gguf_map.c \
                      $(CONVERT)/ft_gguf_dequant.c \
                      $(CONVERT)/ft_gguf_write.c \
                      $(CONVERT)/ft_tt_precision_ledger.c \
                      $(CONVERT)/ft_tt_weight_convert.c \
                      $(CONVERT)/ft_tt_format_map.c \
                      $(TT)/ft_tt_tilize.c \
                      synthetic_hybrid_checkpoint.c \
                      test_format_map.c
FORMAT_MAP_TARGET  := $(BUILD)/test_format_map

# SHA-256, the MXTL tile file and the FTTW splice, on the same synthetic checkpoint.
SPLICE_SOURCES := $(CONVERT)/ft_bfp.c \
                  $(CONVERT)/ft_float_convert.c \
                  $(CONVERT)/ft_gguf_map.c \
                  $(CONVERT)/ft_gguf_dequant.c \
                  $(CONVERT)/ft_gguf_write.c \
                  $(CONVERT)/ft_tt_precision_ledger.c \
                  $(CONVERT)/ft_tt_weight_convert.c \
                  $(CONVERT)/ft_tt_format_map.c \
                  $(CONVERT)/ft_tt_artifact_provenance.c \
                  $(CONVERT)/ft_sha256.c \
                  $(CONVERT)/ft_tt_tile_file.c \
                  $(CONVERT)/ft_tt_artifact_splice.c \
                  $(TT)/ft_tt_tilize.c \
                  synthetic_hybrid_checkpoint.c \
                  test_artifact_splice.c
SPLICE_TARGET  := $(BUILD)/test_artifact_splice

# Q8_0->Bfp4 (converter) vs Q8_0->Bfp8->Bfp4 (load-time repack) byte agreement per class.
# Synthetic checkpoint by default; `./build/route_agreement_bfp4 model.gguf` for a real one.
ROUTE_SOURCES := $(CONVERT)/ft_bfp.c \
                 $(CONVERT)/ft_float_convert.c \
                 $(CONVERT)/ft_gguf_map.c \
                 $(CONVERT)/ft_gguf_dequant.c \
                 $(CONVERT)/ft_gguf_write.c \
                 $(CONVERT)/ft_tt_precision_ledger.c \
                 $(CONVERT)/ft_tt_weight_convert.c \
                 $(TT)/ft_tt_tilize.c \
                 synthetic_hybrid_checkpoint.c \
                 route_agreement_bfp4.c
ROUTE_TARGET  := $(BUILD)/route_agreement_bfp4

.PHONY: all check red clean bfp-reference bench format-map splice

# ns a value of the block-float unpack (the CPU oracle's per-weight cost). Not part of check.
bench: | $(BUILD)
	$(CC) $(CFLAGS) $(CONVERT)/ft_bfp.c bench_bfp_unpack.c -o $(BUILD)/bench_bfp_unpack $(LDLIBS)
	./$(BUILD)/bench_bfp_unpack

all: $(BFP_TARGET) $(BFP6_TARGET) $(SPLIT_TARGET) $(CONVERT_TARGET) $(MOE_TARGET) $(CACHE_TARGET) \
     $(FORMAT_MAP_TARGET) $(SPLICE_TARGET) $(ROUTE_TARGET)

$(BUILD):
	mkdir -p $(BUILD)

# One direct cc per program on purpose: whole-program builds cannot serve a
# stale object out of the GNU make 3.81 same-second staleness trap.
$(BFP_TARGET): $(BFP_SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(BFP_SOURCES) -o $@ $(LDLIBS)

$(BFP6_TARGET): $(BFP6_SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(BFP6_SOURCES) -o $@ $(LDLIBS)

$(SPLIT_TARGET): $(SPLIT_SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(SPLIT_SOURCES) -o $@ $(LDLIBS)

$(CONVERT_TARGET): $(CONVERT_SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(CONVERT_SOURCES) -o $@ $(LDLIBS)

$(MOE_TARGET): $(MOE_SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(MOE_SOURCES) -o $@ $(LDLIBS)

$(CACHE_TARGET): $(CACHE_SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(CACHE_SOURCES) -o $@ $(LDLIBS)

$(FORMAT_MAP_TARGET): $(FORMAT_MAP_SOURCES) synthetic_hybrid_checkpoint.h | $(BUILD)
	$(CC) $(CFLAGS) $(FORMAT_MAP_SOURCES) -o $@ $(LDLIBS)

$(SPLICE_TARGET): $(SPLICE_SOURCES) synthetic_hybrid_checkpoint.h | $(BUILD)
	$(CC) $(CFLAGS) $(SPLICE_SOURCES) -o $@ $(LDLIBS)

$(ROUTE_TARGET): $(ROUTE_SOURCES) synthetic_hybrid_checkpoint.h | $(BUILD)
	$(CC) $(CFLAGS) $(ROUTE_SOURCES) -o $@ $(LDLIBS)

# Only the format-map pieces: a small build for a small disk.
format-map: $(FORMAT_MAP_TARGET) $(ROUTE_TARGET)
	./$(FORMAT_MAP_TARGET)
	./$(ROUTE_TARGET)

check: $(BFP_TARGET) $(BFP6_TARGET) $(SPLIT_TARGET) $(CONVERT_TARGET) $(MOE_TARGET) $(CACHE_TARGET) \
       $(FORMAT_MAP_TARGET) $(SPLICE_TARGET) $(ROUTE_TARGET)
	./$(BFP_TARGET)
	./$(BFP6_TARGET)
	./$(SPLIT_TARGET)
	./$(CONVERT_TARGET)
	./$(MOE_TARGET)
	./$(CACHE_TARGET)
	./$(FORMAT_MAP_TARGET)
	./$(SPLICE_TARGET)
	./$(ROUTE_TARGET)

# Full rebuild, then run. Use this for every red/green pass.
red: clean $(BFP_TARGET) $(CONVERT_TARGET) $(MOE_TARGET) $(CACHE_TARGET) $(FORMAT_MAP_TARGET) $(SPLICE_TARGET) \
     $(ROUTE_TARGET)
	-./$(BFP_TARGET)
	-./$(CONVERT_TARGET)
	-./$(MOE_TARGET)
	-./$(CACHE_TARGET)
	-./$(FORMAT_MAP_TARGET)
	-./$(SPLICE_TARGET)
	-./$(ROUTE_TARGET)

# Only the splice pieces.
splice: $(SPLICE_TARGET)
	./$(SPLICE_TARGET)

# Regenerate ../reference/ft_bfp_reference_vectors.h. The generator refuses to
# write unless its independent reference agrees with ft_bfp.c on every word.
bfp-reference: | $(BUILD)
	$(CC) $(CFLAGS) $(CONVERT)/reference/generate_bfp_reference.c $(CONVERT)/ft_bfp.c -o $(BUILD)/generate_bfp_reference $(LDLIBS)
	./$(BUILD)/generate_bfp_reference $(BUILD)/ft_bfp_reference_vectors.h
	cp $(BUILD)/ft_bfp_reference_vectors.h $(CONVERT)/reference/ft_bfp_reference_vectors.h

clean:
	rm -rf $(BUILD)
