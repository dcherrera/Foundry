# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Foundry Blackhole backend — Phase 1a tests.
#
# The device test is READ-ONLY against the card. It runs only on the card host
# (the card host); on any other machine `make check` refuses instead of failing.
#
# Two build modes:
#   make            Phase 1a. The NoC fence is CLOSED: CONFIGURE_TLB cannot be
#                   issued, TLB apertures map PROT_NONE, and the TLB register
#                   file maps read-only.
#   make PHASE1B=1  Opens the fence. DO NOT USE UNSUPERVISED — this is what
#                   lets a wrong NoC coordinate hang the card (spec §7.3).
#
# `make red` exists because a green test nobody has watched fail is not
# evidence (coding-best-practices.md §9). It rebuilds from scratch each time:
# Apple ships GNU make 3.81, whose one-second mtime granularity silently skips
# a rebuild of a source edited in the same second as its object, so a mutation
# loop that relies on incremental make can be testing the PREVIOUS binary.

CC      ?= cc
SRC_DIR := ..

# The on-card harness opens the card through the product transport seam.
include hardware_session_sources.mk
BUILD   := build

CFLAGS  := -std=c99 -O1 -g \
           -Wall -Wextra -Werror -pedantic \
           -Wshadow -Wconversion -Wstrict-prototypes -Wformat=2 \
           -fstack-protector-strong \
           -I$(SRC_DIR)

# Header dependency tracking. Without it, editing a HEADER rebuilds nothing and
# the link silently mixes objects compiled against two different struct layouts
# — which is not a build error, it is a wrong offset and a segfault (or worse, a
# wrong answer). Adding a field to ft_tt_grid_info reproduced exactly that on
# 2026-08-12. `-MMD -MP` makes every .o depend on the headers it read.
DEPFLAGS := -MMD -MP


ifdef PHASE1B
CFLAGS += -DFT_TT_PHASE1B_NOC_ENABLED
endif

SOURCES := $(SRC_DIR)/ft_tt_kmd.c \
           $(SRC_DIR)/ft_tt_arch.c \
           $(SRC_DIR)/ft_tt_arch_blackhole.c \
           $(SRC_DIR)/ft_tt_device.c \
           $(SRC_DIR)/ft_tt_tlb.c \
           $(SRC_DIR)/ft_tt_coord.c \
           $(SRC_DIR)/ft_tt_arc_telemetry.c \
           test_device.c

OBJECTS := $(sort $(addprefix $(BUILD)/,$(notdir $(SOURCES:.c=.o))))
TARGET  := $(BUILD)/test_device

# Phase 2 run control. Runs ANYWHERE — it needs no card, because every device touch
# goes through the ft_tt_noc_access vtable and the test supplies a simulated Tensix.
LAUNCH_SOURCES := $(SRC_DIR)/ft_tt_kmd.c \
                  $(SRC_DIR)/ft_tt_arch.c \
                  $(SRC_DIR)/ft_tt_arch_blackhole.c \
                  $(SRC_DIR)/ft_tt_device.c \
                  $(SRC_DIR)/ft_tt_tlb.c \
                  $(SRC_DIR)/ft_tt_coord.c \
                  $(SRC_DIR)/ft_tt_noc.c \
                  $(SRC_DIR)/ft_tt_buffer.c \
                  $(SRC_DIR)/ft_tt_elf.c \
                  $(SRC_DIR)/ft_tt_kernel_blob.c \
                  $(SRC_DIR)/ft_tt_udrv_claim.c \
                  $(SRC_DIR)/ft_tt_tensix_claim.c \
                  $(SRC_DIR)/ft_tt_launch.c \
                  ft_tt_phase2_l1_layout.c \
                  test_launch.c

LAUNCH_OBJECTS := $(sort $(addprefix $(BUILD)/,$(notdir $(LAUNCH_SOURCES:.c=.o))))
LAUNCH_TARGET  := $(BUILD)/test_launch

# Phase 2 ON-CARD checklists — docs/tt_launch_notes.md §6 steps 5-13, one binary per
# group of steps. These need the card AND the NoC fence open, so they are built only
# with PHASE2=1 and refuse to build without PHASE1B. Firmware ELFs come from
# FT_TT_FIRMWARE_DIRECTORY at run time; the kernels are embedded in ft_tt_kernels.h.
PHASE2_BACKEND := $(SRC_DIR)/ft_tt_kmd.c \
                  $(SRC_DIR)/ft_tt_arch.c \
                  $(SRC_DIR)/ft_tt_arch_blackhole.c \
                  $(SRC_DIR)/ft_tt_device.c \
                  $(SRC_DIR)/ft_tt_tlb.c \
                  $(SRC_DIR)/ft_tt_arc_telemetry.c \
                  $(SRC_DIR)/ft_tt_coord.c \
                  $(SRC_DIR)/ft_tt_noc.c \
                  $(SRC_DIR)/ft_tt_elf.c \
                  $(SRC_DIR)/ft_tt_kernel_blob.c \
                  $(SRC_DIR)/ft_tt_udrv_claim.c \
                  $(SRC_DIR)/ft_tt_tensix_claim.c \
                  $(SRC_DIR)/ft_tt_launch.c

PHASE2_SUPPORT := ft_tt_test_report.c $(HARDWARE_SESSION_SOURCES) ft_tt_phase2_l1_layout.c

PHASE2_PROGRAMS := test_phase2_boot test_phase2_images test_phase2_launch \
                   test_phase2_vector_add test_phase2_multicore

PHASE2_TARGETS := $(addprefix $(BUILD)/,$(PHASE2_PROGRAMS))

# Phase 3 tile layout. Pure arithmetic, runs anywhere, needs nothing.
TILIZE_SOURCES := $(SRC_DIR)/ft_tt_tilize.c test_tilize.c
TILIZE_TARGET  := $(BUILD)/test_tilize

# Phase 7 row-0-only transfers. Runs anywhere: a simulated DRAM bank stands in for the
# accessor and enforces the real 64 B read / 16 B write alignment, so a mutation that asks
# for a 32 B read is refused here exactly as the card refuses it. ft_tt_noc.c is linked for
# its transfer checks — the proof that the bulk path cannot carry a register access.
ROW0_SOURCES := $(SRC_DIR)/ft_tt_kmd.c \
                $(SRC_DIR)/ft_tt_arch.c \
                $(SRC_DIR)/ft_tt_arch_blackhole.c \
                $(SRC_DIR)/ft_tt_device.c \
                $(SRC_DIR)/ft_tt_tlb.c \
                $(SRC_DIR)/ft_tt_coord.c \
                $(SRC_DIR)/ft_tt_noc.c \
                $(SRC_DIR)/ft_tt_tilize.c \
                $(SRC_DIR)/ft_tt_tile_row0.c \
                test_tile_row0.c
ROW0_TARGET  := $(BUILD)/test_tile_row0

.PHONY: all check red check-launch red-launch check-tilize red-tilize check-row0 red-row0 \
        phase2 clean

all: $(TARGET) $(LAUNCH_TARGET) $(TILIZE_TARGET) $(ROW0_TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/%.o: %.c | $(BUILD)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@

$(LAUNCH_TARGET): $(LAUNCH_OBJECTS)
	$(CC) $(CFLAGS) $(LAUNCH_OBJECTS) -o $@

# Whole-program build: see the note on the Phase 2 rule below.
$(TILIZE_TARGET): $(TILIZE_SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(TILIZE_SOURCES) -o $@

# Whole-program too, for the same reason, and because run_tile_row0_red.sh rebuilds it
# against a mutated copy of the source and must never link a stale object of the original.
$(ROW0_TARGET): $(ROW0_SOURCES) | $(BUILD)
	$(CC) $(CFLAGS) $(ROW0_SOURCES) -o $@

# One direct cc per program: these are whole-program builds on purpose, so the GNU make
# 3.81 same-second staleness trap (best-practices §9) cannot serve a stale object.
$(BUILD)/test_phase2_%: test_phase2_%.c $(PHASE2_BACKEND) $(PHASE2_SUPPORT) | $(BUILD)
ifndef PHASE1B
	$(error Phase 2 hardware tests need PHASE1B=1 — they drive the NoC)
endif
	$(CC) $(CFLAGS) $(PHASE2_BACKEND) $(PHASE2_SUPPORT) $< -o $@

# Build every on-card checklist binary. Running them is deliberately NOT a make target:
# they drive silicon and belong in a supervised, detached run.
phase2: $(PHASE2_TARGETS)

check: $(TARGET)
	@test -c /dev/tenstorrent/0 || { \
	    echo "no /dev/tenstorrent/0 — this test needs the card host"; exit 77; }
	./$(TARGET)

check-launch: $(LAUNCH_TARGET)
	./$(LAUNCH_TARGET)

# Full rebuild, then run. Use this for every red/green pass.
red: clean $(TARGET)
	-./$(TARGET)

red-launch: clean $(LAUNCH_TARGET)
	-./$(LAUNCH_TARGET)

check-tilize: $(TILIZE_TARGET)
	./$(TILIZE_TARGET)

red-tilize: clean $(TILIZE_TARGET)
	-./$(TILIZE_TARGET)

check-row0: $(ROW0_TARGET)
	./$(ROW0_TARGET)

red-row0: clean $(ROW0_TARGET)
	-./$(ROW0_TARGET)

clean:
	rm -rf $(BUILD)

-include $(wildcard $(BUILD)/*.d)
