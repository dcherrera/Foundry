# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# ft_tt_core — per-core targeting, health probing and the recovery ladder.
#
#   make PHASE1B=1
#   ./build/ft_tt_core status all
#
# PHASE1B=1 is mandatory. Without it every device call returns FT_TT_EPERM before touching
# the NoC — the fence exists because a wrong NoC coordinate hangs the chip and costs a
# physical power cycle (spec §7.3). A diagnostic tool is exactly the thing that must not
# be built with the fence quietly bypassed by default.
#
# ---------------------------------------------------------------------------
# WHY THIS LINKS TWO FILES OUT OF tests/
# ---------------------------------------------------------------------------
# ft_tt_hardware_session.c owns the firmware boot sequence, and ft_tt_phase2_l1_layout.c
# derives the L1 map from where the firmware images landed. Both are labelled test
# scaffolding, and both are exactly what an operational tool needs. Copying the boot
# sequence into tools/ would produce a SECOND, subtly different boot sequence — and every
# ordering constraint in it is load-bearing. They are linked,
# not duplicated. If a second consumer appears they belong in the backend proper.

CC      ?= cc
SRC_DIR := ..
TEST_DIR := ../tests
BUILD   := build

CFLAGS  := -std=c99 -O1 -g \
           -Wall -Wextra -Werror -pedantic \
           -Wshadow -Wconversion -Wstrict-prototypes -Wformat=2 \
           -fstack-protector-strong \
           -I$(SRC_DIR) -I$(TEST_DIR)

DEPFLAGS := -MMD -MP

# The regen targets compile nothing for the host and never open the card, so they are
# exempt: demanding PHASE1B=1 to regenerate a header would be a fence in front of the wrong
# door. `verify-programs` is exempt for the stronger reason that it must NEVER be built with
# the fence open — it compares bytes in this binary against files on disk and has no
# business being able to reach a card at all.
ifeq ($(filter regen-kernels regen-kernels-check regen-programs regen-programs-check \
              verify-programs explode-programs check-kernel-compile-time-args \
              check-program-directory \
              check-program-provenance,$(MAKECMDGOALS)),)
ifndef PHASE1B
$(error ft_tt_core drives the card: build with PHASE1B=1 or every device call is refused)
endif
endif
# Guarded, so the exempt targets above cannot pick the define up by accident. Every
# non-exempt goal has already been stopped by the $(error) unless PHASE1B is set, so this
# changes nothing for them.
ifdef PHASE1B
CFLAGS += -DFT_TT_PHASE1B_NOC_ENABLED
endif

BACKEND_SOURCES := $(SRC_DIR)/ft_tt_kmd.c \
                   $(SRC_DIR)/ft_tt_arch.c \
                   $(SRC_DIR)/ft_tt_arch_blackhole.c \
                   $(SRC_DIR)/ft_tt_device.c \
                   $(SRC_DIR)/ft_tt_tlb.c \
                   $(SRC_DIR)/ft_tt_coord.c \
                   $(SRC_DIR)/ft_tt_arc_telemetry.c \
                   $(SRC_DIR)/ft_tt_noc.c \
                   $(SRC_DIR)/ft_tt_buffer.c \
                   $(SRC_DIR)/ft_tt_elf.c \
                   $(SRC_DIR)/ft_tt_kernel_blob.c \
                   $(SRC_DIR)/ft_tt_udrv_claim.c \
                   $(SRC_DIR)/ft_tt_udrv_reset.c \
                   $(SRC_DIR)/ft_tt_udrv_pci_sysfs.c \
                   $(SRC_DIR)/ft_tt_udrv_tlb_encode.c \
                   $(SRC_DIR)/ft_tt_udrv_noc.c \
           $(SRC_DIR)/ft_tt_udrv_noc_bank.c \
                   $(SRC_DIR)/ft_tt_udrv_arc_msg.c \
                   $(SRC_DIR)/ft_tt_udrv_arc_probe.c \
                   $(SRC_DIR)/ft_tt_runtime.c \
                   $(SRC_DIR)/ft_tt_runtime_udrv.c \
                   $(SRC_DIR)/ft_tt_tensix_claim.c \
                   $(SRC_DIR)/ft_tt_launch.c \
                   $(SRC_DIR)/ft_tt_queue.c \
                   $(SRC_DIR)/ft_tt_core_status.c \
                   $(SRC_DIR)/ft_tt_core_readiness.c

SCAFFOLD_SOURCES := $(TEST_DIR)/ft_tt_hardware_session.c \
                    $(TEST_DIR)/ft_tt_phase2_l1_layout.c \
                    $(TEST_DIR)/ft_tt_program_directory.c

# ft_tt_core_status.c LEFT for tt/ on 2026-08-24, and it is in BACKEND_SOURCES above now.
# Pantheon's arming path became a second consumer of its verdict: it must not boot firmware
# onto a core that is still running, and duplicating this classification into the serve would
# have produced a second, subtly different reading of the same registers. What stayed here is
# the PRESENTATION — the three printers and the operational advice they carry, which is
# addressed to a person at a terminal.
TOOL_SOURCES := ft_tt_core_status_print.c \
                ft_tt_core_probe.c \
                ft_tt_core_recover.c \
                ft_tt_device_reset.c \
                ft_tt_card_lock.c \
                ft_tt_core.c

SOURCES := $(BACKEND_SOURCES) $(SCAFFOLD_SOURCES) $(TOOL_SOURCES)
OBJECTS := $(addprefix $(BUILD)/,$(notdir $(SOURCES:.c=.o)))
TARGET  := $(BUILD)/ft_tt_core

# ft_tt_power_request links NOTHING from the backend, deliberately. It opens the chardev,
# issues one SET_POWER_STATE, and reads sysfs; it never aims a TLB and never issues a NoC
# transaction, so linking the backend into it would only give it the ability to do damage it
# has no reason to be able to do.
POWER_TARGET := $(BUILD)/ft_tt_power_request

.PHONY: all clean rebuild explode-programs regen-kernels regen-kernels-check regen-programs \
        regen-programs-check verify-programs check-kernel-compile-time-args \
        check-program-directory \
        check-program-provenance

all: $(TARGET) $(POWER_TARGET)

# ---------------------------------------------------------------------------
# The one guard ft_tensix_compile_time_arg() does not have built in: it indexes a
# `static const` array with NO bounds check, and SFPI GCC 15.1.0 folds an out-of-range
# read to a literal 0 with no diagnostic at any -Warray-bounds level (measured on a second card host,
# 2026-08-18 -- see the script's own header). The argument count written beside each
# kernel's documented build line is the only thing that catches a short build, so this
# makes that count a real, enforced check instead of a comment someone has to remember to
# read.
#
#   make check-kernel-compile-time-args
#
# Needs nothing -- no card, no toolchain, pure comment/source audit. Runs on the Mac.
check-kernel-compile-time-args:
	python3 check_tensix_kernel_compile_time_arg_counts.py

# ---------------------------------------------------------------------------
# IS THE SHIPPING BLOB CURRENT WITH THE KERNEL SOURCES BESIDE IT?
#
# The mtime check that used to answer this is defeated by an `explode`, which rewrites
# every file and every timestamp: all 21 directories reported `current` while `programs/gdn`
# was five commits behind and the four `rtriad_*` readers had no cooperative multicast at
# all. This asks the question by CONTENT — each embedded image carries a hash of every
# in-tree file its compile opened — so nothing about timestamps can change the answer.
#
#   make check-program-provenance
#
# Needs nothing: no card, no toolchain, no program directories. Runs on the Mac, and it is
# the check to run BEFORE trusting any on-card result, not after.
#
# **IT DOES NOT SEE tt/programs/.** It holds the EMBEDDED BLOB against the sources, which is
# the right question for a serve — an unset `PN_TT_PROGRAM_DIR` loads the blob and nothing
# else. It is the wrong question for an ARM, because an arm that sets `PN_TT_PROGRAM_DIR`
# runs the exploded directory instead, and on 2026-08-19 that directory was 22 images behind
# the blob while this target printed `22 current, 0 stale`. See check-program-directory.
PROGRAMS_HEADER ?= ../ft_tt_programs.h

check-program-provenance:
	python3 check_tensix_program_provenance.py --header $(PROGRAMS_HEADER)

# ---------------------------------------------------------------------------
# IS THE EXPLODED DIRECTORY THE SAME PROGRAM THE BLOB CARRIES?
#
# A DIFFERENT QUESTION FROM check-program-provenance, and it was answered NO while that one
# was green. Measured 2026-08-19: `tt/programs/` held 22 images differing from the embedded
# blob across SIX programs — `gdn` (all five images), `probe`, and all four `rtriad_*`. Both
# consequences were silent and large:
#
#   - with the GDN family armed, `PN_TT_PROGRAM_DIR=tt/programs` REFUSED every descriptor
#     ("gdn descriptor 2 did not complete: error", one byte of output), twice, reproducibly,
#     on a card that ran the blob clean immediately before and after;
#   - the on-disk `rtriad` reader is the PRE-COLUMN-BLOCK matmul, so every arm ever taken
#     through a program directory paid 0.181 s/token more than the blob does — the size of
#     the entire column-block win, charged to whatever that arm happened to be testing.
#
# `verify-programs` could always answer this. It built the tool and then printed the command
# for you to run yourself, so nobody ran it. This runs it, and fails on a difference.
#
#   make check-program-directory
#
# Needs a toolchain but no card.
check-program-directory: verify-programs
	$(BUILD)/verify_tensix_program_blob $(PROGRAMS)

# ---------------------------------------------------------------------------
# The ONE entry point that rebuilds every embedded Tensix kernel and regenerates
# ../ft_tt_kernels.h. There was no such entry point, and that is how the committed header
# went stale: it shipped C++ ELFs from before the Foundry move for weeks. The script also
# refuses to finish quietly when the firmware in FT_TT_FIRMWARE_DIR is not what these
# kernels were linked against -- a kernel's .text sits at its own firmware's
# __fw_export_text_end, so a regen without the matching firmware is a wedged core.
#
#   make regen-kernels                     rebuild + write the header; FAIL if firmware moved
#   make regen-kernels INSTALL_FIRMWARE=1  ... and back up + install the new firmware
#   make regen-kernels-check               rebuild + diff only; writes nothing
#
# Needs the SFPI toolchain, i.e. the card host. OUT defaults to /tmp/ftk_regen.
OUT ?= /tmp/ftk_regen
INSTALL_FIRMWARE ?= 0

regen-kernels:
	INSTALL_FIRMWARE=$(INSTALL_FIRMWARE) bash regen_tensix_kernel_header.sh $(OUT)

regen-kernels-check:
	CHECK=1 bash regen_tensix_kernel_header.sh $(OUT)

# ---------------------------------------------------------------------------
# The PROGRAM blob: whole build_tensix_program.sh directories -- five kernel ELFs plus the
# five fw_*.elf they were linked against -- embedded into ../ft_tt_programs.h, so a binary
# carries its Tensix programs instead of reading them out of somebody's home directory.
# That hardcoded path (PN_TT_PROGRAM_ROOT_DEFAULT) is why the serve runs on one machine, and
# it is why rtriad_bfp8 could go missing for a day and cost 1.085x silently.
#
#   make regen-programs         regenerate ../ft_tt_programs.h from ../programs/*
#   make regen-programs-check   diff only; writes nothing
#   make verify-programs        build the byte-for-byte comparison tool
#
# Needs the program directories, i.e. the card host. The regen has NEVER been run for real; the
# mechanism is gated against synthetic fixtures by tests/Makefile.programblob.
PROGRAMS ?= ../programs

regen-programs:
	bash regen_tensix_program_header.sh

regen-programs-check:
	CHECK=1 bash regen_tensix_program_header.sh

# Whole-program link, into its own binary and with NO shared objects: every other object in
# $(BUILD) is compiled with the NoC fence OPEN, and mixing one of those into a tool that is
# supposed to be unable to reach the card would quietly give it that ability.
VERIFY_SOURCES := $(SRC_DIR)/ft_tt_kmd.c \
                  $(SRC_DIR)/ft_tt_arch.c \
                  $(SRC_DIR)/ft_tt_arch_blackhole.c \
                  $(SRC_DIR)/ft_tt_device.c \
                  $(SRC_DIR)/ft_tt_elf.c \
                  $(SRC_DIR)/ft_tt_program_blob.c \
                  verify_tensix_program_blob.c

verify-programs: | $(BUILD)
	rm -f $(BUILD)/verify_tensix_program_blob
	$(CC) $(CFLAGS) $(VERIFY_SOURCES) -o $(BUILD)/verify_tensix_program_blob
	@echo "built $(BUILD)/verify_tensix_program_blob — run it as:"
	@echo "    $(BUILD)/verify_tensix_program_blob $(PROGRAMS)"

# The blob back out as DIRECTORIES, so a machine with no tt/programs/ can still regenerate
# it after one kernel changes. Same whole-program link and the same reason: this tool must
# not be able to reach the card.
EXPLODE_SOURCES := $(SRC_DIR)/ft_tt_kmd.c \
                   $(SRC_DIR)/ft_tt_arch.c \
                   $(SRC_DIR)/ft_tt_arch_blackhole.c \
                   $(SRC_DIR)/ft_tt_device.c \
                   $(SRC_DIR)/ft_tt_elf.c \
                   $(SRC_DIR)/ft_tt_program_blob.c \
                   explode_tensix_program_blob.c

explode-programs: | $(BUILD)
	rm -f $(BUILD)/explode_tensix_program_blob
	$(CC) $(CFLAGS) $(EXPLODE_SOURCES) -o $(BUILD)/explode_tensix_program_blob
	@echo "built $(BUILD)/explode_tensix_program_blob — run it as:"
	@echo "    $(BUILD)/explode_tensix_program_blob <outdir> --verify --recipes <file>"

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/%.o: $(TEST_DIR)/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD)/%.o: %.c | $(BUILD)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ -lm

$(POWER_TARGET): $(BUILD)/ft_tt_power_request.o
	$(CC) $(CFLAGS) $< -o $@

# `rebuild` deletes the tree first. GNU make 3.81 compares mtimes at one-second
# granularity, so a source edited in the same second as its object is silently not
# rebuilt and you test the PREVIOUS binary (coding-best-practices §9). rsync -a preserving
# the Mac's mtimes has the same effect across machines.
rebuild: clean all

clean:
	rm -rf $(BUILD)

-include $(wildcard $(BUILD)/*.d)
