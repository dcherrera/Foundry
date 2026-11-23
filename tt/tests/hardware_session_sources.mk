# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# hardware_session_sources.mk — what an ON-CARD test harness must link.
#
# `tests/ft_tt_hardware_session.c` opens the card through `ft_tt_runtime_open_with_driver()`,
# the SAME seam a serve uses, so `FT_TT_DRIVER=kmd|udrv` selects the transport for every
# on-card suite. That pulls the userspace driver into every such link — which is the point:
# before 2026-08-18 the harness called `ft_tt_init_device()` directly and every on-card suite
# was hard-wired to tt-kmd, so the card could have moved to a board with no vendor module
# where NOT ONE of them would run.
#
# ONE list, included rather than copied into seventeen Makefiles, because a source added to
# sixteen of them and forgotten in the seventeenth is a link error in whichever suite is run
# last. Include it AFTER `SRC_DIR` is defined and use `$(HARDWARE_SESSION_SOURCES)` wherever
# `ft_tt_hardware_session.c` used to appear.
#
# Some Makefiles already list some of these (the udrv suites do). Their object lists are
# wrapped in `$(sort ...)`, which de-duplicates — linking one object twice is a duplicate
# symbol, not a warning.

HARDWARE_SESSION_SOURCES := \
    $(SRC_DIR)/ft_tt_runtime.c \
    $(SRC_DIR)/ft_tt_runtime_udrv.c \
    $(SRC_DIR)/ft_tt_udrv_pci_sysfs.c \
    $(SRC_DIR)/ft_tt_udrv_tlb_encode.c \
    $(SRC_DIR)/ft_tt_udrv_noc.c \
    $(SRC_DIR)/ft_tt_udrv_noc_bank.c \
    $(SRC_DIR)/ft_tt_udrv_claim.c \
    $(SRC_DIR)/ft_tt_udrv_arc_msg.c \
    $(SRC_DIR)/ft_tt_udrv_arc_probe.c \
    $(SRC_DIR)/ft_tt_udrv_reset.c \
    ft_tt_hardware_session.c
