# SPDX-License-Identifier: GPL-2.0-only OR AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
obj-m += bar_swap.o

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean
