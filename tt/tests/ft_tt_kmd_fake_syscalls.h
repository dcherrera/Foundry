/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_kmd_fake_syscalls.h
 * @brief Force-included (`-include`) when Makefile.kmd compiles ft_tt_kmd.c: points the
 *   library's ft_tt_kmd_sys_* seam at fakes in test_kmd.c that record every call and its
 *   argument bytes and replay scripted replies. Never included by production code.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_TT_KMD_FAKE_SYSCALLS_H
#define FT_TT_KMD_FAKE_SYSCALLS_H

#include <stddef.h>
#include <sys/types.h>

int test_kmd_fake_open(const char *path, int flags);
int test_kmd_fake_close(int fd);
int test_kmd_fake_ioctl(int fd, unsigned long request, void *argument);
void *test_kmd_fake_mmap(void *address, size_t length, int protection, int flags, int fd,
                         off_t offset);
int test_kmd_fake_munmap(void *address, size_t length);

#define ft_tt_kmd_sys_open   test_kmd_fake_open
#define ft_tt_kmd_sys_close  test_kmd_fake_close
#define ft_tt_kmd_sys_ioctl  test_kmd_fake_ioctl
#define ft_tt_kmd_sys_mmap   test_kmd_fake_mmap
#define ft_tt_kmd_sys_munmap test_kmd_fake_munmap

#endif /* FT_TT_KMD_FAKE_SYSCALLS_H */
