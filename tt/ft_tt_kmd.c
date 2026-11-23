/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_kmd.c
 * @brief Implementation of the kernel-driver boundary: one fd per device, one ioctl per
 *   request, the window aperture mmap, and the NoC fence on CONFIGURE_WINDOW.
 *
 * Every syscall goes through an ft_tt_kmd_sys_* name that defaults to the libc call. The
 * off-device unit test (tests/test_kmd.c) force-includes a header that points them at fakes
 * recording each request; nothing else should define them.
 *
 * @status Complete; unit-tested against fake syscalls.
 * @issues None
 * @todo None
 */

/* POSIX.1-2008 for O_CLOEXEC and mmap under -std=c99. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_kmd.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef ft_tt_kmd_sys_open
#define ft_tt_kmd_sys_open open
#endif
#ifndef ft_tt_kmd_sys_close
#define ft_tt_kmd_sys_close close
#endif
#ifndef ft_tt_kmd_sys_ioctl
#define ft_tt_kmd_sys_ioctl ioctl
#endif
#ifndef ft_tt_kmd_sys_mmap
#define ft_tt_kmd_sys_mmap mmap
#endif
#ifndef ft_tt_kmd_sys_munmap
#define ft_tt_kmd_sys_munmap munmap
#endif

struct ft_tt_kmd_device {
    int fd;
    ft_tt_kmd_resources resources;
};

struct ft_tt_kmd_window {
    uint32_t id;
    size_t size;
    void *aperture;
};

/* ------------------------------------------------------------------------- */
/* Device                                                                    */
/* ------------------------------------------------------------------------- */

/* Record one QUERY_MAPPINGS entry: a UC id carries the BAR's offset AND length, a WC id only
 * its offset. Unknown ids are ignored. */
static void cache_mapping(ft_tt_kmd_resources *resources, const struct ft_tt_kmd_mapping *map)
{
    switch (map->id) {
    case FT_TT_KMD_MAPPING_RESOURCE0_UC:
        resources->bar0_uc_offset = map->base;
        resources->bar0_len = map->size;
        break;
    case FT_TT_KMD_MAPPING_RESOURCE0_WC:
        resources->bar0_wc_offset = map->base;
        break;
    case FT_TT_KMD_MAPPING_RESOURCE1_UC:
        resources->bar2_uc_offset = map->base;
        resources->bar2_len = map->size;
        break;
    case FT_TT_KMD_MAPPING_RESOURCE1_WC:
        resources->bar2_wc_offset = map->base;
        break;
    case FT_TT_KMD_MAPPING_RESOURCE2_UC:
        resources->bar4_uc_offset = map->base;
        resources->bar4_len = map->size;
        break;
    case FT_TT_KMD_MAPPING_RESOURCE2_WC:
        resources->bar4_wc_offset = map->base;
        break;
    default:
        break;
    }
}

int ft_tt_kmd_open(const char *path, ft_tt_kmd_device **out_device, int extra_flags)
{
    struct ft_tt_kmd_query_mappings_request query;
    ft_tt_kmd_device *device;
    size_t index;

    if (!path || !out_device)
        return -EINVAL;

    device = calloc(1, sizeof(*device));
    if (!device)
        return -ENOMEM;

    device->fd = ft_tt_kmd_sys_open(path, O_RDWR | O_CLOEXEC | extra_flags);
    if (device->fd < 0) {
        int rc = -errno;

        free(device);
        return rc;
    }

    memset(&query, 0, sizeof(query));
    query.in.count = FT_TT_KMD_MAPPING_QUERY_COUNT;
    if (ft_tt_kmd_sys_ioctl(device->fd, FT_TT_KMD_IOCTL_QUERY_MAPPINGS, &query) != 0) {
        int rc = -errno;

        (void)ft_tt_kmd_sys_close(device->fd);
        free(device);
        return rc;
    }

    for (index = 0; index < FT_TT_KMD_MAPPING_QUERY_COUNT; index++)
        cache_mapping(&device->resources, &query.mappings[index]);

    *out_device = device;
    return 0;
}

int ft_tt_kmd_close(ft_tt_kmd_device *device)
{
    int rc = 0;

    if (!device)
        return 0;

    if (ft_tt_kmd_sys_close(device->fd) != 0)
        rc = -errno;
    free(device);
    return rc;
}

/* Window counts follow from the BAR lengths, and only a Blackhole is counted. */
static void derive_window_counts(const ft_tt_kmd_device *device, ft_tt_kmd_device_attrs *attrs)
{
    uint64_t count_2m;

    if (attrs->chip_arch != (uint64_t)FT_TT_KMD_ARCH_BLACKHOLE)
        return;

    count_2m = device->resources.bar0_len / FT_TT_KMD_WINDOW_SIZE_2M;
    if (count_2m > FT_TT_KMD_BLACKHOLE_2M_WINDOW_LIMIT)
        count_2m = FT_TT_KMD_BLACKHOLE_2M_WINDOW_LIMIT;
    attrs->window_count_2m = count_2m;
    attrs->window_count_4g = device->resources.bar4_len / FT_TT_KMD_WINDOW_SIZE_4G;
}

int ft_tt_kmd_get_attrs(ft_tt_kmd_device *device, ft_tt_kmd_device_attrs *out_attrs)
{
    struct ft_tt_kmd_device_info_request request;
    ft_tt_kmd_device_attrs attrs;

    if (!device || !out_attrs)
        return -EINVAL;

    memset(&request, 0, sizeof(request));
    request.in.out_size = (uint32_t)sizeof(request.out);
    if (ft_tt_kmd_sys_ioctl(device->fd, FT_TT_KMD_IOCTL_GET_DEVICE_INFO, &request) != 0)
        return -errno;

    memset(&attrs, 0, sizeof(attrs));
    if (request.out.device_id == FT_TT_KMD_PCI_DEVICE_ID_BLACKHOLE)
        attrs.chip_arch = FT_TT_KMD_ARCH_BLACKHOLE;
    else if (request.out.device_id == FT_TT_KMD_PCI_DEVICE_ID_WORMHOLE)
        attrs.chip_arch = FT_TT_KMD_ARCH_WORMHOLE;
    else
        attrs.chip_arch = FT_TT_KMD_ARCH_UNKNOWN;

    attrs.pci_domain = request.out.pci_domain;
    attrs.pci_bus = (uint64_t)((request.out.bdf >> 8) & 0xFFu);
    attrs.pci_device = (uint64_t)((request.out.bdf >> 3) & 0x1Fu);
    attrs.pci_function = (uint64_t)(request.out.bdf & 0x7u);
    attrs.pci_vendor_id = request.out.vendor_id;
    attrs.pci_device_id = request.out.device_id;
    attrs.pci_subsystem_vendor_id = request.out.subsystem_vendor_id;
    attrs.pci_subsystem_id = request.out.subsystem_id;
    derive_window_counts(device, &attrs);

    *out_attrs = attrs;
    return 0;
}

int ft_tt_kmd_get_resources(const ft_tt_kmd_device *device, ft_tt_kmd_resources *out_resources)
{
    if (!device || !out_resources)
        return -EINVAL;

    *out_resources = device->resources;
    return 0;
}

int ft_tt_kmd_fd(const ft_tt_kmd_device *device)
{
    return device ? device->fd : -1;
}

int ft_tt_kmd_request_max_ai_clock(ft_tt_kmd_device *device, bool want_max)
{
    struct ft_tt_kmd_power_state_request request;

    if (!device)
        return -EINVAL;

    memset(&request, 0, sizeof(request));
    request.argsz = (uint32_t)sizeof(request);
    request.validity = (uint8_t)FT_TT_KMD_POWER_VALIDITY(15u, 0u);
    request.flags_word = want_max
                             ? (uint16_t)FT_TT_KMD_POWER_FLAGS_ALL
                             : (uint16_t)(FT_TT_KMD_POWER_FLAGS_ALL &
                                          ~FT_TT_KMD_POWER_FLAG_MAX_AI_CLOCK);
    if (ft_tt_kmd_sys_ioctl(device->fd, FT_TT_KMD_IOCTL_SET_POWER_STATE, &request) != 0)
        return -errno;
    return 0;
}

/* One NOC_CLEANUP carrying `request`'s payload; argsz is stamped here. */
static int issue_noc_cleanup(ft_tt_kmd_device *device,
                             struct ft_tt_kmd_noc_cleanup_request *request)
{
    request->argsz = (uint32_t)sizeof(*request);
    if (ft_tt_kmd_sys_ioctl(device->fd, FT_TT_KMD_IOCTL_NOC_CLEANUP, request) != 0)
        return -errno;
    return 0;
}

int ft_tt_kmd_set_noc_cleanup(ft_tt_kmd_device *device, uint8_t noc, uint8_t x, uint8_t y,
                              uint64_t addr, uint64_t value)
{
    struct ft_tt_kmd_noc_cleanup_request request;

    if (!device)
        return -EINVAL;

    memset(&request, 0, sizeof(request));
    request.enabled = 1u;
    request.x = x;
    request.y = y;
    request.noc = noc;
    request.addr = addr;
    request.data = value;
    return issue_noc_cleanup(device, &request);
}

int ft_tt_kmd_clear_noc_cleanup(ft_tt_kmd_device *device)
{
    struct ft_tt_kmd_noc_cleanup_request request;

    if (!device)
        return -EINVAL;

    memset(&request, 0, sizeof(request));
    return issue_noc_cleanup(device, &request);
}

int ft_tt_kmd_driver_attr(ft_tt_kmd_device *device, enum ft_tt_kmd_driver_attr attr,
                          uint64_t *out_value)
{
    struct ft_tt_kmd_driver_info_request request;

    if (!out_value)
        return -EINVAL;

    memset(&request, 0, sizeof(request));
    if (device) {
        request.in.out_size = (uint32_t)sizeof(request.out);
        if (ft_tt_kmd_sys_ioctl(device->fd, FT_TT_KMD_IOCTL_GET_DRIVER_INFO, &request) != 0)
            return -errno;
    }

    switch (attr) {
    case FT_TT_KMD_DRIVER_API_VERSION:
        *out_value = FT_TT_KMD_ABI_VERSION;
        return 0;
    case FT_TT_KMD_DRIVER_VERSION_MAJOR:
        *out_value = request.out.major;
        break;
    case FT_TT_KMD_DRIVER_VERSION_MINOR:
        *out_value = request.out.minor;
        break;
    case FT_TT_KMD_DRIVER_VERSION_PATCH:
        *out_value = request.out.patch;
        break;
    default:
        return -EINVAL;
    }
    return device ? 0 : -ENODEV;
}

/* ------------------------------------------------------------------------- */
/* Windows                                                                   */
/* ------------------------------------------------------------------------- */

/* One FREE_WINDOW for `id`. */
static int issue_free_window(ft_tt_kmd_device *device, uint32_t id)
{
    struct ft_tt_kmd_free_window_request request;

    memset(&request, 0, sizeof(request));
    request.in.id = id;
    if (ft_tt_kmd_sys_ioctl(device->fd, FT_TT_KMD_IOCTL_FREE_WINDOW, &request) != 0)
        return -errno;
    return 0;
}

int ft_tt_kmd_window_alloc(ft_tt_kmd_device *device, size_t size,
                           enum ft_tt_kmd_cache_mode cache, int protection,
                           ft_tt_kmd_window **out_window)
{
    struct ft_tt_kmd_allocate_window_request request;
    ft_tt_kmd_window *window;
    uint64_t offset;
    void *aperture;

    if (!device || !out_window)
        return -EINVAL;

    window = calloc(1, sizeof(*window));
    if (!window)
        return -ENOMEM;

    memset(&request, 0, sizeof(request));
    request.in.size = (uint64_t)size;
    if (ft_tt_kmd_sys_ioctl(device->fd, FT_TT_KMD_IOCTL_ALLOCATE_WINDOW, &request) != 0) {
        int rc = -errno;

        free(window);
        return rc;
    }

    offset = cache == FT_TT_KMD_CACHE_UC ? request.out.mmap_offset_uc
                                         : request.out.mmap_offset_wc;
    aperture = ft_tt_kmd_sys_mmap(NULL, size, protection, MAP_SHARED, device->fd,
                                  (off_t)offset);
    if (aperture == MAP_FAILED) {
        int rc = -errno;
        int free_rc = issue_free_window(device, request.out.id);

        if (free_rc != 0)
            rc = free_rc;
        free(window);
        return rc;
    }

    window->id = request.out.id;
    window->size = size;
    window->aperture = aperture;
    *out_window = window;
    return 0;
}

int ft_tt_kmd_window_free(ft_tt_kmd_device *device, ft_tt_kmd_window *window)
{
    int rc = 0;
    int free_rc;

    if (!device || !window)
        return -EINVAL;

    /* munmap FIRST: the driver may hand the id to another client the moment it is freed. */
    if (ft_tt_kmd_sys_munmap(window->aperture, window->size) != 0)
        rc = -errno;
    free_rc = issue_free_window(device, window->id);
    if (free_rc != 0)
        rc = free_rc;
    free(window);
    return rc;
}

int ft_tt_kmd_window_aperture(const ft_tt_kmd_window *window, void **out_aperture)
{
    if (!window || !out_aperture)
        return -EINVAL;
    *out_aperture = window->aperture;
    return 0;
}

int ft_tt_kmd_window_id(const ft_tt_kmd_window *window, uint32_t *out_id)
{
    if (!window || !out_id)
        return -EINVAL;
    *out_id = window->id;
    return 0;
}

int ft_tt_kmd_window_size(const ft_tt_kmd_window *window, size_t *out_size)
{
    if (!window || !out_size)
        return -EINVAL;
    *out_size = window->size;
    return 0;
}

#ifdef FT_TT_PHASE1B_NOC_ENABLED

int ft_tt_kmd_window_aim(ft_tt_kmd_device *device, ft_tt_kmd_window *window,
                         const ft_tt_kmd_window_target *target)
{
    struct ft_tt_kmd_configure_window_request request;

    if (!device || !window || !target)
        return -EINVAL;
    if (window->size == 0 || target->addr % (uint64_t)window->size != 0)
        return -EINVAL;

    memset(&request, 0, sizeof(request));
    request.in.id = window->id;
    request.in.config.addr = target->addr;
    request.in.config.x_end = target->x_end;
    request.in.config.y_end = target->y_end;
    request.in.config.x_start = target->x_start;
    request.in.config.y_start = target->y_start;
    request.in.config.noc = target->noc;
    request.in.config.mcast = target->mcast;
    request.in.config.ordering = target->ordering;
    request.in.config.static_vc = target->static_vc;
    if (ft_tt_kmd_sys_ioctl(device->fd, FT_TT_KMD_IOCTL_CONFIGURE_WINDOW, &request) != 0)
        return -errno;
    return 0;
}

#else /* !FT_TT_PHASE1B_NOC_ENABLED */

int ft_tt_kmd_window_aim(ft_tt_kmd_device *device, ft_tt_kmd_window *window,
                         const ft_tt_kmd_window_target *target)
{
    (void)device;
    (void)window;
    (void)target;
    return -EPERM;
}

#endif /* FT_TT_PHASE1B_NOC_ENABLED */

int ft_tt_kmd_window_aim_unicast(ft_tt_kmd_device *device, ft_tt_kmd_window *window,
                                 uint16_t x, uint16_t y, uint64_t addr)
{
    ft_tt_kmd_window_target target;

    memset(&target, 0, sizeof(target));
    target.addr = addr;
    target.x_end = x;
    target.y_end = y;
    return ft_tt_kmd_window_aim(device, window, &target);
}
