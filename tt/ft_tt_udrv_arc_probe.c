/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_arc_probe.c
 * @brief Implementation of the ARC probe sequence (see ft_tt_udrv_arc_probe.h).
 *
 * @status On silicon 2026-08-17.
 * @issues None
 * @todo None
 */

/* `-std=c99 -pedantic` hides clock_gettime and nanosleep behind a feature-test
 * macro on glibc while macOS declares them unconditionally — the cross-machine
 * trap DEV_NOTES §5 records. Declared before any include. */
#define _POSIX_C_SOURCE 200809L

#include "ft_tt_udrv_arc_probe.h"

#include <string.h>
#include <time.h>

#include "ft_tt_udrv.h"

/** True when `[address, address + 4)` lies inside the ARC's CSM. Byte for byte
 *  the bound tt-kmd's `is_range_within_csm()` applies. */
static bool address_is_within_csm(uint32_t address)
{
    uint64_t limit = (uint64_t)FT_TT_UDRV_ARC_CSM_BASE + (uint64_t)FT_TT_UDRV_ARC_CSM_SIZE;

    return ((uint64_t)address >= (uint64_t)FT_TT_UDRV_ARC_CSM_BASE) &&
           (((uint64_t)address + 4u) <= limit);
}

static double seconds_now(void)
{
    struct timespec now;

    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void sleep_microseconds(unsigned microseconds)
{
    struct timespec request;

    request.tv_sec = (time_t)(microseconds / 1000000u);
    request.tv_nsec = (long)(microseconds % 1000000u) * 1000L;
    (void)nanosleep(&request, NULL);
}

/* ------------------------------------------------------------------------ *
 * The transport
 * ------------------------------------------------------------------------ */

int ft_tt_udrv_arc_noc_read32(ft_tt_udrv_arc_noc_context *context, uint32_t address,
                              uint32_t *out_value)
{
    int rc;

    if (!context || !context->path || !out_value)
        return FT_TT_UDRV_EINVAL;

    rc = ft_tt_udrv_noc_transfer(context->path, context->x, context->y, (uint64_t)address,
                                 out_value, sizeof(*out_value), true, FT_TT_NOC_WIDTH_WORD32);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    context->reads++;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_noc_write32(ft_tt_udrv_arc_noc_context *context, uint32_t address,
                               uint32_t value)
{
    int rc;

    if (!context || !context->path)
        return FT_TT_UDRV_EINVAL;

    rc = ft_tt_udrv_noc_transfer(context->path, context->x, context->y, (uint64_t)address, &value,
                                 sizeof(value), false, FT_TT_NOC_WIDTH_WORD32);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    context->writes++;
    return FT_TT_UDRV_OK;
}

/** Transport read: the raw accessor with the CSM bound the ring is entitled to. */
static int transport_csm_read32(void *opaque, uint32_t address, uint32_t *out_value)
{
    ft_tt_udrv_arc_noc_context *context = (ft_tt_udrv_arc_noc_context *)opaque;

    if (!address_is_within_csm(address))
        return FT_TT_UDRV_EINVAL;

    return ft_tt_udrv_arc_noc_read32(context, address, out_value);
}

static int transport_csm_write32(void *opaque, uint32_t address, uint32_t value)
{
    ft_tt_udrv_arc_noc_context *context = (ft_tt_udrv_arc_noc_context *)opaque;

    if (!address_is_within_csm(address))
        return FT_TT_UDRV_EINVAL;

    return ft_tt_udrv_arc_noc_write32(context, address, value);
}

/**
 * The doorbell: write 0 to ARC_MSI_FIFO on the ARC tile. Outside the CSM, so it
 * uses the raw accessor — the same split tt-kmd makes.
 */
static int transport_trigger(void *opaque)
{
    ft_tt_udrv_arc_noc_context *context = (ft_tt_udrv_arc_noc_context *)opaque;
    int rc;

    if (!context)
        return FT_TT_UDRV_EINVAL;

    rc = ft_tt_udrv_arc_noc_write32(context, FT_TT_UDRV_ARC_MSI_FIFO, 0u);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    context->triggers++;
    return FT_TT_UDRV_OK;
}

static void transport_wait(void *opaque)
{
    (void)opaque;
    sleep_microseconds(FT_TT_UDRV_ARC_POLL_WAIT_US);
}

int ft_tt_udrv_arc_transport_init(ft_tt_udrv_arc_noc_context *context, ft_tt_udrv_noc_path *path,
                                  uint16_t x, uint16_t y,
                                  ft_tt_udrv_arc_transport *out_transport)
{
    if (!context || !path || !out_transport)
        return FT_TT_UDRV_EINVAL;

    /* A write-combined path would let a ring-pointer store overtake the payload
     * it publishes. That is not slow, it is a message the firmware reads half
     * of. */
    if (path->write_combined)
        return FT_TT_UDRV_EINVAL;

    memset(context, 0, sizeof(*context));
    context->path = path;
    context->x = x;
    context->y = y;

    memset(out_transport, 0, sizeof(*out_transport));
    out_transport->csm_read32 = transport_csm_read32;
    out_transport->csm_write32 = transport_csm_write32;
    out_transport->trigger = transport_trigger;
    out_transport->wait = transport_wait;
    out_transport->context = context;

    return FT_TT_UDRV_OK;
}

/* ------------------------------------------------------------------------ *
 * Locating the queue
 * ------------------------------------------------------------------------ */

int ft_tt_udrv_arc_locate_queue(ft_tt_udrv_arc_noc_context *context, unsigned timeout_ms,
                                ft_tt_udrv_arc_locate_report *out_report,
                                ft_tt_udrv_arc_queue *out_queue)
{
    ft_tt_udrv_arc_locate_report report;
    double deadline;
    uint32_t boot_status = 0u;
    uint32_t qcb_pointer = 0u;
    int rc;

    if (!context || !out_queue)
        return FT_TT_UDRV_EINVAL;

    memset(&report, 0, sizeof(report));
    deadline = seconds_now() + ((double)timeout_ms / 1000.0);

    for (;;) {
        rc = ft_tt_udrv_arc_noc_read32(context, FT_TT_UDRV_ARC_BOOT_STATUS, &boot_status);
        if (rc != FT_TT_UDRV_OK) {
            report.failure_stage = "the NoC read of ARC_BOOT_STATUS did not complete — nothing "
                                   "was read, so the boot status below is not a value the card "
                                   "returned. Check PCI_COMMAND bit 1 (memory decode) first";
            goto done;
        }
        /* Recorded on EVERY exit, not just the one that succeeds. It used to be
         * assigned after the loop, so both failure paths reported 0x00000000 —
         * a value nobody read, and one that reads as a plausible "the ARC is
         * asleep". On a second card host the true value was 0xFFFFFFFF (memory decode off),
         * which names the cause on sight; the zero cost a bisect. */
        report.boot_status = boot_status;

        /* All ones is what a hung fabric or a device that has gone away reads
         * as, and it also happens to have the ready bit set. */
        if (boot_status == 0xFFFFFFFFu) {
            report.failure_stage = "ARC_BOOT_STATUS reads all ones — the card is not answering "
                                   "at all (memory decode off, or the device has gone away)";
            rc = FT_TT_UDRV_EIO;
            goto done;
        }

        if (boot_status & FT_TT_UDRV_ARC_BOOT_STATUS_READY_FOR_MSG)
            break;

        if (seconds_now() >= deadline) {
            report.failure_stage = "the ARC never set ARC_BOOT_STATUS bit 0 (ready for a "
                                   "message) within the timeout";
            rc = FT_TT_UDRV_EIO;
            goto done;
        }

        report.boot_status_polls++;
        sleep_microseconds(1000u);
    }

    rc = ft_tt_udrv_arc_noc_read32(context, FT_TT_UDRV_ARC_MSG_QCB_PTR, &qcb_pointer);
    if (rc != FT_TT_UDRV_OK) {
        report.failure_stage = "the NoC read of RESET_SCRATCH(11), the queue-control-block "
                               "pointer, did not complete";
        goto done;
    }
    report.qcb_pointer = qcb_pointer;

    /* Bounds-checked BEFORE it is dereferenced, so "no queue published yet"
     * reports as ENODEV instead of arriving as an EIO from the accessor. */
    if (!address_is_within_csm(qcb_pointer) || !address_is_within_csm(qcb_pointer + 4u)) {
        report.failure_stage = "RESET_SCRATCH(11) does not point inside the ARC CSM — firmware "
                               "has not published a message queue";
        rc = FT_TT_UDRV_ENODEV;
        goto done;
    }

    rc = ft_tt_udrv_arc_noc_read32(context, qcb_pointer, &report.qcb_word0);
    if (rc != FT_TT_UDRV_OK)
        goto done;

    rc = ft_tt_udrv_arc_noc_read32(context, qcb_pointer + 4u, &report.qcb_word1);
    if (rc != FT_TT_UDRV_OK)
        goto done;

    rc = ft_tt_udrv_arc_queue_init(report.qcb_word0, report.qcb_word1, out_queue);

done:
    if (out_report)
        *out_report = report;
    return rc;
}

/* ------------------------------------------------------------------------ *
 * Exchanges
 * ------------------------------------------------------------------------ */

int ft_tt_udrv_arc_exchange_send(const ft_tt_udrv_arc_transport *transport,
                                 const ft_tt_udrv_arc_queue *queue,
                                 const ft_tt_udrv_arc_msg *request, uint32_t max_polls,
                                 ft_tt_udrv_arc_exchange *out_exchange)
{
    ft_tt_udrv_arc_msg message;
    double started;
    int rc;

    if (!out_exchange)
        return FT_TT_UDRV_EINVAL;

    memset(out_exchange, 0, sizeof(*out_exchange));

    if (!transport || !queue || !request) {
        out_exchange->rc = FT_TT_UDRV_EINVAL;
        return out_exchange->rc;
    }

    message = *request;
    out_exchange->request_header = request->header;
    memcpy(out_exchange->request_payload, request->payload, sizeof(out_exchange->request_payload));

    started = seconds_now();
    rc = ft_tt_udrv_arc_send_sync(transport, queue, &message,
                                  (max_polls == 0u) ? FT_TT_UDRV_ARC_DEFAULT_POLLS : max_polls);
    out_exchange->elapsed_seconds = seconds_now() - started;

    /* The response overwrote `message` in place whether the status was good or
     * bad, and the firmware's own words are what a caller needs to see. */
    out_exchange->response_header = message.header;
    memcpy(out_exchange->response_payload, message.payload,
           sizeof(out_exchange->response_payload));
    out_exchange->rc = rc;

    return rc;
}

int ft_tt_udrv_arc_power_request_build(bool max_ai_clock, ft_tt_udrv_arc_msg *out_msg)
{
    uint16_t flags = (uint16_t)FT_TT_UDRV_ARC_POWER_FLAG_ALL;

    if (!max_ai_clock)
        flags = (uint16_t)(flags & ~(uint16_t)FT_TT_UDRV_ARC_POWER_FLAG_MAX_AI_CLK);

    /* Validity 0x0F is FT_TT_KMD_POWER_VALIDITY(15, 0): fifteen valid flags, no valid
     * settings — exactly what tt_cdev_open() builds. */
    return ft_tt_udrv_arc_msg_power_setting(0x0Fu, flags, NULL, out_msg);
}

/* ------------------------------------------------------------------------ *
 * The sequence
 * ------------------------------------------------------------------------ */

int ft_tt_udrv_arc_probe_hardware(const ft_tt_udrv_arc_transport *transport, uint32_t watchdog_ms,
                                  bool max_ai_clock, ft_tt_udrv_arc_probe_report *out_report)
{
    ft_tt_udrv_arc_probe_report report;
    ft_tt_udrv_arc_noc_context *context;
    ft_tt_udrv_arc_queue queue;
    ft_tt_udrv_arc_msg message;
    int rc;

    if (!transport || !transport->context)
        return FT_TT_UDRV_EINVAL;

    context = (ft_tt_udrv_arc_noc_context *)transport->context;
    memset(&report, 0, sizeof(report));

    rc = ft_tt_udrv_arc_locate_queue(context, FT_TT_UDRV_ARC_READY_TIMEOUT_MS, &report.locate,
                                     &queue);
    if (rc != FT_TT_UDRV_OK)
        goto done;

    rc = ft_tt_udrv_arc_msg_asic_state0(&message);
    if (rc != FT_TT_UDRV_OK)
        goto done;
    rc = ft_tt_udrv_arc_exchange_send(transport, &queue, &message, 0u, &report.asic_state0);
    if (rc != FT_TT_UDRV_OK)
        goto done;

    rc = ft_tt_udrv_arc_msg_set_wdt_timeout(
        (watchdog_ms == 0u) ? FT_TT_UDRV_ARC_WATCHDOG_DEFAULT_MS : watchdog_ms, &message);
    if (rc != FT_TT_UDRV_OK)
        goto done;
    rc = ft_tt_udrv_arc_exchange_send(transport, &queue, &message, 0u, &report.set_wdt_timeout);
    if (rc != FT_TT_UDRV_OK)
        goto done;

    rc = ft_tt_udrv_arc_power_request_build(max_ai_clock, &message);
    if (rc != FT_TT_UDRV_OK)
        goto done;
    rc = ft_tt_udrv_arc_exchange_send(transport, &queue, &message, 0u, &report.power_setting);

done:
    if (out_report)
        *out_report = report;
    return rc;
}
