/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_tensix_claim.c
 * @brief Implementation of per-core Tensix claims (see ft_tt_tensix_claim.h for
 *   the card-wide/per-core split and the reasoning behind every part of it).
 *
 * @status Off-hardware complete; this module never touches the card.
 * @issues None
 * @todo None
 */

#include "ft_tt_tensix_claim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_udrv.h"

struct ft_tt_tensix_claim_set {
    ft_tt_udrv_claim_namespace ns;
    ft_tt_udrv_claim **claims;
    uint32_t *indices;
    size_t count;
};

/** The reserved set is small by nature — a handful of families plus a serve. */
#define RESERVED_CAPACITY 128u

static void refusal_reset(ft_tt_tensix_claim_refusal *refusal)
{
    if (refusal)
        memset(refusal, 0, sizeof(*refusal));
}

static void refusal_set(ft_tt_tensix_claim_refusal *refusal,
                        ft_tt_tensix_claim_outcome outcome, uint32_t index, int detail)
{
    if (!refusal)
        return;
    refusal->outcome = outcome;
    refusal->index = index;
    refusal->detail = detail;
}

/**
 * Name the live holder of `index` for a refusal message.
 *
 * The pid comes from the advisory record next to the lock, and the record can be
 * absent — a holder that died between taking the flock and writing the record
 * leaves none. That is not a failure: a refusal without a name is still a
 * refusal, and printing pid 0 is honest about not knowing.
 */
static void describe_holder(const ft_tt_udrv_claim_namespace *ns, uint32_t index,
                            ft_tt_tensix_claim_refusal *refusal)
{
    ft_tt_udrv_claim_info *info;

    if (!refusal)
        return;

    info = calloc((size_t)ns->count, sizeof(*info));
    if (!info)
        return;

    if (ft_tt_udrv_claim_scan(ns, info, (size_t)ns->count, NULL) == FT_TT_UDRV_OK &&
        index < ns->count) {
        refusal->pid = info[index].pid;
        memcpy(refusal->reason, info[index].reason, sizeof(refusal->reason));
        refusal->reason[sizeof(refusal->reason) - 1u] = '\0';
    }
    free(info);
}

static bool index_is_reserved(uint32_t index, const uint32_t *reserved, size_t reserved_count)
{
    size_t slot;

    for (slot = 0u; slot < reserved_count; slot++) {
        if (reserved[slot] == index)
            return true;
    }
    return false;
}

static bool request_has_duplicate(const uint32_t *indices, size_t count, uint32_t *out_index)
{
    size_t outer;

    for (outer = 0u; outer < count; outer++) {
        size_t inner;

        for (inner = outer + 1u; inner < count; inner++) {
            if (indices[outer] == indices[inner]) {
                *out_index = indices[outer];
                return true;
            }
        }
    }
    return false;
}

/** Free every claim already taken, in reverse order, and the set itself. */
static void release_partial(ft_tt_tensix_claim_set *set, size_t held)
{
    size_t slot = held;

    while (slot > 0u) {
        slot--;
        (void)ft_tt_udrv_claim_release(set->claims[slot]);
    }
    free(set->claims);
    free(set->indices);
    free(set);
}

const char *ft_tt_tensix_claim_resolve_card(const char *card)
{
    const char *value;

    if (card && card[0] != '\0')
        return card;

    value = getenv(FT_TT_TENSIX_CLAIM_CARD_ENVIRONMENT);
    return (value && value[0] != '\0') ? value : FT_TT_TENSIX_CLAIM_CARD_DEFAULT;
}

const char *ft_tt_tensix_claim_resolve_root(const char *root)
{
    const char *value;

    if (root && root[0] != '\0')
        return root;

    value = getenv(FT_TT_TENSIX_CLAIM_ROOT_ENVIRONMENT);
    /* NULL, not a literal: ft_tt_udrv_claim owns the default and must stay the
     * single place it is written down. */
    return (value && value[0] != '\0') ? value : NULL;
}

int ft_tt_tensix_claim_parse_indices(const char *text, uint32_t *out, size_t capacity,
                                     size_t *out_count)
{
    const char *cursor = text;
    size_t count = 0u;

    if (!out || !out_count || capacity == 0u)
        return FT_TT_UDRV_EINVAL;

    *out_count = 0u;
    if (!text || text[0] == '\0')
        return FT_TT_UDRV_OK;

    while (*cursor != '\0') {
        char *end = NULL;
        unsigned long first;
        unsigned long last;
        unsigned long value;

        /* strtoul SKIPS leading whitespace and accepts a sign, so "1, 2" and
         * "1,-2" would both parse without this — the first as a valid two-core
         * list and the second by wrapping through ULONG_MAX. Demanding a digit
         * is what makes a quoting mistake a refusal instead of a wrong core set. */
        if (*cursor < '0' || *cursor > '9')
            return FT_TT_UDRV_EINVAL;
        first = strtoul(cursor, &end, 10);
        if (end == cursor || first >= FT_TT_TENSIX_CLAIM_INDEX_COUNT)
            return FT_TT_UDRV_EINVAL;
        cursor = end;

        last = first;
        if (*cursor == '-') {
            cursor++;
            if (*cursor < '0' || *cursor > '9')
                return FT_TT_UDRV_EINVAL;
            last = strtoul(cursor, &end, 10);
            if (end == cursor || last >= FT_TT_TENSIX_CLAIM_INDEX_COUNT || last < first)
                return FT_TT_UDRV_EINVAL;
            cursor = end;
        }

        for (value = first; value <= last; value++) {
            if (count >= capacity)
                return FT_TT_UDRV_EINVAL;
            out[count++] = (uint32_t)value;
        }

        if (*cursor == ',')
            cursor++;
        else if (*cursor != '\0')
            return FT_TT_UDRV_EINVAL;
    }

    *out_count = count;
    return FT_TT_UDRV_OK;
}

int ft_tt_tensix_claim_reserved_from_environment(uint32_t *out, size_t capacity,
                                                 size_t *out_count)
{
    return ft_tt_tensix_claim_parse_indices(getenv(FT_TT_TENSIX_CLAIM_RESERVED_ENVIRONMENT), out,
                                            capacity, out_count);
}

int ft_tt_tensix_claim_acquire(const ft_tt_tensix_claim_request *request,
                               ft_tt_tensix_claim_set **out_set,
                               ft_tt_tensix_claim_refusal *out_refusal)
{
    uint32_t environment_reserved[RESERVED_CAPACITY];
    const uint32_t *reserved;
    size_t reserved_count = 0u;
    ft_tt_tensix_claim_set *set = NULL;
    uint32_t duplicate = 0u;
    size_t slot;
    int rc;

    refusal_reset(out_refusal);

    if (!request || !out_set || !request->indices || request->count == 0u ||
        request->count > FT_TT_TENSIX_CLAIM_MAX_CORES) {
        refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_FAILED, 0u, FT_TT_UDRV_EINVAL);
        return FT_TT_UDRV_EINVAL;
    }

    /* A duplicated index would take the flock, then be refused by its own
     * holder's flock on the second attempt — a self-deadlock reported as
     * "held by pid <self>", which is the most confusing message available. */
    if (request_has_duplicate(request->indices, request->count, &duplicate)) {
        refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_DUPLICATE, duplicate, FT_TT_UDRV_EINVAL);
        return FT_TT_UDRV_EINVAL;
    }

    if (request->reserved) {
        reserved = request->reserved;
        reserved_count = request->reserved_count;
    } else {
        rc = ft_tt_tensix_claim_reserved_from_environment(environment_reserved, RESERVED_CAPACITY,
                                                          &reserved_count);
        if (rc != FT_TT_UDRV_OK) {
            refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_FAILED, 0u, rc);
            return rc;
        }
        reserved = environment_reserved;
    }

    /* Reserved is checked for the WHOLE request before anything is claimed, so a
     * request that names a reserved core never takes a lock it will hand back. */
    for (slot = 0u; slot < request->count; slot++) {
        if (request->indices[slot] >= FT_TT_TENSIX_CLAIM_INDEX_COUNT) {
            refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_FAILED, request->indices[slot],
                        FT_TT_UDRV_EINVAL);
            return FT_TT_UDRV_EINVAL;
        }
        if (index_is_reserved(request->indices[slot], reserved, reserved_count)) {
            refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_RESERVED, request->indices[slot],
                        FT_TT_UDRV_EAGAIN);
            return FT_TT_UDRV_EAGAIN;
        }
    }

    set = calloc(1u, sizeof(*set));
    if (!set) {
        refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_FAILED, 0u, FT_TT_UDRV_ENOMEM);
        return FT_TT_UDRV_ENOMEM;
    }
    set->claims = calloc(request->count, sizeof(*set->claims));
    set->indices = calloc(request->count, sizeof(*set->indices));
    if (!set->claims || !set->indices) {
        release_partial(set, 0u);
        refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_FAILED, 0u, FT_TT_UDRV_ENOMEM);
        return FT_TT_UDRV_ENOMEM;
    }

    rc = ft_tt_udrv_claim_namespace_init(&set->ns,
                                         ft_tt_tensix_claim_resolve_root(request->root),
                                         ft_tt_tensix_claim_resolve_card(request->card),
                                         FT_TT_TENSIX_CLAIM_NAMESPACE,
                                         FT_TT_TENSIX_CLAIM_INDEX_COUNT);
    if (rc != FT_TT_UDRV_OK) {
        release_partial(set, 0u);
        refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_FAILED, 0u, rc);
        return rc;
    }

    for (slot = 0u; slot < request->count; slot++) {
        uint32_t index = request->indices[slot];

        rc = ft_tt_udrv_claim_acquire(&set->ns, index, request->reason, &set->claims[slot]);
        if (rc != FT_TT_UDRV_OK) {
            ft_tt_tensix_claim_outcome outcome = (rc == FT_TT_UDRV_EAGAIN)
                                                     ? FT_TT_TENSIX_CLAIM_HELD
                                                     : FT_TT_TENSIX_CLAIM_FAILED;

            refusal_set(out_refusal, outcome, index, rc);
            if (outcome == FT_TT_TENSIX_CLAIM_HELD)
                describe_holder(&set->ns, index, out_refusal);
            /* Everything already taken goes back. A half-claimed set is a set
             * nobody can use and nobody else can claim. */
            release_partial(set, slot);
            return rc;
        }
        set->indices[slot] = index;
        set->count = slot + 1u;
    }

    /* RECOVERY-ON-ACQUIRE, and only once the whole set is held: a retirement that
     * ran while a later core was still unclaimed could be retiring a core another
     * process is about to take. */
    if (request->retire) {
        for (slot = 0u; slot < set->count; slot++) {
            int retired = request->retire(request->retire_context, set->indices[slot]);

            if (retired != 0) {
                refusal_set(out_refusal, FT_TT_TENSIX_CLAIM_UNRETIRED, set->indices[slot],
                            retired);
                release_partial(set, set->count);
                return FT_TT_UDRV_EAGAIN;
            }
        }
    }

    *out_set = set;
    return FT_TT_UDRV_OK;
}

void ft_tt_tensix_claim_release(ft_tt_tensix_claim_set *set)
{
    if (!set)
        return;
    release_partial(set, set->count);
}

size_t ft_tt_tensix_claim_count(const ft_tt_tensix_claim_set *set)
{
    return set ? set->count : 0u;
}

uint32_t ft_tt_tensix_claim_index_at(const ft_tt_tensix_claim_set *set, size_t slot)
{
    if (!set || slot >= set->count)
        return UINT32_MAX;
    return set->indices[slot];
}

void ft_tt_tensix_claim_refusal_print(const ft_tt_tensix_claim_refusal *refusal)
{
    if (!refusal || refusal->outcome == FT_TT_TENSIX_CLAIM_GRANTED)
        return;

    switch (refusal->outcome) {
    case FT_TT_TENSIX_CLAIM_HELD:
        if (refusal->pid > 0) {
            (void)fprintf(stderr, "refused: Tensix %u is held by pid %ld (\"%s\")\n",
                          refusal->index, (long)refusal->pid,
                          refusal->reason[0] != '\0' ? refusal->reason : "no reason recorded");
        } else {
            (void)fprintf(stderr,
                          "refused: Tensix %u is held by another process, which did not "
                          "record its pid\n",
                          refusal->index);
        }
        (void)fprintf(stderr,
                      "  Nothing was claimed. Wait for that process, or name different cores "
                      "with %s / %s.\n",
                      FT_TT_TENSIX_CLAIM_RESERVED_ENVIRONMENT, "FT_TT_TENSIX_INDICES");
        break;
    case FT_TT_TENSIX_CLAIM_RESERVED:
        (void)fprintf(stderr,
                      "refused: Tensix %u is RESERVED by %s — it belongs to another op family "
                      "or to a live serve.\n  Nothing was claimed.\n",
                      refusal->index, FT_TT_TENSIX_CLAIM_RESERVED_ENVIRONMENT);
        break;
    case FT_TT_TENSIX_CLAIM_DUPLICATE:
        (void)fprintf(stderr,
                      "refused: Tensix %u was named twice in one request. Nothing was "
                      "claimed.\n",
                      refusal->index);
        break;
    case FT_TT_TENSIX_CLAIM_UNRETIRED:
        (void)fprintf(stderr,
                      "refused: Tensix %u was claimed but could not be retired (rc=%d), so it "
                      "is not in a known state.\n"
                      "  A released claim is not a retired residency — the previous holder may "
                      "have died with the core still running.\n"
                      "  The whole set was released. Check `ft_tt_core status %u --detail`.\n",
                      refusal->index, refusal->detail, refusal->index);
        break;
    case FT_TT_TENSIX_CLAIM_FAILED:
    case FT_TT_TENSIX_CLAIM_GRANTED:
    default:
        (void)fprintf(stderr,
                      "refused: could not claim Tensix %u (rc=%d). Nothing was claimed.\n",
                      refusal->index, refusal->detail);
        break;
    }
}
