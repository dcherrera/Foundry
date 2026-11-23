/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_gguf_slim.c
 * @brief CLI: the METADATA GGUF a card host keeps beside its FTTW artifacts — every key
 *   (tokenizer, config) and every non-matmul tensor, with the matmul weights left out.
 *
 * The serve reads tokenizer, config, the 1-D norms and the conv taps from a GGUF; every
 * matmul weight is on the card from the FTTW artifact. a 32 GB-host p150 box cannot hold Qwen3.6's 37.8 GB
 * GGUF beside its 29.5 GB of artifacts, so it gets this instead (qwen36_blackhole_spec.md
 * §4.2). Kept: tensors of rank < 2 and the NORM / BIAS / SSM_CONV / SSM_DECAY roles.
 * Dropped: everything stored MATMUL_NMAJOR or as the embedding.
 *
 * The key/value section is copied BYTE FOR BYTE (ft_gguf_map_metadata_raw), so no key is
 * re-encoded. After writing, the output is read back: same keys (raw bytes), and every kept
 * tensor's name, rank, dims, type and bytes equal to the source's. `FT_TT_GGUF_SLIM_RED=1`
 * writes the first kept tensor shifted by one byte; the read-back must then fail.
 *
 *   ft_tt_gguf_slim model.gguf model.meta.gguf
 *   ft_tt_gguf_slim model.gguf model.first4.gguf --truncate-blocks 4
 *   ft_tt_gguf_slim model.gguf model.meta+router.gguf --keep ffn_gate_inp
 *
 * `--keep SUB` also keeps every tensor whose name contains SUB (metadata mode). An attached
 * artifact never replaces a tensor the GGUF holds, so this is how the oracle runs one tensor
 * family at its SOURCE values (the fp32 router arm of plan Phase 4.3).
 *
 * `--truncate-blocks N` writes a TRUNCATED checkpoint instead: every tensor of blocks < N
 * and every non-block tensor (embedding, head, final norm), with `<arch>.block_count`
 * patched to N in the copied key/value section — the 4-block (3 GDN + 1 attention) shape
 * the oracle's exactness test and the card's per-layer MoE parity run on.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_gguf_map.h"
#include "ft_gguf_write.h"
#include "ft_tt_precision_ledger.h"

/**
 * Whether a tensor is written. Metadata mode (`truncate_blocks` 0): what the serve reads from
 * the GGUF rather than the artifact. Truncate mode: blocks below N and every non-block tensor.
 */
typedef struct {
    unsigned long truncate_blocks; /**< 0 = metadata mode */
    const char *keep_also;         /**< metadata mode: also keep names containing this */
} slim_options;

static int slim_keeps(const ft_gguf_tensor *tensor, const slim_options *options)
{
    ft_tt_tensor_role role = ft_tt_ledger_role(tensor->name);
    unsigned long truncate_blocks = options->truncate_blocks;

    if (options->keep_also && strstr(tensor->name, options->keep_also))
        return 1;
    if (truncate_blocks)
        return strncmp(tensor->name, "blk.", 4) != 0 || strtoul(tensor->name + 4, NULL, 10) < truncate_blocks;
    return tensor->n_dims < 2u || role == FT_TT_ROLE_NORM || role == FT_TT_ROLE_BIAS ||
           role == FT_TT_ROLE_SSM_CONV || role == FT_TT_ROLE_SSM_DECAY;
}

/**
 * Copy the key/value section and, in truncate mode, patch `<arch>.block_count` to N. The
 * caller frees `*out`. Returns 0, or -1 when the key is missing or not a u32/u64.
 */
static int slim_kv_copy(const ft_gguf_map *map, const void *kv, uint64_t kv_bytes, unsigned long truncate_blocks,
                        unsigned char **out)
{
    char architecture[64] = "", key[128];
    uint64_t offset;
    uint32_t type;

    *out = malloc((size_t)kv_bytes ? (size_t)kv_bytes : 1u);
    if (!*out)
        return -1;
    memcpy(*out, kv, (size_t)kv_bytes);
    if (!truncate_blocks)
        return 0;
    (void)ft_gguf_map_metadata_string(map, "general.architecture", architecture, sizeof(architecture));
    snprintf(key, sizeof(key), "%s.block_count", architecture);
    if (ft_gguf_map_metadata_value_offset(map, key, &type, &offset) != FT_GGUF_OK || (type != 4u && type != 10u)) {
        fprintf(stderr, "ft_tt_gguf_slim: '%s' is missing or not an integer; cannot truncate\n", key);
        return -1;
    }
    for (uint32_t b = 0; b < (type == 4u ? 4u : 8u); b++) /* little-endian u32 / u64 */
        (*out)[offset + b] = (unsigned char)((uint64_t)truncate_blocks >> (8u * b));
    return 0;
}

/** Read the written file back against the source. Returns the number of disagreements. */
static uint32_t slim_verify(const ft_gguf_map *source, const char *path, const unsigned char *want_kv,
                            const ft_gguf_write_tensor *kept, uint32_t kept_count)
{
    const void *source_kv, *got_kv;
    uint64_t want_bytes, got_bytes, want_count, got_count;
    ft_gguf_map *written = NULL;
    uint32_t bad = 0;

    if (ft_gguf_map_open(path, &written) != FT_GGUF_OK) {
        fprintf(stderr, "ft_tt_gguf_slim: the written file '%s' does not open as GGUF\n", path);
        return 1;
    }
    (void)ft_gguf_map_metadata_raw(source, &source_kv, &want_bytes, &want_count);
    (void)ft_gguf_map_metadata_raw(written, &got_kv, &got_bytes, &got_count);
    if (want_bytes != got_bytes || want_count != got_count || memcmp(want_kv, got_kv, (size_t)want_bytes) != 0) {
        fprintf(stderr, "ft_tt_gguf_slim: key/value section differs from the source's\n");
        bad++;
    }
    if (ft_gguf_map_tensor_count(written) != kept_count) {
        fprintf(stderr, "ft_tt_gguf_slim: %u tensors written, %u read back\n", kept_count,
                ft_gguf_map_tensor_count(written));
        bad++;
    }
    for (uint32_t i = 0; i < kept_count; i++) {
        const ft_gguf_tensor *want = ft_gguf_map_find(source, kept[i].name);
        const ft_gguf_tensor *got = ft_gguf_map_find(written, kept[i].name);

        if (!want || !got || want->n_dims != got->n_dims || memcmp(want->dims, got->dims, sizeof(want->dims)) != 0 ||
            want->ggml_type != got->ggml_type || want->byte_count != got->byte_count ||
            memcmp(want->data, got->data, (size_t)want->byte_count) != 0) {
            fprintf(stderr, "ft_tt_gguf_slim: '%s' read back differently from the source\n", kept[i].name);
            bad++;
        }
    }
    ft_gguf_map_close(written);
    return bad;
}

/** Collect the kept tensors (pointing into the source mapping) and their total bytes. */
static uint32_t slim_collect(const ft_gguf_map *map, const slim_options *options, ft_gguf_write_tensor *kept,
                             uint64_t *kept_bytes)
{
    uint32_t count = 0;

    *kept_bytes = 0;
    for (uint32_t i = 0; i < ft_gguf_map_tensor_count(map); i++) {
        const ft_gguf_tensor *t = ft_gguf_map_tensor(map, i);

        if (!slim_keeps(t, options))
            continue;
        kept[count].name = t->name;
        kept[count].n_dims = t->n_dims;
        memcpy(kept[count].dims, t->dims, sizeof(t->dims));
        kept[count].ggml_type = t->ggml_type;
        kept[count].data = t->data;
        kept[count].byte_count = t->byte_count;
        *kept_bytes += t->byte_count;
        count++;
    }
    return count;
}

int main(int argc, char **argv)
{
    ft_gguf_write_tensor *kept = NULL;
    ft_gguf_map *map = NULL;
    unsigned char *kv_out = NULL;
    const void *kv;
    uint64_t kv_bytes, kv_count, kept_bytes;
    uint32_t kept_count, bad;
    slim_options options = {0, NULL};
    int rc = 1;

    if (argc == 5 && strcmp(argv[3], "--truncate-blocks") == 0)
        options.truncate_blocks = strtoul(argv[4], NULL, 10);
    else if (argc == 5 && strcmp(argv[3], "--keep") == 0)
        options.keep_also = argv[4];
    if ((argc != 3 && argc != 5) || (argc == 5 && !options.truncate_blocks && !options.keep_also)) {
        fprintf(stderr, "usage: %s <model.gguf> <out.gguf> [--truncate-blocks N | --keep SUBSTRING]\n", argv[0]);
        return 2;
    }
    if (ft_gguf_map_open(argv[1], &map) != FT_GGUF_OK)
        return 1;
    if (ft_gguf_map_part_count(map) != 1u) {
        fprintf(stderr, "%s: '%s' is a split checkpoint; its split.* keys would be copied into a "
                        "single file that is not part of a set — refusing\n", argv[0], argv[1]);
        goto done;
    }
    kept = calloc(ft_gguf_map_tensor_count(map) + 1u, sizeof(*kept));
    if (!kept || ft_gguf_map_metadata_raw(map, &kv, &kv_bytes, &kv_count) != FT_GGUF_OK ||
        slim_kv_copy(map, kv, kv_bytes, options.truncate_blocks, &kv_out) != 0)
        goto done;
    kept_count = slim_collect(map, &options, kept, &kept_bytes);
    if (getenv("FT_TT_GGUF_SLIM_RED") && atoi(getenv("FT_TT_GGUF_SLIM_RED")) == 1 && kept_count > 0u) {
        fprintf(stderr, "%s: RED — '%s' written one byte off; the read-back must FAIL\n", argv[0], kept[0].name);
        kept[0].data = (const unsigned char *)kept[0].data + 1;
    }
    if (ft_gguf_write_file(argv[2], kv_out, kv_bytes, kv_count, kept, kept_count, ft_gguf_map_alignment(map)) !=
        FT_GGUF_WRITE_OK)
        goto done;
    bad = slim_verify(map, argv[2], kv_out, kept, kept_count);
    printf("%s: %llu keys, %u of %u tensors kept (%.2f MiB); read-back %s (%u disagreements)\n", argv[2],
           (unsigned long long)kv_count, kept_count, ft_gguf_map_tensor_count(map),
           (double)kept_bytes / (1024.0 * 1024.0), bad ? "FAIL" : "IDENTICAL", bad);
    rc = bad ? 1 : 0;

done:
    free(kv_out);
    free(kept);
    ft_gguf_map_close(map);
    return rc;
}
