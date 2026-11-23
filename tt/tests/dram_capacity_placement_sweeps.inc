/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * dram_capacity_placement_sweeps.inc — stage `place` of test_dram_capacity.c (tt-place lane,
 * 2026-10-22). Included by that file, after its sweep driver; it uses run_sweep() and the
 * explicit reader map (`map_slot` / `map_bank` / `map_port` / `map_core`) and nothing else.
 *
 * THE QUESTIONS (tt_ornith_optimized_spec.md, "DECODE WEIGHT PLACEMENT"):
 *   pairs   per (member core, bank, door): one reader alone — bandwidth at the matmul's read
 *           shape, and round-trip latency (64 B reads, one in flight). Does NoC distance
 *           change either?
 *   maps    the 16 production members together, under the production rotation (all eight
 *           cursors), nearest-bank and farthest-bank maps, at the matmul shape, the peak
 *           shape and a column-strided walk (the matmul's DRAM page pattern).
 *   l1src   members reading ANOTHER core's L1 at the matmul shape (the L1-staging question):
 *           1:1, 2:1, and 1:1 while the staging cores themselves stream DRAM.
 *
 * Booted slots: FT_TT_DRAMCAP_CORE_LIST names them; slots 0..15 must be the 16 matmul members
 * (member 0's core first), slots 16..31 the idle staging cores. FT_TT_DRAMCAP_NOC is the NoC
 * the reader kernel was built for (0 or 1) — it sets the distance the maps minimise.
 *
 * Status: NEW 2026-10-22. Issues: none. Todo: none.
 */
#include "ft_tt_dram_nearest.h"
#include "ft_tt_dram_stripe.h"

#define PLACE_MEMBERS 16u
#define PLACE_STAGING_FIRST 16u
#define PLACE_PAIR_MS 40u

static uint8_t place_slot[PLACE_MEMBERS * 2u];

/** Read-response hops for booted slot `slot` from bank/port, on the kernel's NoC. */
static uint32_t place_hops(const capacity_fixture *fixture, uint32_t slot, uint32_t bank,
                           uint32_t port, uint32_t noc)
{
    uint32_t sx = 0u, sy = 0u;

    (void)ft_tt_coord_noc_size(&fixture->session->grid, &sx, &sy);
    return ft_tt_dram_response_hops(fixture->dram_noc0[bank][port], fixture->core_noc0[slot],
                                    noc, sx, sy);
}

/**
 * Pairs: one reader alone, every member x every door. 384 doors x 2 shapes do not fit one
 * sweep, so one sweep per member (48 conditions), and the matrices print at the end.
 */
static void place_run_pairs(capacity_fixture *fixture, uint32_t noc)
{
    static sweep_condition conditions[MAX_CONDITIONS];
    static uint8_t slot_one[PLACE_MEMBERS][1], bank_one[24][1], port_one[24][1];
    static double bandwidth[PLACE_MEMBERS][24], latency_ns[PLACE_MEMBERS][24];
    uint32_t member, door, bank, port;

    for (door = 0u; door < 24u; door++) {
        bank_one[door][0] = (uint8_t)(door / 3u);
        port_one[door][0] = (uint8_t)(door % 3u);
    }
    for (member = 0u; member < PLACE_MEMBERS; member++) {
        uint32_t count = 0u;

        slot_one[member][0] = (uint8_t)member;
        for (door = 0u; door < 24u; door++) {
            sweep_condition *c = &conditions[count++];

            set_condition(c, "pair bw", 1u, 1u, 1u, 0u, 16uLL * MIB, 8704u, 9u, 9u,
                          STREAM_TARGET);
            (void)snprintf(c->label, sizeof(c->label), "m%u b%u p%u bw 8704x9", member,
                           door / 3u, door % 3u);
            c->map_slot = slot_one[member];
            c->map_bank = bank_one[door];
            c->map_port = port_one[door];
            c->duration_ms = PLACE_PAIR_MS;
            c = &conditions[count++];
            /* One 64 B read in flight: the rate IS 64 B over the round trip. */
            set_condition(c, "pair lat", 1u, 1u, 1u, 0u, 1uLL * MIB, 64u, 32u, 1u,
                          STREAM_TARGET);
            (void)snprintf(c->label, sizeof(c->label), "m%u b%u p%u lat 64x1", member,
                           door / 3u, door % 3u);
            c->map_slot = slot_one[member];
            c->map_bank = bank_one[door];
            c->map_port = port_one[door];
            c->duration_ms = PLACE_PAIR_MS;
        }
        run_sweep(fixture, "PLACE pairs — one member alone, every door", conditions, count);
        for (door = 0u; door < 24u; door++) {
            bandwidth[member][door] = g_sweep_median[2u * door];
            latency_ns[member][door] = g_sweep_median[2u * door + 1u] > 0.0
                                           ? 64.0 / g_sweep_median[2u * door + 1u]
                                           : 0.0;
        }
    }

    printf("\n  PLACE PAIRS MATRIX (NoC %u) — per member: core noc0, then per bank the best\n"
           "  door's GB/s / latency ns / response hops, and the door-averaged values\n", noc);
    for (member = 0u; member < PLACE_MEMBERS; member++) {
        printf("  m%-2u core %3u (%2u,%2u) |", member, fixture->core_index[member],
               fixture->core_noc0[member].x, fixture->core_noc0[member].y);
        for (bank = 0u; bank < MAX_BANKS; bank++) {
            double best_bw = 0.0, lat = 0.0;
            uint32_t hops = 99u;

            for (port = 0u; port < MAX_PORTS; port++) {
                uint32_t h = place_hops(fixture, member, bank, port, noc);

                if (h < hops)
                    hops = h;
                if (bandwidth[member][bank * 3u + port] > best_bw)
                    best_bw = bandwidth[member][bank * 3u + port];
                lat += latency_ns[member][bank * 3u + port] / 3.0;
            }
            printf(" b%u %5.1f %4.0f %2u |", bank, best_bw, lat, hops);
        }
        printf("\n");
    }
    printf("\n  PLACE PAIRS RAW (member bank port hops GB/s ns)\n");
    for (member = 0u; member < PLACE_MEMBERS; member++)
        for (door = 0u; door < 24u; door++)
            printf("  PAIR %u %u %u %u %.2f %.1f\n", member, door / 3u, door % 3u,
                   place_hops(fixture, member, door / 3u, door % 3u, noc),
                   bandwidth[member][door], latency_ns[member][door]);
    fflush(stdout);
}

/** One 16-reader map condition over the member slots. */
static void place_map_condition(sweep_condition *c, const char *label, const uint8_t *banks,
                                const uint8_t *ports, uint32_t block, uint32_t depth,
                                uint32_t stride)
{
    set_condition(c, label, PLACE_MEMBERS, 8u, 1u, 0u, 16uLL * MIB, block, depth, depth,
                  STREAM_TARGET);
    c->map_slot = place_slot;
    c->map_bank = banks;
    c->map_port = ports;
    c->stride_bytes = stride;
    if (stride != 0u)
        c->working_set = 4uLL * MIB; /* 481 distinct pages, 67 MiB span a reader */
    c->duration_ms = SWEEP_DURATION_MS;
}

/** Total response hops of a map, for the table. */
static uint32_t place_map_hops(const capacity_fixture *fixture, const uint8_t *banks,
                               const uint8_t *ports, uint32_t noc)
{
    uint32_t r, total = 0u;

    for (r = 0u; r < PLACE_MEMBERS; r++)
        total += place_hops(fixture, r, banks[r], ports[r], noc);
    return total;
}

/**
 * Maps: rotation cursor 0..7 (doors by the production chooser), nearest, farthest; each at
 * the matmul shape (8704 x 9), the peak shape (2048 x 32) and the matmul's column stride
 * (8704 B reads 136 KiB apart = Kt 128 x 1088 B, one DRAM page per read).
 */
static void place_run_maps(capacity_fixture *fixture, uint32_t noc)
{
    static sweep_condition conditions[MAX_CONDITIONS];
    static uint8_t banks[14][PLACE_MEMBERS], ports[14][PLACE_MEMBERS];
    static char names[14][16] = {"rot0", "rot1", "rot2", "rot3", "rot4",
                                 "rot5", "rot6", "rot7", "NEAR", "FAR"};
    uint32_t map, r, count = 0u, maps = 10u;

    for (map = 0u; map < 8u; map++) {
        uint32_t b[PLACE_MEMBERS];
        ft_tt_dram_endpoint doors[PLACE_MEMBERS];

        for (r = 0u; r < PLACE_MEMBERS; r++)
            b[r] = (map + r) % 8u;
        if (ft_tt_dram_stripe_choose_ports(&fixture->session->grid, b, PLACE_MEMBERS,
                                           FT_TT_COORD_TRANSLATED, doors) != FT_TT_OK)
            return;
        for (r = 0u; r < PLACE_MEMBERS; r++) {
            banks[map][r] = (uint8_t)b[r];
            ports[map][r] = (uint8_t)doors[r].port;
        }
    }
    for (map = 8u; map < 10u; map++) {
        uint32_t b[PLACE_MEMBERS], p[PLACE_MEMBERS];

        if (ft_tt_dram_nearest_assign(&fixture->session->grid, fixture->core_noc0,
                                      PLACE_MEMBERS, noc, 2u,
                                      map == 8u ? FT_TT_DRAM_NEAREST_GOAL_NEAREST
                                                : FT_TT_DRAM_NEAREST_GOAL_FARTHEST,
                                      b, p) != FT_TT_OK)
            return;
        for (r = 0u; r < PLACE_MEMBERS; r++) {
            banks[map][r] = (uint8_t)b[r];
            ports[map][r] = (uint8_t)p[r];
        }
    }
    /* EXTRA MAPS under test: FT_TT_DRAMCAP_MAPS="NAME=b.p,b.p,...;NAME2=..." (16 entries). */
    {
        const char *cursor = getenv("FT_TT_DRAMCAP_MAPS");

        while (cursor && *cursor && maps < 14u) {
            const char *eq = strchr(cursor, '=');
            size_t length = eq ? (size_t)(eq - cursor) : 0u;

            if (!eq || length == 0u || length >= sizeof(names[0]))
                break;
            memcpy(names[maps], cursor, length);
            names[maps][length] = '\0';
            cursor = eq + 1;
            for (r = 0u; r < PLACE_MEMBERS; r++) {
                char *end = NULL;
                unsigned long b = strtoul(cursor, &end, 10), p;

                if (end == cursor || *end != '.')
                    return;
                p = strtoul(end + 1, &end, 10);
                banks[maps][r] = (uint8_t)b;
                ports[maps][r] = (uint8_t)p;
                cursor = (*end == ',') ? end + 1 : end;
            }
            maps++;
            cursor = *cursor == ';' ? cursor + 1 : cursor;
        }
    }
    for (map = 0u; map < maps; map++) {
        char label[48];

        printf("  map %-4s hops %3u:", names[map], place_map_hops(fixture, banks[map],
                                                                    ports[map], noc));
        for (r = 0u; r < PLACE_MEMBERS; r++)
            printf(" %u.%u", banks[map][r], ports[map][r]);
        printf("\n");
        (void)snprintf(label, sizeof(label), "MAP %s 8704x9", names[map]);
        place_map_condition(&conditions[count++], label, banks[map], ports[map], 8704u, 9u, 0u);
        (void)snprintf(label, sizeof(label), "MAP %s 2048x32", names[map]);
        place_map_condition(&conditions[count++], label, banks[map], ports[map], 2048u, 32u,
                            0u);
        (void)snprintf(label, sizeof(label), "MAP %s 8704x9 colstride", names[map]);
        place_map_condition(&conditions[count++], label, banks[map], ports[map], 8704u, 9u,
                            139264u);
    }
    run_sweep(fixture, "PLACE maps — the 16 members together, rotation vs nearest vs farthest",
              conditions, count);
    printf("  PLACE MAPS per-reader GB/s (fastest repetition), 8704x9:\n");
    for (map = 0u; map < maps; map++) {
        printf("  MAPREADERS %-4s", names[map]);
        for (r = 0u; r < PLACE_MEMBERS; r++)
            printf(" %5.1f", g_sweep_reader_best[3u * map][r]);
        printf("\n");
    }
    fflush(stdout);
}

/**
 * L1 source: members read a staging core's L1 (+0x0C0000, 512 KiB walk) instead of DRAM.
 * 1:1 (member r <- staging r), 2:1 (member r <- staging r/2), and 1:1 while the 16 staging
 * cores themselves stream DRAM under the rotation-0 map (the ping-pong overlap).
 */
static void place_run_l1src(capacity_fixture *fixture)
{
    static sweep_condition conditions[MAX_CONDITIONS];
    static uint8_t slot32[32], src_one[32], src_two[32], port32[32], core32[32];
    uint32_t r, count = 0u;
    static const uint32_t shapes[3][2] = {{8704u, 9u}, {2048u, 32u}, {8704u, 18u}};
    uint32_t s;

    for (r = 0u; r < PLACE_MEMBERS; r++) {
        slot32[r] = (uint8_t)r;
        src_one[r] = (uint8_t)(PLACE_STAGING_FIRST + r);
        src_two[r] = (uint8_t)(PLACE_STAGING_FIRST + r / 2u);
        core32[r] = 1u;
        port32[r] = 0u;
        /* staging readers: slot 16+r streams DRAM bank (r % 8), port (r / 8) */
        slot32[PLACE_MEMBERS + r] = (uint8_t)(PLACE_STAGING_FIRST + r);
        src_one[PLACE_MEMBERS + r] = (uint8_t)(r % 8u);
        port32[PLACE_MEMBERS + r] = (uint8_t)((r / 8u) % 3u);
        core32[PLACE_MEMBERS + r] = 0u;
    }
    for (s = 0u; s < 3u; s++) {
        sweep_condition *c = &conditions[count++];

        set_condition(c, "", PLACE_MEMBERS, 8u, 1u, 0u, 512uLL * KIB, shapes[s][0],
                      shapes[s][1], shapes[s][1], STREAM_TARGET);
        (void)snprintf(c->label, sizeof(c->label), "L1SRC 1:1 %ux%u", shapes[s][0],
                       shapes[s][1]);
        c->map_slot = slot32;
        c->map_bank = src_one;
        c->map_port = port32;
        c->map_core = core32;
        c->duration_ms = SWEEP_DURATION_MS;

        c = &conditions[count++];
        set_condition(c, "", PLACE_MEMBERS, 8u, 1u, 0u, 512uLL * KIB, shapes[s][0],
                      shapes[s][1], shapes[s][1], STREAM_TARGET);
        (void)snprintf(c->label, sizeof(c->label), "L1SRC 2:1 %ux%u", shapes[s][0],
                       shapes[s][1]);
        c->map_slot = slot32;
        c->map_bank = src_two;
        c->map_port = port32;
        c->map_core = core32;
        c->duration_ms = SWEEP_DURATION_MS;

        c = &conditions[count++];
        set_condition(c, "", 2u * PLACE_MEMBERS, 8u, 1u, 0u, 512uLL * KIB, shapes[s][0],
                      shapes[s][1], shapes[s][1], STREAM_TARGET);
        (void)snprintf(c->label, sizeof(c->label), "L1SRC 1:1 + staging DRAM %ux%u",
                       shapes[s][0], shapes[s][1]);
        c->map_slot = slot32;
        c->map_bank = src_one;
        c->map_port = port32;
        c->map_core = core32;
        c->duration_ms = SWEEP_DURATION_MS;

        c = &conditions[count++];
        set_condition(c, "", PLACE_MEMBERS, 8u, 1u, 0u, 512uLL * KIB, shapes[s][0],
                      shapes[s][1], shapes[s][1], STREAM_TARGET);
        (void)snprintf(c->label, sizeof(c->label), "L1SRC staging DRAM only %ux%u",
                       shapes[s][0], shapes[s][1]);
        c->map_slot = slot32 + PLACE_MEMBERS;
        c->map_bank = src_one + PLACE_MEMBERS;
        c->map_port = port32 + PLACE_MEMBERS;
        c->map_core = core32 + PLACE_MEMBERS;
        c->duration_ms = SWEEP_DURATION_MS;
    }
    run_sweep(fixture, "PLACE l1src — members reading staging cores' L1", conditions, count);
    printf("  L1SRC per-reader GB/s (fastest repetition):\n");
    for (s = 0u; s < count; s++) {
        printf("  L1READERS %-34s", conditions[s].label);
        for (r = 0u; r < conditions[s].core_count && r < 32u; r++)
            printf(" %5.1f", g_sweep_reader_best[s][r]);
        printf("\n");
    }
    fflush(stdout);
}

/**
 * Rows: does a SAME-ROW door help under load? On a one-way torus the round trip is the whole
 * ring in every dimension the core and the door differ in, so a door in the reader's own NoC
 * row skips the Y ring (pairs: ~280 vs ~370 ns alone). Booted slots 0..7 = x 1, y 3..10;
 * slots 8..15 = x 16, y 3..10 (FT_TT_DRAMCAP_CORE_LIST="14,28,...,112,27,41,...,125").
 * SAMEROW and OFFROW use the SAME cores, banks and doors, only paired differently; ROT is the
 * production rotation on the same cores.
 */
static void place_run_rows(capacity_fixture *fixture)
{
    static sweep_condition conditions[MAX_CONDITIONS];
    /* west reader at y = 3 + i: bank.port */
    static const uint8_t same_bank[8] = {1u, 2u, 3u, 3u, 0u, 2u, 0u, 1u};
    static const uint8_t same_port[8] = {2u, 1u, 0u, 2u, 0u, 2u, 1u, 1u};
    static const uint8_t off_port[8] = {0u, 2u, 1u, 0u, 0u, 1u, 1u, 2u};
    static uint8_t banks[3][PLACE_MEMBERS], ports[3][PLACE_MEMBERS];
    static const char *const names[3] = {"SAMEROW", "OFFROW", "ROT"};
    static const uint32_t shapes[3][2] = {{8704u, 9u}, {4352u, 9u}, {2048u, 32u}};
    uint32_t r, map, s, count = 0u;

    for (r = 0u; r < 8u; r++) {
        banks[0][r] = same_bank[r];
        ports[0][r] = same_port[r];
        banks[0][8u + r] = (uint8_t)(same_bank[r] + 4u);
        ports[0][8u + r] = same_port[r];
        banks[1][r] = same_bank[r];
        ports[1][r] = off_port[r];
        banks[1][8u + r] = (uint8_t)(same_bank[r] + 4u);
        ports[1][8u + r] = off_port[r];
    }
    {
        uint32_t b[PLACE_MEMBERS];
        ft_tt_dram_endpoint doors[PLACE_MEMBERS];

        for (r = 0u; r < PLACE_MEMBERS; r++)
            b[r] = r % 8u;
        if (ft_tt_dram_stripe_choose_ports(&fixture->session->grid, b, PLACE_MEMBERS,
                                           FT_TT_COORD_TRANSLATED, doors) != FT_TT_OK)
            return;
        for (r = 0u; r < PLACE_MEMBERS; r++) {
            banks[2][r] = (uint8_t)b[r];
            ports[2][r] = (uint8_t)doors[r].port;
        }
    }
    for (map = 0u; map < 3u; map++) {
        uint32_t same = 0u;

        for (r = 0u; r < PLACE_MEMBERS; r++)
            same += fixture->dram_noc0[banks[map][r]][ports[map][r]].y ==
                    fixture->core_noc0[r].y;
        printf("  rows map %-7s same-row readers %2u:", names[map], same);
        for (r = 0u; r < PLACE_MEMBERS; r++)
            printf(" (%u,%u)<-%u.%u", fixture->core_noc0[r].x, fixture->core_noc0[r].y,
                   banks[map][r], ports[map][r]);
        printf("\n");
        for (s = 0u; s < 3u; s++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "ROWS %s %ux%u", names[map], shapes[s][0],
                           shapes[s][1]);
            place_map_condition(&conditions[count++], label, banks[map], ports[map],
                                shapes[s][0], shapes[s][1], 0u);
        }
    }
    /* ORDER: the decode reader's per-block pattern — 8 column chunks of `depth` tiles, each
     * Kt x 1088 B = 136 KiB apart (one DRAM page per read) — against the same chunks laid
     * contiguously (a block-major panel). Production rotation on these cores. */
    {
        static const uint32_t order[4][3] = {
            {4352u, 9u, 139264u}, {4352u, 9u, 0u}, {4352u, 18u, 139264u}, {4352u, 18u, 0u}};

        for (s = 0u; s < 4u; s++) {
            char label[48];

            (void)snprintf(label, sizeof(label), "ORDER ROT %ux%u %s", order[s][0], order[s][1],
                           order[s][2] ? "colstride" : "contiguous");
            place_map_condition(&conditions[count++], label, banks[2], ports[2], order[s][0],
                                order[s][1], order[s][2]);
        }
    }
    run_sweep(fixture, "PLACE rows — same-row doors vs the same doors off-row", conditions,
              count);
    fflush(stdout);
}

static void run_placement_sweeps(capacity_fixture *fixture, const char *stages,
                                 sweep_condition *unused)
{
    const uint32_t noc = environment_u32("FT_TT_DRAMCAP_NOC", 0u);
    uint32_t r;

    (void)unused;
    if (!stage_selected(stages, "pairs") && !stage_selected(stages, "maps") &&
        !stage_selected(stages, "l1src") && !stage_selected(stages, "rows"))
        return;
    if (fixture->core_count < PLACE_MEMBERS) {
        ft_tt_test_check("stage place needs >= 16 booted cores", false, "too few");
        return;
    }
    for (r = 0u; r < PLACE_MEMBERS * 2u; r++)
        place_slot[r] = (uint8_t)r;
    printf("  PLACE stage: reader NoC %u, kernel_config_base 0x%x, L1 source window "
           "0x%x..0x%llx\n", noc, fixture->layout.kernel_config_base,
           fixture->layout.kernel_config_base + L1_SOURCE_OFFSET,
           (unsigned long long)(fixture->layout.kernel_config_base + L1_SOURCE_OFFSET +
                                L1_SOURCE_SPAN));
    if (stage_selected(stages, "rows"))
        place_run_rows(fixture);
    if (stage_selected(stages, "maps"))
        place_run_maps(fixture, noc);
    if (stage_selected(stages, "l1src") && fixture->core_count >= 2u * PLACE_MEMBERS) {
        if ((uint64_t)fixture->layout.kernel_config_base + L1_SOURCE_OFFSET + L1_SOURCE_SPAN >
            (uint64_t)(1536u * 1024u))
            ft_tt_test_check("the L1 source window fits L1", false, "skipping l1src");
        else
            place_run_l1src(fixture);
    }
    if (stage_selected(stages, "pairs"))
        place_run_pairs(fixture, noc);
}
