// SPDX-License-Identifier: AGPL-3.0-or-later
//
// ft_tt_memory_device_shim.c
//   A flat C API over the arithmetic half of kernels/resident_memory_protocol.h, so the
//   OFF-HARDWARE suite exercises THE KERNEL'S OWN FUNCTIONS instead of a C transcription of
//   them.
//
//   THIS FILE WAS C++ UNTIL 2026-08-13, AND ONLY FOR LINKAGE. The protocol header was C++,
//   so reaching the kernel's own inline functions from a C suite needed an `extern "C"`
//   wrapper. Phase 6.6 made that header C99, the linkage reason evaporated, and this is now
//   plain C. What it still earns its place for is the FLATTENING at the bottom: `ftm_run`
//   becomes two parallel arrays, so the suite needs no struct of the kernel's.
//
//   THIS IS THE DIFFERENCE BETWEEN TESTING THE KERNEL AND TESTING A COPY OF IT. Face order
//   is the trap docs/tt_kernel_model.md §9 exists for, and it is invisible to any metric
//   computed through the same wrong code — so a test that re-implements the chunk offsets
//   in C and compares them with the kernel's would agree with itself. Here the suite calls
//   the real inline functions and scores them against ft_tt_tilize.c, which was written for
//   another purpose entirely and is verified against tt-metal's own tilize_utils.cpp.
//
//   It compiles because resident_memory_protocol.h was written to be compilable: it
//   includes <stdint.h>/<stdbool.h> and nothing else, names no device API, and touches no
//   L1 pointer. That is a property to KEEP — the moment it needs the stock dataflow API, this shim
//   stops building and the off-hardware gate loses the only part of the kernel it can see.
//
//   Status: built by tests/Makefile.memory as part of the off-hardware gate.
//   Issues: None
//   Todo:   None

#include "../kernels/resident_memory_protocol.h"


uint32_t ft_tt_device_chunk_bytes(uint32_t tile_bytes)
{
    return ftm_chunk_bytes(tile_bytes);
}

uint32_t ft_tt_device_face_bytes(uint32_t tile_bytes)
{
    return ftm_face_bytes(tile_bytes);
}

uint32_t ft_tt_device_chunk_offset(uint32_t row, uint32_t half, uint32_t tile_bytes)
{
    return ftm_chunk_offset(row, half, tile_bytes);
}

uint32_t ft_tt_device_group_row_bytes(uint32_t group_tiles, uint32_t tile_bytes)
{
    return ftm_group_row_bytes(group_tiles, tile_bytes);
}

void ft_tt_device_place_row_from_row_major(uint32_t* destination_block, const uint32_t* source_row,
                                           uint32_t destination_row, uint32_t group_tiles,
                                           uint32_t tile_bytes)
{
    ftm_place_row_from_row_major(destination_block, source_row, destination_row, group_tiles,
                                 tile_bytes);
}

void ft_tt_device_place_row_from_tiles(uint32_t* destination_block, const uint32_t* source_block,
                                       uint32_t source_row, uint32_t destination_row,
                                       uint32_t group_tiles, uint32_t tile_bytes)
{
    ftm_place_row_from_tiles(destination_block, source_block, source_row, destination_row,
                             group_tiles, tile_bytes);
}

void ft_tt_device_zero_words(uint32_t* destination, uint32_t words)
{
    ftm_zero_words(destination, words);
}

int ft_tt_device_banked_destination(uint32_t flags, uint32_t packed_x, uint32_t packed_y,
                                    uint32_t group, uint32_t* x, uint32_t* y, uint32_t* slot)
{
    return ftm_banked_destination(flags, packed_x, packed_y, group, x, y, slot) ? 1 : 0;
}

/** The runs, flattened into two parallel arrays so the C side needs no struct of the
 *  kernel's. `offsets` and `bytes` must hold kMemMaxRuns entries each. */
uint32_t ft_tt_device_destination_runs(uint32_t row0, uint32_t row_count, uint32_t tile_bytes,
                                       uint32_t* offsets, uint32_t* bytes)
{
    ftm_run runs[kMemMaxRuns];
    const uint32_t count = ftm_destination_runs(row0, row_count, tile_bytes, runs);

    for (uint32_t index = 0; index < count; index++) {
        offsets[index] = runs[index].offset;
        bytes[index] = runs[index].bytes;
    }
    return count;
}

uint32_t ft_tt_device_max_runs(void)
{
    return kMemMaxRuns;
}

/** The zero-length guard itself, so the suite can assert the kernel HAS one rather than
 *  trusting a comment. A zero-length NoC read is suspected of taking the whole card down
 *  (resident_memory_protocol.h), so the guard is load-bearing rather than defensive. */
int ft_tt_device_length_is_issuable(uint32_t bytes)
{
    return ftm_length_is_issuable(bytes) ? 1 : 0;
}

