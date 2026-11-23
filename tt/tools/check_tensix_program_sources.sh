#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_tensix_program_sources.sh
#   Refuses a Tensix PROGRAM DIRECTORY whose `.src` sidecars name kernel sources that are
#   absent from the tree, or that are NEWER than the ELF built from them.
#
#   Usage:  check_tensix_program_sources.sh <program-dir> [more dirs...]
#   Env:    KERNELS  the kernel source directory (default: ../kernels beside this script)
#   Exit:   0 every directory is current; 1 at least one is stale (and says which file)
#
#   WHY THIS EXISTS. `build_tensix_program.sh` and `build_tensix_dataflow_kernel.sh` already
#   write, beside every `<proc>.elf`, a `<proc>.elf.src` naming the source it was built from.
#   NOTHING READ THEM, and the cost of that was not theoretical:
#
#     * All seven Phase 6 program directories carried sidecars naming `.cpp` files deleted
#       weeks earlier, so `test_phase6_vector`'s greens -- and the four mutation results
#       quoted in its own comment -- were produced by pre-C, pre-guard images and said
#       nothing about the tree they were read as evidence for.
#     * A "silent CB-wrap corruption" was investigated as a live defect for a round. It was
#       a program directory built before the guard it was supposedly bypassing existed.
#     * The committed `ft_tt_kernels.h` shipped C++ ELFs from before the Foundry move.
#
#   Three incidents, one class: an artifact built from sources that have since moved, with
#   nothing in the loop to notice. The sidecars recorded the truth the whole time.
#
#   WHAT IT CANNOT DO, stated so the guarantee is not overread. This is a HOST-SIDE, tree-side
#   check: it needs the kernel sources next to the program directory, so it belongs in a build
#   or test driver and NOT inside `ft_tt_program_directory_load()`, which runs from a binary
#   that has no path to the source tree. A directory copied to a machine without the tree
#   cannot be checked at all, and this script says so rather than passing.
#
#   THIS IS NO LONGER THE STALENESS CHECK — it is the file-EXISTENCE half of one.
#   check_tensix_program_provenance.py is the other half and is the one that decides. The
#   mtime comparison below is defeated by `explode_tensix_program_blob`, which writes every
#   image and every sidecar fresh, so all 21 program directories report `current` the
#   instant they are exploded whatever the blob inside them held. That is how `programs/gdn`
#   ran five commits behind and the four `rtriad_*` readers shipped with no cooperative
#   multicast for a day, with this script green on both. Keep running it: the
#   "names a source that is not in the tree" half is exact, and it is the half that caught
#   the three incidents above.
#
#   Status: added 2026-08-13 with the Phase 6.6 reduce conversion, after all seven Phase 6
#           program directories were found stale. Demoted 2026-08-19 to the existence half.
#   Issues: mtime, not content -- see above. Superseded, not fixed, because the two checks
#           answer different questions and the file-existence one still answers its own.
#   Todo:   None
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KERNELS=${KERNELS:-$HERE/../kernels}

if [ $# -eq 0 ]; then
    echo "usage: $(basename "$0") <program-dir> [more dirs...]" >&2
    exit 2
fi
if [ ! -d "$KERNELS" ]; then
    echo "REFUSED: no kernel source tree at KERNELS=$KERNELS -- this directory cannot be checked" >&2
    exit 1
fi

stale=0
for dir in "$@"; do
    if [ ! -d "$dir" ]; then
        echo "  $dir: MISSING"
        stale=$((stale + 1))
        continue
    fi

    sidecars=$(find "$dir" -maxdepth 1 -name '*.elf.src' | sort)
    if [ -z "$sidecars" ]; then
        echo "  $dir: NO SIDECARS -- built by something that does not record its sources"
        stale=$((stale + 1))
        continue
    fi

    problems=0
    for sidecar in $sidecars; do
        elf=${sidecar%.src}
        source_name=$(tr -d '[:space:]' < "$sidecar")
        source_path=$KERNELS/$source_name

        if [ ! -f "$source_path" ]; then
            echo "  $dir: STALE -- $(basename "$elf") was built from '$source_name', which is NOT in $KERNELS"
            problems=$((problems + 1))
            continue
        fi
        if [ ! -f "$elf" ]; then
            echo "  $dir: BROKEN -- $(basename "$sidecar") has no $(basename "$elf") beside it"
            problems=$((problems + 1))
            continue
        fi
        if [ "$source_path" -nt "$elf" ]; then
            echo "  $dir: STALE -- $source_name is NEWER than $(basename "$elf")"
            problems=$((problems + 1))
        fi
    done

    if [ "$problems" -eq 0 ]; then
        echo "  $dir: current ($(echo "$sidecars" | wc -l | tr -d ' ') images)"
    else
        stale=$((stale + 1))
    fi
done

if [ "$stale" -ne 0 ]; then
    echo "$stale program directory/directories are STALE -- rebuild before believing any result from them"
    exit 1
fi
exit 0
