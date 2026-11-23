#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# record_tensix_build_closure.sh
#   Records EVERY file a Tensix program build opens — headers through every compile, archive
#   members through every link — by rebuilding real program directories through a shim
#   toolchain. Clean-room plan Phase 1: the firmware TUs build without -MMD, so their header
#   closure was only ever inferred from the -I set.
#
#   Usage:  record_tensix_build_closure.sh <out-dir> <program-dir> [program-dir ...]
#   Env:    REAL_SFPI  the real toolchain (default /opt/tenstorrent/sfpi)
#   Output: <out-dir>/log/*.H (one `-H` trace per compile), *.map (one map per link),
#           *.cmd (the command each came from), <out-dir>/<program>/ (the rebuilt images),
#           <out-dir>/summary.txt
#
#   HOW, AND WHY IT CANNOT PERTURB THE BUILD. SFPI is pointed at a shim whose g++/gcc run the
#   REAL command unchanged, then re-run it on the side: a compile again as `-E -H -o
#   /dev/null` (no -c/-o/-M* flags), a link again to a throwaway output with `-Wl,-Map`.
#   Every other tool is a symlink. The shim-built images are then `cmp`ed against the
#   originals: byte-identical is the proof the shim changed nothing.
#
#   Status: added 2026-10-08.
#   Issues: None
#   Todo:   None
#
set -uo pipefail

OUT=${1:?usage: record_tensix_build_closure.sh <out-dir> <program-dir> [program-dir ...]}
shift
[ $# -gt 0 ] || { echo "name at least one program directory" >&2; exit 2; }
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REAL_SFPI=${REAL_SFPI:-/opt/tenstorrent/sfpi}
REPO=$(cd -- "$HERE/../.." && pwd)

mkdir -p "$OUT/log" "$OUT/shim/compiler/bin"
OUT=$(cd "$OUT" && pwd)
SHIM=$OUT/shim
ln -sfn "$REAL_SFPI/include" "$SHIM/include"
for tool in "$REAL_SFPI"/compiler/*; do
    [ "$(basename "$tool")" = bin ] || ln -sfn "$tool" "$SHIM/compiler/$(basename "$tool")"
done
for tool in "$REAL_SFPI"/compiler/bin/*; do
    ln -sfn "$tool" "$SHIM/compiler/bin/$(basename "$tool")"
done

# The recording wrapper, one per compiler driver.
for driver in riscv-tt-elf-g++ riscv-tt-elf-gcc; do
    rm -f "$SHIM/compiler/bin/$driver"
    cat > "$SHIM/compiler/bin/$driver" <<WRAP_EOF
#!/usr/bin/env bash
REAL="$REAL_SFPI/compiler/bin/$driver"
LOG="$OUT/log"
"\$REAL" "\$@"; status=\$?
[ \$status -eq 0 ] || exit \$status
mode=link; output=""; side=(); skip=0
for arg in "\$@"; do
    if [ \$skip -eq 1 ]; then skip=0; [ "\$prev" = "-o" ] && output=\$arg; continue; fi
    case "\$arg" in
        -c) mode=compile ;;
        -E) mode=preprocess ;;
        -o|-MF|-MT|-MQ) skip=1; prev=\$arg; continue ;;
        -MMD|-MD|-MP) continue ;;
    esac
    side+=("\$arg")
done
tag=\$(date +%s%N)_\$\$
case \$mode in
    compile)
        printf '%s\n' "\$output" > "\$LOG/\$tag.cmd"; printf '%q ' "\$@" >> "\$LOG/\$tag.cmd"
        "\$REAL" "\${side[@]}" -E -H -o /dev/null 2> "\$LOG/\$tag.H" ;;
    link)
        printf '%s\n' "\$output" > "\$LOG/\$tag.cmd"; printf '%q ' "\$@" >> "\$LOG/\$tag.cmd"
        "\$REAL" "\${side[@]}" -o "\$LOG/\$tag.throwaway" -Wl,-Map="\$LOG/\$tag.map" \
            > /dev/null 2>&1; rm -f "\$LOG/\$tag.throwaway" ;;
esac
exit 0
WRAP_EOF
    chmod +x "$SHIM/compiler/bin/$driver"
done

identical=0 differing=0
for program in "$@"; do
    program=$(cd "$program" && pwd); name=$(basename "$program")
    mkdir -p "$OUT/$name"
    SFPI=$SHIM bash "$HERE/rebuild_tensix_program.sh" "$program" "$OUT/$name" \
        > "$OUT/$name.rebuild.log" 2>&1 || echo "REBUILD FAILED: $name (see $name.rebuild.log)"
    for image in "$program"/*.elf; do
        if cmp -s "$image" "$OUT/$name/$(basename "$image")"; then
            identical=$((identical + 1))
        else
            differing=$((differing + 1)); echo "DIFFERS: $name/$(basename "$image")"
        fi
    done
done

{
    echo "shim-built images: $identical byte-identical, $differing differing"
    echo
    echo "== sources compiled from outside the repo (must be empty)"
    grep -h -o "/[^ ']*\.\(cc\|cpp\|c\|S\)" "$OUT"/log/*.cmd | grep -v "^$REPO/" | sort | uniq -c
    echo
    echo "== headers opened from outside the repo, the SFPI tree and this build's outputs, per TU (must be empty)"
    for cmd in "$OUT"/log/*.cmd; do
        tu=$(grep -o "[^/ ']*\.\(cc\|cpp\|c\|S\)\b" "$cmd" | tail -1)
        trace=${cmd%.cmd}.H
        [ -f "$trace" ] && sed 's/^\.* //' "$trace" | grep "^/" \
            | grep -v -E "^($REPO|$REAL_SFPI|$SHIM|$OUT)/" | sed "s|^|$tu |"
    done | sort -u > "$OUT/headers_by_tu.txt"
    awk '{print $1}' "$OUT/headers_by_tu.txt" | sort | uniq -c
    echo "unique outside header paths: $(awk '{print $2}' "$OUT/headers_by_tu.txt" | sort -u | wc -l)"
    echo
    echo "== SFPI-tree headers opened (realpath)"
    cat "$OUT"/log/*.H 2>/dev/null | sed 's/^\.* //' | grep -E "^($REAL_SFPI|$SHIM)/" \
        | xargs -r -n1 realpath | sort | uniq -c
    echo
    echo "== archive members linked into any ELF"
    for map in "$OUT"/log/*.map; do
        awk '/^Archive member included/ { on = 1; next }
             /^(Discarded input sections|Memory Configuration|Allocating common)/ { on = 0 }
             on && /\.a\(/ { print $1 }' "$map"
    done | sort | uniq -c
    echo "maps read: $(ls "$OUT"/log/*.map 2>/dev/null | wc -l);" \
         "LOADed archives: $(grep -h '^LOAD .*\.a$' "$OUT"/log/*.map 2>/dev/null | sort -u | tr '\n' ' ')"
    echo "(no member lines above = no libgcc / libc object in any image)"
} > "$OUT/summary.txt" 2>&1
echo DONE > "$OUT/DONE"
