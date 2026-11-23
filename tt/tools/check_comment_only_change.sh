#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_comment_only_change.sh
#   Proves a change touched COMMENTS ONLY: every changed C/C++/asm source between a git ref
#   and the working tree must have the same comment-stripped token stream on both sides.
#
#   Usage:  check_comment_only_change.sh <base-ref> [path ...]
#           Paths default to the whole repository. Exit 0 = comment-only, 1 = a code change
#           (each one named), 2 = usage or tool error.
#
#   HOW. A tokenizer that knows string and character literals replaces every comment with a
#   space; whitespace is then squeezed and blank lines dropped, so re-wrapping a comment
#   cannot read as a change. Python files are compared on their AST with docstrings removed.
#   Any other changed file is reported as unchecked, never passed silently.
#
#   PROVEN RED 2026-10-08: the first version stripped with `clang -fpreprocessed`, which Apple
#   clang does not have; the error was swallowed, both sides were empty, and it passed a real
#   code change. The stripper now fails loudly on an error or on an empty result.
#
#   Status: added 2026-10-08 for the clean-room Phase 0 (licence notices are comments only).
#   Issues: None
#   Todo:   None
#
set -uo pipefail

BASE=${1:?usage: check_comment_only_change.sh <base-ref> [path ...]}
shift
ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2

WORK=$(mktemp -d "${TMPDIR:-/tmp}/comment_only.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/strip_c.py" <<'PYEOF'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="surrogateescape").read()
token = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|\s+|.', re.S)
out = []
for match in token.finditer(text):
    piece = match.group(0)
    if piece.startswith("//") or piece.startswith("/*"):
        out.append(" ")
    elif piece.isspace():
        out.append("\n" if "\n" in piece else " ")
    else:
        out.append(piece)
code = [" ".join(line.split()) for line in "".join(out).split("\n")]
code = [line for line in code if line]
if not code:
    sys.exit("strip_c: %s has no code left after stripping" % sys.argv[1])
print("\n".join(code))
PYEOF

cat > "$WORK/strip_py.py" <<'PYEOF'
import ast, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
for node in ast.walk(tree):
    body = getattr(node, "body", None)
    if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
            and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        node.body = body[1:]
print(ast.dump(tree))
PYEOF

code=0 checked=0
while IFS= read -r path; do
    [ -f "$path" ] || { echo "DELETED   $path"; code=1; continue; }
    old="$WORK/old"
    git show "$BASE:$path" > "$old" 2>/dev/null || { echo "NEW FILE  $path"; code=1; continue; }
    case "$path" in
        *.c|*.h|*.cc|*.cpp|*.hpp|*.S|*.ld) stripper=$WORK/strip_c.py ;;
        *.py)                              stripper=$WORK/strip_py.py ;;
        *)  echo "UNCHECKED $path (not a C or Python source)"; code=1; continue ;;
    esac
    python3 "$stripper" "$old" > "$WORK/a" && python3 "$stripper" "$path" > "$WORK/b" ||
        { echo "ERROR     $path (stripper failed)"; exit 2; }
    checked=$((checked + 1))
    if cmp -s "$WORK/a" "$WORK/b"; then
        echo "comments  $path"
    else
        echo "CODE      $path"; diff "$WORK/a" "$WORK/b" | head -10 | sed 's/^/    /'
        code=1
    fi
done < <(git diff --name-only "$BASE" -- "$@")

echo "checked $checked file(s): $([ $code -eq 0 ] && echo COMMENT-ONLY || echo NOT comment-only)"
exit $code
