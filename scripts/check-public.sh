#!/bin/sh
# Content scan for public repositories: flags private paths, hosts, internal
# names, closed technology and credentials before they leave a private clone.
#
#   check-public.sh -p PATTERNS range BASE HEAD   added lines of BASE..HEAD, full
#                                               contents of every file the range
#                                               changes (at HEAD), commit messages
#   check-public.sh -p PATTERNS tree REV          every text file at REV
#
# BASE may be the empty tree (4b825dc642cb6eb9a060e54bf8d69288fbee4904).
# PATTERNS lines are "<class> <ERE>" with class leak, closed, secret or allow;
# allow rules drop hits whose line "<class> <path>:<line>: <text>" they match.
# Contents are scanned, never file names. Hits on lines the range adds are
# marked [new]; other hits sit in a file the range touches. Commit messages
# show as path "commit-<sha12>".
#
# Exit: 0 clean, 1 hits found, 2 usage or git error.
# POSIX sh + git + grep + awk + sed. Scratch files go under the git dir.

set -eu

# Byte semantics: generated sources hold invalid UTF-8.
LC_ALL=C
export LC_ALL

EMPTY_TREE=4b825dc642cb6eb9a060e54bf8d69288fbee4904

usage() {
	sed -n '4,8p' "$0" | sed 's/^# \{0,1\}//' >&2
	exit 2
}

PATTERNS=""
while getopts p: opt; do
	case "$opt" in
	p) PATTERNS=$OPTARG ;;
	*) usage ;;
	esac
done
shift $((OPTIND - 1))

[ -n "$PATTERNS" ] && [ -r "$PATTERNS" ] || {
	echo "public-scan: pattern file missing or unreadable: ${PATTERNS:-<none>}" >&2
	exit 2
}
[ $# -ge 1 ] || usage
MODE=$1
shift
case "$MODE" in
range) [ $# -eq 2 ] || usage; BASE=$1; HEAD=$2 ;;
tree) [ $# -eq 1 ] || usage; BASE=$EMPTY_TREE; HEAD=$1 ;;
*) usage ;;
esac

git rev-parse --verify --quiet "$HEAD^{commit}" >/dev/null || {
	echo "public-scan: not a commit: $HEAD" >&2
	exit 2
}
git rev-parse --verify --quiet "$BASE^{tree}" >/dev/null || {
	echo "public-scan: not a commit or tree: $BASE" >&2
	exit 2
}

GITDIR=$(git rev-parse --absolute-git-dir)
WORK=$(mktemp -d "$GITDIR/public-scan.XXXXXX")
trap 'rm -rf "$WORK"' EXIT INT TERM

G="git -c core.quotepath=off"

# Lines added by the range, as "path:line" (new-file numbering).
$G diff -U0 --no-color --no-ext-diff --no-renames \
	--src-prefix=a/ --dst-prefix=b/ "$BASE" "$HEAD" |
	awk '
	/^\+\+\+ / { path = ($0 == "+++ /dev/null") ? "" : substr($0, 7); next }
	/^@@ / { split($3, a, ","); ln = substr(a[1], 2) + 0; next }
	/^\+/ { if (path != "") print path ":" ln; ln++ }
	' >"$WORK/added"

# Files the range leaves present at HEAD, extracted under $WORK/t. Symlinks are
# replaced by their targets under $WORK/l; commit messages go under $WORK/m.
$G diff --raw --no-abbrev --no-renames --diff-filter=d "$BASE" "$HEAD" >"$WORK/raw"
sed 's/^[^	]*	//' "$WORK/raw" >"$WORK/files"
mkdir "$WORK/t" "$WORK/l" "$WORK/m"
if [ -s "$WORK/files" ]; then
	if [ "$MODE" = tree ]; then
		git archive --format=tar "$HEAD" | (cd "$WORK/t" && tar -xf -)
	else
		sed 's/^/:(literal)/' "$WORK/files" | tr '\n' '\000' |
			xargs -0 sh -c 'rev=$1 dir=$2; shift 2
				git archive --format=tar "$rev" -- "$@" >"$dir.tar" || exit 255
				(cd "$dir" && tar -xf -) <"$dir.tar" || exit 255' \
				sh "$HEAD" "$WORK/t" 2>"$WORK/archive.err" || {
			echo "public-scan: extracting changed files failed:" >&2
			cat "$WORK/archive.err" >&2
			exit 2
		}
		rm -f "$WORK/t.tar"
	fi
	# Every regular file must have arrived, or a broken extraction reads as clean.
	want=$(awk '{ split($0, m, " ") } m[2] != "120000" && m[2] != "160000"' "$WORK/raw" | wc -l | tr -d ' ')
	got=$(find "$WORK/t" -type f | wc -l | tr -d ' ')
	if [ "$want" != "$got" ]; then
		echo "public-scan: extracted $got of $want changed files; refusing to report clean" >&2
		exit 2
	fi
	find "$WORK/t" -type l -exec rm -f {} +
	awk -F '\t' '{ split($1, m, " "); if (m[2] == "120000") print m[4] "\t" $2 }' "$WORK/raw" |
		while IFS='	' read -r sha f; do
			mkdir -p "$WORK/l/$(dirname "$f")"
			git cat-file blob "$sha" >"$WORK/l/$f"
		done
fi

if [ "$MODE" = range ]; then
	if [ "$BASE" = "$EMPTY_TREE" ]; then
		revs=$(git rev-list "$HEAD")
	else
		revs=$(git rev-list "$BASE..$HEAD")
	fi
	for c in $revs; do
		git log -1 --format=%B "$c" >"$WORK/m/commit-$(echo "$c" | cut -c1-12)"
	done
fi

rules() {
	sed -n "s/^$1[[:space:]]\{1,\}//p" "$PATTERNS"
}

: >"$WORK/hits"
for class in secret leak closed; do
	rules "$class" >"$WORK/p.$class"
	[ -s "$WORK/p.$class" ] || continue
	icase=-i
	[ "$class" = secret ] && icase=
	# shellcheck disable=SC2086
	grep -rnI -E $icase -f "$WORK/p.$class" "$WORK/t" "$WORK/l" "$WORK/m" 2>"$WORK/grep.err" |
		awk -v c="$class" -v W="$WORK" '
		{
			s = substr($0, length(W) + 4)
			if (!match(s, /:[0-9]+:/)) next
			t = substr(s, RSTART + RLENGTH)
			if (length(t) > 200) t = substr(t, 1, 200) "..."
			print c " " substr(s, 1, RSTART + RLENGTH - 1) " " t
		}' >>"$WORK/hits" || true
	if [ -s "$WORK/grep.err" ]; then
		echo "public-scan: grep failed on $class rules:" >&2
		cat "$WORK/grep.err" >&2
		exit 2
	fi
done

rules allow >"$WORK/p.allow"
if [ -s "$WORK/p.allow" ]; then
	grep -v -E -f "$WORK/p.allow" "$WORK/hits" >"$WORK/kept" || true
else
	cp "$WORK/hits" "$WORK/kept"
fi

if [ ! -s "$WORK/kept" ]; then
	echo "public-scan: clean ($(wc -l <"$WORK/files" | tr -d ' ') files, range $(echo "$BASE" | cut -c1-12)..$(echo "$HEAD" | cut -c1-12))"
	exit 0
fi

awk -v A="$WORK/added" '
BEGIN { while ((getline l <A) > 0) add[l] = 1 }
{
	split($2, a, ":"); key = a[1] ":" a[2]
	sub(/:$/, "", key)
	print ((key in add) || a[1] ~ /^commit-/ ? "[new] " : "      ") $0
}' "$WORK/kept"
echo "public-scan: $(wc -l <"$WORK/kept" | tr -d ' ') hit(s). Remove them, or add an allow rule if it is a known exception." >&2
exit 1
