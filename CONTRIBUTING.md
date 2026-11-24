# Contributing to Foundry

Thanks for your interest in contributing to Foundry! This document covers the basics.

## Licensing of contributions — read this first

Foundry ships under **AGPL-3.0** (see [`LICENSE`](./LICENSE)).

> **Contributions are made under the Contributor License Agreement in
> [`CLA.md`](./CLA.md)**, which grants me a broad copyright and patent license,
> including the right to sublicense your contribution under any terms and to
> transfer that right along with the project.

**You keep your copyright.** The CLA is a license, not an assignment — nothing
is transferred away from you. Your contribution stays yours to use, publish,
sell, or license to anyone else, anywhere, forever.

Signing takes one commit: add your entry to
[`.cla/signatures.md`](./.cla/signatures.md) in your first pull request. Nothing
to print, scan, or email. Organizations whose employees contribute on company
time sign [`CLA-CORPORATE.md`](./CLA-CORPORATE.md) by email instead.

If you would rather not sign, open an issue describing the change instead of a
pull request. An idea costs nothing to accept.

### Sign your commits (DCO)

Every commit must carry a `Signed-off-by` line certifying the
[Developer Certificate of Origin](https://developercertificate.org/) — that you
wrote the code, or otherwise have the right to submit it:

```bash
git commit -s -m "your message"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and a working email address — the same address as your CLA
roster entry, since that is what CI matches on. If you are contributing work your
employer owns, confirm you actually have the right to do so before you sign off.

Both checks run in CI on every pull request, and you can run them yourself first:

```bash
./scripts/check-dco.sh     # every commit signed off, matching its author
./scripts/check-cla.sh     # every author on the CLA roster
./scripts/check-headers.sh # every source file you ADD has SPDX + the @file block
```

CI also runs `public-scan`, a content check for private paths, internal names and
credentials. Its rules are not published, so on a fork pull request a maintainer
runs it instead.

### License header on new files

Every source file starts with:

```c
/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
```

Add it to new files you create (`.c`, `.h`, `.cu`, `.cuh`, `.m`). Do not edit the
copyright line on existing files — you keep your copyright in what you contribute
without needing a line in the header, and a per-file list of names becomes
unmaintainable fast. `git log` and `.cla/signatures.md` are the authoritative
record of who wrote what.

## Getting Started

```bash
# Clone and build
git clone https://github.com/dcherrera/Foundry.git
cd Foundry
make all

# Run tests
make test
```

## Development Workflow

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure `make all` compiles with zero warnings (`-Wall -Wextra`)
5. Ensure `make test` passes all tests
6. Submit a pull request

## Code Style

- **C11 standard** (`-std=c11`)
- **Two-letter module prefixes** (e.g., `tm_` for tensor, `ft_` for tokenizer)
- **Header guards** in all `.h` files
- **`extern "C"` guards** in all public headers for C++ compatibility
- **NULL checks** on all pointer parameters
- **`snprintf`** instead of `sprintf` (never use `sprintf`)
- No compiler warnings with `-Wall -Wextra`

Some comments still cite notes from the pre-release history (`DEV_NOTES`, `docs/tt_*.md`,
`results/*.md`, plan and spec files). Those files are not in this repo; the comment itself is the
record. Comments in `tt/kernels/` and `tt/fw/` are left as built, because the embedded program
images record a hash of every source file.

## Module Structure

Each module follows this layout:

```
module/
├── module.h          # Public API
├── module.c          # Implementation
└── tests/
    └── test_module.c # Tests using lib/test.h
```

## Adding Tests

Use the test framework in `lib/test.h`:

```c
#include "lib/test.h"
#include "module/module.h"

static void test_my_feature(void) {
    ASSERT_TRUE(condition);
    ASSERT_INT_EQ(expected, actual);
    ASSERT_FLT_NEAR(expected, actual, tolerance);
}

int main(void) {
    RUN_TEST(my_feature);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
```

Add a corresponding test target in the `Makefile`.

## Pull Requests

Code is reviewed against
[`docs/coding-best-practices.md`](docs/coding-best-practices.md), cited by rule
number. This section covers the PR itself.

**One logical change per PR.**

**Size.** Reviewable lines = added + deleted, excluding generated files, pure
renames, and vendored imports.

| Reviewable lines | |
|---|---|
| ≤ 400 | normal |
| 400–1,000 | expect a request to split |
| > 1,000, or > 15 files | state in one line why it can't be split |

**Exempt from the size limits — but must be alone in their own PR:**

1. **Generated sources** (`fc_cuda_ptx.h` and friends). State the regeneration
   command, toolchain version, and target arch. Reviewed by rebuilding, not by
   reading — see `tools/ptx_modules.py --verify`.
2. **Pure moves and renames.** No content change in the same commit.
3. **Mechanical sweeps.** Include the command that reproduces the diff.
4. **Vendored or backend imports.**

**Evidence.**

- **Performance claim** — the command, the device, before/after in tok/s, and the
  noise floor. A multiplier on its own is not a result.
- **Forward pass or kernel** — parity against the reference path, argmax-exact or
  ulp-0, naming the model and quantization.
- **A test you call green** — show it can go red (rule 8). Delete the `.o` *and*
  the test binary first, or you may be running the old one.
- **Untested** — say so plainly rather than leaving it implied.

**Keep it short.** Title is `scope: imperative summary`, lowercase scope, ≤72
characters. The body says what changed and why in a few lines. Commit messages
the same: a subject line, and a body only when the reason isn't visible in the
diff. Long rationale belongs in a spec file, not a commit message. No tool,
assistant, or generator attribution anywhere — title, body, or comments.

**Review comments carry a severity prefix:**

- `blocker:` — must change before merge
- `question:` — answer before merge
- `nit:` — optional, never blocks

Unprefixed reads as `nit:`. The author does not resolve a `blocker:` thread — the
reviewer does. Approval goes stale when new commits land, unless they are a
rebase or a fixup.

**Merge bar:** builds clean with zero warnings, tests pass, DCO and CLA checks
green, and the code meets the practices doc.

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Minimal reproduction steps
- Your platform and compiler version

## Philosophy

- **Full feature parity** with Python equivalents - when in doubt, implement it
- **CPU first**, GPU extends - every operation must work on CPU
- **Simple over clever** - readable C over micro-optimizations
- **Test everything** - if it's not tested, it's not done
