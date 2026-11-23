## What this changes

<!-- One logical change. A few lines — what, and why. Not an essay.
If this is over 1,000 reviewable lines or 15 files, say why it can't be split. -->

## Evidence

<!-- Kernel, op, or anything numeric: parity against the CPU reference — what
tolerance, what shapes. Performance: the command, the device, before/after, and
the noise floor. Untested? Say so. "It runs and the loss goes down" is not
evidence. -->

## Checklist

- [ ] **I license this contribution under the CLA in [`CLA.md`](../CLA.md)** — my entry
      is in [`.cla/signatures.md`](../.cla/signatures.md). I keep my copyright.
- [ ] Every commit signed off (`git commit -s`) — `./scripts/check-dco.sh`
- [ ] `make all` clean, zero warnings under `-Wall -Wextra`; `make test` passes
- [ ] Every operation still works on CPU — GPU extends, it does not replace
- [ ] New files carry the SPDX pair and the `@file` header block
- [ ] Docs updated if the public API changed

<!-- Contributing on behalf of an employer that owns your work? Your organization
signs CLA-CORPORATE.md by email instead — see that file. -->
