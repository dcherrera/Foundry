# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Foundry, please report it responsibly.

**Do not open a public issue for security vulnerabilities.**

Email **coreyh55@gmail.com** with:

- a description of the vulnerability
- steps to reproduce
- potential impact
- a suggested fix, if you have one

You will get an acknowledgement within 48 hours. Critical issues are aimed at a
fix within 7 days. If you want credit in the fix commit, say so and give me the
name to use.

## Scope

Security-relevant areas in Foundry include:

- **safetensors/** - File parsing and memory-mapped I/O
- **tokenizer/** - Input parsing and encoding
- **data/** - File loading (JSONL, CSV)
- **swarm/** - Network protocol handling (coordinator/worker sockets)

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | Yes       |

## Best Practices

When contributing to Foundry:

- Use `snprintf` instead of `sprintf`
- Validate all array indices before access
- Check all `malloc`/`calloc` return values for NULL
- Use bounds-checked operations for buffer handling
- Avoid integer overflow in size calculations
