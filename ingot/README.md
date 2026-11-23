# ingot — exact model file compression

Makes the model files people download smaller, **losslessly**: decoding gives back the exact
original file (same sha256).

- **BF16 safetensors:** ~10.5 bits per weight instead of 16. Format version 3 codes a tensor
  given the tensors that hold the same units (a down_proj column given its neuron's gate and up
  rows; also up, z, out_proj, o_proj): −0.021 bits per weight on Qwen3.8-27B layers 4–7.
- **GGUF Q8_0:** ~7.9 instead of 8.5. The scales are coded as the BF16 values they came from.
- **GGUF K-quants:** a few percent smaller (the scale and code fields are coded).
- **Fine-tunes:** a fine-tune of a model you already have can be stored as a delta against it,
  often a small fraction of the file.

Files decode as a stream: a downloader can rebuild the original while the `.ingot` arrives,
holding only a few MB. Format: [`FORMAT.md`](FORMAT.md).

## Build and test

```
make -C ingot            # CLI (ingot/build/ingot) + tests
make -C ingot test       # C suites + CLI checks
```

## Convert your own model

```
# a GGUF or safetensors file on disk -> model.gguf.ingot
ingot encode model.gguf

# straight from a download, nothing stored but the .ingot (stdin needs an output name)
curl -sL https://huggingface.co/ORG/REPO/resolve/REV/model.gguf | ingot encode - model.gguf.ingot

# a fine-tune as a delta against the base model the user already has
ingot encode --base base-Q8_0.gguf finetune-Q8_0.gguf

# check it: decode without writing, compare every tensor hash and the file's sha256
ingot verify model.gguf.ingot                  # (+ the original, byte for byte: ... model.gguf)
ingot verify --base base-Q8_0.gguf --check-base finetune-Q8_0.gguf.ingot

# get the original back
ingot decode model.gguf.ingot                  # -> model.gguf
curl -sL .../model.gguf.ingot | ingot decode - model.gguf

ingot info model.gguf.ingot                    # sizes per tensor, sha256s, the base it needs
```

Encode adds `.ingot` and decode removes it; give DST to choose the name. An existing default
output is never overwritten. Files are recognised by magic and version, not by name. Encode and
decode show progress on a terminal (`-q` turns it off) and use 4 threads (`-j N`). The usual
errors say what to do: a truncated download, a corrupt file, a missing or wrong base (with the
base's sha256 from the header).

Scripts:
- `scripts/ingot_gguf.sh REPO REV FILE OUTDIR` streams one Hugging Face GGUF through encode and
  verify and checks the rebuilt sha256 against Hugging Face's (`--base FILE` or
  `--base-hf REPO REV FILE` for a delta).
- `scripts/ingot_convert.sh` converts a whole safetensors repo, one shard at a time:

```
ingot/scripts/ingot_convert.sh --rev <commit> Org/Model out/
ingot/scripts/ingot_convert.sh --base-repo Org/Base --base-rev <commit> Org/FineTune out/
```

## Library

| Path | What |
|---|---|
| `core/` | errors, byte inputs (file / pipe / callback), BF16, ggml types, hashes, bit packing, parallel-for, rANS |
| `codec/` | block framing; plain and delta (BF16); the field coder and the Q8_0, K-quant and block-delta coders; one block decoder over all modes |
| `container/` | source layout (safetensors, GGUF), header and directory, streaming encoder and decoder, random-access reader, base set |
| `cli/`, `scripts/` | the `ingot` CLI, converters |

A program that downloads `.ingot` files links `container/ingot_stream.h`. It passes the download
as an `ingot_input` callback and gets the exact file through a write callback. Every tensor hash
and the file's sha256 are checked on the way.
