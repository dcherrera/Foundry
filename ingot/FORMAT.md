# .ingot format, version 4

Exact (lossless) storage of a safetensors or GGUF file. Decoding gives back every tensor
bit-identical and the whole source file byte-identical (same SHA-256). All integers are
little-endian.

**Name.** The original file name plus `.ingot`: `model-00001-of-00018.safetensors.ingot`,
`Qwen3.6-35B-A3B-Q8_0.gguf.ingot`. The name is only a convention: readers recognise a file by its
magic and version, and refuse any version they do not know.

**Streaming.** The directory comes before the payloads, so a decoder can read the file once, front
to back (e.g. straight from a download), and write the source file front to back. Version 2
payloads are in source order: the decoder holds only a few blocks. Version 3 payloads are in
decode order (a predicted tensor after its partners); the decoder holds a tensor decoded ahead of
its place in the output, and a partner until its last dependent is decoded (about one layer's
tensors), and still writes the source in order. Version 4 adds match entries (mode 7),
whose sources may also be tensors of earlier files of the same model (external refs, read from
the files already rebuilt); the directory tail lists every source, so the decoder knows what to
hold before it reaches the payloads.

## File

| Offset | Type | Field |
|---|---|---|
| 0 | u8[4] | magic `INGT` |
| 4 | u32 | format version: 4 if the file has match targets (a directory tail), else 3 if it has a mode 6 entry or payloads out of source order, else 2 (same header; readers refuse versions they do not know) |
| 8 | u32 | entry count |
| 12 | u32 | source kind: 1 safetensors, 2 GGUF |
| 16 | u64 | source header offset (= 144) |
| 24 | u64 | source header size |
| 32 | u64 | directory offset |
| 40 | u64 | directory size |
| 48 | u64 | source file size |
| 56 | u64 | this file's size (truncation check) |
| 64 | u8[32] | SHA-256 of the source file |
| 96 | u8[32] | SHA-256 of the base file (delta files over one base file; zero otherwise) |
| 128 | u64 | base file size (0 if unknown or none) |
| 136 | u64 | reserved, 0 |

Then the source header region verbatim (safetensors: 8-byte length + JSON; GGUF: everything
before the tensor data, alignment padding included), padded to 8; then the directory, padded to
8; then one payload per entry in entry order, each starting 8-aligned.

## Directory entry

| Type | Field |
|---|---|
| u16, u8[n] | name length, name (empty = filler bytes between or after tensors) |
| u8 | dtype: 0 other, 1 BF16, 2 F16, 3 F32, 4 F64, 5 I8, 6 U8, 7 I16, 8 U16, 9 I32, 10 U32, 11 I64, 12 U64, 13 BOOL; 64 + ggml type for other GGUF types (72 = Q8_0, 75..78 = Q3_K..Q6_K) |
| u8 | mode: 0 raw, 1 plain, 2 delta, 3 quant, 4 quant delta, 5 block delta, 6 predict (version 3 and later), 7 match (version 4); a mode newer than the file version is refused as unsupported |
| u8 | ndim (≤ 8), u8 zero |
| u64[ndim] | shape, row-major (the last dimension is contiguous; GGUF dims are listed reversed) |
| u64 ×6 | source offset, source size, payload offset, payload size, hash, base hash |

Entries are in payload order and tile the source data region; version 2 requires source order.
`hash` is over the source bytes; `base hash` (delta modes) is over the base tensor's bytes, so a
decoder refuses a wrong base tensor (mode 6: the partner entries, see below); the header's base
SHA-256 identifies the whole base file. Hash:
`h = 0x6A09E667F3BCC909 ^ size`; per 64-bit word `w` (tail zero-padded):
`h = (h ^ w) * 0x9E3779B97F4A7C15; h ^= h >> 32`. It detects corruption, not tampering.

Every tensor takes the smallest of its candidate modes; anything else is raw.

## Shared pieces

**Payload frame.** Every coded payload (modes 1–6) starts with `u64 prefix_bytes` (where the
block area starts, frame included) and `u64 block_count`, then the mode prefix, then the block
area. A raw payload is the source bytes.

**rANS.** Static 16-bit tables, 32-bit state in [2^16, 2^32), 16-bit renormalization. Table on
disk: u16 entry count, then entries (u16 symbol, u16 freq − 1), symbols ascending, frequencies
summing to 2^16 (zero entries = unused context). Decode one symbol on state x:

```
slot = x & 0xFFFF;  s = symbol with start[s] <= slot < start[s] + freq[s]
x = freq[s] * (x >> 16) + slot - start[s]
if (x < 2^16) x = (x << 16) | next word
```

**Blocks.** Symbol n of a block uses lane n % 32. Layout:
`u32 raw_bytes, u32 word_count, u32 states[32], u8 raw[raw_bytes], pad to 2,
u16 words[word_count], pad to 8`. Words are read forward, shared by all lanes in symbol order.
After the block every lane state is 2^16, every word and every raw byte is consumed (a check).
The raw stream packs fields LSB-first in order.

**Block area.** `u64 offsets[block_count + 1]` relative to the first block (the last entry is the
area size), then the blocks.

**Step code** (delta, quant): a signed difference d (|d| < 2^16) as one symbol for the bit
length L of |d|, its sign and the two bits after the leading one; the low L − 3 bits raw. L = 0 →
symbol 0; L = 1 → 1 + sign; L = 2 → 3 + 2·sign + (|d| & 1); L ≥ 3 → 7 + 8(L − 3) + 4·sign + next
two bits. Alphabet 119.

## Mode 1: plain (BF16)

Symbol = BF16 bits 14..5 (exponent + top 2 mantissa bits, 0..1023). Raw = 6 bits per weight,
`(sign << 5) | (mantissa & 31)`. Blocks of 2^18 weights.

Prefix: `u32 group_count, u32 block_log2 (18), u64 group_weights, u64 block_count`,
group_count tables, pad to 8. Groups are consecutive slices with their own table: the leading
dimension of a 3-D tensor (MoE experts) when each slice holds ≥ 2^18 weights, else one group.
Blocks run group by group; the last block of a group may be short.

## Mode 2: delta (BF16)

Coded against a base tensor of the same element count. Rows are the last dimension. Per row,
`row_scale` (i8) = round(2·log2(mean |t − b|)) clamped to ±127, or −128 if the row equals the
base (no symbols). Per weight of other rows: `e = max(exp(b), 1)`, `k = row_scale − 2(e − 134)`;
`k > 12`: fallback context 29, plain symbol of the target, 6 raw bits; else context
`max(k, −16) + 16` (0..28), step code of `d = ord(t) − ord(b)` (ord maps BF16 bits monotonically
to integers: +0 → 0, −0 → −1). Blocks of 2^18 weights over the flat index.

Prefix: `u32 context_count (30), u32 block_log2, u64 rows, u64 cols, u64 block_count`,
`i8 row_scale[rows]`, pad to 8, 30 tables (context 29 has alphabet 1024), pad to 8.

## Modes 3–5: quantized tensors (GGUF)

A tensor is rows × cols weights stored as ggml blocks along rows (cols = the contiguous
dimension). Coded blocks hold whole rows: `rows_per_block = cols ≥ 2^18 ? 1 : 2^18 / cols`.
Every symbol below is coded in the order given, through one context model per tensor.

Prefix: `u32 ggml_type, u32 flags (bit 0: base), u64 rows, u64 cols, u64 rows_per_block,
u64 block_count, u32 context_count, u32 0`, context_count tables (in the context order below),
pad to 8.

**Base flags** (modes 4 and 5): before a ggml block's fields, one symbol (alphabet 2, context =
the previous flag in the row, starting at 0): 1 = the block equals the base tensor's block at
the same place; it has no further symbols and is copied from the base.

**Mode 3, Q8_0** (block: fp16 d, int8 codes[32]). Contexts: [2 flags], 2 scale (alphabet 120),
1 fp16 step (119), 1 position (33), 8 codes (256). Per row: first every block's flag and scale,
then every coded block's codes.
- Scale. ggml quantizes from BF16 with `d = fp16(f32(amax / 127))`, so a scale is coded as its
  BF16 amax: the step code of `amax − ref` in context 0 (first block of the row) or 1, where ref
  is the amax of the previous amax-coded block in the coded block (0 at its start); the decoder sets
  `d = fp16(f32(amax / 127))` (both roundings to nearest even). A scale that no BF16 amax
  produces is symbol 119 followed by the step code of `ord(d) − ref16` in the fp16-step context
  (ref16 = ord of the previous coded block's scale, 0 at the start). Both refs follow every
  coded block (an amax-coded block sets ref16 = ord(its d)).
- Codes. The position p of the first code equal to ±127 (32 = none); if p < 32 one raw bit, its
  sign. The other codes, `code + 128`, in code context
  `clamp(floor((d − median) / 512) + 4, 0, 7)` with d and the row's median scale as unsigned
  16-bit patterns (median = the (n/2)-th smallest of the row's n scales, identical blocks
  included).

**Mode 3, K-quants** (Q3_K, Q4_K, Q5_K, Q6_K: 256 weights per block, ggml `block_q3_K` ..
`block_q6_K`). A block splits bijectively into fields (`codec/ingot_kquant_fields.h`): fp16 d;
fp16 dmin (Q4_K, Q5_K); S sub-block scales (Q3_K 16 x 6 bit, Q4_K / Q5_K 8 x 6 bit, Q6_K 16 int8
stored as int8 + 128); 8 six-bit mins (Q4_K, Q5_K); 256 codes in weight order (Q3_K low2 + 4 x
hmask bit, 0..7; Q4_K 0..15; Q5_K 0..31; Q6_K 0..63). Sub-block k holds weights
[k·256/S, (k+1)·256/S). Contexts: [2 flags], 2 d steps (119), [2 dmin steps (119)], scales
(Q3_K..Q5_K 5 of alphabet 64; Q6_K 9 of 256), [8 mins (64)], codes (Q3_K 32 of 8, Q4_K / Q5_K
64 of 16 / 32, Q6_K 16 of 64). Blocks in row order, every field of a block before the next:
- Flag (with a base): context = the previous block's flag in the coded block (0 at its start).
- d: the step code of `ord(d) − ref` in context 0 (first block of the row) or 1; ref = ord(d) of
  the previous coded block in the coded block (0 at its start); ord as for BF16. dmin the same
  with its own ref and contexts.
- Scale k: context `k == 0 ? F : scale[k−1] >> s` (6 bit: s = 4, F = 4; Q6_K: s = 5, F = 8).
- Min k: context `scale[k] >> 3`.
- Codes of sub-block k, one context: Q3_K `scale[k] >> 1`; Q6_K `scale[k] >> 4`; Q4_K / Q5_K
  `4 z + (scale[k] >> 4)` with z the zero point `dmin·min[k] / (d·scale[k])` in units of
  2^(bits − 4) codes, exactly in integers: fp16 magnitudes as `m·2^e` (sign ignored; exponent
  field E > 0: m = 1024 + mantissa, e = E − 25; E = 0: m = mantissa, e = −24), num = m_dmin·min,
  den = m_d·scale·2^(bits − 4), t = e_dmin − e_d; z = 15 if den = 0 or t > 30, else 0 if num = 0
  or t < −30, else min(15, floor(num·2^t / den)).

**Mode 4, quant delta.** Mode 3 with base flags.

**Mode 5, block delta** (any ggml type with known geometry; same type and shape as the base).
Unit = the ggml block, or 32 / block_weights blocks when a block holds fewer than 32 weights and
cols is a multiple of 32. Per unit: a base flag (contexts 0..1); a differing unit's bytes go raw,
8 bits each.

## Mode 6: predict (BF16, version 3)

A dependent tensor D coded given 1–2 partner tensors that hold the same units, all in the
residual basis (an FFN neuron's gate and up rows predict its down column). The writer predicts
by tensor name and shape, within one file (layer prefix X):

| D | Units | Partners (unit u → partner row) |
|---|---|---|
| `X.mlp.down_proj.weight` [H, I] | columns | `X.mlp.gate_proj.weight`, `X.mlp.up_proj.weight` [I, H]: row u |
| `X.mlp.up_proj.weight` [I, H] | rows | gate_proj: row u |
| `X.linear_attn.in_proj_z.weight` [V, H] | rows | `X.linear_attn.in_proj_qkv.weight` [R, H]: row R − V + u |
| `X.linear_attn.out_proj.weight` [H, V] | columns | in_proj_qkv row R − V + u; in_proj_z row u |
| `X.self_attn.o_proj.weight` [H, nh·hd] | columns | `X.self_attn.v_proj.weight` [nkv·hd, H]: grouped-query row (hd = length of `X.self_attn.q_norm.weight`, g = nh / nkv) |

**Units.** D holds U unit vectors of length H: unit u is row u (axis 0, D is [U, H]) or column u
(axis 1, D is [H, U]). Partner j is a BF16 tensor [rows, H]; unit u's partner vector is its row
`off + ((u / hd) / g)·hd + u % hd` (hd = g = 1: row off + u). Partners are earlier entries
(directory order), decoded and hash-checked before D.

**Predictor** (integers only, so every decoder computes the same contexts):
- Partner value x (BF16 bits) → fixed point `X = m·2^(e + F_j)` truncated toward zero, with
  m = 128 + mantissa, e = exp − 134 (exp > 0) or m = mantissa, e = −133 (exp = 0); |X| clamped to
  2^40; exp = 255 (inf, NaN) gives ±2^40. F_j = C − E_j.
- `p = a_u·X1 + b_u·X2` in int64 (a, b: the unit's int16 coefficients at exponents E_1, E_2, so p
  is the prediction at scale 2^C).
- Thresholds of unit u with scale s_u (unit RMS ≈ 2^(s_u/4)): `T_k = floor(t_k·r·2^(floor(s_u/4)
  + C))`, r = 2^((s_u mod 4)/4) as Q16 (65536, 77936, 92682, 110218), t_k in 1/256 units
  {16, 32, 64, 96, 128, 192, 256, 320, 384, 512, 768} (so T_k = Q8·Q16·2^(floor(s_u/4) + C − 24)),
  clamped to 2^62.
- Bucket = number of k with |p| ≥ T_k (0..11). Class = clamp(floor((s_u − class_ref) / width) +
  S/2, 0, S − 1). Context z = class·12 + bucket.

**Coding** per weight w of D, in D's row-major order: symbol `sign(w) xor (p < 0)` in context z
(alphabet 2); then the plain symbol (bits 14..5, alphabet 1024) in context 12·S + 2z + that bit;
then the low 5 mantissa bits raw. Blocks of 2^18 weights.

Prefix (after the frame): `u32 partner_count J, u32 axis, u64 U, u64 H, i32 C, u32 block_log2
(18), u64 block_count, u32 S (1..8), u32 width (1..16), i32 class_ref, u32 context_count (36·S),
u64 param_hash`; J × `{u32 entry, u32 0, u64 off, u64 hd, u64 g, i32 E_j, u32 0}`;
`i8 s[U]`, pad to 2, `i16 coef[U][J]`, pad to 8; context_count tables (S·12 binary, then 2·S·12
of alphabet 1024), pad to 8. `param_hash` is the entry hash of `s[]` followed by `coef[]` (as
stored, pads excluded): any changed coefficient or scale is refused. The directory entry's base
hash field names the same partners (`entry + 1` in the low 32 bits, the second partner's
`entry + 1` or 0 in the high 32 bits); a decoder refuses a mismatch, a partner at or after D, or a
partner that is not a 2-D BF16 [rows, H] tensor covering every mapped row.

The encoder fits a, b per unit by least squares and picks C, E, S and width (smallest exact
coded size, tables included); none of that is part of the format.

## Mode 7: match (BF16, version 4)

A target tensor D coded given up to 8 partner units per unit, each chosen from any earlier
tensor: another entry of this file, D's own earlier rows, or a tensor of an earlier file of the
same model (an external ref). It generalises mode 6: the predictor is a per-unit linear sum and
the contexts and weight coding are mode 6's.

**Units.** As mode 6: D holds U units of length H, row u (axis 0, D is [U, H]) or column u
(axis 1, D is [H, U]). Source s holds n_s units of length H: rows of a [n_s, H] tensor (axis 0)
or columns of a [H, n_s] tensor (axis 1). Its ref is a directory index earlier in payload order,
`entry_count + r` for external ref r (directory tail), or `0xFFFFFFFF` for D itself (self: axis
0 only, D's axis 0, n = U; unit u uses rows i < u only).

**Side stream** (one block in the prefix, its own static tables; units in order 0 .. U − 1):
- scale s_u + 128 (context 0, alphabet 256; unit RMS ≈ 2^(s_u/4));
- K_u, 0..8 (context 1, alphabet 9);
- per partner k < K_u: link = 2·src + same (context 2, alphabet 2·NS); same = 1 means index u
  (never for the self source); otherwise the index as `ceil(log2 n_src)` raw bits (< n_src;
  self: < u); then the coefficient q (|q| ≤ 32767) as the step code in context 3 + src.

**Predictor** (integers only): X = mode 6's fixed point of the partner value with shift
F = C − E_src; `p = Σ_k q_k · X_k` in int64. |X| ≤ 2^40, |q| < 2^15 and K ≤ 8, so |p| < 2^58: the
sum is exact, and any order or split (lanes, limbs, RNS) gives the same p. Thresholds, classes,
context z, the sign-agree bit, the plain symbol in (z, bit) and the 5 raw bits are exactly mode
6's (K_u = 0: p = 0).

**Block order.** Blocks of 2^18 weights in D's row-major order. With a self source (flag bit 0)
blocks decode in order, and each row's contexts are computed after the rows before it are
decoded; otherwise blocks are independent.

Prefix (after the frame): `u32 NS (1..64), u32 axis, u64 U, u64 H, i32 C, u32 block_log2 (18),
u64 block_count, u32 S (1..8), u32 width (1..16), i32 class_ref, u32 context_count (36·S),
u64 param_hash, u32 flags (bit 0: a self source), u32 0, u64 side_table_bytes,
u64 side_block_bytes`; NS × `{u32 ref, u32 axis, u64 n, i32 E, u32 0}`; the side tables, pad to
8; the side block (block layout above), pad to 8; context_count weight tables, pad to 8.
`param_hash` is the entry hash of the decoded side values: per unit u8 s_u, u8 K_u, then per
partner u8 src, u64 index, i16 q (little-endian); a mismatch is refused. The directory entry's
base hash is the entry hash of the NS refs (u32 each, in record order), and must match the
directory tail's list for the entry.

The encoder's search (which partners, K, E, C, classes) is not part of the format.

## Directory tail (version 4)

After the directory entries, inside `directory size`:
`u32 R`, R external refs `{u16 name length, name, u8 dtype, u8 ndim, u16 0, u64 shape[ndim],
u64 size, u64 hash}`; `u32 L`, L source lists `{u32 entry, u32 n, u32 source[n]}`, one for every
mode 7 entry (its refs, as in its payload); then zero bytes up to the directory size (the writer
reserves the space before it knows the lists). A decoder finds an external ref by name in the
earlier files it is given (`--refs`), checks dtype, size and hash, and holds it until the last
entry that names it is decoded. The reference encoder only refers to files that sort before the
file being written, so decoding shards in name order always has every ref.

## Version 1

Version 1 files (safetensors only, plain / delta / raw) put the directory at the end, have a
64-byte header (no source kind, digests or base fields) and payloads without the frame.
Readers still decode them by random access; they cannot be streamed.

## Not in version 4

F16/F32 entropy coding, chained deltas (a delta base must be a model file or a plain/raw/quant
.ingot entry), a row × column scale context for BF16 (measured: −0.003 to −0.025 bits/weight for
~30% slower decode and 8× the tables), the Q8_0 code comb (−0.01 bits/weight, needs per-weight
tables at decode), MoE expert down | own gate, up (measured +0.037 bits per expert down weight on
Qwen3.6), residual coding of strongly predicted weights (match contexts condition the symbol on
|p|; they do not code w − p), and the cross-layer covariance (U5) predictor (floating-point
eigen/Cholesky, ~2,560 MACs per weight).
