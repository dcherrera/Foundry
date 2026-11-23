# tokenizer

Fast text tokenization with BPE support. Python counterpart: HuggingFace Tokenizers.

**Header:** `tokenizer/tokenizer.h`
**Source:** `tokenizer/tokenizer.c`
**Status:** API Complete - Pending Verification

---

## Overview

The tokenizer module provides:
- BPE (Byte-Pair Encoding) tokenization
- HuggingFace JSON format loading
- Special token handling (BOS, EOS, PAD, UNK)
- Batch encoding/decoding

---

## Quick Start

```c
#include "tokenizer/tokenizer.h"

// Load tokenizer from HuggingFace format
ft_tokenizer* tok = ft_load("tokenizer.json");

// Encode text
int len;
int* ids = ft_encode(tok, "Hello, world!", &len);

// Decode back
char* text = ft_decode(tok, ids, len);
printf("%s\n", text);  // "Hello, world!"

// Cleanup
free(text);
free(ids);
ft_free(tok);
```

---

## API Reference

### Loading

```c
ft_tokenizer* ft_load(const char* path);                              // HuggingFace JSON
ft_tokenizer* ft_load_bpe(const char* vocab_path, const char* merges_path);  // BPE files
ft_tokenizer* ft_new(void);                                            // Empty tokenizer
void ft_free(ft_tokenizer* tok);
```

### Vocabulary

```c
int ft_add_token(ft_tokenizer* tok, const char* token);   // Returns token ID
void ft_add_merge(ft_tokenizer* tok, const char* a, const char* b);
int ft_vocab_size(ft_tokenizer* tok);
const char* ft_id_to_token(ft_tokenizer* tok, int id);
int ft_token_to_id(ft_tokenizer* tok, const char* token);
```

### Special Tokens

```c
void ft_set_unk_token(ft_tokenizer* tok, const char* token);
void ft_set_bos_token(ft_tokenizer* tok, const char* token);
void ft_set_eos_token(ft_tokenizer* tok, const char* token);
void ft_set_pad_token(ft_tokenizer* tok, const char* token);

int ft_unk_id(ft_tokenizer* tok);  // -1 if not set
int ft_bos_id(ft_tokenizer* tok);
int ft_eos_id(ft_tokenizer* tok);
int ft_pad_id(ft_tokenizer* tok);
```

### Encoding

```c
int* ft_encode(ft_tokenizer* tok, const char* text, int* length);
int* ft_encode_with_special(ft_tokenizer* tok, const char* text, int* length,
                            bool add_bos, bool add_eos);
int** ft_encode_batch(ft_tokenizer* tok, const char** texts, int n, int** lengths);
```

### Decoding

```c
char* ft_decode(ft_tokenizer* tok, int* ids, int length);
char* ft_decode_skip_special(ft_tokenizer* tok, int* ids, int length);
```

---

## See Also

- [data](data.md) - Data loading
- [safetensors](safetensors.md) - Model loading
