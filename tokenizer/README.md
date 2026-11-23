# tokenizer/

C implementation of fast text tokenization.

## Python counterpart

**tokenizers** - https://github.com/huggingface/tokenizers

## Why tokenizers?

HuggingFace tokenizers is the standard for fast tokenization. Written in Rust with Python bindings. Supports:

- BPE (Byte-Pair Encoding) - Used by GPT, LLaMA
- WordPiece - Used by BERT
- Unigram - Used by T5, ALBERT
- SentencePiece compatibility

Alternatives (SentencePiece, tiktoken) implement specific algorithms. tokenizers is the most general.

## Scope

This module implements:

1. **BPE tokenizer** - Most common for modern LLMs
2. **Vocabulary loading** - From JSON vocab files
3. **Special tokens** - PAD, EOS, BOS, UNK handling
4. **Batch tokenization** - Parallel processing
5. **Decoding** - Tokens back to text

## Getting Started

### Dependencies

- None (standalone module)

### Key Files to Study

From tokenizers:
- `tokenizers/src/models/bpe/` - BPE implementation
- `tokenizers/src/tokenizer/` - Core tokenizer

From SentencePiece:
- `src/bpe_model.cc` - BPE in C++

From tiktoken:
- `tiktoken/_tiktoken.pyi` - Simple BPE

### Core API

```c
// Load tokenizer from file
ft_tokenizer* ft_load(const char* path);
ft_tokenizer* ft_load_bpe(const char* vocab_path, const char* merges_path);

// Tokenize text
int* ft_encode(ft_tokenizer* tok, const char* text, int* length);
int** ft_encode_batch(ft_tokenizer* tok, const char** texts, int n, int** lengths);

// Decode tokens
char* ft_decode(ft_tokenizer* tok, int* ids, int length);

// Special tokens
int ft_pad_id(ft_tokenizer* tok);
int ft_eos_id(ft_tokenizer* tok);
int ft_bos_id(ft_tokenizer* tok);

// Vocabulary
int ft_vocab_size(ft_tokenizer* tok);
const char* ft_id_to_token(ft_tokenizer* tok, int id);
int ft_token_to_id(ft_tokenizer* tok, const char* token);
```

## Implementation Order

1. Vocabulary loading (JSON)
2. Simple tokenization (whitespace + vocab lookup)
3. BPE merge algorithm
4. Special token handling
5. Batch tokenization
6. Decoding

## References

- [BPE Paper](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://github.com/google/sentencepiece)
- [tiktoken](https://github.com/openai/tiktoken)
- [minbpe](https://github.com/karpathy/minbpe) - Minimal BPE implementation
