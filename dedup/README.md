# Dedup

ASIC-accelerated training data deduplication using MinHash/LSH.

## Why This Module?

Every major LLM training pipeline deduplicates data:
- Removes exact duplicates (wasted compute)
- Removes near-duplicates (memorization, overfitting)
- Improves model quality
- Reduces training cost

Current approaches use GPU clusters. We use $30 ASIC miners.

## How MinHash Deduplication Works

```
Document: "the quick brown fox jumps over the lazy dog"
    │
    ▼
┌─────────────────────────────────────────┐
│  1. Shingling (create n-grams)          │
│     "the quick brown"                   │
│     "quick brown fox"                   │
│     "brown fox jumps"                   │
│     ... (k shingles)                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  2. MinHash (for each permutation i)    │  ← ASIC accelerates this
│     sig[i] = min(hash(shingle, i)       │
│              for shingle in shingles)   │
│                                         │
│     Result: 128-dimensional signature   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  3. LSH Bucketing                       │
│     Split signature into b bands        │
│     Hash each band → bucket             │
│     Documents in same bucket = candidates│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  4. Candidate Verification              │
│     For each candidate pair:            │
│     Jaccard = |A ∩ B| / |A ∪ B|        │
│     If Jaccard > threshold → duplicate  │
└─────────────────────────────────────────┘
```

## Why ASIC Helps

The bottleneck is step 2: computing MinHash signatures.

For each document:
- ~1000 shingles (13-grams)
- × 128 permutations
- = **128,000 hashes per document**

For Common Crawl (3B documents):
- 3B × 128K = **384 trillion hashes**

| Hardware | Hashes/sec | Time for 384T hashes |
|----------|------------|----------------------|
| CPU (1 core) | 10M | 1.2 years |
| CPU (100 cores) | 1B | 4.4 days |
| GPU cluster | 100B | 1 hour |
| ASIC array (100×) | 1T+ | 6 minutes |

## API Overview

### Types

```c
fd_shingle      // Single n-gram shingle
fd_signature    // MinHash signature (128+ dimensions)
fd_index        // LSH index for candidate detection
fd_pipeline     // Complete dedup pipeline
fd_result       // Deduplication results
fd_config       // Configuration
```

### Core Functions

```c
// Shingling
fd_shingle* fd_shingle_text(const char *text, int n, size_t *count);

// MinHash signature
fd_signature* fd_minhash(fd_shingle *shingles, size_t n, fd_config *config);

// LSH indexing
int fd_index_add(fd_index *idx, uint64_t doc_id, fd_signature *sig);
fd_candidates* fd_index_query(fd_index *idx, fd_signature *sig);

// Full pipeline
fd_pipeline* fd_pipeline_new(fd_config *config);
int fd_pipeline_add_file(fd_pipeline *pipe, const char *path);
fd_result* fd_pipeline_run(fd_pipeline *pipe);

// ASIC backend
void fd_pipeline_set_backend(fd_pipeline *pipe, asic_device *dev);
```

## Usage

### Basic Deduplication

```c
#include "dedup/dedup.h"
#include "asic.h"  // external ASIC interface, not in this tree

int main() {
    // Open ASIC (falls back to CPU if unavailable)
    asic_device *dev = asic_open_auto();

    // Configure pipeline
    fd_config config = fd_config_default();
    config.shingle_size = 13;           // 13-character n-grams
    config.num_permutations = 128;      // MinHash dimensions
    config.num_bands = 8;               // LSH bands
    config.similarity_threshold = 0.8;  // Jaccard threshold

    // Create pipeline
    fd_pipeline *pipe = fd_pipeline_new(&config);
    fd_pipeline_set_backend(pipe, dev);

    // Add input files (JSONL format)
    fd_pipeline_add_file(pipe, "data/crawl-001.jsonl");
    fd_pipeline_add_file(pipe, "data/crawl-002.jsonl");

    // Run deduplication
    fd_result *result = fd_pipeline_run(pipe);

    // Print results
    printf("Documents processed: %zu\n", result->total_docs);
    printf("Duplicates found:    %zu\n", result->duplicates);
    printf("Unique documents:    %zu\n", result->unique);
    printf("Total hashes:        %zu\n", result->hashes_computed);
    printf("Throughput:          %.2f docs/sec\n", result->docs_per_sec);
    printf("Hash throughput:     %.2f MH/sec\n", result->hashes_per_sec / 1e6);

    // Write deduplicated output
    fd_pipeline_write_output(pipe, "data/clean.jsonl");

    // Cleanup
    fd_result_free(result);
    fd_pipeline_free(pipe);
    asic_close(dev);

    return 0;
}
```

### Streaming Large Datasets

```c
// For datasets that don't fit in memory
fd_config config = fd_config_default();
config.streaming = true;
config.batch_size = 10000;  // Process 10K docs at a time
config.index_path = "/tmp/dedup-index";  // Memory-mapped index

fd_pipeline *pipe = fd_pipeline_new(&config);

// Add files (will be streamed, not loaded into memory)
for (int i = 0; i < num_files; i++) {
    fd_pipeline_add_file(pipe, files[i]);
}

// Progress callback
fd_pipeline_on_progress(pipe, progress_callback, NULL);

// Run (streams through data)
fd_result *result = fd_pipeline_run(pipe);
```

## Input Format

JSONL with `text` field:

```json
{"id": "doc1", "text": "The quick brown fox..."}
{"id": "doc2", "text": "Another document here..."}
```

Or specify custom field:

```c
config.text_field = "content";  // Use "content" instead of "text"
```

## Output Format

JSONL with duplicate annotations:

```json
{"id": "doc1", "text": "...", "duplicate_of": null}
{"id": "doc2", "text": "...", "duplicate_of": "doc1"}
```

Or filtered (only unique documents):

```c
config.output_mode = FD_OUTPUT_UNIQUE_ONLY;
```

## Performance Targets

| Dataset Size | CPU Time | ASIC Time | Speedup |
|--------------|----------|-----------|---------|
| 1M docs | 10 min | 10 sec | 60x |
| 100M docs | 16 hours | 15 min | 64x |
| 1B docs | 7 days | 2.5 hours | 67x |

## Comparison with Existing Tools

| Tool | Language | ASIC Support | Streaming |
|------|----------|--------------|-----------|
| datasketch | Python | No | No |
| text-dedup | Python | No | Partial |
| **fd_dedup** | C | **Yes** | Yes |

## File Format

### Index Format (`.fdi`)

```
[Header: 16 bytes]
  magic: "FDI1"
  version: uint16
  num_docs: uint64
  config_hash: uint32

[Config: variable]
  shingle_size: uint8
  num_permutations: uint16
  num_bands: uint8
  band_size: uint8

[Signatures: num_docs × sig_size]
  For each doc:
    doc_id: uint64
    signature: uint32[num_permutations]

[LSH Buckets: num_bands × variable]
  For each band:
    num_buckets: uint32
    For each bucket:
      bucket_hash: uint32
      doc_ids: uint64[]
```

## Dependencies

- An external ASIC device abstraction (not in this tree)
- Standard C library only
