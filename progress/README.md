# progress/

C implementation of progress reporting and training metrics.

## Python counterpart

**tqdm** - https://github.com/tqdm/tqdm

## Why tqdm?

tqdm provides progress bars for Python loops. Simple but essential for:

- Training progress visualization
- ETA estimation
- Throughput metrics (samples/sec, tokens/sec)
- Nested progress (epochs > batches)

## Scope

This module implements:

1. **Progress bar** - Visual progress indicator
2. **ETA estimation** - Time remaining calculation
3. **Rate tracking** - Throughput metrics
4. **Training metrics** - Loss, learning rate, etc.
5. **Logging** - Structured output for training runs

## Getting Started

### Dependencies

- None (standalone module)

### Key Files to Study

From tqdm:
- `tqdm/std.py` - Core implementation

Simple enough to implement from scratch.

### Core API

```c
// Create progress bar
fp_progress* fp_new(int64_t total, const char* desc);

// Update progress
void fp_update(fp_progress* p, int64_t n);
void fp_set_postfix(fp_progress* p, const char* fmt, ...);

// Close (print final state)
void fp_close(fp_progress* p);

// Training-specific
typedef struct {
    int epoch;
    int step;
    float loss;
    float lr;
    float throughput;  // samples/sec or tokens/sec
} fp_train_state;

void fp_log_step(fp_train_state* state);
void fp_log_epoch(fp_train_state* state);

// Nested progress
fp_progress* fp_new_nested(fp_progress* parent, int64_t total, const char* desc);

// Simple iteration
#define FP_FOREACH(p, i, n) \
    for (fp_progress* _p = fp_new(n, NULL); \
         (i) < (n) && (fp_update(_p, 1), 1); \
         (i)++)
```

## Output Format

```
Training: 100%|██████████| 1000/1000 [00:45<00:00, 22.1 step/s, loss=0.234, lr=1e-5]
```

Components:
- Description
- Percentage
- Progress bar (unicode blocks)
- Current/total
- Elapsed/remaining time
- Rate
- Custom postfix metrics

## Implementation Order

1. Basic progress bar
2. Rate calculation
3. ETA estimation
4. Postfix metrics
5. Nested progress
6. Training-specific helpers

## Implementation Notes

Terminal handling:
```c
// Carriage return to overwrite line
printf("\r");

// Clear to end of line
printf("\033[K");

// Get terminal width
struct winsize w;
ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
int width = w.ws_col;
```

Unicode progress blocks:
```c
const char* blocks[] = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
```

## References

- [tqdm](https://github.com/tqdm/tqdm)
- [indicatif](https://github.com/console-rs/indicatif) - Rust progress bars
