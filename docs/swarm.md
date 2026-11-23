# swarm

Docker Swarm distributed training for heterogeneous CPU clusters.

**Directory:** `swarm/`
**Status:** Complete with 14 passing tests

---

## Overview

The swarm module provides distributed training across heterogeneous hardware using a parameter server architecture:

- **Coordinator:** Holds master weights, aggregates gradients, distributes updates
- **Workers:** Compute forward/backward passes, send compressed gradients

Key features:
- Async gradient updates (workers don't block each other)
- TopK gradient compression (90% bandwidth reduction)
- FP16 compression option
- Hardware profile auto-detection
- Fault tolerance (workers can drop and rejoin)
- Docker Swarm deployment support

---

## Architecture

```
                    ┌─────────────────┐
                    │   Coordinator   │
                    │  (holds weights)│
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │ Worker  │        │ Worker  │        │ Worker  │
    │  (Pi 4) │        │ (Laptop)│        │ (Server)│
    └─────────┘        └─────────┘        └─────────┘
```

---

## Quick Start

### Running Locally

```bash
# Terminal 1: Start coordinator
./bin/swarm_coordinator --port 7100

# Terminal 2: Start worker
./bin/swarm_worker --host localhost --port 7100
```

### Docker Swarm Deployment

```bash
cd swarm/docker
./deploy.sh
```

---

## Files

| File | Description |
|------|-------------|
| `protocol.h/c` | Wire protocol, compression, message handling |
| `network.h/c` | TCP socket abstraction |
| `config.h/c` | TOML-like config file parsing |
| `coordinator.h/c` | Parameter server |
| `worker.h/c` | Training worker |
| `coordinator_main.c` | Coordinator CLI entry point |
| `worker_main.c` | Worker CLI entry point |

---

## Protocol

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `REGISTER` | Worker → Coord | Worker registration |
| `REGISTER_ACK` | Coord → Worker | Registration confirmed |
| `WEIGHTS` | Coord → Worker | Full weight tensor |
| `GRADIENTS` | Worker → Coord | Compressed gradients |
| `UPDATE` | Coord → Worker | Weight update delta |
| `START` | Coord → Worker | Start training |
| `PAUSE/RESUME` | Coord → Worker | Training control |
| `SHUTDOWN` | Coord → Worker | Graceful shutdown |
| `HEARTBEAT` | Both | Connection keepalive |
| `CHECKPOINT` | Coord → Worker | Save checkpoint |

### Compression

| Type | Ratio | Description |
|------|-------|-------------|
| `NONE` | 1:1 | No compression |
| `FP16` | 2:1 | Float16 conversion |
| `TOPK` | 10:1 | Top-K by magnitude |
| `QSGD` | Variable | Quantized SGD |

---

## API Reference

### Coordinator

```c
typedef struct {
    uint16_t port;
    float learning_rate;
    float beta1, beta2;
    float epsilon;
    int min_workers;
    int checkpoint_interval;
    char checkpoint_dir[SW_MAX_PATH];
    sw_compress_type compression;
    float topk_ratio;
} sw_coordinator_config;

sw_coordinator* sw_coordinator_new(tm_tensor** weights, int num_weights,
                                    sw_coordinator_config* config);
void sw_coordinator_free(sw_coordinator* coord);

bool sw_coordinator_start(sw_coordinator* coord);
void sw_coordinator_stop(sw_coordinator* coord);
bool sw_coordinator_wait_for_workers(sw_coordinator* coord, int timeout_sec);

int sw_coordinator_num_workers(sw_coordinator* coord);
int sw_coordinator_get_step(sw_coordinator* coord);
float sw_coordinator_avg_loss(sw_coordinator* coord);
```

### Worker

```c
typedef struct {
    char coordinator_host[256];
    uint16_t coordinator_port;
    int reconnect_attempts;
    int reconnect_delay_sec;
    int num_threads;
    int batch_size;
    char data_path[SW_MAX_PATH];
    int shard_index;
    sw_compress_type compression;
    float topk_ratio;
    char worker_id[SW_MAX_WORKER_ID];
} sw_worker_config;

sw_worker* sw_worker_new(sw_worker_config* config);
void sw_worker_free(sw_worker* w);

bool sw_worker_connect(sw_worker* w);
void sw_worker_disconnect(sw_worker* w);

void sw_worker_set_weights(sw_worker* w, tm_tensor** weights, int num_weights);
void sw_worker_set_forward(sw_worker* w, sw_forward_fn fn, void* ctx);
void sw_worker_set_backward(sw_worker* w, sw_backward_fn fn, void* ctx);
void sw_worker_set_data_loader(sw_worker* w, sw_next_batch_fn next,
                                sw_reset_data_fn reset, sw_free_batch_fn free_batch,
                                void* ctx);

bool sw_worker_fetch_weights(sw_worker* w);
void sw_worker_train(sw_worker* w);  // Blocking
void sw_worker_stop(sw_worker* w);
```

### Training Callbacks

```c
// Forward: compute loss from batch
typedef float (*sw_forward_fn)(tm_tensor** weights, int num_weights,
                                void* batch_data, void* ctx);

// Backward: compute gradients
typedef void (*sw_backward_fn)(tm_tensor** weights, int num_weights,
                                void* batch_data, void* ctx);

// Data loading
typedef void* (*sw_next_batch_fn)(void* ctx);
typedef void (*sw_reset_data_fn)(void* ctx);
typedef void (*sw_free_batch_fn)(void* batch, void* ctx);
```

---

## Configuration Files

### coordinator.toml
```toml
[network]
port = 7100

[training]
learning_rate = 1e-4
beta1 = 0.9
beta2 = 0.999
min_workers = 2

[checkpointing]
interval = 1000
directory = "/checkpoints"

[compression]
type = "topk"
topk_ratio = 0.1
```

### worker.toml
```toml
[network]
coordinator_host = "coordinator"
coordinator_port = 7100
reconnect_attempts = 10
reconnect_delay_sec = 5

[hardware]
num_threads = 0  # auto
max_memory_mb = 0  # auto

[training]
batch_size = 0  # auto from profile
accumulation_steps = 0  # auto

[data]
path = "/data/train.jsonl"
shard_index = -1  # auto-assign

[compression]
type = "topk"
topk_ratio = 0.1
```

---

## Hardware Profiles

Auto-detected profiles with recommended batch sizes:

| Profile | Memory | Batch Size | Description |
|---------|--------|------------|-------------|
| `RPI3` | <1GB | 2 | Raspberry Pi 3 |
| `RPI4` | 1-4GB | 4-8 | Raspberry Pi 4 |
| `RPI5` | 4-8GB | 8-16 | Raspberry Pi 5 |
| `LAPTOP` | 8-16GB | 16-32 | Standard laptop |
| `DESKTOP` | 16-32GB | 32-64 | Desktop PC |
| `WORKSTATION` | 32-64GB | 64-128 | Workstation |
| `SERVER` | >64GB | 128-256 | Server |

---

## Docker Deployment

See `swarm/docker/` for:
- `Dockerfile.coordinator`
- `Dockerfile.worker`
- `docker-compose.yml` (local testing)
- `docker-stack.yml` (Swarm deployment)
- `deploy.sh` (deployment script)

---

## See Also

- [distributed](distributed.md) - Local distributed primitives
- [tensor](tensor.md) - Tensor operations
- [optim](optim.md) - Optimizers
