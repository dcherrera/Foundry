# swarm/

**Custom Extension** - Distributed CPU training across Docker Swarm.

This is a project-specific addition to Foundry, not part of the core toolset. It enables training models across heterogeneous hardware (Raspberry Pis, laptops, desktops, servers) connected via WiFi.

## Purpose

Train a custom 1B parameter RWKV-x model using:
- ~20 Raspberry Pi 4s
- 4-10 laptops
- ~10 desktops (3rd-5th gen Intel i-series)
- A few old servers

All connected via WiFi/Starlink with Docker Swarm orchestration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DOCKER SWARM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────┐                                       │
│   │    COORDINATOR      │  (1 instance, on server)              │
│   │  - Holds weights    │                                       │
│   │  - Aggregates grads │                                       │
│   │  - Checkpoints      │                                       │
│   └──────────┬──────────┘                                       │
│              │                                                  │
│              │ TCP (port 7100)                                  │
│              │                                                  │
│   ┌──────────┴──────────────────────────────────────────┐       │
│   │                                                     │       │
│   ▼                    ▼                    ▼           ▼       │
│ ┌──────────┐     ┌──────────┐     ┌──────────┐   ┌──────────┐   │
│ │ WORKER   │     │ WORKER   │     │ WORKER   │   │ WORKER   │   │
│ │ server   │     │ desktop  │     │ laptop   │   │ rpi4     │   │
│ │ batch=128│     │ batch=64 │     │ batch=32 │   │ batch=8  │   │
│ └──────────┘     └──────────┘     └──────────┘   └──────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Async gradient updates** - Don't wait for slow nodes (critical for WiFi)
- **Gradient compression** - TopK/FP16/QSGD to reduce bandwidth
- **Auto hardware detection** - Adjusts batch size per device capability
- **Fault tolerance** - Workers can drop and rejoin without killing training
- **LoRA training mode** - Train specialized adapters with minimal bandwidth

## Files

| File | Purpose |
|------|---------|
| `protocol.h/.c` | Message types, serialization, compression |
| `network.h/.c` | TCP socket communication layer |
| `coordinator.h/.c` | Central training server |
| `worker.h/.c` | Training worker client |
| `docker/` | Swarm deployment configs |

## Dependencies

This module builds on:
- `tensor/` - Tensor operations and autograd
- `optim/` - Adam optimizer, LR scheduling
- `distributed/` - Gradient accumulation primitives

## Usage

### Local Development

```bash
# Build images
cd swarm/docker
./deploy.sh build

# Start with docker-compose (2 workers by default)
docker-compose up -d

# Scale to more workers
docker-compose up -d --scale worker=5
```

### Production Swarm Deployment

```bash
# From swarm/docker/
# On manager node - initialize swarm
./deploy.sh init

# On worker nodes - join swarm
./deploy.sh join <token> <manager-ip>:2377

# Label nodes by hardware type
./deploy.sh label <node-id> server
./deploy.sh label <node-id> desktop
./deploy.sh label <node-id> laptop
./deploy.sh label <node-id> rpi

# Build and push images
./deploy.sh build
REGISTRY=myregistry:5000 ./deploy.sh push

# Deploy the stack
./deploy.sh deploy

# Monitor
./deploy.sh status
./deploy.sh logs coordinator

# Scale workers
./deploy.sh scale worker-rpi 20
```

### Manual Binary Usage

```bash
# Build: make swarm-bin
# On coordinator node (server)
./bin/swarm_coordinator --config swarm/docker/config/coordinator.toml

# On worker nodes
./bin/swarm_worker --config swarm/docker/config/worker.toml
```

## Integration with RWKV-x

This module is designed to be pulled into the RWKV-x repo as a git submodule:

```
rwkv-x/
├── model/           # RWKV-x architecture
├── data/            # Training data
└── foundry/         # ← Git submodule
    ├── tensor/
    ├── optim/
    ├── swarm/       # ← This module
    └── ...
```

## See Also

- `distributed/` - Low-level primitives (all-reduce, gradient accumulation)
- `optim/` - Optimizers and LR scheduling
