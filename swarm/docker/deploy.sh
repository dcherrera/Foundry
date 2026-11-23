#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# Foundry Swarm Deployment Script
#
# This script helps deploy the training swarm across heterogeneous hardware.
#
# Usage:
#   ./deploy.sh init           # Initialize swarm on this node (manager)
#   ./deploy.sh join <token>   # Join an existing swarm
#   ./deploy.sh build          # Build Docker images
#   ./deploy.sh push           # Push images to registry
#   ./deploy.sh deploy         # Deploy the stack
#   ./deploy.sh status         # Show swarm status
#   ./deploy.sh logs           # Show coordinator logs
#   ./deploy.sh stop           # Stop the stack

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOUNDRY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
REGISTRY="${REGISTRY:-}"
STACK_NAME="foundry"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Initialize Docker Swarm
cmd_init() {
    log_info "Initializing Docker Swarm..."

    if docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_warn "Swarm already initialized"
        docker swarm join-token worker
        return
    fi

    docker swarm init

    log_info "Swarm initialized! To add workers, run on each node:"
    docker swarm join-token worker

    log_info "Labeling this node as 'server'..."
    NODE_ID=$(docker node ls --format '{{.ID}}' | head -1)
    docker node update --label-add hw=server "$NODE_ID"
}

# Join existing swarm
cmd_join() {
    local TOKEN="$1"
    local MANAGER="$2"

    if [ -z "$TOKEN" ] || [ -z "$MANAGER" ]; then
        log_error "Usage: $0 join <token> <manager-ip:port>"
        exit 1
    fi

    log_info "Joining swarm..."
    docker swarm join --token "$TOKEN" "$MANAGER"
}

# Build Docker images
cmd_build() {
    log_info "Building coordinator image..."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.coordinator" \
        -t foundry/swarm-coordinator:$IMAGE_TAG \
        "$FOUNDRY_ROOT"

    log_info "Building worker image (amd64)..."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.worker" \
        -t foundry/swarm-worker:$IMAGE_TAG \
        "$FOUNDRY_ROOT"

    # Build ARM images for Raspberry Pi
    if command -v docker buildx &> /dev/null; then
        log_info "Building worker image (arm64 for Raspberry Pi)..."
        docker buildx build \
            --platform linux/arm64 \
            -f "$SCRIPT_DIR/Dockerfile.worker" \
            -t foundry/swarm-worker:arm64 \
            "$FOUNDRY_ROOT"
    else
        log_warn "docker buildx not available, skipping ARM build"
        log_warn "Install buildx for Raspberry Pi support"
    fi

    log_info "Build complete!"
}

# Push images to registry
cmd_push() {
    if [ -z "$REGISTRY" ]; then
        log_error "Set REGISTRY environment variable first"
        log_error "Example: REGISTRY=myregistry.local:5000 $0 push"
        exit 1
    fi

    log_info "Pushing images to $REGISTRY..."

    docker tag foundry/swarm-coordinator:$IMAGE_TAG "$REGISTRY/swarm-coordinator:$IMAGE_TAG"
    docker push "$REGISTRY/swarm-coordinator:$IMAGE_TAG"

    docker tag foundry/swarm-worker:$IMAGE_TAG "$REGISTRY/swarm-worker:$IMAGE_TAG"
    docker push "$REGISTRY/swarm-worker:$IMAGE_TAG"

    if docker image inspect foundry/swarm-worker:arm64 &>/dev/null; then
        docker tag foundry/swarm-worker:arm64 "$REGISTRY/swarm-worker:arm64"
        docker push "$REGISTRY/swarm-worker:arm64"
    fi

    log_info "Push complete!"
}

# Deploy the stack
cmd_deploy() {
    log_info "Deploying Foundry Swarm stack..."

    # Check if swarm is initialized
    if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
        log_error "Docker Swarm not initialized. Run '$0 init' first."
        exit 1
    fi

    # Deploy
    docker stack deploy -c "$SCRIPT_DIR/docker-stack.yml" "$STACK_NAME"

    log_info "Stack deployed! Check status with:"
    echo "  docker service ls"
    echo "  docker service ps ${STACK_NAME}_coordinator"
}

# Show status
cmd_status() {
    log_info "Swarm Status:"
    echo ""

    echo "=== Nodes ==="
    docker node ls
    echo ""

    echo "=== Services ==="
    docker service ls --filter "name=$STACK_NAME"
    echo ""

    echo "=== Coordinator ==="
    docker service ps "${STACK_NAME}_coordinator" 2>/dev/null || echo "Not deployed"
    echo ""

    echo "=== Workers ==="
    for svc in worker-server worker-desktop worker-laptop worker-rpi; do
        count=$(docker service ls --filter "name=${STACK_NAME}_${svc}" --format '{{.Replicas}}' 2>/dev/null || echo "0/0")
        echo "  $svc: $count"
    done
}

# Show logs
cmd_logs() {
    local SERVICE="${1:-coordinator}"
    docker service logs -f "${STACK_NAME}_${SERVICE}"
}

# Stop the stack
cmd_stop() {
    log_info "Stopping Foundry Swarm stack..."
    docker stack rm "$STACK_NAME"
    log_info "Stack removed"
}

# Label a node by hardware type
cmd_label() {
    local NODE="$1"
    local HW_TYPE="$2"

    if [ -z "$NODE" ] || [ -z "$HW_TYPE" ]; then
        log_error "Usage: $0 label <node-id> <hw-type>"
        log_error "hw-type: server, desktop, laptop, rpi"
        exit 1
    fi

    docker node update --label-add "hw=$HW_TYPE" "$NODE"
    log_info "Labeled node $NODE as $HW_TYPE"
}

# Scale workers
cmd_scale() {
    local SERVICE="$1"
    local COUNT="$2"

    if [ -z "$SERVICE" ] || [ -z "$COUNT" ]; then
        log_error "Usage: $0 scale <service> <count>"
        log_error "Services: worker-server, worker-desktop, worker-laptop, worker-rpi"
        exit 1
    fi

    docker service scale "${STACK_NAME}_${SERVICE}=${COUNT}"
}

# Main
case "${1:-help}" in
    init)
        cmd_init
        ;;
    join)
        cmd_join "$2" "$3"
        ;;
    build)
        cmd_build
        ;;
    push)
        cmd_push
        ;;
    deploy)
        cmd_deploy
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs "$2"
        ;;
    stop)
        cmd_stop
        ;;
    label)
        cmd_label "$2" "$3"
        ;;
    scale)
        cmd_scale "$2" "$3"
        ;;
    help|*)
        echo "Foundry Swarm Deployment"
        echo ""
        echo "Usage: $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  init              Initialize Docker Swarm on this node"
        echo "  join <token> <ip> Join an existing swarm"
        echo "  build             Build Docker images"
        echo "  push              Push images to registry (set REGISTRY env)"
        echo "  deploy            Deploy the training stack"
        echo "  status            Show swarm and service status"
        echo "  logs [service]    Show service logs (default: coordinator)"
        echo "  stop              Stop and remove the stack"
        echo "  label <node> <hw> Label a node by hardware type"
        echo "  scale <svc> <n>   Scale a worker service"
        echo ""
        echo "Examples:"
        echo "  $0 init                           # Initialize swarm"
        echo "  $0 label abc123 desktop           # Label node as desktop"
        echo "  $0 build                          # Build images"
        echo "  $0 deploy                         # Deploy stack"
        echo "  $0 scale worker-rpi 20            # Scale RPi workers to 20"
        ;;
esac
