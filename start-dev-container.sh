#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-autocomp-devtools}"
CONTAINER_NAME="${CONTAINER_NAME:-autocomp-devtools}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-$SCRIPT_DIR/Dockerfile.devtools}"
HOST_WORKSPACE="${HOST_WORKSPACE:-$PWD}"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace}"
SHELL_BIN="${SHELL_BIN:-/bin/bash}"
DOCKER_RUN_FLAGS=(--rm)

if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_RUN_FLAGS+=(-it)
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required but was not found in PATH" >&2
    exit 1
fi

if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "missing Dockerfile: $DOCKERFILE_PATH" >&2
    exit 1
fi

if [ ! -d "$HOST_WORKSPACE" ]; then
    echo "host workspace does not exist: $HOST_WORKSPACE" >&2
    exit 1
fi

echo "Building image $IMAGE_NAME from $DOCKERFILE_PATH"
docker build \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    -t "$IMAGE_NAME" \
    -f "$DOCKERFILE_PATH" \
    "$SCRIPT_DIR"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting container $CONTAINER_NAME with $HOST_WORKSPACE mounted at $CONTAINER_WORKSPACE"
docker run "${DOCKER_RUN_FLAGS[@]}" \
    --name "$CONTAINER_NAME" \
    --hostname "$CONTAINER_NAME" \
    --cap-add SYS_PTRACE \
    --cap-add PERFMON \
    --security-opt seccomp=unconfined \
    -v "$HOST_WORKSPACE:$CONTAINER_WORKSPACE" \
    -w "$CONTAINER_WORKSPACE" \
    "$IMAGE_NAME" \
    "$SHELL_BIN"
