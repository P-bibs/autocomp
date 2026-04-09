#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 /path/to/program [args...]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "${script_dir}/.." && pwd)"
preload="${root}/build/cuda/libcuda_trace_preload.so"

if [[ ! -f "${preload}" ]]; then
    echo "missing ${preload}; build the preload library first" >&2
    exit 1
fi

if [[ -n "${LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="${preload}:${LD_PRELOAD}"
else
    export LD_PRELOAD="${preload}"
fi

exec "$@"
