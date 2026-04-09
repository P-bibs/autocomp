#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root}/build/pytorch_ext"
jit_dir="${build_dir}/jit"
vendor_dir="${build_dir}/vendor"
log_dir="${root}/logs/pytorch_ext"
script="${1:-${root}/examples/pytorch/test.py}"
json_path="${2:-${log_dir}/run.jsonl}"

mkdir -p "${jit_dir}" "${log_dir}" "${build_dir}"
: > "${json_path}"

export PATH="${vendor_dir}/bin:${PATH}"
export PYTHONPATH="${vendor_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_EXTENSIONS_DIR="${jit_dir}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5+PTX}"
export CUDA_TRACE_FILTER_PATH_SUBSTR="${CUDA_TRACE_FILTER_PATH_SUBSTR:-${jit_dir}}"
export CUDA_TRACE_JSON_PATH="${CUDA_TRACE_JSON_PATH:-${json_path}}"
export LD_PRELOAD="${build_dir}/libcuda_trace_module_preload.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export MAX_JOBS="${MAX_JOBS:-1}"

exec python3 "${script}"
