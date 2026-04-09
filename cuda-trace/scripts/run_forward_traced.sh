#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root}/build/pytorch_ext"
vendor_dir="${build_dir}/vendor"
log_dir="${root}/logs/forward_trace"
tracer_path="${build_dir}/libcuda_trace_module_preload.so"
cli_path="${root}/src/pytorch/forward_pipeline_trace.py"
input_py="${1:-${root}/examples/pytorch/pipeline_single.py}"
io_file="${3:-}"
input_stem="$(basename "${input_py}")"
input_stem="${input_stem%.*}"
input_stem="$(printf '%s' "${input_stem}" | tr -c '[:alnum:]_-' '_')"
input_hash="$(sha256sum "${input_py}" | awk '{print substr($1, 1, 12)}')"
run_name="${input_stem}_${input_hash}"
jit_dir_default="${build_dir}/jit/${run_name}"
jit_dir="${TORCH_EXTENSIONS_DIR:-${jit_dir_default}}"
output_json="${2:-${log_dir}/${run_name}.json}"

mkdir -p "${jit_dir}" "${log_dir}" "${build_dir}"

if [ ! -x "${vendor_dir}/bin/ninja" ]; then
    python3 -m pip install --target "${vendor_dir}" ninja
fi

if [ ! -f "${tracer_path}" ]; then
    make -C "${root}" pytorch-tracer
fi

export PATH="${vendor_dir}/bin:${PATH}"
export PYTHONPATH="${vendor_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export FORWARD_TRACE_TRACER_PATH="${FORWARD_TRACE_TRACER_PATH:-${tracer_path}}"
export TORCH_EXTENSIONS_DIR="${jit_dir}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5+PTX}"

argv=(python3 "${cli_path}" "${input_py}" "${output_json}")
if [ -n "${io_file}" ]; then
    argv+=("--io-file" "${io_file}")
fi

exec "${argv[@]}"
