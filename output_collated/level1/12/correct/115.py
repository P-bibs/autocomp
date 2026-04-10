# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, *args) -> torch.Tensor:
        return functional_model(*args, **extract_state_kwargs(self))


def build_reference_model():
    init_inputs = list(get_init_inputs())
    model = ModelNew(*init_inputs)
    model.eval()
    return model


def extract_state_kwargs(model):
    flat_state = {}
    for name, value in model.named_parameters():
        flat_state[name.replace('.', '_')] = value
    for name, value in model.named_buffers():
        flat_state[name.replace('.', '_')] = value
    state_kwargs = {}
    init_inputs = list(get_init_inputs())
    init_arg_map = {name: value for name, value in zip(INIT_PARAM_NAMES, init_inputs)}
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
#  Custom CUDA kernel for broadcast multiplication (high performance)
# ----------------------------------------------------------------------

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Vectorized kernel using float4 for maximum memory bandwidth utilization
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    int m_idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    int n = blockIdx.y;

    if (n < N && m_idx < M) {
        float a_val = A[n];
        
        // Load, Multiply, Store using float4 for coalesced 128-bit memory access
        float4 b_val = reinterpret_cast<const float4*>(&B[n * M + m_idx])[0];
        
        float4 out_val;
        out_val.x = a_val * b_val.x;
        out_val.y = a_val * b_val.y;
        out_val.z = a_val * b_val.z;
        out_val.w = a_val * b_val.w;
        
        reinterpret_cast<float4*>(&output[n * M + m_idx])[0] = out_val;
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // 256 threads per block (64 float4 segments)
    const int threads = 256;
    const int elements_per_thread = 4;
    
    dim3 grid((M / elements_per_thread + threads - 1) / threads, N);
    
    broadcast_mul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute ``output[i, j] = A[i] * B[i, j]`` using a custom CUDA kernel
    optimized for memory bandwidth.

    Parameters
    ----------
    A : torch.Tensor
        1-D tensor of shape (N,). Must be on CUDA device.
    B : torch.Tensor
        2-D tensor of shape (N, M). Must be on CUDA device.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, M) with the broadcasted multiplication result.
    """
    # Ensure both tensors are on CUDA
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Create output tensor
    output = torch.empty_like(B)

    # Launch custom CUDA kernel
    fused_ext.broadcast_mul(A, B, output)

    return output

# ----------------------------------------------------------------------
#  Helper used by the test harness
# ----------------------------------------------------------------------
def get_inputs():
    """
    Returns a pair of tensors that match the shapes used in the original
    benchmark: ``A`` is (N,) and ``B`` is (N, M). Tensors are placed on CUDA.
    """
    N, M = 4096, 4096
    A = torch.rand(N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, M, device="cuda", dtype=torch.float32)
    return A, B

# ----------------------------------------------------------------------
#  Mini-benchmark
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time

    A, B = get_inputs()

    # Warm-up
    _ = functional_model(A, B)

    torch.cuda.synchronize()
    start = time.time()
    out = functional_model(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Output shape: {out.shape}")
    print(f"Latency (custom CUDA kernel): {elapsed:.6f} seconds")
