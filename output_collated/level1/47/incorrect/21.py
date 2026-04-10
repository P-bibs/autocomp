# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_19.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




import torch
from torch.utils.cpp_extension import load_inline

# CUDA kernel for optimized sum reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int B, const int D1, const int D2,
    const int reduce_dim
) {
    if (reduce_dim == 1) {
        // Reduce along dim1: (B, D1, D2) -> (B, 1, D2)
        // threadIdx.x handles D2, blockIdx.x handles B
        int b = blockIdx.x;
        int d2 = threadIdx.x;
        if (b < B && d2 < D2) {
            float sum = 0.0f;
            for (int d1 = 0; d1 < D1; ++d1) {
                sum += input[b * D1 * D2 + d1 * D2 + d2];
            }
            output[b * D2 + d2] = sum;
        }
    } else {
        // Reduce along dim2: (B, D1, D2) -> (B, D1, 1)
        // Each block reduces one row of D2
        extern __shared__ float sdata[];
        int b = blockIdx.y;
        int d1 = blockIdx.x;
        int tid = threadIdx.x;
        
        float sum = 0.0f;
        for (int d2 = tid; d2 < D2; d2 += blockDim.x) {
            sum += input[b * D1 * D2 + d1 * D2 + d2];
        }
        sdata[tid] = sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        
        if (tid == 0) {
            output[b * D1 + d1] = sdata[0];
        }
    }
}

void sum_reduce(const at::Tensor input, at::Tensor output, int reduce_dim) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    const float* d_in = input.data_ptr<float>();
    float* d_out = output.data_ptr<float>();

    if (reduce_dim == 1) {
        dim3 grid(B);
        dim3 block(std::min(D2, 1024));
        sum_reduce_kernel<<<grid, block>>>(d_in, d_out, B, D1, D2, 1);
    } else {
        dim3 grid(D1, B);
        int threads = 256; 
        sum_reduce_kernel<<<grid, threads, threads * sizeof(float)>>>(d_in, d_out, B, D1, D2, 2);
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void sum_reduce(const at::Tensor input, at::Tensor output, int reduce_dim);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduce", &sum_reduce, "Custom sum reduction");
}
"""

fused_ext = load_inline(
    name='fused_sum_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, dim):
    B, D1, D2 = x.shape
    if dim == 1:
        out = torch.empty((B, 1, D2), device=x.device, dtype=x.dtype)
    else:
        out = torch.empty((B, D1, 1), device=x.device, dtype=x.dtype)
    
    fused_ext.sum_reduce(x, out, dim)
    return out

# Global vars for compatibility
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    return [torch.randn(batch_size, dim1, dim2, device='cuda')]
