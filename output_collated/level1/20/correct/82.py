# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_213956/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['negative_slope']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['negative_slope']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float=0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

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
    if 'negative_slope' in flat_state:
        state_kwargs['negative_slope'] = flat_state['negative_slope']
    else:
        state_kwargs['negative_slope'] = getattr(model, 'negative_slope')
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

# Constants
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

# Optimized CUDA kernel with increased vectorization and loop unrolling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vectorized_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t n) 
{
    // Each thread processes 8 elements (2 float4 vectors) for better ILP
    const size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx  = id * 8;

    if (idx + 7 < n) {
        // Load two float4 vectors (8 elements total)
        const float4 in_vec0 = __ldg(reinterpret_cast<const float4*>(input) + id * 2);
        const float4 in_vec1 = __ldg(reinterpret_cast<const float4*>(input) + id * 2 + 1);

        // Branch-less Leaky ReLU: out = fmax(x,0) + slope * fmin(x,0)
        // Unrolled computation with explicit statements for max ILP
        float4 out_vec0;
        out_vec0.x = fmaf(negative_slope, fminf(in_vec0.x, 0.0f), fmaxf(in_vec0.x, 0.0f));
        out_vec0.y = fmaf(negative_slope, fminf(in_vec0.y, 0.0f), fmaxf(in_vec0.y, 0.0f));
        out_vec0.z = fmaf(negative_slope, fminf(in_vec0.z, 0.0f), fmaxf(in_vec0.z, 0.0f));
        out_vec0.w = fmaf(negative_slope, fminf(in_vec0.w, 0.0f), fmaxf(in_vec0.w, 0.0f));

        float4 out_vec1;
        out_vec1.x = fmaf(negative_slope, fminf(in_vec1.x, 0.0f), fmaxf(in_vec1.x, 0.0f));
        out_vec1.y = fmaf(negative_slope, fminf(in_vec1.y, 0.0f), fmaxf(in_vec1.y, 0.0f));
        out_vec1.z = fmaf(negative_slope, fminf(in_vec1.z, 0.0f), fmaxf(in_vec1.z, 0.0f));
        out_vec1.w = fmaf(negative_slope, fminf(in_vec1.w, 0.0f), fmaxf(in_vec1.w, 0.0f));

        // Write results
        reinterpret_cast<float4*>(output)[id * 2] = out_vec0;
        reinterpret_cast<float4*>(output)[id * 2 + 1] = out_vec1;
    } else if (idx + 3 < n) {
        // Handle remaining 4-7 elements
        const float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + id * 2);

        float4 out_vec;
        out_vec.x = fmaf(negative_slope, fminf(in_vec.x, 0.0f), fmaxf(in_vec.x, 0.0f));
        out_vec.y = fmaf(negative_slope, fminf(in_vec.y, 0.0f), fmaxf(in_vec.y, 0.0f));
        out_vec.z = fmaf(negative_slope, fminf(in_vec.z, 0.0f), fmaxf(in_vec.z, 0.0f));
        out_vec.w = fmaf(negative_slope, fminf(in_vec.w, 0.0f), fmaxf(in_vec.w, 0.0f));

        reinterpret_cast<float4*>(output)[id * 2] = out_vec;

        // Handle remaining 1-3 elements
        for (int i = 4; i < 8 && idx + i < n; ++i) {
            const float val = input[idx + i];
            output[idx + i] = fmaf(negative_slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
        }
    } else {
        // Handle final (<4) elements one-by-one
        #pragma unroll
        for (int i = 0; i < 8 && idx + i < n; ++i) {
            const float val = input[idx + i];
            output[idx + i] = fmaf(negative_slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // One thread per 8 elements (2 float4 vectors)
    const int blocks = (n / 8 + threads - 1) / threads;

    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward,
          "Vectorized Leaky ReLU with loop unrolling for increased ILP");
}
"""

leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional model using loop unrolling.
    
    Improvements:
    - Each thread processes 8 elements (2 float4 vectors) instead of 4
    - Explicit unrolled computation increases instruction-level parallelism
    - Better pipelining of floating-point operations
    - Reduced loop overhead
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output
