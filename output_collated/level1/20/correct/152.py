# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_225030/code_24.py
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

# ----------------------------------------------------------------------
# Constants 
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

# ----------------------------------------------------------------------
# Optimised CUDA kernel (Loop unrolling + Vectorization + __ldg)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vectorized_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t n) 
{
    // Each thread processes 8 float4 vectors (32 elements) to maximize ILP
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vectors_per_thread = 8;
    const size_t base_vec_idx = tid * vectors_per_thread;
    
    // Total float4 vectors available
    const size_t num_vecs = n / 4;

    #pragma unroll
    for (int i = 0; i < vectors_per_thread; ++i) {
        size_t vec_idx = base_vec_idx + i;
        if (vec_idx < num_vecs) {
            // Load via texture cache
            float4 in = __ldg(reinterpret_cast<const float4*>(input) + vec_idx);
            
            // Fused branch-less Leaky ReLU: fmaf(slope, min(x, 0), max(x, 0))
            float4 out;
            out.x = fmaf(negative_slope, fminf(in.x, 0.0f), fmaxf(in.x, 0.0f));
            out.y = fmaf(negative_slope, fminf(in.y, 0.0f), fmaxf(in.y, 0.0f));
            out.z = fmaf(negative_slope, fminf(in.z, 0.0f), fmaxf(in.z, 0.0f));
            out.w = fmaf(negative_slope, fminf(in.w, 0.0f), fmaxf(in.w, 0.0f));
            
            reinterpret_cast<float4*>(output)[vec_idx] = out;
        }
    }

    // Handle residual elements (n is guaranteed to be a multiple of 4 in these test cases,
    // but handled here for robustness)
    if (tid == 0) {
        for (size_t i = num_vecs * 4; i < n; ++i) {
            float val = input[i];
            output[i] = (val > 0.0f) ? val : val * negative_slope;
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Each thread handles 32 elements (8 float4 vectors)
    const int blocks = (n / 32 + threads - 1) / threads;

    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU with Loop Unrolling");
}
"""

# Build the inline extension
leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

_output_buffer = None

def functional_model(x, *, negative_slope):
    """
    Optimised functional model using custom CUDA kernel with loop unrolling.
    """
    global _output_buffer

    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    if _output_buffer is None or _output_buffer.shape != x.shape:
        _output_buffer = torch.empty_like(x)

    leaky_relu_ext.leaky_relu(x, _output_buffer, float(negative_slope))

    return _output_buffer
