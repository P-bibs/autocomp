# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_211408/code_24.py
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

# ----------------------------------------------------------------------
# Optimized CUDA Kernel: Increased Work per Thread (16 floats per thread)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Each thread handles 16 floats (4 float4 vectors)
__global__ void leaky_relu_tiled_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t n) 
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t floats_per_thread = 16;
    const size_t base_idx = tid * floats_per_thread;

    if (base_idx + 15 < n) {
        // Unrolled processing of 4 float4 vectors
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 in_vec = reinterpret_cast<const float4*>(input + base_idx + i * 4)[0];
            
            float4 out_vec;
            out_vec.x = fmaf(negative_slope, fminf(in_vec.x, 0.0f), fmaxf(in_vec.x, 0.0f));
            out_vec.y = fmaf(negative_slope, fminf(in_vec.y, 0.0f), fmaxf(in_vec.y, 0.0f));
            out_vec.z = fmaf(negative_slope, fminf(in_vec.z, 0.0f), fmaxf(in_vec.z, 0.0f));
            out_vec.w = fmaf(negative_slope, fminf(in_vec.w, 0.0f), fmaxf(in_vec.w, 0.0f));
            
            reinterpret_cast<float4*>(output + base_idx + i * 4)[0] = out_vec;
        }
    } else {
        // Cleanup loop for remaining elements
        for (int i = 0; i < floats_per_thread; ++i) {
            if (base_idx + i < n) {
                float val = input[base_idx + i];
                output[base_idx + i] = fmaf(negative_slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
            }
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // 16 floats per thread (4 float4 vectors), so divide n by 16
    const int blocks = (n / 16 + threads - 1) / threads;

    leaky_relu_tiled_kernel<<<blocks, threads>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional model using 16-element-per-thread tiling and #pragma unroll.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output
