# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_225030/code_16.py
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
# Constants (kept for compatibility with the reference test harness)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

# ----------------------------------------------------------------------
# Optimised CUDA kernel using texture memory
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vectorized_kernel_tex(
        cudaTextureObject_t input_tex,
        float* __restrict__ output,
        float negative_slope,
        size_t n)
{
    // Each thread processes 4 elements (1 float4)
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idx = id * 4;

    if (idx + 3 < n) {
        // Use texture fetch for float4 vector
        float4 in_vec = tex1Dfetch<float4>(input_tex, id);

        // Branch-less Leaky ReLU using fmaf
        // out = (x > 0) ? x : x * slope
        // Equivalent: fmaf(slope, fminf(x, 0), fmaxf(x, 0))
        float4 out_vec;
        out_vec.x = fmaf(negative_slope, fminf(in_vec.x, 0.0f), fmaxf(in_vec.x, 0.0f));
        out_vec.y = fmaf(negative_slope, fminf(in_vec.y, 0.0f), fmaxf(in_vec.y, 0.0f));
        out_vec.z = fmaf(negative_slope, fminf(in_vec.z, 0.0f), fmaxf(in_vec.z, 0.0f));
        out_vec.w = fmaf(negative_slope, fminf(in_vec.w, 0.0f), fmaxf(in_vec.w, 0.0f));

        reinterpret_cast<float4*>(output)[id] = out_vec;
    } else {
        // Scalar leftovers
        for (int i = 0; i < 4; ++i) {
            int current_idx = idx + i;
            if (current_idx < n) {
                float val = tex1Dfetch<float>(input_tex, current_idx);
                output[current_idx] = fmaf(negative_slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
            }
        }
    }
}

void leaky_relu_forward_tex(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    const int blocks = (n / 4 + threads - 1) / threads;

    // Create texture object for input to optimize read access
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = input.data_ptr<float>();
    res_desc.res.linear.sizeInBytes = n * sizeof(float);
    res_desc.res.linear.desc = cudaCreateChannelDesc<float>();

    cudaTextureDesc tex_desc = {};
    tex_desc.readMode = cudaReadModeElementType;

    cudaTextureObject_t input_tex = 0;
    cudaCreateTextureObject(&input_tex, &res_desc, &tex_desc, NULL);

    leaky_relu_vectorized_kernel_tex<<<blocks, threads>>>(
        input_tex,
        output.data_ptr<float>(),
        negative_slope,
        n
    );

    cudaDestroyTextureObject(input_tex);
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_forward_tex(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward_tex, "Leaky ReLU with texture memory optimization");
}
"""

# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
leaky_relu_ext = load_inline(
    name='leaky_relu_tex',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

_output_buffer = None

def functional_model(x, *, negative_slope):
    global _output_buffer

    # Ensure memory is compatible with our kernel
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # Initialize / Re-initialize output buffer
    if _output_buffer is None or _output_buffer.shape != x.shape:
        _output_buffer = torch.empty_like(x)

    leaky_relu_ext.leaky_relu(x, _output_buffer, float(negative_slope))
    return _output_buffer
