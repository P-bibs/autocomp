# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_13.py
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

# -------------------------------------------------------------------------
# CUDA source – two kernels (vector + tail) and their launchers
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// Vectorised kernel: one thread -> one float4 (branch‑less)
// ------------------------------------------------------------------
__global__ void leaky_relu_vec_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       float negative_slope,
                                       int vec_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vec_count) return;                     // out‑of‑range guard

    float4 in_vec = reinterpret_cast<const float4*>(input)[idx];
    float4 out_vec;
    // branch‑less leaky ReLU: max(0,x) + slope * min(0,x)
    out_vec.x = fmaxf(in_vec.x, 0.0f) + negative_slope * fminf(in_vec.x, 0.0f);
    out_vec.y = fmaxf(in_vec.y, 0.0f) + negative_slope * fminf(in_vec.y, 0.0f);
    out_vec.z = fmaxf(in_vec.z, 0.0f) + negative_slope * fminf(in_vec.z, 0.0f);
    out_vec.w = fmaxf(in_vec.w, 0.0f) + negative_slope * fminf(in_vec.w, 0.0f);

    reinterpret_cast<float4*>(output)[idx] = out_vec;
}

// ------------------------------------------------------------------
// Tail kernel: handles the last 0‑3 elements (also branch‑less)
// ------------------------------------------------------------------
__global__ void leaky_relu_tail_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        float negative_slope,
                                        int start_idx,
                                        int rem_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rem_count) return;

    int g_idx = start_idx + idx;
    float val = input[g_idx];
    output[g_idx] = fmaxf(val, 0.0f) + negative_slope * fminf(val, 0.0f);
}

// ------------------------------------------------------------------
// Launchers (public C++ API)
// ------------------------------------------------------------------
void launch_leaky_relu_vec(int vec_count, float* input, float* output,
                           float negative_slope) {
    const int block_size = 256;
    int grid_size = (vec_count + block_size - 1) / block_size;
    leaky_relu_vec_kernel<<<grid_size, block_size>>>(input, output,
                                                     negative_slope, vec_count);
}

void launch_leaky_relu_tail(int rem_count, float* input, float* output,
                            float negative_slope, int start_idx) {
    const int block_size = 256;
    int grid_size = (rem_count + block_size - 1) / block_size;
    leaky_relu_tail_kernel<<<grid_size, block_size>>>(input, output,
                                                       negative_slope,
                                                       start_idx, rem_count);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PyBind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu_vec(int vec_count, float* input, float* output,
                           float negative_slope);
void launch_leaky_relu_tail(int rem_count, float* input, float* output,
                            float negative_slope, int start_idx);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward_vec", &launch_leaky_relu_vec,
          "Vectorised Leaky ReLU forward (full float4 blocks)");
    m.def("leaky_relu_forward_tail", &launch_leaky_relu_tail,
          "Leaky ReLU forward (tail <4 elements)");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Functional model required by the harness
# -------------------------------------------------------------------------
def functional_model(x, *, negative_slope):
    # Make sure the input is contiguous on the GPU
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x)

    numel = x.numel()
    vec_count = numel // 4          # number of complete float4 blocks
    rem = numel % 4                 # leftover elements (0..3)

    if vec_count > 0:
        leaky_relu_ext.leaky_relu_forward_vec(
            vec_count, x.data_ptr<float>(), output.data_ptr<float>(), float(negative_slope))

    if rem > 0:
        start = vec_count * 4
        leaky_relu_ext.leaky_relu_forward_tail(
            rem, x.data_ptr<float>(), output.data_ptr<float>(), float(negative_slope), start)

    return output


# -------------------------------------------------------------------------
# Boilerplate required by the evaluation harness (batch size, input gen)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216


def get_init_inputs():
    return []


def get_inputs():
    # Random input tensor on GPU, float32
    return [torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)]
