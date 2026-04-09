# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103120/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




#!/usr/bin/env python3
import torch
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------
# 1. CUDA source (kernel + host wrapper)
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256                     // threads per block

// ----- device‑side mish -------------------------------------------------
// mish(x) = x * tanh(softplus(x))
// softplus(x) = log1p(exp(x))
__device__ __forceinline__ float mish(float x) {
    float sp = log1pf(expf(x));            // softplus
    float t = tanhf(sp);
    return x * t;
}

// ----- fused kernel ----------------------------------------------------
__global__ void fused_op_kernel(
    const float* __restrict__ input,       // (batch, in_features)
    const float* __restrict__ weight,      // (out_features, in_features)
    const float* __restrict__ bias,        // (out_features)
    float* __restrict__ output,            // (batch, out_features)
    const int batch_size,
    const int in_features,
    const int out_features)
{
    const int batch_idx = blockIdx.x;                     // batch dimension
    const int out_idx   = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    if (out_idx >= out_features) return;

    const int input_offset = batch_idx * in_features;     // row start in input
    const int weight_offset = out_idx * in_features;      // row start in weight

    // shared memory for one tile of the input vector (BLOCK_SIZE elements)
    __shared__ float s_input[BLOCK_SIZE];

    float sum = 0.0f;

    // ---- tiled matrix‑vector product ---------------------------------
    // Loop over the input dimension in chunks of BLOCK_SIZE
    for (int i = 0; i < in_features; i += BLOCK_SIZE) {
        // each thread loads one element of the input tile (coalesced)
        int idx = i + threadIdx.x;
        if (idx < in_features) {
            s_input[threadIdx.x] = input[input_offset + idx];
        }
        __syncthreads();

        // compute partial dot product for the loaded tile
        if (idx < in_features) {
            sum += weight[weight_offset + idx] * s_input[threadIdx.x];
        }
        __syncthreads();   // ready for next tile
    }

    // ---- add bias ----------------------------------------------------
    sum += bias[out_idx];

    // ---- apply mish twice -------------------------------------------
    float y = mish(sum);
    y = mish(y);

    // ---- write result ------------------------------------------------
    output[batch_idx * out_features + out_idx] = y;
}

// ----- host wrapper (PyTorch binding) ---------------------------------
void fused_op_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output)
{
    const int batch_size   = input.size(0);
    const int in_features  = input.size(1);
    const int out_features = weight.size(0);

    const int blocks_y = (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(batch_size, blocks_y);        // (batch, output‑tile)
    dim3 block(BLOCK_SIZE);                 // threads per block

    const size_t shared_mem = BLOCK_SIZE * sizeof(float);

    fused_op_kernel<<<grid, block, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features);
}
"""

# --------------------------------------------------------------
# 2. C++ binding (PYBIND11)
# --------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused linear + double mish (CUDA)");
}
"""

# --------------------------------------------------------------
# 3. Build the extension
# --------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------
# 4. Replace functional_model
# --------------------------------------------------------------
def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
):
    """
    Fused linear layer + two mish activations.
    All arguments are PyTorch tensors; they are moved to the current CUDA device
    if they are not already there.
    """
    # Ensure we are running on GPU
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()

    batch_size   = x.size(0)
    out_features = linear_weight.size(0)

    # Allocate output tensor on the same device
    out = torch.empty((batch_size, out_features),
                      dtype=x.dtype, device=x.device)

    # Launch the fused CUDA kernel
    fused_ext.fused_op(x, linear_weight, linear_bias, out)

    return out


# --------------------------------------------------------------
# 5. Helper functions required by the harness (not used in evaluation)
# --------------------------------------------------------------
def get_init_inputs():
    in_features = 8192
    out_features = 8192
    return [in_features, out_features]

def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features)]
