# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103120/code_7.py
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




import torch
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------
# 1. CUDA source (kernel + host wrapper)
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// mish(x) = x * tanh(softplus(x)) = x * tanh(log1p(exp(x)))
__device__ __forceinline__ float mish(float x) {
    // Optimization: avoid overflow/underflow for large x
    // For x > 20, softplus(x) approx x, tanh(x) approx 1, so mish(x) approx x
    if (x > 20.0f) return x;
    if (x < -20.0f) return x * expf(x);
    return x * tanhf(log1pf(expf(x)));
}

// Fused kernel: Linear(x) + Mish(x) + Mish(x)
// Strategy: Thread computes one output element. 
// Uses shared memory to cache input features tile.
__global__ void fused_linear_mish_kernel(
    const float* __restrict__ input,       // (batch_size, in_features)
    const float* __restrict__ weight,      // (out_features, in_features)
    const float* __restrict__ bias,        // (out_features)
    float* __restrict__ output,            // (batch_size, out_features)
    const int in_features,
    const int out_features)
{
    const int batch_idx = blockIdx.x;
    const int out_idx   = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    
    if (out_idx >= out_features) return;

    extern __shared__ float s_input[];

    float sum = 0.0f;
    const float* weight_row = weight + (out_idx * in_features);
    const float* input_row = input + (batch_idx * in_features);

    for (int i = 0; i < in_features; i += BLOCK_SIZE) {
        // Coalesced load into shared memory
        if (i + threadIdx.x < in_features) {
            s_input[threadIdx.x] = input_row[i + threadIdx.x];
        } else {
            s_input[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += weight_row[i + j] * s_input[j];
        }
        __syncthreads();
    }

    sum += bias[out_idx];
    
    // Apply double mish
    float y = mish(sum);
    output[batch_idx * out_features + out_idx] = mish(y);
}

void fused_op_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output)
{
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    dim3 block(BLOCK_SIZE);
    dim3 grid(batch_size, (out_features + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fused_linear_mish_kernel<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        in_features,
        out_features
    );
}
"""

# --------------------------------------------------------------
# 2. C++ binding
# --------------------------------------------------------------
cpp_source = r"""
void fused_op_forward(at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused linear + 2xMish");
}
"""

# --------------------------------------------------------------
# 3. Compile
# --------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------
# 4. Functional Model Implementation
# --------------------------------------------------------------
def functional_model(x, *, linear_weight, linear_bias):
    batch_size, out_features = x.shape[0], linear_weight.shape[0]
    out = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x.contiguous(), linear_weight.contiguous(), linear_bias.contiguous(), out)
    return out

def get_init_inputs():
    return [8192, 8192]

def get_inputs():
    return [torch.rand(1024, 8192, device='cuda')]
