# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_074911/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'subtract_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias', 'subtract_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

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
    if 'subtract_value' in flat_state:
        state_kwargs['subtract_value'] = flat_state['subtract_value']
    else:
        state_kwargs['subtract_value'] = getattr(model, 'subtract_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
# CUDA source – the fused kernel that replaces linear + subtract/multiply/ReLU
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Fused kernel: computes  out = ( x @ W^T + bias - sub ) * mul  with ReLU
__global__ void fused_op_forward_kernel(
    const float* __restrict__ x,          // (batch, in_features)
    const float* __restrict__ weight,     // (out_features, in_features)
    const float* __restrict__ bias,       // (out_features)
    float subtract_value,
    float multiply_value,
    float* __restrict__ out,              // (batch, out_features)
    int batch,
    int in_features,
    int out_features)
{
    // dynamic shared memory for the reduction of partial dot-products
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_features) return;

    int row = idx / out_features;   // which batch sample
    int col = idx % out_features;   // which output feature

    // ---- 1) partial dot-product (strided access for coalescing) ----
    float sum = 0.0f;
    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
        sum += x[row * in_features + k] * weight[col * in_features + k];
    }

    // ---- 2) store partial result in shared memory ----
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // ---- 3) reduce partial results within the block ----
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total = sdata[0];   // now every thread can read the full dot-product

    // ---- 4) bias, subtract, multiply, ReLU (all fused) ----
    total += bias[col];
    total = (total - subtract_value) * multiply_value;
    total = fmaxf(total, 0.0f);   // ReLU

    out[row * out_features + col] = total;
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value,
    torch::Tensor out)
{
    const int batch      = x.size(0);
    const int in_feat    = x.size(1);
    const int out_feat   = weight.size(0);

    const int block_size = BLOCK_SIZE;
    const int grid_size  = (batch * out_feat + block_size - 1) / block_size;

    fused_op_forward_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        subtract_value,
        multiply_value,
        out.data_ptr<float>(),
        batch,
        in_feat,
        out_feat
    );
}
"""

# -------------------------------------------------------------------------
# C++ bindings – expose the fused_op function to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value,
    torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused linear (GEMM) + element-wise (sub, mul, ReLU) kernel");
}
"""

# -------------------------------------------------------------------------
# Compile the extension (this happens once when the module is imported)
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# The only function that will be imported by the evaluator
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value
):
    """
    Fused implementation:  out = ReLU( (x @ W^T + bias - subtract) * multiply )
    All work is performed in a single custom CUDA kernel.
    """
    # Ensure inputs reside on the GPU and are contiguous
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()

    x = x.contiguous()
    linear_weight = linear_weight.contiguous()
    linear_bias = linear_bias.contiguous()

    # Allocate output tensor
    batch_size = x.size(0)
    out_features = linear_weight.size(0)
    out = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)

    # Launch the fused kernel
    fused_ext.fused_op(
        x,
        linear_weight,
        linear_bias,
        subtract_value,
        multiply_value,
        out
    )
    
    return out
