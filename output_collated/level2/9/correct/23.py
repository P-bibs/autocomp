# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_075823/code_2.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused linear + bias + subtract + multiply + relu
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_linear_ops_kernel(
    const float* __restrict__ x,           // [batch_size, in_features]
    const float* __restrict__ weight,      // [out_features, in_features]
    const float* __restrict__ bias,        // [out_features]
    float* __restrict__ output,            // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    float subtract_value,
    float multiply_value
) {
    // Each thread block handles one output feature for one batch element
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Shared memory for partial sums
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // Initialize shared memory
    sdata[tid] = 0.0f;
    
    // Compute dot product with shared memory reduction
    for (int i = tid; i < in_features; i += blockDim.x) {
        sdata[tid] += weight[out_idx * in_features + i] * x[batch_idx * in_features + i];
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 applies bias, subtract, multiply and ReLU
    if (tid == 0) {
        float result = sdata[0] + bias[out_idx];
        result = (result - subtract_value) * multiply_value;
        result = fmaxf(0.0f, result);  // ReLU
        output[batch_idx * out_features + out_idx] = result;
    }
}

void fused_linear_ops_launch(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features,
    float subtract_value,
    float multiply_value
) {
    // Grid: (batch_size, out_features)
    dim3 grid(batch_size, out_features);
    // Block: threads per block for reduction (power of 2, max 512)
    int threads_per_block = 256;
    if (in_features < threads_per_block) {
        threads_per_block = 1 << (int)log2f((float)in_features);
        if (threads_per_block == 0) threads_per_block = 1;
    }
    
    // Shared memory size
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    fused_linear_ops_kernel<<<grid, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        subtract_value,
        multiply_value
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_linear_ops_launch(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_features,
    int out_features,
    float subtract_value,
    float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_ops", &fused_linear_ops_launch, 
          "Fused linear + bias + subtract + multiply + relu operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_linear_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = linear_weight.shape[0]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)
    
    # Call fused CUDA kernel
    fused_ext.fused_linear_ops(
        x,
        linear_weight,
        linear_bias,
        output,
        batch_size,
        in_features,
        out_features,
        float(subtract_value),
        float(multiply_value)
    )
    
    return output


# Test configuration
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
