# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_20.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
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

# The kernel uses float4 vectorized loads/stores and a warp-level scan
# ensuring high throughput on the RTX 2080Ti.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int vec_cols) {
    int row = blockIdx.x;
    const float4* input_v = reinterpret_cast<const float4*>(input + row * (vec_cols * 4));
    float4* output_v = reinterpret_cast<float4*>(output + row * (vec_cols * 4));

    float carry = 0.0f;
    // Each thread processes 4 elements via float4
    for (int i = 0; i < vec_cols; i += 32) {
        int idx = i + threadIdx.x;
        float4 val_v = (idx < vec_cols) ? input_v[idx] : make_float4(0,0,0,0);
        
        float vals[4] = {val_v.x, val_v.y, val_v.z, val_v.w};
        // Local inclusive scan within the float4 elements
        for(int j = 1; j < 4; ++j) vals[j] += vals[j-1];

        // Warp-level scan across the 32 threads using __shfl_up_sync
        for (int offset = 1; offset < 32; offset <<= 1) {
            float temp = __shfl_up_sync(0xFFFFFFFF, vals[3], offset);
            if (threadIdx.x >= offset) {
                #pragma unroll
                for(int j = 0; j < 4; ++j) vals[j] += temp;
            }
        }

        // Apply incoming carry from previous segments
        #pragma unroll
        for(int j = 0; j < 4; ++j) vals[j] += carry;

        if (idx < vec_cols) {
            output_v[idx] = make_float4(vals[0], vals[1], vals[2], vals[3]);
        }
        // Broadcast the final sum of the current float4 group to all threads in the warp
        carry = __shfl_sync(0xFFFFFFFF, vals[3], 31);
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    int vec_cols = (int)(cols / 4);
    dim3 threads(32);
    dim3 blocks((int)rows);
    cumsum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        (int)rows, 
        vec_cols
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Vectorized fused cumsum");
}
"""

# Compile the extension inline
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    original_dtype = x.dtype
    orig_shape = list(x.shape)
    
    # Handle dimension transposition to ensure scan is along the last dimension
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    
    # Pad to multiple of 4 for float4 vectorization
    last_dim = x.shape[-1]
    pad = (4 - (last_dim % 4)) % 4
    if pad != 0:
        x = torch.nn.functional.pad(x, (0, pad))
    
    x_f32 = x.to(torch.float32)
    output = torch.empty_like(x_f32)
    
    rows = x_f32.numel() // x_f32.shape[-1]
    cols = x_f32.shape[-1]
    
    # Launch CUDA kernel
    fused_ext.fused_op(rows, cols, x_f32, output)
    
    # Remove padding
    if pad != 0:
        output = output[..., :last_dim]
        
    # Revert transpose if needed
    if dim != -1 and dim != len(orig_shape) - 1:
        output = output.permute(*permute_dims)
        
    return output.to(original_dtype)
