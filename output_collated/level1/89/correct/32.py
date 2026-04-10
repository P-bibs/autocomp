# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_4.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Pad cols to nearest multiple of 4
    int padded_cols = ((cols + 3) / 4) * 4;
    
    const float4* input_v = reinterpret_cast<const float4*>(input + row * cols);
    float4* output_v = reinterpret_cast<float4*>(output + row * cols);
    
    int vec_cols = padded_cols / 4;
    int valid_vec_cols = cols / 4;

    float carry = 0.0f;
    
    for (int i = 0; i < vec_cols; i += 32) {
        int idx = i + threadIdx.x;
        
        // Load data - handle both valid and padding elements
        float4 val_v;
        if (idx < valid_vec_cols) {
            val_v = input_v[idx];
        } else if (idx < vec_cols) {
            // Handle partial vector with padding
            const float* base_ptr = input + row * cols;
            int base_idx = idx * 4;
            float v0 = (base_idx < cols) ? base_ptr[base_idx] : 0.0f;
            float v1 = (base_idx + 1 < cols) ? base_ptr[base_idx + 1] : 0.0f;
            float v2 = (base_idx + 2 < cols) ? base_ptr[base_idx + 2] : 0.0f;
            float v3 = (base_idx + 3 < cols) ? base_ptr[base_idx + 3] : 0.0f;
            val_v = make_float4(v0, v1, v2, v3);
        } else {
            val_v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // Convert float4 to array for processing
        float vals[4] = {val_v.x, val_v.y, val_v.z, val_v.w};
        
        // Inclusive scan within the 4-element group
        vals[1] += vals[0];
        vals[2] += vals[1];
        vals[3] += vals[2];

        // Warp-level inclusive scan using shfl_up
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float temp = __shfl_up_sync(0xFFFFFFFF, vals[3], offset);
            if (threadIdx.x >= offset) {
                for(int j = 0; j < 4; j++) {
                    vals[j] += temp;
                }
            }
        }

        // Add carry from previous segment
        for(int j = 0; j < 4; j++) {
            vals[j] += carry;
        }

        // Write back to global memory
        if (idx < valid_vec_cols) {
            output_v[idx] = make_float4(vals[0], vals[1], vals[2], vals[3]);
        } else if (idx < vec_cols) {
            // Handle partial writes
            float* base_ptr = output + row * cols;
            int base_idx = idx * 4;
            if (base_idx < cols) base_ptr[base_idx] = vals[0];
            if (base_idx + 1 < cols) base_ptr[base_idx + 1] = vals[1];
            if (base_idx + 2 < cols) base_ptr[base_idx + 2] = vals[2];
            if (base_idx + 3 < cols) base_ptr[base_idx + 3] = vals[3];
        }

        // Update carry with the sum of this segment (last element of the warp)
        carry = __shfl_sync(0xFFFFFFFF, vals[3], 31);
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    dim3 threads(32);
    dim3 blocks(rows);
    cumsum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        (int)rows, 
        (int)cols
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

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    # Handle dimension permutation if needed
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    # Reverse permutation if needed
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
