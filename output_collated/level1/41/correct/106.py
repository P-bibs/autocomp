# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_050814/code_9.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """

    def __init__(self, kernel_size: int, stride: int=None, padding: int=0, dilation: int=1, return_indices: bool=False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices)

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
    # State for maxpool (nn.MaxPool1d)
    state_kwargs['maxpool_kernel_size'] = model.maxpool.kernel_size
    state_kwargs['maxpool_stride'] = model.maxpool.stride
    state_kwargs['maxpool_padding'] = model.maxpool.padding
    state_kwargs['maxpool_dilation'] = model.maxpool.dilation
    state_kwargs['maxpool_ceil_mode'] = model.maxpool.ceil_mode
    state_kwargs['maxpool_return_indices'] = model.maxpool.return_indices
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Vectorized load helpers
inline __device__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

inline __device__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

inline __device__ float maxf4(float4 v) {
    float m = fmaxf(v.x, v.y);
    m = fmaxf(m, v.z);
    return fmaxf(m, v.w);
}

__global__ void maxpool1d_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Calculate global offset for the batch and channel
    const float* in_ptr = input + (batch_idx * channels + channel_idx) * input_length;
    float* out_ptr = output + (batch_idx * channels + channel_idx) * output_length;
    
    // Process output positions with stride 4 (vectorized output writes)
    int output_pos_step = 4;
    
    for (int base_out_pos = tid * output_pos_step; base_out_pos < output_length; base_out_pos += blockDim.x * output_pos_step) {
        float4 results = make_float4(-3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F, -3.402823466e+38F);
        
        // Process up to 4 output positions
        int positions[4];
        int valid_count = 0;
        
        #pragma unroll 4
        for (int i = 0; i < output_pos_step && base_out_pos + i < output_length; i++) {
            positions[i] = base_out_pos + i;
            valid_count++;
        }
        
        // For each output position in this group
        for (int pos_idx = 0; pos_idx < valid_count; pos_idx++) {
            int out_pos = positions[pos_idx];
            int start_pos = out_pos * stride - padding;
            float max_val = -3.402823466e+38F;
            
            // Read kernel window with vectorized loads where possible
            int k = 0;
            
            // Try to read 4 elements at once when aligned
            while (k + 4 <= kernel_size) {
                int in_pos_base = start_pos + k * dilation;
                
                // Check if we can do a vectorized read
                if (dilation == 1 && 
                    in_pos_base >= 0 && 
                    in_pos_base + 3 < input_length &&
                    ((size_t)(in_ptr + in_pos_base) & 15) == 0) {
                    
                    float4 vals = load_float4(in_ptr + in_pos_base);
                    max_val = fmaxf(max_val, vals.x);
                    max_val = fmaxf(max_val, vals.y);
                    max_val = fmaxf(max_val, vals.z);
                    max_val = fmaxf(max_val, vals.w);
                    k += 4;
                } else {
                    // Fall back to scalar reads for misaligned or strided access
                    int in_pos = in_pos_base;
                    if (in_pos >= 0 && in_pos < input_length) {
                        max_val = fmaxf(max_val, in_ptr[in_pos]);
                    }
                    k++;
                }
            }
            
            // Handle remaining elements
            #pragma unroll
            for (; k < kernel_size; ++k) {
                int in_pos = start_pos + k * dilation;
                if (in_pos >= 0 && in_pos < input_length) {
                    max_val = fmaxf(max_val, in_ptr[in_pos]);
                }
            }
            
            // Store result in vectorized structure
            if (pos_idx == 0) results.x = max_val;
            else if (pos_idx == 1) results.y = max_val;
            else if (pos_idx == 2) results.z = max_val;
            else if (pos_idx == 3) results.w = max_val;
        }
        
        // Vectorized write if aligned and full 4 elements
        if ((base_out_pos + 4 <= output_length) && 
            ((size_t)(out_ptr + base_out_pos) & 15) == 0) {
            store_float4(out_ptr + base_out_pos, results);
        } else {
            // Scalar writes for boundary or misaligned case
            if (base_out_pos < output_length) out_ptr[base_out_pos] = results.x;
            if (base_out_pos + 1 < output_length) out_ptr[base_out_pos + 1] = results.y;
            if (base_out_pos + 2 < output_length) out_ptr[base_out_pos + 2] = results.z;
            if (base_out_pos + 3 < output_length) out_ptr[base_out_pos + 3] = results.w;
        }
    }
}

void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    
    // Grid dimensions: x = channels, y = batch
    dim3 grid(channels, batch_size);
    // Increased thread block for better occupancy with wider vectorization
    int threads = 256;
    
    maxpool1d_vectorized_kernel<<<grid, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        channels, input_length, output_length, k, s, p, d
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_forward, "Vectorized MaxPool1D forward with float4 optimization");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='maxpool1d_vec',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    total_len = x.size(2)
    
    # Calculate output dimensions according to ceiling/floor logic
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = (total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    # Ensure input is on device and contiguous
    x_gpu = x.cuda().contiguous()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    # Call the optimized kernel
    fused_ext.maxpool1d(x_gpu, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
