# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_8.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for max_pool1d with optimized memory coalescing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void maxpool1d_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int num_features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Grid-stride loop approach for maximum occupancy
    // Each thread processes one output element
    
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    
    if (output_idx >= output_length || feature_idx >= num_features || batch_idx >= batch_size) {
        return;
    }
    
    // Calculate the starting input position for this output
    int input_start = output_idx * stride - padding;
    
    // Find the maximum value across the kernel window
    float max_val = -1e30f; // Use a large negative number instead of -infinity
    
    #pragma unroll 8
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = input_start + k * dilation;
        
        // Bounds checking
        if (input_pos >= 0 && input_pos < input_length) {
            // Coalesced memory access: threads in same warp read adjacent positions
            int input_flat_idx = batch_idx * num_features * input_length + 
                                 feature_idx * input_length + 
                                 input_pos;
            float val = input[input_flat_idx];
            max_val = fmaxf(max_val, val);
        }
    }
    
    // Write output with coalesced access pattern
    int output_flat_idx = batch_idx * num_features * output_length + 
                          feature_idx * output_length + 
                          output_idx;
    output[output_flat_idx] = max_val;
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

void maxpool1d_fused_forward(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d_fused", &maxpool1d_fused_forward, "Fused max_pool1d CUDA kernel");
}
"""

cuda_kernel_impl = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void maxpool1d_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int num_features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    
    if (output_idx >= output_length || feature_idx >= num_features || batch_idx >= batch_size) {
        return;
    }
    
    int input_start = output_idx * stride - padding;
    float max_val = -1e30f;
    
    #pragma unroll 8
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = input_start + k * dilation;
        
        if (input_pos >= 0 && input_pos < input_length) {
            int input_flat_idx = batch_idx * num_features * input_length + 
                                 feature_idx * input_length + 
                                 input_pos;
            float val = input[input_flat_idx];
            max_val = fmaxf(max_val, val);
        }
    }
    
    int output_flat_idx = batch_idx * num_features * output_length + 
                          feature_idx * output_length + 
                          output_idx;
    output[output_flat_idx] = max_val;
}

void maxpool1d_fused_forward(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int num_features = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    
    // Optimal block size: 32 threads per warp, 1024 total threads per block
    // Layout: (256 output positions, 4 features, 1 batch) per block
    dim3 block_size(256, 4, 1);
    
    dim3 grid_size(
        (output_length + block_size.x - 1) / block_size.x,
        (num_features + block_size.y - 1) / block_size.y,
        batch_size
    );
    
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    maxpool1d_fused_kernel<<<grid_size, block_size>>>(
        input_ptr,
        output_ptr,
        batch_size,
        num_features,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    cudaDeviceSynchronize();
}
"""

# Load the CUDA extension
try:
    maxpool_fused = load_inline(
        name='maxpool1d_fused_ext',
        cpp_sources=cpp_source,
        cuda_sources=cuda_kernel_impl,
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        with_cuda=True,
        verbose=False
    )
except:
    maxpool_fused = None

def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    """
    Custom CUDA-optimized max_pool1d with memory coalescing.
    """
    if maxpool_return_indices:
        raise NotImplementedError("return_indices not supported in custom kernel")
    
    batch_size, num_features, input_length = x.shape
    
    # Calculate output length
    if maxpool_ceil_mode:
        output_length = (input_length + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride + maxpool_stride - 1) // maxpool_stride
    else:
        output_length = (input_length + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    # Ensure input is contiguous and in float32
    x = x.contiguous().float()
    output = torch.empty(batch_size, num_features, output_length, dtype=x.dtype, device=x.device)
    
    # Call custom CUDA kernel
    maxpool_fused.maxpool1d_fused(
        x,
        output,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
        maxpool_dilation
    )
    
    return output


batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride = 1
padding = 4
dilation = 3
return_indices = False

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]
