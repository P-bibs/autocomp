# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_044333/code_29.py
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

# -------------------------------------------------------------------------
# CUDA implementation using block-wide shared memory for caching input
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define MIN_FLOAT -3.402823466e+38F

__global__ void maxpool1d_shared_kernel(
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
    // Each block processes exactly one (batch, channel) slice
    extern __shared__ float sdata[];
    
    int batch_idx = blockIdx.y;
    int ch_idx = blockIdx.x;
    
    // Calculate global offsets for current (batch, channel)
    const float* in_ptr = input + (batch_idx * channels + ch_idx) * input_length;
    float* out_ptr = output + (batch_idx * channels + ch_idx) * output_length;
    
    const int padded_len = input_length + 2 * padding;
    
    // 1. Cooperative load: fill shared memory with padded input
    for (int i = threadIdx.x; i < padded_len; i += blockDim.x) {
        int in_pos = i - padding;
        if (in_pos >= 0 && in_pos < input_length) {
            sdata[i] = in_ptr[in_pos];
        } else {
            sdata[i] = MIN_FLOAT;
        }
    }
    __syncthreads();
    
    // 2. Compute max pool for assigned output positions
    for (int out_pos = threadIdx.x; out_pos < output_length; out_pos += blockDim.x) {
        int start_pos = out_pos * stride; // sdata is already padded
        float max_val = MIN_FLOAT;
        
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int in_shared_pos = start_pos + k * dilation;
            float val = sdata[in_shared_pos];
            if (val > max_val) max_val = val;
        }
        out_ptr[out_pos] = max_val;
    }
}

void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length = output.size(2);
    
    dim3 grid(channels, batch_size);
    int threads_per_block = 256;
    size_t shared_mem_size = (input_length + 2 * p) * sizeof(float);
    
    maxpool1d_shared_kernel<<<grid, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        channels, input_length, output_length, k, s, p, d
    );
}
"""

cpp_source = r"""
void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_forward, "Shared-memory optimized 1D MaxPool");
}
"""

# Compile the extension inline
fused_ext = load_inline(
    name='maxpool1d_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    total_len = x.size(2)
    
    # Calculate output dimensions
    denom = maxpool_stride
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) + denom - 1) // denom + 1
    else:
        output_length = (total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // denom + 1
    
    # Prepare inputs: Ensure contiguous and on GPU
    x_gpu = x.cuda().contiguous()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    # Compute using optimized kernel
    fused_ext.maxpool1d(x_gpu, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
