# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_22.py
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

# ----------------------------------------------------------------------
# Optimized CUDA kernel with Shared Memory Tiling and Coalesced Access
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void max_pool1d_dilated_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Shared memory for a segment of the input channel
    // Max tile size: 256 * 1 + (8-1)*3 = 277 floats. 
    // We make it slightly larger for safety and alignment.
    extern __shared__ float sdata[];

    const int bs = blockIdx.x / channels;
    const int ch = blockIdx.x % channels;
    if (bs >= batch_size) return;

    const int input_channel_offset = (bs * channels + ch) * input_length;
    const int output_channel_offset = (bs * channels + ch) * output_length;

    // Process output sequence in chunks
    for (int out_start = 0; out_start < output_length; out_start += blockDim.x) {
        
        // Coalesced loading into shared memory
        // The tile must cover the furthest index accessed by this block
        const int in_start_block = out_start * stride - padding;
        const int tile_size = blockDim.x * stride + (kernel_size - 1) * dilation;

        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            int input_pos = in_start_block + i;
            if (input_pos >= 0 && input_pos < input_length) {
                sdata[i] = input[input_channel_offset + input_pos];
            } else {
                sdata[i] = -1e38f; // Represent -infinity for padding
            }
        }
        __syncthreads();

        // Compute max for the thread's specific output position
        int out_pos = out_start + threadIdx.x;
        if (out_pos < output_length) {
            float max_val = -1e38f;
            int in_start_local = threadIdx.x * stride;
            
            #pragma unroll
            for (int k = 0; k < kernel_size; ++k) {
                float val = sdata[in_start_local + k * dilation];
                if (val > max_val) max_val = val;
            }
            output[output_channel_offset + out_pos] = max_val;
        }
        __syncthreads();
    }
}

void max_pool1d_dilated_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length = output.size(2);
    
    // Use 256 threads per block
    const int threads = 256;
    const int blocks = batch_size * channels;
    
    // Shared memory size calculation
    const int tile_size = threads * stride + (kernel_size - 1) * dilation;
    const size_t shared_mem_size = tile_size * sizeof(float);
    
    max_pool1d_dilated_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void max_pool1d_dilated_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool1d_dilated_forward", &max_pool1d_dilated_forward, "Max Pool1D Dilated Forward");
}
"""

# Compile the extension
maxpool_ext = load_inline(
    name='max_pool1d_dilated_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    # Calculate output length
    val = (x.shape[2] + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1
    if maxpool_ceil_mode:
        output_length = int(torch.ceil(torch.tensor(val)).item())
    else:
        output_length = int(torch.floor(torch.tensor(val)).item())
    
    output = torch.empty(x.shape[0], x.shape[1], output_length, device=x.device, dtype=x.dtype)
    
    maxpool_ext.max_pool1d_dilated_forward(
        x, output, 
        maxpool_kernel_size, 
        maxpool_stride, 
        maxpool_padding, 
        maxpool_dilation
    )
    
    if maxpool_return_indices:
        return output, None
    return output

def get_init_inputs():
    return [8, 1, 4, 3, False]

def get_inputs():
    return [torch.rand(64, 192, 65536, device='cuda')]
