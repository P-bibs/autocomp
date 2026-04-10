# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113951/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['avg_pool_kernel_size', 'avg_pool_stride', 'avg_pool_padding', 'avg_pool_ceil_mode', 'avg_pool_count_include_pad']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int=1, padding: int=0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

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
    # State for avg_pool (nn.AvgPool1d)
    state_kwargs['avg_pool_kernel_size'] = model.avg_pool.kernel_size
    state_kwargs['avg_pool_stride'] = model.avg_pool.stride
    state_kwargs['avg_pool_padding'] = model.avg_pool.padding
    state_kwargs['avg_pool_ceil_mode'] = model.avg_pool.ceil_mode
    state_kwargs['avg_pool_count_include_pad'] = model.avg_pool.count_include_pad
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

__global__ void avg_pool1d_kernel(const float* input, float* output,
                                  int batch_size, int channels, int input_length,
                                  int kernel_size, int stride, int padding,
                                  int output_length) {
    extern __shared__ float shared_data[];

    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.z;
    
    // Calculate global output index
    int global_out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_out_idx >= output_length) return;
    
    // Calculate the starting input index for this output element
    int input_start_idx = global_out_idx * stride - padding;
    
    // Shared memory configuration
    int shared_size = blockDim.x * stride + kernel_size - 1;
    
    // Load data into shared memory with padding handling
    for (int i = threadIdx.x; i < shared_size; i += blockDim.x) {
        int input_idx = input_start_idx + i;
        if (input_idx >= 0 && input_idx < input_length) {
            shared_data[i] = input[(batch_idx * channels + channel_idx) * input_length + input_idx];
        } else {
            shared_data[i] = 0.0f; // Padding with zeros
        }
    }
    
    __syncthreads();
    
    // Perform average pooling
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        sum += shared_data[threadIdx.x * stride + i];
    }
    
    output[(batch_idx * channels + channel_idx) * output_length + global_out_idx] = sum / kernel_size;
}

void avg_pool1d_forward(const torch::Tensor& input, torch::Tensor& output,
                        int kernel_size, int stride, int padding) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    
    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid_x = (output_length + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_grid_x, batch_size, channels);
    dim3 block(threads_per_block);
    
    // Calculate shared memory size
    int shared_size = (threads_per_block * stride + kernel_size - 1) * sizeof(float);
    
    avg_pool1d_kernel<<<grid, block, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_length
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_forward(const torch::Tensor& input, torch::Tensor& output,
                        int kernel_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_forward", &avg_pool1d_forward, "1D Average Pooling Forward Pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='avg_pool1d_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,
    avg_pool_count_include_pad,
):
    # Calculate output length (simplified for floor mode and count_include_pad=True)
    # This matches the original F.avg_pool1d behavior for the common case
    output_length = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    # Create output tensor on the same device as input
    out = torch.empty((x.shape[0], x.shape[1], output_length), device=x.device, dtype=x.dtype)
    
    # Call custom CUDA kernel
    fused_ext.avg_pool1d_forward(x.contiguous(), out, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    
    return out

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length, device='cuda')
    return [x]
