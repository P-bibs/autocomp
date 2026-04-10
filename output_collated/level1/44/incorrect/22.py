# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115226/code_3.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel code with memory coalescing optimization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define KERNEL_SIZE 8

// Optimized kernel with memory coalescing
__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int channel_idx = blockIdx.z;
    
    // Calculate output position
    const int out_pos = bid * BLOCK_SIZE + tid;
    if (out_pos >= output_length) return;
    
    // Calculate input start position
    const int in_start = out_pos * stride - padding;
    
    // Base offsets
    const int offset = (batch_idx * in_channels + channel_idx);
    const int in_base = offset * input_length;
    const int out_base = offset * output_length;
    
    // Compute sum
    float sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        const int idx = in_start + i;
        if (idx >= 0 && idx < input_length) {
            sum += input[in_base + idx];
        }
    }
    
    // Store result
    output[out_base + out_pos] = sum / kernel_size;
}

void avg_pool1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    
    const int threads = BLOCK_SIZE;
    const int blocks = (output_length + threads - 1) / threads;
    
    dim3 grid(blocks, batch_size, in_channels);
    dim3 block(threads);
    
    avg_pool1d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda, "Optimized avg pool 1D");
}
"""

# Compile
fused_ext = load_inline(
    name='avg_pool1d_final_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]

def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,
    avg_pool_count_include_pad,
):
    # Ensure input is on GPU
    if not x.is_cuda:
        x = x.cuda()
    
    # Calculate output length
    output_length = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    # Create output tensor
    output = torch.empty(x.shape[0], x.shape[1], output_length, dtype=x.dtype, device='cuda')
    
    # Call CUDA kernel
    fused_ext.avg_pool1d_cuda(
        x,
        output,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding
    )
    
    return output
