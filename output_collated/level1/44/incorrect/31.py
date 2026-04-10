# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115947/code_4.py
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

# --- CUDA Kernel ---
# Optimizations:
# 1. Use shared memory tiling if possible, though for 1D pooling, 
#    thread-local work is efficient when input is globally cached.
# 2. Loop unrolling: The compiler handles small loops well.
# 3. Memory Coalescing: Ensure threads in a warp access sequential addresses.
#    Threads are mapped to (batch, channel, output_idx).
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_length;

    if (tid >= total_elements) return;

    // Optimized index decomposition: 
    // Tid mapped to: batch -> channel -> output_pos
    // In memory order: [batch, channel, output_pos]
    int out_pos = tid % output_length;
    int rest = tid / output_length;
    int ch = rest % channels;
    int batch = rest / channels;

    int in_start = out_pos * stride - padding;
    int in_end = in_start + kernel_size;

    // Boundary conditions
    int start = max(in_start, 0);
    int end = min(in_end, input_length);

    float sum = 0.0f;
    int count = end - start;

    if (count > 0) {
        int base_idx = (batch * channels + ch) * input_length;
        #pragma unroll
        for (int i = start; i < end; ++i) {
            sum += input[base_idx + i];
        }
        output[tid] = sum / (float)count;
    } else {
        output[tid] = 0.0f;
    }
}

void avg_pool1d_launcher(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size, int stride, int padding
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    
    int total_elements = batch_size * channels * output_length;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    avg_pool1d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, input_length, output_length,
        kernel_size, stride, padding
    );
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_launcher(const torch::Tensor& input, torch::Tensor& output, int kernel_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_fused", &avg_pool1d_launcher, "Fused 1D AvgPool kernel");
}
"""

# Compile
fused_ext = load_inline(
    name='avg_pool1d_fused',
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
    # Calculate output shape (matching original logic)
    # ceil_mode=False is assumed based on original snippet requirements
    out_length = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    output = torch.empty((x.shape[0], x.shape[1], out_length), device=x.device, dtype=x.dtype)
    
    fused_ext.avg_pool1d_fused(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    return output

# Inputs Config
batch_size, in_channels, input_length = 64, 128, 65536
kernel_size, stride, padding = 8, 1, 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, input_length, device='cuda')]
