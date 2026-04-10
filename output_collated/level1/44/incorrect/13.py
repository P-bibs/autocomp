# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114325/code_4.py
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

# CUDA kernel for optimized average pooling
# The strategy uses shared memory to load tiles of input data, reducing redundant global reads.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad) {
    
    // Shared memory size calculation: we need to cover the range of inputs accessed by the block
    extern __shared__ unsigned char shared_mem[];
    scalar_t* shm = reinterpret_cast<scalar_t*>(shared_mem);

    int bc_idx = blockIdx.y; 
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx >= output_length) return;

    // Each block processes a range of output indices. 
    // Load necessary input range into shared memory to facilitate reuse.
    int start_load_idx = blockIdx.x * blockDim.x * stride - padding;
    int end_load_idx = (blockIdx.x * blockDim.x + blockDim.x - 1) * stride - padding + kernel_size;
    
    // Clamp to valid range
    int shm_start = max(start_load_idx, 0);
    int shm_end = min(end_load_idx, input_length);

    // Cooperative loading into shared memory
    for (int i = threadIdx.x; i < (shm_end - shm_start); i += blockDim.x) {
        shm[i] = input[bc_idx * input_length + shm_start + i];
    }
    __syncthreads();

    // Computation
    int start_idx = output_idx * stride - padding;
    int end_idx = start_idx + kernel_size;
    
    scalar_t sum = 0.0;
    int count = 0;
    for (int k = start_idx; k < end_idx; ++k) {
        if (k >= 0 && k < input_length) {
            sum += shm[k - shm_start];
            count++;
        }
    }
    
    float divisor = count_include_pad ? (float)kernel_size : (float)count;
    output[bc_idx * output_length + output_idx] = sum / divisor;
}

void avg_pool1d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool ceil_mode,
    const bool count_include_pad) {
    
    auto sizes = input.sizes();
    int batch_channels = sizes[0] * sizes[1];
    int input_length = sizes[2];
    int output_length = output.size(2);
    
    int threads = 256;
    int blocks = (output_length + threads - 1) / threads;
    
    // Size of shared memory: max input range needed for a block
    int shm_size = (threads * stride + kernel_size) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool1d_forward", ([&] {
        avg_pool1d_kernel<scalar_t><<<dim3(blocks, batch_channels), threads, shm_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_channels,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding,
            count_include_pad
        );
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>
void avg_pool1d_forward(const torch::Tensor& input, torch::Tensor& output, const int kernel_size, const int stride, const int padding, const bool ceil_mode, const bool count_include_pad);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_forward", &avg_pool1d_forward, "AVG Pool 1D Forward");
}
"""

optimized_pool = load_inline(
    name='optimized_pool',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad):
    in_len = x.shape[2]
    if avg_pool_ceil_mode:
        out_len = ((in_len + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride) + 1
    else:
        out_len = ((in_len + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride) + 1
    out_len = max(out_len, 1)
    
    output = torch.empty((x.shape[0], x.shape[1], out_len), device=x.device, dtype=x.dtype)
    optimized_pool.avg_pool1d_forward(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad)
    return output
