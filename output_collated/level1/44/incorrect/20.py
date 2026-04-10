# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114736/code_5.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void avg_pool1d_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int length,
    int kernel_size, int stride, int padding,
    int output_length) {

    extern __shared__ float tile[];

    int tid = threadIdx.x;
    int b = blockIdx.z; 
    int c = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + tid;

    if (out_idx >= output_length) return;

    // Load necessary segment into shared memory
    // Each thread block covers a segment of input length: (threads * stride + kernel_size - stride)
    int load_range = blockDim.x * stride + kernel_size - stride;
    int start_pos = blockIdx.x * blockDim.x * stride - padding;

    for (int i = tid; i < load_range; i += blockDim.x) {
        int input_pos = start_pos + i;
        if (input_pos >= 0 && input_pos < length)
            tile[i] = input[((int64_t)b * channels + c) * length + input_pos];
        else
            tile[i] = 0.0f;
    }
    __syncthreads();

    // Compute average for this thread's output index
    float sum = 0.0f;
    int local_start = tid * stride;
    for (int k = 0; k < kernel_size; ++k) {
        sum += tile[local_start + k];
    }
    output[((int64_t)b * channels + c) * output_length + out_idx] = sum / kernel_size;
}

torch::Tensor avg_pool1d_tiled(torch::Tensor x, int kernel_size, int stride, int padding) {
    int B = x.size(0), C = x.size(1), L = x.size(2);
    int out_L = (L + 2 * padding - kernel_size) / stride + 1;
    auto output = torch::zeros({B, C, out_L}, x.options());

    int threads = 256;
    int blocks_x = (out_L + threads - 1) / threads;
    dim3 grid(blocks_x, C, B);
    
    // Shared memory size calculation
    size_t shared_mem = (threads * stride + kernel_size) * sizeof(float);
    
    avg_pool1d_tiled_kernel<<<grid, threads, shared_mem>>>(
        x.data_ptr<float>(), output.data_ptr<float>(),
        B, C, L, kernel_size, stride, padding, out_L
    );
    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor avg_pool1d_tiled(torch::Tensor x, int kernel_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_tiled", &avg_pool1d_tiled, "Optimized Tiled 1D Average Pooling");
}
"""

# Compile the extension
avg_pool_ext = load_inline(
    name='avg_pool_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad):
    # Optimized kernel call
    return avg_pool_ext.avg_pool1d_tiled(x.contiguous(), avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
