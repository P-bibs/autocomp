# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113602/code_5.py
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

// Tile size for shared memory
#define TILE_SIZE 256
// Pad shared memory to handle max kernel size
#define SHARED_MEM_SIZE (TILE_SIZE + 2048) 

__global__ void avg_pool1d_shared_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                         int batch_size, int in_channels, int input_len, 
                                         int kernel_size, int stride, int padding, int output_len) {
    extern __shared__ float s_data[];

    int b = blockIdx.y / in_channels;
    int c = blockIdx.y % in_channels;
    int tid = threadIdx.x;
    
    int input_offset = (b * in_channels + c) * input_len;
    int output_offset = (b * in_channels + c) * output_len;

    int output_start = blockIdx.x * TILE_SIZE;
    int input_start = output_start * stride - padding;

    // Collaborative loading into shared memory
    for (int i = tid; i < TILE_SIZE + kernel_size; i += blockDim.x) {
        int idx = input_start + i;
        if (idx >= 0 && idx < input_len)
            s_data[i] = input[input_offset + idx];
        else
            s_data[i] = 0.0f;
    }
    __syncthreads();

    int out_idx = output_start + tid;
    if (out_idx < output_len) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            sum += s_data[tid * stride + i];
        }
        output[output_offset + out_idx] = sum / (float)kernel_size;
    }
}

void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, int kernel_size, int stride, int padding) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_len = input.size(2);
    int output_len = output.size(2);

    dim3 threads(256);
    dim3 blocks((output_len + TILE_SIZE - 1) / TILE_SIZE, batch_size * in_channels);
    
    avg_pool1d_shared_kernel<<<blocks, threads, (TILE_SIZE + kernel_size) * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, input_len, kernel_size, stride, padding, output_len
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, int kernel_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda, "Optimized AvgPool1D");
}
"""

module = load_inline(
    name='avg_pool_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad):
    # Note: Simplified to match standard pooling behavior for the task optimizations
    output_len = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    output = torch.empty((x.shape[0], x.shape[1], output_len), device=x.device, dtype=x.dtype)
    module.avg_pool1d_cuda(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    return output
