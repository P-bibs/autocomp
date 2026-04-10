# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115226/code_5.py
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

# The strategy uses a tiled approach. Given the memory bandwidth bound nature of 
# average pooling, we maximize throughput by letting threads cooperatively load 
# tiles into shared memory, minimizing global memory latency.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256
#define SHARED_MEM_SIZE 512

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input, float* __restrict__ output, 
    int batch_size, int channels, int in_len, 
    int kernel_size, int stride, int padding, 
    int out_len) {

    int tid = threadIdx.x;
    int ch_idx = blockIdx.y; 
    int b_idx = blockIdx.z;

    // Each block processes a stripe of the output sequence
    int out_start = blockIdx.x * TILE_SIZE;
    
    // Shared memory to cache input segment for the current channel
    __shared__ float s_data[SHARED_MEM_SIZE];

    const float* in_ptr = input + (b_idx * channels + ch_idx) * in_len;
    float* out_ptr = output + (b_idx * channels + ch_idx) * out_len;

    // Load necessary input range into shared memory
    int load_start = out_start * stride - padding;
    int load_end = (out_start + TILE_SIZE) * stride - padding + kernel_size;
    
    // Coalesced load into shared memory
    for (int i = tid + load_start; i < load_end; i += blockDim.x) {
        int smem_idx = i - load_start;
        if (smem_idx >= 0 && smem_idx < SHARED_MEM_SIZE) {
            if (i >= 0 && i < in_len)
                s_data[smem_idx] = in_ptr[i];
            else
                s_data[smem_idx] = 0.0f;
        }
    }
    __syncthreads();

    // Compute output
    if (out_start + tid < out_len) {
        int out_idx = out_start + tid;
        int start = out_idx * stride - padding;
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            int smem_pos = (start + i) - load_start;
            sum += s_data[smem_pos];
        }
        out_ptr[out_idx] = sum / kernel_size;
    }
}

void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, int ks, int st, int pad) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_len = input.size(2);
    const int out_len = output.size(2);

    dim3 blocks((out_len + TILE_SIZE - 1) / TILE_SIZE, channels, batch_size);
    dim3 threads(TILE_SIZE);

    avg_pool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, in_len, ks, st, pad, out_len
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, int ks, int st, int pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda, "Optimized 1D Avg Pool");
}
"""

fused_ext = load_inline(
    name='fused_pool',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, 
                     avg_pool_ceil_mode, avg_pool_count_include_pad):
    # Standard 1D pooling output shape calculation
    out_len = int(((x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride) + 1)
    output = torch.empty((x.size(0), x.size(1), out_len), device=x.device, dtype=x.dtype)
    
    fused_ext.avg_pool1d_cuda(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    return output
