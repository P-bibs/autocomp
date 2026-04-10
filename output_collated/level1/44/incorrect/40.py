# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_120345/code_5.py
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

# The custom CUDA kernel optimizes global memory access by loading input segments 
# into shared memory. This is particularly efficient for sliding window operations 
# where elements are reused across adjacent output positions.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                int in_len, int out_len, 
                                int kernel_size, int stride, int padding) {
    // Each block processes one channel of one batch element
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int channel_offset = blockIdx.x * in_len;
    
    // Cooperative loading of input into shared memory
    for (int i = tid; i < in_len; i += blockDim.x) {
        s_data[i] = input[channel_offset + i];
    }
    __syncthreads();

    // Compute output elements
    int output_offset = blockIdx.x * out_len;
    for (int i = tid; i < out_len; i += blockDim.x) {
        int start = i * stride - padding;
        float sum = 0.0f;
        int count = 0;
        
        // This loop utilizes the cached shared memory (s_data)
        for (int k = 0; k < kernel_size; ++k) {
            int idx = start + k;
            if (idx >= 0 && idx < in_len) {
                sum += s_data[idx];
                count++;
            }
        }
        output[output_offset + i] = (count > 0) ? (sum / (float)count) : 0.0f;
    }
}

void avg_pool_cuda(torch::Tensor input, torch::Tensor output, int ks, int stride, int padding) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_len = input.size(2);
    int out_len = output.size(2);
    
    int total_channels = batch_size * in_channels;
    
    // Launch one block per channel
    const dim3 threads(256);
    const dim3 blocks(total_channels);
    size_t shared_mem_size = in_len * sizeof(float);
    
    avg_pool_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        in_len, out_len, ks, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void avg_pool_cuda(torch::Tensor input, torch::Tensor output, int ks, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool_cuda", &avg_pool_cuda, "Optimized Average Pool 1D");
}
"""

# Compile extension
avg_pool_ext = load_inline(
    name='avg_pool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    # Calculate output dimension
    in_len = x.size(2)
    out_len = (in_len + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    output = torch.empty((x.size(0), x.size(1), out_len), device=x.device, dtype=x.dtype)
    
    # Custom kernel invocation
    avg_pool_ext.avg_pool_cuda(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    
    return output
