# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_034611/code_5.py
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

# Shared memory optimized MaxPool1D
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define SHARED_MEM_SIZE 1024 

__global__ void max_pool1d_shared_kernel(const float* __restrict__ input, float* __restrict__ output,
                                         int batch, int features, int seq_len,
                                         int k_size, int stride, int padding, int dilation) {
    extern __shared__ float s_data[];

    int out_len = (seq_len + 2 * padding - (dilation * (k_size - 1) + 1)) / stride + 1;
    int b = blockIdx.z;
    int f = blockIdx.y;
    int tid = threadIdx.x;

    // Load input into shared memory (with padding)
    // Here we load a block's worth of data + required context
    // This simple approach treats the feature map as a stream and utilizes throughput
    for (int i = tid; i < seq_len; i += blockDim.x) {
        s_data[i] = input[b * features * seq_len + f * seq_len + i];
    }
    __syncthreads();

    for (int o = tid; o < out_len; o += blockDim.x) {
        int input_start = o * stride - padding;
        float max_val = -1e38f;

        for (int k = 0; k < k_size; ++k) {
            int in_idx = input_start + k * dilation;
            if (in_idx >= 0 && in_idx < seq_len) {
                float val = s_data[in_idx];
                if (val > max_val) max_val = val;
            }
        }
        output[b * (features * out_len) + f * out_len + o] = max_val;
    }
}

void max_pool1d_cuda(torch::Tensor input, torch::Tensor output, int k, int s, int p, int d) {
    int batch = input.size(0);
    int features = input.size(1);
    int seq_len = input.size(2);
    int out_len = output.size(2);
    
    dim3 grid(1, features, batch);
    int threads = 256;
    
    max_pool1d_shared_kernel<<<grid, threads, seq_len * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, features, seq_len, k, s, p, d);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool1d_cuda(torch::Tensor input, torch::Tensor output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool1d_cuda", &max_pool1d_cuda, "Optimized MaxPool1D for CUDA");
}
"""

max_pool_ext = load_inline(
    name='max_pool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    batch, feat, seq = x.shape
    # Calculate output length based on standard padding rules
    out_len = (seq + 2 * maxpool_padding - (maxpool_dilation * (maxpool_kernel_size - 1) + 1)) // maxpool_stride + 1
    output = torch.empty((batch, feat, out_len), device=x.device)
    
    max_pool_ext.max_pool1d_cuda(
        x, output, 
        maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    )
    return output
