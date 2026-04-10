# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_045555/code_17.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// Shared memory max pooling kernel
__global__ void maxpool1d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int input_length,
    const int output_length,
    const int k,
    const int s,
    const int p,
    const int d
) {
    // Dynamic shared memory buffer
    extern __shared__ float s_data[];

    const int batch_idx = blockIdx.y;
    const int channel_idx = blockIdx.x;
    
    // Total elements to process in parallel
    const int base_out_pos = blockIdx.z * blockDim.x;
    
    // Calculate input range required for this block
    const int start_in_idx = max(0, (int)(base_out_pos * s) - p);
    const int end_in_idx = min(input_length, (int)((base_out_pos + blockDim.x - 1) * s - p + (k - 1) * d + 1));
    
    // Coalesced load into shared memory
    for (int i = threadIdx.x; i < (end_in_idx - start_in_idx); i += blockDim.x) {
        s_data[i] = input[(batch_idx * channels + channel_idx) * input_length + start_in_idx + i];
    }
    __syncthreads();

    int out_pos = base_out_pos + threadIdx.x;
    if (out_pos < output_length) {
        float max_val = -3.402823466e+38F;
        int start_pos = out_pos * s - p;
        
        #pragma unroll
        for (int i = 0; i < k; ++i) {
            int in_pos = start_pos + i * d;
            if (in_pos >= start_in_idx && in_pos < end_in_idx) {
                float val = s_data[in_pos - start_in_idx];
                if (val > max_val) max_val = val;
            }
        }
        output[(batch_idx * channels + channel_idx) * output_length + out_pos] = max_val;
    }
}

void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d) {
    const int b = input.size(0);
    const int c = input.size(1);
    const int in_l = input.size(2);
    const int out_l = output.size(2);

    const int threads = 256;
    const int blocks_per_channel = (out_l + threads - 1) / threads;
    
    // Max input needed by a block: (threads-1)*s + (k-1)*d + 1
    // We limit shared memory usage to safe device bounds (e.g., 48KB)
    int shared_size = (threads * s + (k + 1) * d) * sizeof(float);
    
    dim3 grid(c, b, blocks_per_channel);
    maxpool1d_shared_kernel<<<grid, threads, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), c, in_l, out_l, k, s, p, d
    );
}
"""

cpp_source = r"""
void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_forward, "Optimized MaxPool1D");
}
"""

fused_ext = load_inline(
    name='maxpool1d_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    total_len = x.size(2)
    # Calculate output length
    if maxpool_ceil_mode:
        output_length = (total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = (total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    x_gpu = x.cuda().contiguous()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    fused_ext.maxpool1d(x_gpu, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    return output
