# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_045555/code_15.py
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

# Optimized CUDA kernel with grid-stride loops (each thread handles multiple output elements)
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool1d_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Grid-stride loop: each thread handles multiple output elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes multiple output elements
    for (int out_idx = tid; out_idx < output_length; out_idx += total_threads) {
        int bf_idx = blockIdx.y;
        
        int start_pos = out_idx * stride - padding;
        float max_val = -1e38f; 

        #pragma unroll 4
        for (int k = 0; k < kernel_size; ++k) {
            int pos = start_pos + k * dilation;
            if (pos >= 0 && pos < input_length) {
                float val = input[bf_idx * input_length + pos];
                if (val > max_val) max_val = val;
            }
        }
        output[bf_idx * output_length + out_idx] = max_val;
    }
}

void maxpool1d_cuda(torch::Tensor input, torch::Tensor output, int k, int s, int p, int d) {
    int batch = input.size(0);
    int feat = input.size(1);
    int out_len = output.size(2);
    
    // Optimize for RTX 2080Ti with fewer blocks and more work per thread
    int total_threads = 256 * 64; // 64 blocks * 256 threads = 16384 threads
    dim3 threads(256);
    dim3 blocks(64, batch * feat);
    
    maxpool1d_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), 
        feat, input.size(2), out_len, k, s, p, d
    );
}
"""

cpp_source = r"""
void maxpool1d_cuda(torch::Tensor input, torch::Tensor output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_cuda, "Optimized MaxPool1d CUDA");
}
"""

_maxpool_ext = load_inline(
    name='maxpool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    if maxpool_return_indices:
        raise NotImplementedError("Indices not supported")
    
    x = x.contiguous()
    batch, feat, seq_len = x.shape
    
    # Standard formula for MaxPool output size
    if maxpool_ceil_mode:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride - 1) // maxpool_stride + 1
    else:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
        
    output = torch.empty((batch, feat, out_len), device=x.device, dtype=x.dtype)
    
    _maxpool_ext.maxpool1d(
        x, output, int(maxpool_kernel_size), int(maxpool_stride), 
        int(maxpool_padding), int(maxpool_dilation)
    )
    
    return output
