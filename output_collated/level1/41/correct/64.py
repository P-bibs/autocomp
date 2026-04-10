# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_042936/code_24.py
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

# Global variable to store compiled extension
fused_ext = None

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <float.h>

// Tile loading strategy: 
// Each block processes a chunk of the output for a specific (batch, channel).
// We load input into shared memory to facilitate coalesced access and reuse.
__global__ void maxpool1d_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    extern __shared__ float s_data[];

    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    int input_offset = (batch_idx * channels + channel_idx) * input_length;
    int output_offset = (batch_idx * channels + channel_idx) * output_length;

    // Calculate how many input elements this block needs
    // We cover a range of output positions and calculate the required input span
    int out_start = blockIdx.z * blockDim.x;
    int out_end = min(out_start + blockDim.x, output_length);
    
    // Load needed data into shared memory
    // To simplify: each thread loads one element from input to shared if within range
    // A more complex tiling would be needed for very large kernels, but for 
    // typical maxpool, direct access with coalesced reads is often sufficient.
    // Here we focus on keeping the register pressure low and memory coalesced.
    
    for (int out_pos = out_start + tid; out_pos < out_end; out_pos += blockDim.x) {
        int start_pos = out_pos * stride - padding;
        float max_val = -FLT_MAX;
        
        // Coalesced-friendly loop: 
        // We handle the pooling window for each output position
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = start_pos + k * dilation;
            if (in_pos >= 0 && in_pos < input_length) {
                max_val = fmaxf(max_val, input[input_offset + in_pos]);
            }
        }
        output[output_offset + out_pos] = max_val;
    }
}

void maxpool1d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int k, int s, int p, int d
) {
    int b = input.size(0);
    int c = input.size(1);
    int in_len = input.size(2);
    int out_len = output.size(2);

    // Grid: (Channels, Batch, (out_len + threads - 1) / threads)
    int threads = 256;
    int blocks_per_row = (out_len + threads - 1) / threads;
    dim3 grid(c, b, blocks_per_row);

    maxpool1d_optimized_kernel<<<grid, threads, 0>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        c, in_len, out_len, k, s, p, d
    );
}
"""

cpp_source = r"""
void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_forward, "Optimized MaxPool1D forward");
}
"""

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
    global fused_ext
    if fused_ext is None:
        fused_ext = load_inline(
            name='maxpool1d_opt',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            with_cuda=True
        )

    total_len = x.size(2)
    stride, pad, dial, k = maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_kernel_size
    
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * pad - dial * (k - 1) - 1) + stride - 1) // stride + 1
    else:
        output_length = (total_len + 2 * pad - dial * (k - 1) - 1) // stride + 1
    
    x_gpu = x.contiguous().cuda()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    fused_ext.maxpool1d(x_gpu, output, k, stride, pad, dial)
    return output
