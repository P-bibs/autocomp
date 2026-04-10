# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_053509/code_23.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding', 'dilation']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

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
    # State for maxpool (nn.MaxPool2d)
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

# Optimized CUDA kernel with memory coalescing and minimal integer arithmetic
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void max_pool2d_coalesced_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int in_h, int in_w,
    int out_h, int out_w, int k, int s, int p, int d
) {
    // Map thread to output planar index: (batch * channels * out_h * out_w)
    // We use a 1D grid to ensure coalesced access where possible
    int total_output_elements = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_size = gridDim.y; 
    
    if (total_output_elements >= (channels * out_h * out_w)) return;

    int c = (total_output_elements / (out_h * out_w)) % channels;
    int h_out = (total_output_elements / out_w) % out_h;
    int w_out = total_output_elements % out_w;

    // Iterate over batches for the current spatial/channel output
    for (int n = 0; n < batch_size; ++n) {
        int h_start = h_out * s - p;
        int w_start = w_out * s - p;

        float max_val = -3.40282e+38F; // -FLT_MAX
        const float* input_ptr = input + (n * channels + c) * (in_h * in_w);

        #pragma unroll
        for (int i = 0; i < k; ++i) {
            int h_in = h_start + i * d;
            if (h_in >= 0 && h_in < in_h) {
                int row_offset = h_in * in_w;
                #pragma unroll
                for (int j = 0; j < k; ++j) {
                    int w_in = w_start + j * d;
                    if (w_in >= 0 && w_in < in_w) {
                        float val = input_ptr[row_offset + w_in];
                        if (val > max_val) max_val = val;
                    }
                }
            }
        }
        output[(n * channels * out_h * out_w) + (c * out_h * out_w) + (h_out * out_w) + w_out] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor input, torch::Tensor output, int k, int s, int p, int d) {
    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    int total_threads = channels * out_h * out_w;
    dim3 threads(256);
    dim3 blocks((total_threads + 255) / 256, batch);
    
    max_pool2d_coalesced_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        channels, in_h, in_w, out_h, out_w, k, s, p, d
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor input, torch::Tensor output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_forward", &max_pool2d_forward, "Coalesced Max Pool 2D");
}
"""

max_pool_ext = load_inline(
    name='max_pool2d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    k, s, p, d = maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    
    # Calculate output dimensions
    if maxpool_ceil_mode:
        OH = (x.shape[2] + 2 * p - d * (k - 1) - 1 + (s - 1)) // s + 1
        OW = (x.shape[3] + 2 * p - d * (k - 1) - 1 + (s - 1)) // s + 1
    else:
        OH = (x.shape[2] + 2 * p - d * (k - 1) - 1) // s + 1
        OW = (x.shape[3] + 2 * p - d * (k - 1) - 1) // s + 1
    
    out = torch.empty((x.shape[0], x.shape[1], OH, OW), device=x.device, dtype=x.dtype)
    max_pool_ext.max_pool2d_forward(x, out, k, s, p, d)
    
    if maxpool_return_indices:
        return out, torch.zeros_like(out, dtype=torch.long)
    return out
