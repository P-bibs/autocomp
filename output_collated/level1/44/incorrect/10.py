# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113951/code_7.py
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

# ----------------------------------------------------------------------
# CUDA kernel + host code
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad)
{
    // Shared memory: tile_width * stride + kernel_size - 1
    extern __shared__ float sdata[];

    const int block_id = blockIdx.x; // maps to batch * channels
    const int tile_width = blockDim.x;
    
    const int input_base = block_id * input_length;
    const int output_base = block_id * output_length;

    // Load tile into shared memory
    const int tile_start = blockIdx.x * tile_width * stride - padding;
    const int needed = tile_width * stride + kernel_size - 1;

    for (int i = threadIdx.x; i < needed; i += blockDim.x) {
        int idx = tile_start + i;
        if (idx >= 0 && idx < input_length) {
            sdata[i] = input[input_base + idx];
        } else {
            sdata[i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute output
    const int out_i = blockIdx.x * tile_width + threadIdx.x;
    if (out_i < output_length) {
        float sum = 0.0f;
        int valid_count = 0;
        
        for (int k = 0; k < kernel_size; ++k) {
            int local_idx = threadIdx.x * stride + k;
            sum += sdata[local_idx];
            
            // Re-verify valid range for division if count_include_pad is false
            if (!count_include_pad) {
                int global_idx = out_i * stride - padding + k;
                if (global_idx >= 0 && global_idx < input_length) {
                    valid_count++;
                }
            }
        }

        if (count_include_pad) {
            output[output_base + out_i] = sum / (float)kernel_size;
        } else {
            output[output_base + out_i] = (valid_count > 0) ? (sum / (float)valid_count) : 0.0f;
        }
    }
}

void avg_pool1d_forward(
    at::Tensor input,
    at::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad)
{
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length = output.size(2);
    
    const int threads = 128;
    const int blocks = batch * channels;
    const int shared_size = (threads * stride + kernel_size - 1) * sizeof(float);

    avg_pool1d_kernel<<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, input_length, output_length,
        kernel_size, stride, padding, count_include_pad
    );
}
"""

cpp_source = r"""
void avg_pool1d_forward(
    at::Tensor input, at::Tensor output, int kernel_size, int stride, 
    int padding, bool ceil_mode, bool count_include_pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d", &avg_pool1d_forward, "1D Avg Pool Forward");
}
"""

fused_ext = load_inline(
    name='avg_pool_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad):
    x = x.contiguous().cuda()
    batch, channels, in_len = x.shape
    
    temp = in_len + 2 * avg_pool_padding - avg_pool_kernel_size
    if avg_pool_ceil_mode:
        out_len = (temp + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        out_len = temp // avg_pool_stride + 1
        
    out = torch.empty((batch, channels, out_len), device='cuda')
    fused_ext.avg_pool1d(x, out, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad)
    return out
