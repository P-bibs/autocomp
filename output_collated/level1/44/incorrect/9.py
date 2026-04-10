# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113951/code_5.py
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

# The CUDA kernel performs tiled loading into shared memory.
# It handles boundary checks and zero-padding, then computes the average.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_kernel(const float* input, float* output, 
                                int batch, int channels, int in_len, 
                                int kernel_size, int stride, int padding, 
                                int out_len) {
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int b = blockIdx.y;
    int c = blockIdx.z;
    
    // Each block processes a range of the output
    int block_out_start = blockIdx.x * blockDim.x;
    
    // Shared memory: sufficient space for the input window needed by the block
    // s_len = (blockDim.x - 1) * stride + kernel_size
    // This covers all elements touched by the threads in this block
    int s_len = (blockDim.x - 1) * stride + kernel_size;
    int input_offset_start = block_out_start * stride - padding;
    
    // Parallel loading into shared memory
    for (int i = tid; i < s_len; i += blockDim.x) {
        int idx = input_offset_start + i;
        if (idx >= 0 && idx < in_len)
            s_data[i] = input[(b * channels + c) * in_len + idx];
        else
            s_data[i] = 0.0f; // Implicit zero-padding
    }
    __syncthreads();

    // Compute average for each thread's assigned output index
    int out_idx = block_out_start + tid;
    if (out_idx < out_len) {
        float sum = 0.0f;
        int start_in = tid * stride;
        for (int i = 0; i < kernel_size; ++i) {
            sum += s_data[start_in + i];
        }
        output[(b * channels + c) * out_len + out_idx] = sum / (float)kernel_size;
    }
}

void fused_op_forward(torch::Tensor x, int ks, int st, int pad, torch::Tensor out) {
    int batch = x.size(0);
    int channels = x.size(1);
    int in_len = x.size(2);
    int out_len = out.size(2);
    
    const int threads = 256;
    int blocks_x = (out_len + threads - 1) / threads;
    dim3 grid(blocks_x, batch, channels);
    
    // Size of shared memory based on kernel and stride requirements
    size_t shared_size = ((threads - 1) * st + ks) * sizeof(float);
    
    avg_pool_kernel<<<grid, threads, shared_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), 
        batch, channels, in_len, ks, st, pad, out_len
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, int ks, int st, int pad, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized AvgPool1d");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_avg_pool',
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
    # Output length calculation based on PyTorch formula
    out_len = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    out = torch.zeros((x.shape[0], x.shape[1], out_len), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x.contiguous(), avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, out)
    return out
