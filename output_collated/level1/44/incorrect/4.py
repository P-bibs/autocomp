# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113602/code_7.py
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

# -------------------------------------------------------------------------
# CUDA kernel for optimized 1D Average Pooling
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int in_len,
    const int out_len,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad)
{
    extern __shared__ float tile[];

    const int tid = threadIdx.x;
    const int b = blockIdx.y; 
    const int c = blockIdx.z;
    const int out_idx = blockIdx.x * blockDim.x + tid;

    if (out_idx >= out_len) return;

    // Load necessary input segment into shared memory
    // Each thread block handles BLOCK_SIZE output elements
    // Input segment size: blockDim.x + kernel_size - 1
    const int base_in_idx = blockIdx.x * blockDim.x * stride - padding;
    
    // Cooperative load of tile
    for (int i = tid; i < blockDim.x + kernel_size - 1; i += blockDim.x) {
        int idx = base_in_idx + i;
        if (idx >= 0 && idx < in_len) {
            tile[i] = input[(b * channels + c) * in_len + idx];
        } else {
            tile[i] = 0.0f;
        }
    }
    __syncthreads();

    // Computation
    float sum = 0.0f;
    int count = 0;
    for (int k = 0; k < kernel_size; ++k) {
        int pos = out_idx * stride - padding + k;
        if (pos >= 0 && pos < in_len) {
            sum += tile[tid + k];
            count++;
        } else if (count_include_pad) {
            count++; // even if out of bounds, count as pad
        }
    }

    if (count_include_pad) {
        output[(b * channels + c) * out_len + out_idx] = sum / (float)kernel_size;
    } else {
        output[(b * channels + c) * out_len + out_idx] = (count > 0) ? (sum / (float)count) : 0.0f;
    }
}

void fused_op_forward(
    const torch::Tensor &input,
    torch::Tensor &output,
    int kernel_size, int stride, int padding, bool count_include_pad)
{
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_len = input.size(2);
    const int out_len = output.size(2);
    
    const int BLOCK_SIZE = 256;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((out_len + BLOCK_SIZE - 1) / BLOCK_SIZE, batch, channels);
    size_t shared_mem = (BLOCK_SIZE + kernel_size) * sizeof(float);

    avg_pool1d_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        channels, in_len, out_len, kernel_size, stride, padding, count_include_pad
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor &input, torch::Tensor &output, int kernel_size, int stride, int padding, bool count_include_pad);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized AvgPool1d");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
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
    x = x.contiguous().cuda()
    batch, channels, in_len = x.shape
    
    if avg_pool_ceil_mode:
        out_len = (in_len + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        out_len = (in_len + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
        
    output = torch.empty(batch, channels, out_len, device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_count_include_pad)
    return output
