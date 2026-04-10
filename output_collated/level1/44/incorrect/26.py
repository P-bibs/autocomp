# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115226/code_6.py
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

# 1. CUDA implementation of tiled average pooling
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define OUT_PER_BLOCK 256

__global__ void avg_pool1d_tile_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N_rows,
    int L,
    int L_out,
    int kernel,
    int stride,
    int padding) 
{
    // Shared memory: enough to hold the tile required for OUT_PER_BLOCK outputs
    // SMEM_LEN = (OUT_PER_BLOCK - 1) * stride + kernel
    extern __shared__ float smem[];

    int global_row = blockIdx.y;
    int block_start_out = blockIdx.x * OUT_PER_BLOCK;
    int block_start_in = block_start_out * stride - padding;

    // Load data into shared memory
    int smem_len = (OUT_PER_BLOCK - 1) * stride + kernel;
    for (int i = threadIdx.x; i < smem_len; i += blockDim.x) {
        int in_idx = block_start_in + i;
        if (in_idx >= 0 && in_idx < L) {
            smem[i] = x[global_row * L + in_idx];
        } else {
            smem[i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute phase
    for (int i = 0; i < OUT_PER_BLOCK; ++i) {
        int out_idx = block_start_out + i;
        if (out_idx < L_out) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < kernel; ++w) {
                sum += smem[i * stride + w];
            }
            y[global_row * L_out + out_idx] = sum / (float)kernel;
        }
    }
}

void avg_pool1d_tile(
    int N_rows, int L, int L_out, int kernel, int stride, int padding,
    torch::Tensor x, torch::Tensor y)
{
    dim3 blocks((L_out + OUT_PER_BLOCK - 1) / OUT_PER_BLOCK, N_rows);
    dim3 threads(256);
    
    // Shared memory size for the buffer: (OUT_PER_BLOCK - 1) * stride + kernel
    size_t smem_size = ((OUT_PER_BLOCK - 1) * stride + kernel) * sizeof(float);
    
    avg_pool1d_tile_kernel<<<blocks, threads, smem_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), 
        N_rows, L, L_out, kernel, stride, padding
    );
}
"""

cpp_source = r"""
void avg_pool1d_tile(int N_rows, int L, int L_out, int kernel, int stride, int padding, torch::Tensor x, torch::Tensor y);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_tile", &avg_pool1d_tile, "Tiled AvgPool1D");
}
"""

fused_ext = load_inline(
    name='avg_pool1d_tile',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def avg_pool1d_tile_wrapper(x, kernel, stride, padding):
    n, c, l = x.shape
    l_out = (l + 2 * padding - kernel) // stride + 1
    out = torch.empty(n, c, l_out, device=x.device, dtype=x.dtype)
    fused_ext.avg_pool1d_tile(n * c, l, l_out, kernel, stride, padding, x, out)
    return out

def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,
    avg_pool_count_include_pad,
):
    return avg_pool1d_tile_wrapper(
        x, 
        avg_pool_kernel_size, 
        avg_pool_stride, 
        avg_pool_padding
    )

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length).cuda()
    return [x]
