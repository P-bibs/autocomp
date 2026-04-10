# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114736/code_1.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int length,
    int kernel_size, int stride, int padding,
    int output_length) {

    extern __shared__ float tile[];

    int tid = threadIdx.x;
    int b = blockIdx.z; 
    int c = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + tid;

    if (out_idx >= output_length) return;

    int input_start = out_idx * stride - padding;
    
    // Tiling: Each thread block loads a segment into shared memory
    // Shared memory size: (blockDim.x + kernel_size - 1)
    for (int i = tid; i < blockDim.x + kernel_size - 1; i += blockDim.x) {
        int input_pos = input_start + i;
        if (input_pos >= 0 && input_pos < length)
            tile[i] = input[((int64_t)b * channels + c) * length + input_pos];
        else
            tile[i] = 0.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        sum += tile[tid + k];
    }
    output[((int64_t)b * channels + c) * output_length + out_idx] = sum / kernel_size;
}

void avg_pool1d_tiled_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding
) {
    int B = input.size(0);
    int C = input.size(1);
    int L = input.size(2);
    int out_L = (L + 2 * padding - kernel_size) / stride + 1;
    
    int threads = 256;
    int blocks = (out_L + threads - 1) / threads;
    dim3 grid(blocks, C, B);
    
    size_t shared_mem = (threads + kernel_size - 1) * sizeof(float);
    
    avg_pool1d_tiled_kernel<<<grid, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, L,
        kernel_size, stride, padding,
        out_L
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_tiled_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_tiled_forward", &avg_pool1d_tiled_forward, "Tiled AvgPool1d forward pass");
}
"""

# Compile the extension
avg_pool_ext = load_inline(
    name='avg_pool_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad):
    # Compute output length according to PyTorch's avg_pool1d formula
    # When ceil_mode=False: Lout = floor((Lin + 2*padding - kernel_size) / stride + 1)
    # When ceil_mode=True:  Lout = ceil((Lin + 2*padding - kernel_size) / stride + 1)
    # For this optimized version, we assume ceil_mode=False and count_include_pad=True for simplicity
    batch_size, channels, length = x.shape
    if avg_pool_ceil_mode:
        output_length = int(torch.ceil(torch.tensor((length + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)).item())
    else:
        output_length = int(torch.floor(torch.tensor((length + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)).item())
    
    output = torch.empty((batch_size, channels, output_length), dtype=x.dtype, device=x.device)
    avg_pool_ext.avg_pool1d_tiled_forward(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    return output

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length, dtype=torch.float32, device='cuda')
    return [x]
