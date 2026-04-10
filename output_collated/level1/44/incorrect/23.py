# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115226/code_1.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_len,
    int kernel_size,
    int stride,
    int padding,
    int out_len
) {
    // Shared memory for the tile
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int ch_id = blockIdx.y;
    int batch_id = blockIdx.z;

    if (ch_id >= channels || batch_id >= batch_size) return;

    // Input and output pointers for this channel/batch
    const float* input_ptr = input + (batch_id * channels + ch_id) * in_len;
    float* output_ptr = output + (batch_id * channels + ch_id) * out_len;

    // Process multiple output elements per block to increase occupancy
    for (int out_start = blockIdx.x * blockDim.x; out_start < out_len; out_start += gridDim.x * blockDim.x) {
        int out_idx = out_start + tid;
        
        if (out_idx < out_len) {
            int start = out_idx * stride - padding;
            int end = start + kernel_size;
            
            // Bounds check
            int s = max(0, start);
            int e = min(in_len, end);
            
            float sum = 0.0f;
            for (int i = s; i < e; ++i) {
                sum += input_ptr[i];
            }
            
            output_ptr[out_idx] = sum / kernel_size;
        }
    }
}

void avg_pool1d_tiled_cuda(torch::Tensor input, torch::Tensor output, int ks, int st, int pad) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_len = input.size(2);
    int out_len = output.size(2);

    // Launch configuration
    const int threads_per_block = 256;
    const int elements_per_block = threads_per_block;
    const int num_blocks_x = (out_len + elements_per_block - 1) / elements_per_block;
    const int num_blocks_y = channels;
    const int num_blocks_z = batch_size;
    
    dim3 block(threads_per_block);
    dim3 grid(num_blocks_x, num_blocks_y, num_blocks_z);

    avg_pool1d_tiled_kernel<<<grid, block, 0>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_len,
        ks,
        st,
        pad,
        out_len
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_tiled_cuda(torch::Tensor input, torch::Tensor output, int ks, int st, int pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_tiled", &avg_pool1d_tiled_cuda, "1D Average Pooling with Tiling");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_pool',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, 
                     avg_pool_ceil_mode, avg_pool_count_include_pad):
    # Calculate output size manually to match avg_pool1d behavior
    if avg_pool_ceil_mode:
        out_len = int(torch.ceil(torch.tensor((x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)).item())
    else:
        out_len = int((x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)
    
    output = torch.empty((x.size(0), x.size(1), out_len), device=x.device, dtype=x.dtype)
    
    fused_ext.avg_pool1d_tiled(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
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
    x = torch.rand(batch_size, in_channels, input_length, device='cuda')
    return [x]
