# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_120345/code_1.py
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
#include <cmath>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad
) {
    // Shared memory for input tile
    extern __shared__ float shared_input[];
    
    int batch_idx = blockIdx.z / channels;
    int channel_idx = blockIdx.z % channels;
    int tid = threadIdx.x;
    
    // Calculate input and output offsets
    int input_batch_channel_offset = (batch_idx * channels + channel_idx) * input_length;
    int output_batch_channel_offset = (batch_idx * channels + channel_idx) * output_length;
    
    // Load input data into shared memory with padding handling
    for (int i = tid; i < input_length; i += blockDim.x) {
        shared_input[i] = input[input_batch_channel_offset + i];
    }
    __syncthreads();
    
    // Compute output elements
    for (int out_idx = blockIdx.x * blockDim.x + tid; out_idx < output_length; out_idx += blockDim.x * gridDim.x) {
        int start_idx = out_idx * stride - padding;
        int end_idx = start_idx + kernel_size;
        
        float sum = 0.0f;
        int count = 0;
        
        // Handle boundary conditions
        for (int k = start_idx; k < end_idx; k++) {
            if (k >= 0 && k < input_length) {
                sum += shared_input[k];
                count++;
            } else if (count_include_pad && k >= -padding && k < input_length + padding) {
                // Only count padding if count_include_pad is true
                count++;
            }
        }
        
        // Avoid division by zero
        if (count > 0) {
            output[output_batch_channel_offset + out_idx] = sum / (float)count;
        } else {
            output[output_batch_channel_offset + out_idx] = 0.0f;
        }
    }
}

void avg_pool1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    
    // Configure kernel launch parameters
    dim3 grid((output_length + 127) / 128, 1, batch_size * channels);
    dim3 block(128);
    size_t shared_mem_size = input_length * sizeof(float);
    
    avg_pool1d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda, "1D Average Pooling CUDA implementation");
}
"""

# Compile the extension
avg_pool_ext = load_inline(
    name='avg_pool_ext',
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
    # Calculate output length based on PyTorch's avg_pool1d formula
    if avg_pool_ceil_mode:
        output_length = int(torch.ceil(torch.tensor(
            (x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1
        )).item())
    else:
        output_length = int(torch.floor(torch.tensor(
            (x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1
        )).item())
    
    # Ensure non-negative output length
    output_length = max(0, output_length)
    
    # Create output tensor
    output = torch.empty(
        (x.size(0), x.size(1), output_length),
        dtype=x.dtype,
        device=x.device
    )
    
    # Call custom CUDA kernel
    avg_pool_ext.avg_pool1d_cuda(
        x, output,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding,
        avg_pool_ceil_mode,
        avg_pool_count_include_pad
    )
    
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
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]
