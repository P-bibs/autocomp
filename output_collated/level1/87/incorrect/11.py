# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_070444/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Implementation ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv2d_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w
) {
    // Shared memory for input tile and weight
    extern __shared__ float shared_mem[];
    float* input_tile = shared_mem;
    float* weight_tile = shared_mem + 16 * 16 * 64; // Assuming max 16x16 threads per block and 64 input channels

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;

    if (out_x >= out_width || out_y >= out_height || out_ch >= out_channels) return;

    float sum = 0.0f;

    // Load bias into register
    if (bias && threadIdx.x == 0 && threadIdx.y == 0) {
        sum = bias[out_ch];
    }

    // Iterate over input channels
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        // Iterate over kernel elements
        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int in_x = out_x * stride_w - pad_w + kx * dilation_w;
                int in_y = out_y * stride_h - pad_h + ky * dilation_h;

                float val = 0.0f;
                if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                    int input_idx = ((/*batch*/ 0 * in_channels + in_ch) * in_height + in_y) * in_width + in_x;
                    val = input[input_idx];
                }

                int weight_idx = ((out_ch * in_channels + in_ch) * kernel_h + ky) * kernel_w + kx;
                float weight_val = weight[weight_idx];
                
                sum += val * weight_val;
            }
        }
    }

    // Write to output
    int output_idx = ((/*batch*/ 0 * out_channels + out_ch) * out_height + out_y) * out_width + out_x;
    output[output_idx] = sum;
}

void conv2d_fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    const int batch_size = input_sizes[0];
    const int in_channels = input_sizes[1];
    const int in_height = input_sizes[2];
    const int in_width = input_sizes[3];
    
    const int out_channels = weight_sizes[0];
    const int kernel_h = weight_sizes[2];
    const int kernel_w = weight_sizes[3];
    
    const int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Configure kernel launch parameters
    const int threads_x = 16;
    const int threads_y = 16;
    const int threads_z = 1;
    
    const int blocks_x = (out_width + threads_x - 1) / threads_x;
    const int blocks_y = (out_height + threads_y - 1) / threads_y;
    const int blocks_z = out_channels;
    
    const dim3 threads(threads_x, threads_y, threads_z);
    const dim3 blocks(blocks_x, blocks_y, blocks_z);
    
    // Shared memory size (simplified - could be optimized further based on actual usage)
    const int shared_mem_input = 16 * 16 * 64 * sizeof(float); // Tile size * channels
    const int shared_mem_weight = 3 * 3 * 128 * 64 * sizeof(float); // Max possible filter size
    const int shared_mem_size = shared_mem_input + shared_mem_weight;

    // Launch kernel for each sample in the batch
    for(int b = 0; b < batch_size; b++) {
        float* input_ptr = input.data_ptr<float>() + b * in_channels * in_height * in_width;
        float* output_ptr = output.data_ptr<float>() + b * out_channels * out_height * out_width;
        
        conv2d_fused_kernel<<<blocks, threads, shared_mem_size>>>(
            input_ptr,
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output_ptr,
            1, // Process one batch at a time
            in_channels,
            out_channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w
        );
    }
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void conv2d_fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_fused_forward", &conv2d_fused_forward, "Fused Conv2D forward pass");
}
"""

# Compile the extension
fused_conv_ext = load_inline(
    name='fused_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Global variables for model dimensions
batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    return [x]

# Optimized functional model using custom CUDA kernel
def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    # Ensure inputs are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not conv1d_weight.is_cuda:
        conv1d_weight = conv1d_weight.cuda()
    if conv1d_bias is not None and not conv1d_bias.is_cuda:
        conv1d_bias = conv1d_bias.cuda()
    
    # Handle stride, padding, dilation tuples
    if isinstance(conv1d_stride, int):
        stride_h, stride_w = conv1d_stride, conv1d_stride
    else:
        stride_h, stride_w = conv1d_stride[0], conv1d_stride[1]
        
    if isinstance(conv1d_padding, int):
        pad_h, pad_w = conv1d_padding, conv1d_padding
    else:
        pad_h, pad_w = conv1d_padding[0], conv1d_padding[1]
        
    if isinstance(conv1d_dilation, int):
        dilation_h, dilation_w = conv1d_dilation, conv1d_dilation
    else:
        dilation_h, dilation_w = conv1d_dilation[0], conv1d_dilation[1]
    
    # Calculate output dimensions
    in_height, in_width = x.shape[2], x.shape[3]
    kernel_h, kernel_w = conv1d_weight.shape[2], conv1d_weight.shape[3]
    
    out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(x.shape[0], conv1d_weight.shape[0], out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call custom CUDA kernel
    fused_conv_ext.conv2d_fused_forward(
        x, conv1d_weight, conv1d_bias if conv1d_bias is not None else torch.empty(0, device=x.device),
        output,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w
    )
    
    return output
