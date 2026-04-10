# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140617/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_conv_subtract_mish_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    float subtract_value_1,
    float subtract_value_2) {
    
    extern __shared__ float shared_weight[];
    
    int out_y = blockIdx.y;
    int out_x = blockIdx.x;
    int out_c = blockIdx.z;
    int tid = threadIdx.x;
    
    if (out_c >= out_channels) return;
    
    int out_h = (height - kernel_size + 2 * padding) / stride + 1;
    int out_w = (width - kernel_size + 2 * padding) / stride + 1;
    
    if (out_y >= out_h || out_x >= out_w) return;
    
    // Each block processes one output channel
    // Use all threads in the block to process this output position
    
    float sum = (bias != nullptr && tid == 0) ? bias[out_c] : 0.0f;
    
    // Load weights into shared memory
    int weights_per_thread = (in_channels * kernel_size * kernel_size + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < weights_per_thread && (tid + i * blockDim.x) < (in_channels * kernel_size * kernel_size); i++) {
        int idx = tid + i * blockDim.x;
        if (idx < in_channels * kernel_size * kernel_size) {
            int weight_idx = out_c * in_channels * kernel_size * kernel_size + idx;
            shared_weight[idx] = weight[weight_idx];
        }
    }
    __syncthreads();
    
    // Perform convolution
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int in_y = out_y * stride + i - padding;
            int in_x = out_x * stride + j - padding;
            
            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                for (int in_c = tid; in_c < in_channels; in_c += blockDim.x) {
                    int input_idx = ((/*batch*/0 * in_channels + in_c) * height + in_y) * width + in_x;
                    int local_weight_idx = in_c * kernel_size * kernel_size + i * kernel_size + j;
                    sum += input[input_idx] * shared_weight[local_weight_idx];
                }
            }
        }
    }
    
    // Reduction across threads for this output position
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    if (tid == 0) {
        // Apply subtractions
        sum -= subtract_value_1;
        sum -= subtract_value_2;
        
        // Apply Mish activation
        float result = mish_activation(sum);
        
        // Write to output
        int output_idx = ((/*batch*/0 * out_channels + out_c) * out_h + out_y) * out_w + out_x;
        output[output_idx] = result;
    }
}

void fused_conv_subtract_mish_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride,
    int padding,
    float subtract_value_1,
    float subtract_value_2) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    
    const int out_h = (height - kernel_size + 2 * padding) / stride + 1;
    const int out_w = (width - kernel_size + 2 * padding) / stride + 1;
    
    // Set up CUDA context
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    dim3 grid(out_w, out_h, out_channels);
    dim3 block(256); // Use 256 threads per block for better occupancy
    
    const int shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);
    
    fused_conv_subtract_mish_kernel_opt<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        subtract_value_1,
        subtract_value_2
    );
    
    cudaDeviceSynchronize();
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_subtract_mish_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride,
    int padding,
    float subtract_value_1,
    float subtract_value_2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_subtract_mish_forward", &fused_conv_subtract_mish_forward, "Fused Convolution, Subtract, and Mish forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    subtract_value_1,
    subtract_value_2,
):
    # Ensure inputs are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not conv_weight.is_cuda:
        conv_weight = conv_weight.cuda()
    if not conv_bias.is_cuda:
        conv_bias = conv_bias.cuda()
    
    # Create output tensor
    out_h = (x.size(2) - conv_weight.size(2) + 2 * conv_padding) // conv_stride + 1
    out_w = (x.size(3) - conv_weight.size(3) + 2 * conv_padding) // conv_stride + 1
    output = torch.empty(x.size(0), conv_weight.size(0), out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_conv_subtract_mish_forward(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding,
        subtract_value_1, subtract_value_2
    )
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
