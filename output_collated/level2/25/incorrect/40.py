# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_085904/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

# Optimization: Fusing Conv2D, Min-reduction, and Tanh operations
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int out_channels
) {
    // Calculate output dimensions (assuming kernel=3, stride=1, padding=0)
    int out_height = height - 2;
    int out_width = width - 2;
    
    // Get global thread indices
    int batch_idx = blockIdx.x;
    int out_h = blockIdx.y;
    int out_w = blockIdx.z;
    
    if (batch_idx >= batch_size || out_h >= out_height || out_w >= out_width) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_min[];
    
    // Each thread computes one channel's convolution result
    int tid = threadIdx.x;
    int channels_per_thread = (out_channels + blockDim.x - 1) / blockDim.x;
    int start_channel = tid * channels_per_thread;
    int end_channel = min(start_channel + channels_per_thread, out_channels);
    
    float local_min = INFINITY;
    
    // Compute conv + bias for assigned channels
    for (int oc = start_channel; oc < end_channel; oc++) {
        float conv_result = bias[oc];
        
        // 3x3 convolution
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int in_h = out_h + kh;
                    int in_w = out_w + kw;
                    int in_idx = ((batch_idx * in_channels + ic) * height + in_h) * width + in_w;
                    int weight_idx = ((oc * in_channels + ic) * 3 + kh) * 3 + kw;
                    conv_result += input[in_idx] * weight[weight_idx];
                }
            }
        }
        
        // Apply first tanh
        conv_result = tanhf(conv_result);
        
        // Apply second tanh (redundant, but required by original code)
        conv_result = tanhf(conv_result);
        
        // Update local minimum
        local_min = fminf(local_min, conv_result);
    }
    
    // Store local result in shared memory
    shared_min[tid] = local_min;
    __syncthreads();
    
    // Reduce within warp first
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        int out_idx = ((batch_idx * 1 + 0) * out_height + out_h) * out_width + out_w;
        output[out_idx] = shared_min[0];
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    
    // Output dimensions for 3x3 kernel, stride=1, padding=0
    const int out_height = height - 2;
    const int out_width = width - 2;
    
    // Grid and block dimensions
    dim3 grid(batch_size, out_height, out_width);
    dim3 block(min(out_channels, 256));
    
    // Shared memory size
    size_t shared_mem_size = block.x * sizeof(float);
    
    fused_conv_min_tanh_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        out_channels
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused Conv2D + Min + Tanh forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
):
    # Validate that parameters match our fused kernel assumptions
    assert conv_stride == 1
    assert conv_padding == 0
    assert conv_dilation == 1
    assert conv_groups == 1
    assert conv_weight.size(2) == 3 and conv_weight.size(3) == 3
    
    batch_size, _, height, width = x.shape
    out_height = height - 2  # For 3x3 kernel with stride=1, padding=0
    out_width = width - 2
    output = torch.empty((batch_size, 1, out_height, out_width), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op_forward(x, conv_weight, conv_bias, output)
    return output

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
