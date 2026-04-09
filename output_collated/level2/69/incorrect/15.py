# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051208/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int out_height,
    int out_width
) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = out_height * out_width;
    
    if (spatial_idx >= total_spatial) return;
    
    int out_y = spatial_idx / out_width;
    int out_x = spatial_idx % out_width;
    
    int oc = blockIdx.y;
    int n = blockIdx.z;
    
    float accumulator = bias[oc];
    
    // Perform convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y + ky;
                int in_x = out_x + kx;
                
                float pixel_val = input[(((n * in_channels) + ic) * in_height + in_y) * in_width + in_x];
                float weight_val = weight[(((oc * in_channels) + ic) * kernel_size + ky) * kernel_size + kx];
                
                accumulator += pixel_val * weight_val;
            }
        }
    }
    
    // Fused activations: HardSwish + ReLU
    float hs_val = hardswish(accumulator);
    float final_val = fmaxf(hs_val, 0.0f);
    
    output[(((n * out_channels) + oc) * out_height + out_y) * out_width + out_x] = final_val;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = in_height - kernel_size + 1;
    const int out_width = in_width - kernel_size + 1;
    
    const int total_spatial = out_height * out_width;
    const int threads_per_block = 256;
    const int blocks_per_spatial = (total_spatial + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_spatial, out_channels, batch_size);
    dim3 block(threads_per_block);
    
    fused_conv_act_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        out_height,
        out_width
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv + HardSwish + ReLU forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
):
    # Validate assumptions (matching original configuration)
    assert conv_stride == 1, "Only stride=1 supported"
    assert conv_padding == 0, "Only padding=0 supported"
    assert conv_dilation == 1, "Only dilation=1 supported"
    assert conv_groups == 1, "Only groups=1 supported"
    
    batch_size, _, height, width = x.shape
    out_channels, _, kernel_size, _ = conv_weight.shape
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, out)
    return out

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
