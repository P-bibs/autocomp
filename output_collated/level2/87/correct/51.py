# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145317/code_30.py
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
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA kernel source – Optimization: Shared Memory Weight Tiling
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

extern __shared__ float weight_cache[];

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int out_c,
    const int k,
    const float sub1,
    const float sub2)
{
    const int oc = blockIdx.y;
    const int b  = blockIdx.z;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_threads = blockDim.x * blockDim.y;
    const int weight_size = k * k * in_c;

    // Load weight tile for this specific output channel into shared memory
    const float* weight_ptr = weight + (oc * weight_size);
    for (int idx = tid; idx < weight_size; idx += block_threads) {
        weight_cache[idx] = weight_ptr[idx];
    }
    __syncthreads();

    // Compute pixel
    const int spatial_idx = blockIdx.x * block_threads + tid;
    if (spatial_idx >= out_h * out_w) return;

    const int oh = spatial_idx / out_w;
    const int ow = spatial_idx % out_w;

    float acc = bias[oc];
    const float* i_base = input + (b * in_c * in_h * in_w);

    // Convolve
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            const int offset = (oh + i) * in_w + (ow + j);
            const int w_base = (i * k + j) * in_c;
            for (int ic = 0; ic < in_c; ++ic) {
                acc += i_base[ic * in_h * in_w + offset] * weight_cache[w_base + ic];
            }
        }
    }

    // Mish Activation
    const float val = acc - sub1 - sub2;
    const float out_val = val * tanhf(logf(1.0f + expf(val)));
    
    output[((b * out_c + oc) * out_h + oh) * out_w + ow] = out_val;
}

void launch_fused_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                         torch::Tensor output, float sub1, float sub2) {
    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weight.size(0);
    const int k = weight.size(2);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    const int threads = 256;
    const int blocks_x = (out_h * out_w + threads - 1) / threads;
    dim3 grid(blocks_x, out_c, batch);
    dim3 block(32, 8);

    size_t shared_size = k * k * in_c * sizeof(float);

    fused_conv_mish_kernel<<<grid, block, shared_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), in_c, in_h, in_w, out_h, out_w, out_c, k, sub1, sub2);
}
"""

cpp_source = r"""
void launch_fused_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                         torch::Tensor output, float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_kernel, "Fused Conv Mish");
}
"""

module = load_inline(
    name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    # Prepare data for kernel: [OC, H, W, IC] layout for coalesced weight reading
    w_reordered = conv_weight.permute(0, 2, 3, 1).contiguous()
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out = torch.empty((batch, conv_weight.size(0), h - k + 1, w - k + 1), device=x.device)
    
    module.fused_conv(x, w_reordered, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
