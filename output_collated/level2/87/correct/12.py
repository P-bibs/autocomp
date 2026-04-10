# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_16.py
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
# Optimized CUDA kernel: Shared memory weight caching + Loop Unrolling
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch, const int in_c, const int in_h, const int in_w,
    const int out_c, const int k,
    const int out_h, const int out_w,
    const float sub1, const float sub2) {

    extern __shared__ float s_weight[];

    int oc = blockIdx.y;
    int b = blockIdx.z;

    // Load weights into shared memory for the current output channel
    int weight_size = in_c * k * k;
    for (int i = threadIdx.x; i < weight_size; i += blockDim.x) {
        s_weight[i] = weight[oc * weight_size + i];
    }
    __syncthreads();

    // Each thread calculates one output pixel
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_h * out_w) return;

    int oh = tid / out_w;
    int ow = tid % out_w;

    float acc = bias[oc];

    // Compute convolution with unrolled inner dimensions
    for (int ic = 0; ic < in_c; ++ic) {
        const float* input_ptr = &input[((b * in_c + ic) * in_h + oh) * in_w + ow];
        const float* weight_ptr = &s_weight[ic * k * k];
        
        #pragma unroll
        for (int i = 0; i < k; ++i) {
            #pragma unroll
            for (int j = 0; j < k; ++j) {
                acc += input_ptr[i * in_w + j] * weight_ptr[i * k + j];
            }
        }
    }

    // Fused activation: Mish(val) = val * tanh(softplus(val))
    float val = acc - sub1 - sub2;
    output[((b * out_c + oc) * out_h + oh) * out_w + ow] = val * tanhf(logf(1.0f + expf(val)));
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2) {
    const int batch = input.size(0);
    const int in_c = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_c = weight.size(0);
    const int k = weight.size(2);
    const int out_h = in_h - k + 1;
    const int out_w = in_w - k + 1;

    int threads = 256;
    int blocks_x = (out_h * out_w + threads - 1) / threads;
    dim3 grid(blocks_x, out_c, batch);
    
    // Shared memory size is dynamic based on kernel size
    size_t shared_size = in_c * k * k * sizeof(float);

    fused_conv_mish_kernel<<<grid, threads, shared_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k, 
        out_h, out_w, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_mish(torch::Tensor i, torch::Tensor w, torch::Tensor b, torch::Tensor o, float s1, float s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish, "Fused Convolution, Subtraction, and Mish");
}
"""

fused_ext = load_inline(
    name='fused_ext', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_source, 
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=0, 
                     conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    batch, _, h, w = x.shape
    k = conv_weight.shape[2]
    out_h, out_w = h - k + 1, w - k + 1
    out = torch.empty((batch, conv_weight.size(0), out_h, out_w), device=x.device, dtype=torch.float32)
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
