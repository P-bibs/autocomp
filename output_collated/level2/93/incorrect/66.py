# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160727/code_8.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose2d_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, int B, int IC, int OC, int IH, int IW, int OH, int OW,
    int KH, int KW, int stride, int padding, float add_val, float mul_val) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * OH * OW;
    if (tid >= total_elements) return;

    int tmp = tid;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int oc = tmp % OC; tmp /= OC;
    int b = tmp;

    float acc = bias[oc];
    
    // Reverse mapping for Transposed Conv
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = (oh + padding - kh);
                int iw = (ow + padding - kw);
                
                if (ih >= 0 && ih < IH * stride && ih % stride == 0 &&
                    iw >= 0 && iw < IW * stride && iw % stride == 0) {
                    
                    int in_h = ih / stride;
                    int in_w = iw / stride;
                    
                    int in_idx = ((b * IC + ic) * IH + in_h) * IW + in_w;
                    // Weight layout: [IC, OC, KH, KW] assumed for direct access, 
                    // or permuted. Standard torch weight is [IC, OC, KH, KW]
                    int w_idx = ((ic * OC + oc) * KH + kh) * KW + kw;
                    
                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    float val = acc + add_val;
    val = fminf(val, 0.0f);
    val = fast_gelu(val);
    output[tid] = val * mul_val;
}

void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                       int stride, int padding, float add_val, float mul_val) {
    int B = input.size(0); int IC = input.size(1);
    int IH = input.size(2); int IW = input.size(3);
    int OC = weight.size(1);
    int KH = weight.size(2); int KW = weight.size(3);
    int OH = output.size(2); int OW = output.size(3);

    int threads = 256;
    int blocks = (B * OC * OH * OW + threads - 1) / threads;
    fused_conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        B, IC, OC, IH, IW, OH, OW, KH, KW, stride, padding, add_val, mul_val
    );
}
"""

cpp_source = r"""
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                       int stride, int padding, float add_val, float mul_val);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_conv, "Fused Transposed Conv + Activation");
}
"""

fused_ext = load_inline(name='fused_conv', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    # Dimensions: Weight is [IC, OC, KH, KW]
    IC, OC, KH, KW = conv_transpose_weight.shape
    B, _, IH, IW = x.shape
    OH = (IH - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KH + conv_transpose_output_padding
    OW = (IW - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KW + conv_transpose_output_padding
    
    out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.fused_conv(x, conv_transpose_weight, conv_transpose_bias, out, 
                         conv_transpose_stride, conv_transpose_padding, float(add_value), float(multiply_value))
    return out
