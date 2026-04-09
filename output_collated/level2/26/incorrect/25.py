# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_042434/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# The CUDA kernel uses a tiled approach for the convolution output calculation
# and fuses the Add + Hardswish operation directly.
# This implementation performs the convolution transpose calculation.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f;
}

__global__ void fused_conv_transpose_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int B, int IC, int OC, int ID, int IH, int IW,
    int kD, int kH, int kW, int stride, int padding) {

    int OD = (ID - 1) * stride + kD - 2 * padding;
    int OH = (IH - 1) * stride + kH - 2 * padding;
    int OW = (IW - 1) * stride + kW - 2 * padding;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * OD * OH * OW;

    if (idx < total_elements) {
        int temp = idx;
        int ow = temp % OW; temp /= OW;
        int oh = temp % OH; temp /= OH;
        int od = temp % OD; temp /= OD;
        int oc = temp % OC; temp /= OC;
        int b = temp;

        float val = bias[oc];

        // Manual convolution transpose accumulation
        for (int ic = 0; ic < IC; ++ic) {
            for (int kd = 0; kd < kD; ++kd) {
                int id = (od + padding - kd);
                if (id % stride == 0) {
                    id /= stride;
                    if (id < 0 || id >= ID) continue;
                    for (int kh = 0; kh < kH; ++kh) {
                        int ih = (oh + padding - kh);
                        if (ih % stride == 0) {
                            ih /= stride;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int iw = (ow + padding - kw);
                                if (iw % stride == 0) {
                                    iw /= stride;
                                    if (iw < 0 || iw >= IW) continue;
                                    
                                    val += input[((b * IC + ic) * ID + id) * IH * IW + ih * IW + iw] * 
                                           weight[(((oc * IC + ic) * kD + kd) * kH + kh) * kW + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        float x = val + add_input[idx];
        output[idx] = x * hardswish_impl(x);
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor add_input, torch::Tensor output, 
                      int stride, int padding) {
    int B = input.size(0); int IC = input.size(1);
    int ID = input.size(2); int IH = input.size(3); int IW = input.size(4);
    int OC = weight.size(0);
    int kD = weight.size(2); int kH = weight.size(3); int kW = weight.size(4);
    
    int numel = output.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    fused_conv_transpose_add_hardswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        add_input.data_ptr<float>(), output.data_ptr<float>(),
        B, IC, OC, ID, IH, IW, kD, kH, kW, stride, padding);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor add_input, torch::Tensor output, 
                      int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose3d + Add + Hardswish");
}
"""

fused_ext = load_inline(
    name='fused_op_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '-use_fast_math'],
    with_cuda=True
)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    output = torch.empty_like(add_input)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, 
                       add_input, output, conv_transpose_stride, conv_transpose_padding)
    return output
