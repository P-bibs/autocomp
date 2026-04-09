# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_044115/code_8.py
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f;
}

__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int B, const int IC, const int OC,
    const int ID, const int IH, const int IW,
    const int OD, const int OH, const int OW,
    const int K, const int S, const int P) {

    // Global spatial index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int od = bz;
    int oh = by;
    int ow = bx;
    
    // Iterate over output channels
    for (int oc = 0; oc < OC; ++oc) {
        float val = bias[oc];
        
        // Compute ConvTranspose3D for this pixel (output channel oc, location od, oh, ow)
        for (int ic = 0; ic < IC; ++ic) {
            for (int kd = 0; kd < K; ++kd) {
                int id = od + P - kd;
                if (id < 0 || id >= ID * S || id % S != 0) continue;
                int ix = id / S;
                
                for (int kh = 0; kh < K; ++kh) {
                    int ih = oh + P - kh;
                    if (ih < 0 || ih >= IH * S || ih % S != 0) continue;
                    int iy = ih / S;

                    for (int kw = 0; kw < K; ++kw) {
                        int iw = ow + P - kw;
                        if (iw < 0 || iw >= IW * S || iw % S != 0) continue;
                        int iz = iw / S;
                        
                        // Input Index: [B, IC, ID, IH, IW]
                        int inp_idx = ((((0 * IC + ic) * ID + ix) * IH + iy) * IW + iz);
                        int wgt_idx = (((oc * IC + ic) * K + kd) * K + kh) * K + kw;
                        val += input[inp_idx] * weight[wgt_idx];
                    }
                }
            }
        }
        
        int out_idx = (((0 * OC + oc) * OD + od) * OH + oh) * OW + ow;
        float add_val = val + add_input[out_idx];
        output[out_idx] = hardswish_impl(add_val) * add_val;
    }
}

void launch_fused_conv_transpose3d_add_hardswish(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& add_input, at::Tensor& output,
    int K, int S, int P) {
    const int B = input.size(0);
    const int IC = input.size(1);
    const int ID = input.size(2), IH = input.size(3), IW = input.size(4);
    const int OC = weight.size(0);
    const int OD = output.size(2), OH = output.size(3), OW = output.size(4);

    dim3 grid(OW, OH, OD);
    fused_conv_transpose3d_add_hardswish_kernel<<<grid, 1>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        add_input.data_ptr<float>(), output.data_ptr<float>(),
        B, IC, OC, ID, IH, IW, OD, OH, OW, K, S, P
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv_transpose3d_add_hardswish(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_conv_transpose3d_add_hardswish, "Fused Op");
}
"""

module = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    B, IC, ID, IH, IW = x.shape
    OC, _, K, _, _ = conv_transpose_weight.shape
    OD = (ID - 1) * conv_transpose_stride + K - 2 * conv_transpose_padding + conv_transpose_output_padding
    OH = (IH - 1) * conv_transpose_stride + K - 2 * conv_transpose_padding + conv_transpose_output_padding
    OW = (IW - 1) * conv_transpose_stride + K - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    output = torch.empty((B, OC, OD, OH, OW), device=x.device)
    module.fused_op(x, conv_transpose_weight, conv_transpose_bias, add_input, output, K, conv_transpose_stride, conv_transpose_padding)
    return output
