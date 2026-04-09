# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040909/code_11.py
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

# CUDA kernel implementing a fused 3D Transposed Convolution, Bias Add, Addition, and HardSwish.
# We focus on a manual sliding-window approach that accumulates directly into registers.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add_input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int IC, int OC, 
    int iD, int iH, int iW,
    int oD, int oH, int oW,
    int kD, int kH, int kW,
    int strD, int strH, int strW,
    int padD, int padH, int padW) {

    int od = blockIdx.z; // Output depth index
    int oh = blockIdx.y; // Output height index
    int ow = blockIdx.x; // Output width index
    
    // Each thread in local block processes one output channel (or a subset)
    int oc = threadIdx.x; 
    
    if (oc >= OC) return;

    for (int n = 0; n < B; ++n) {
        float val = bias[oc];
        
        // Manual 3D loop for transposed convolution:
        // Input index is derived from output index using stride
        for (int ic = 0; ic < IC; ++ic) {
            for (int kd = 0; kd < kD; ++kd) {
                int id = (od + padD - kd);
                if (id % strD != 0) continue;
                id /= strD;
                if (id < 0 || id >= iD) continue;

                for (int kh = 0; kh < kH; ++kh) {
                    int ih = (oh + padH - kh);
                    if (ih % strH != 0) continue;
                    ih /= strH;
                    if (ih < 0 || ih >= iH) continue;

                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = (ow + padW - kw);
                        if (iw % strW != 0) continue;
                        iw /= strW;
                        if (iw < 0 || iw >= iW) continue;

                        float input_val = x[((n * IC + ic) * iD + id) * iH * iW + ih * iW + iw];
                        float weight_val = weight[((ic * OC + oc) * kD + kd) * kH * kW + kh * kW + kw];
                        val += input_val * weight_val;
                    }
                }
            }
        }
        
        // Fused Post-ops: Add input, then Hardswish
        int out_idx = (((n * OC + oc) * oD + od) * oH + oh) * oW + ow;
        val += add_input[out_idx];
        output[out_idx] = hardswish(val);
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor add_input, torch::Tensor weight, 
                      torch::Tensor bias, torch::Tensor output,
                      int strD, int strH, int strW, int padD, int padH, int padW) {
    int B = x.size(0); int IC = x.size(1);
    int iD = x.size(2); int iH = x.size(3); int iW = x.size(4);
    int OC = weight.size(1);
    int oD = output.size(2); int oH = output.size(3); int oW = output.size(4);
    int kD = weight.size(2); int kH = weight.size(3); int kW = weight.size(4);

    dim3 blocks(oW, oH, oD);
    dim3 threads(min(OC, 512));
    
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), add_input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), output.data_ptr<float>(),
        B, IC, OC, iD, iH, iW, oD, oH, oW, kD, kH, kW, strD, strH, strW, padD, padH, padW
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor add_input, torch::Tensor weight, 
                      torch::Tensor bias, torch::Tensor output,
                      int strD, int strH, int strW, int padD, int padH, int padW);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused 3D Transposed Conv + Add + HardSwish");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Weight shape expectation: [IC, OC, kD, kH, kW]
    B, IC, iD, iH, iW = x.shape
    OC = conv_transpose_weight.shape[1]
    strD, strH, strW = conv_transpose_stride
    padD, padH, padW = conv_transpose_padding
    
    oD = (iD - 1) * strD - 2 * padD + conv_transpose_weight.size(2) + conv_transpose_output_padding[0]
    oH = (iH - 1) * strH - 2 * padH + conv_transpose_weight.size(3) + conv_transpose_output_padding[1]
    oW = (iW - 1) * strW - 2 * padW + conv_transpose_weight.size(4) + conv_transpose_output_padding[2]
    
    output = torch.empty((B, OC, oD, oH, oW), device='cuda')
    
    fused_ext.fused_op(x, add_input, conv_transpose_weight, conv_transpose_bias, output,
                       strD, strH, strW, padD, padH, padW)
    return output
