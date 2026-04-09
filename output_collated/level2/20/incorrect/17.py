# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int IC, int OC, int ID, int IH, int IW, 
    int OD, int OH, int OW, int K, int S, int P
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * OD * OH * OW;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int od = tmp % OD; tmp /= OD;
    int oc = tmp % OC; tmp /= OC;
    int b  = tmp;

    float acc = (conv_bias != nullptr) ? conv_bias[oc] : 0.0f;

    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < K; ++kd) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int id_in = od + P - kd;
                    int ih_in = oh + P - kh;
                    int iw_in = ow + P - kw;
                    if (id_in % S == 0 && ih_in % S == 0 && iw_in % S == 0) {
                        int id = id_in / S;
                        int ih = ih_in / S;
                        int iw = iw_in / S;
                        if (id >= 0 && id < ID && ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            float val = input[(((b * IC + ic) * ID + id) * IH + ih) * IW + iw];
                            float w = weight[(((oc * IC + ic) * K + kd) * K + kh) * K + kw];
                            acc += val * w;
                        }
                    }
                }
            }
        }
    }
    
    // Fused element-wise: original_x = acc; x = (acc + bias) + acc + acc*acc + acc
    float b_val = bias[oc];
    output[idx] = (acc + b_val) + acc + (acc * acc) + acc;
}

void launch_fused_conv(
    const torch::Tensor& input, const torch::Tensor& weight, 
    const torch::Tensor& conv_bias, const torch::Tensor& bias, 
    torch::Tensor& output, int S, int P
) {
    int B = input.size(0); int IC = input.size(1);
    int ID = input.size(2); int IH = input.size(3); int IW = input.size(4);
    int OC = weight.size(0); int K = weight.size(2);
    int OD = output.size(2); int OH = output.size(3); int OW = output.size(4);
    
    int total = B * OC * OD * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    const float* cb_ptr = conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr;
    
    conv_transpose3d_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), cb_ptr, bias.data_ptr<float>(),
        output.data_ptr<float>(), B, IC, OC, ID, IH, IW, OD, OH, OW, K, S, P
    );
}
"""

cpp_source = "void launch_fused_conv(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int, int);"

fused_lib = load_inline(
    name='fused_conv_transpose', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'], functions=['launch_fused_conv']
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Output dim calculation for ConvTranspose3d
    D_in, H_in, W_in = x.shape[2:]
    K = conv_transpose_weight.shape[2]
    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + K + conv_transpose_output_padding
    
    output = torch.empty((x.shape[0], conv_transpose_weight.shape[0], D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    fused_lib.launch_fused_conv(x, conv_transpose_weight, conv_transpose_bias, bias.view(-1), 
                                output, conv_transpose_stride, conv_transpose_padding)
    return output
