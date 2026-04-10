# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_132110/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# The custom CUDA implementation performs the 3D Transpose Convolution
# via a direct gather-based approach, fused with clamping and division
# to maximize memory bandwidth utilization and reduce kernel launch overhead.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_transpose_conv3d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int IC, int ID, int IH, int IW,
    int OC, int KD, int KH, int KW,
    int OD, int OH, int OW,
    int stride, int padding,
    float min_val, float divisor) {
    
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OC * OD * OH * OW;

    if (g_idx < total_elements) {
        int temp = g_idx;
        int ow = temp % OW; temp /= OW;
        int oh = temp % OH; temp /= OH;
        int od = temp % OD; temp /= OD;
        int oc = temp % OC; temp /= OC;
        int n = temp;

        float val = bias[oc];
        
        // Loop over input channels and kernel weights
        for (int ic = 0; ic < IC; ++ic) {
            for (int kd = 0; kd < KD; ++kd) {
                int id = od + padding - kd;
                if (id % stride == 0) {
                    int id_src = id / stride;
                    if (id_src >= 0 && id_src < ID) {
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh + padding - kh;
                            if (ih % stride == 0) {
                                int ih_src = ih / stride;
                                if (ih_src >= 0 && ih_src < IH) {
                                    for (int kw = 0; kw < KW; ++kw) {
                                        int iw = ow + padding - kw;
                                        if (iw % stride == 0) {
                                            int iw_src = iw / stride;
                                            if (iw_src >= 0 && iw_src < IW) {
                                                val += input[(((n * IC + ic) * ID + id_src) * IH + ih_src) * IW + iw_src] * 
                                                       weight[(((oc * IC + ic) * KD + kd) * KH + kh) * KW + kw];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output[g_idx] = fmaxf(val, min_val) / divisor;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, 
                      int stride, int padding, float min_val, float divisor) {
    const int N = x.size(0), IC = x.size(1), ID = x.size(2), IH = x.size(3), IW = x.size(4);
    const int OC = weight.size(0), KD = weight.size(2), KH = weight.size(3), KW = weight.size(4);
    const int OD = out.size(2), OH = out.size(3), OW = out.size(4);
    
    int total = N * OC * OD * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_transpose_conv3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, 
        stride, padding, min_val, divisor
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, 
                      int stride, int padding, float min_val, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transpose Conv3d + Clamp + Div");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, min_value, divisor):
    
    # Calculate dimensions for the outputs based on Transpose Conv 3D formula
    # OD = (ID - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    C_in, C_out = conv_transpose_weight.shape[0], conv_transpose_weight.shape[1]
    K_size = conv_transpose_weight.shape[2]
    
    device = x.device
    N, IC, ID, IH, IW = x.shape
    OD = (ID - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K_size - 1) + 1 + conv_transpose_output_padding[0]
    OH = (IH - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K_size - 1) + 1 + conv_transpose_output_padding[1]
    OW = (IW - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K_size - 1) + 1 + conv_transpose_output_padding[2]
    
    out = torch.empty((N, C_out, OD, OH, OW), device=device)
    
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, out, 
                       conv_transpose_stride, conv_transpose_padding, 
                       float(min_value), float(divisor))
    return out
