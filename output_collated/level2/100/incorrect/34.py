# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_132110/code_1.py
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

# Optimization: Fused CUDA kernel for Transpose Convolution + Clamp + Divide
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_transpose_conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int IC, int ID, int IH, int IW,
    int OC, int OD, int OH, int OW,
    int KD, int KH, int KW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    float min_value,
    float divisor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OC * OD * OH * OW;
    
    if (idx >= total_elements) return;
    
    int ow = idx % OW; idx /= OW;
    int oh = idx % OH; idx /= OH;
    int od = idx % OD; idx /= OD;
    int oc = idx % OC; idx /= OC;
    int n = idx;
    
    int group = oc / (OC / groups);
    int oc_in_group = oc % (OC / groups);
    
    float sum = bias[oc];
    
    for (int ic_group = 0; ic_group < (IC / groups); ++ic_group) {
        int ic = group * (IC / groups) + ic_group;
        
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    // Compute input position
                    int in_d = od * stride_d - pad_d + kd * dilation_d;
                    int in_h = oh * stride_h - pad_h + kh * dilation_h;
                    int in_w = ow * stride_w - pad_w + kw * dilation_w;
                    
                    if (in_d >= 0 && in_d < ID &&
                        in_h >= 0 && in_h < IH &&
                        in_w >= 0 && in_w < IW) {
                        
                        float in_val = input[(((n * IC + ic) * ID + in_d) * IH + in_h) * IW + in_w];
                        float w_val = weight[(((oc_in_group * (IC/groups) + ic_group) * KD + kd) * KH + kh) * KW + kw];
                        sum += in_val * w_val;
                    }
                }
            }
        }
    }
    
    // Clamp and divide
    sum = fmaxf(sum, min_value);
    sum = sum / divisor;
    
    output[(((n * OC + oc) * OD + od) * OH + oh) * OW + ow] = sum;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    float min_value,
    float divisor
) {
    int N = input.size(0);
    int IC = input.size(1);
    int ID = input.size(2);
    int IH = input.size(3);
    int IW = input.size(4);
    
    int OC = weight.size(0);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);
    
    int OD = output.size(2);
    int OH = output.size(3);
    int OW = output.size(4);
    
    int total_threads = N * OC * OD * OH * OW;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    fused_transpose_conv3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        min_value,
        divisor
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    float min_value,
    float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused 3D Transpose Convolution with Clamp and Divide");
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
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    min_value,
    divisor,
):
    # Calculate output dimensions
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    in_depth = x.shape[2]
    in_height = x.shape[3]
    in_width = x.shape[4]
    
    out_channels = conv_transpose_weight.shape[1]
    
    # Handle tuple or int stride/padding/output_padding
    if isinstance(conv_transpose_stride, int):
        stride_d = stride_h = stride_w = conv_transpose_stride
    else:
        stride_d, stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        pad_d = pad_h = pad_w = conv_transpose_padding
    else:
        pad_d, pad_h, pad_w = conv_transpose_padding
        
    if isinstance(conv_transpose_output_padding, int):
        out_pad_d = out_pad_h = out_pad_w = conv_transpose_output_padding
    else:
        out_pad_d, out_pad_h, out_pad_w = conv_transpose_output_padding
        
    if isinstance(conv_transpose_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    # Calculate output size using the formula for conv_transpose3d
    out_depth = (in_depth - 1) * stride_d - 2 * pad_d + dilation_d * (conv_transpose_weight.shape[2] - 1) + 1 + out_pad_d
    out_height = (in_height - 1) * stride_h - 2 * pad_h + dilation_h * (conv_transpose_weight.shape[3] - 1) + 1 + out_pad_h
    out_width = (in_width - 1) * stride_w - 2 * pad_w + dilation_w * (conv_transpose_weight.shape[4] - 1) + 1 + out_pad_w
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, out_depth, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call the fused kernel
    fused_ext.fused_op_forward(
        x, conv_transpose_weight, conv_transpose_bias, out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups,
        min_value,
        divisor
    )
    
    return out

# Provided values (to make the file complete)
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
