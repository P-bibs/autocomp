# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_125214/code_1.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Ci, int Co, 
    int D, int H, int W,
    int OD, int OH, int OW,
    int KD, int KH, int KW,
    int SD, int SH, int SW,
    int PD, int PH, int PW,
    float min_val,
    float divisor
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * Co * OD * OH * OW;

    if (tid >= total_threads) return;

    int b  = tid / (Co * OD * OH * OW);
    int c  = (tid / (OD * OH * OW)) % Co;
    int od = (tid / (OH * OW)) % OD;
    int oh = (tid / OW) % OH;
    int ow = tid % OW;

    float acc = bias[c];

    for (int ci = 0; ci < Ci; ci++) {
        for (int kd = 0; kd < KD; kd++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int id = od + PD - kd;
                    int ih = oh + PH - kh;
                    int iw = ow + PW - kw;

                    if (id >= 0 && ih >= 0 && iw >= 0 &&
                        id < D * SD && ih < H * SH && iw < W * SW &&
                        id % SD == 0 && ih % SH == 0 && iw % SW == 0) {
                        
                        int in_d = id / SD;
                        int in_h = ih / SH;
                        int in_w = iw / SW;

                        if (in_d < D && in_h < H && in_w < W) {
                            float val = input[(((b * Ci + ci) * D + in_d) * H + in_h) * W + in_w];
                            float wgt = weight[(((ci * Co + c) * KD + kd) * KH + kh) * KW + kw];
                            acc += val * wgt;
                        }
                    }
                }
            }
        }
    }

    output[tid] = fmaxf(acc, min_val) / divisor;
}

void launch_fused_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int SD, int SH, int SW,
    int PD, int PH, int PW,
    float min_val,
    float divisor
) {
    int B = input.size(0);
    int Ci = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int Co = weight.size(1);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);

    int OD = (D - 1) * SD + KD - 2 * PD;
    int OH = (H - 1) * SH + KH - 2 * PH;
    int OW = (W - 1) * SW + KW - 2 * PW;

    int total_elements = B * Co * OD * OH * OW;
    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, Ci, Co, D, H, W, OD, OH, OW, KD, KH, KW,
        SD, SH, SW, PD, PH, PW,
        min_val, divisor
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_fused_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int SD, int SH, int SW,
    int PD, int PH, int PW,
    float min_val,
    float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &launch_fused_op, "Fused ConvTranspose3d + clamp + div");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_fused',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    B, Ci, D, H, W = x.shape
    Co, _, KD, KH, KW = conv_transpose_weight.shape
    
    # Calculate output dimensions
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    OD = (D - 1) * stride + KD - 2 * padding + conv_transpose_output_padding
    OH = (H - 1) * stride + KH - 2 * padding + conv_transpose_output_padding
    OW = (W - 1) * stride + KW - 2 * padding + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((B, Co, OD, OH, OW), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_op_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        stride, stride, stride,
        padding, padding, padding,
        min_value, divisor
    )
    
    return output

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
