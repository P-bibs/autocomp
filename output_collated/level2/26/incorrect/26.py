# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_042434/code_11.py
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

# CUDA kernel with fused Hardswish and Add logic
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv_tr_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, const float* __restrict__ add_input,
    float* __restrict__ output, int N, int IC, int OC, int D, int H, int W,
    int KD, int KH, int KW, int OD, int OH, int OW,
    int stride, int pad, int opad) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OC * OD * OH * OW;
    if (tid >= total_elements) return;

    // Linear index mapping
    int tmp = tid;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int od = tmp % OD; tmp /= OD;
    int oc = tmp % OC; tmp /= OC;
    int n  = tmp;

    float acc = bias[oc];
    // Optimized ConvTranspose3D accumulation logic
    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int id = (od + pad - kd);
                    int ih = (oh + pad - kh);
                    int iw = (ow + pad - kw);
                    
                    if (id % stride == 0 && ih % stride == 0 && iw % stride == 0) {
                        id /= stride; ih /= stride; iw /= stride;
                        if (id >= 0 && id < D && ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            acc += input[((n * IC + ic) * D + id) * H * W + ih * W + iw] * 
                                   weight[((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw];
                        }
                    }
                }
            }
        }
    }
    output[tid] = hardswish(acc + add_input[tid]);
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor add_input, torch::Tensor output, int stride, int pad, int opad) {
    const int N = input.size(0), IC = input.size(1), D = input.size(2), H = input.size(3), W = input.size(4);
    const int OC = weight.size(1), KD = weight.size(2), KH = weight.size(3), KW = weight.size(4);
    const int OD = output.size(2), OH = output.size(3), OW = output.size(4);
    
    int count = N * OC * OD * OH * OW;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    
    fused_conv_tr_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        add_input.data_ptr<float>(), output.data_ptr<float>(), 
        N, IC, OC, D, H, W, KD, KH, KW, OD, OH, OW, stride, pad, opad);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor add_input, torch::Tensor output, int stride, int pad, int opad);
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_op_forward'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    output = torch.empty_like(add_input)
    # Note: conv_transpose_groups/dilation assumed 1/1 for simplicity in custom kernel
    fused_ext.fused_op_forward(
        x, conv_transpose_weight, conv_transpose_bias, add_input, output,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
    )
    return output
