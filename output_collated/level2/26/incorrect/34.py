# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_043234/code_11.py
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

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, const float* __restrict__ add_input, 
    float* __restrict__ output,
    int B, int Ci, int Co, int D, int H, int W, 
    int OD, int OH, int OW, int KD, int KH, int KW,
    int stride, int padding, int output_padding) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * Co * OD * OH * OW) return;

    int tmp = tid;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int od = tmp % OD; tmp /= OD;
    int co = tmp % Co;
    int b = tmp / Co;

    float acc = bias[co];

    for (int ci = 0; ci < Ci; ++ci) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int id = (od + padding - kd);
                    int ih = (oh + padding - kh);
                    int iw = (ow + padding - kw);

                    if (id % stride == 0 && ih % stride == 0 && iw % stride == 0) {
                        id /= stride; ih /= stride; iw /= stride;
                        if (id >= 0 && id < D && ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            int weight_idx = (((ci * Co + co) * KD + kd) * KH + kh) * KW + kw;
                            int input_idx = (((b * Ci + ci) * D + id) * H + ih) * W + iw;
                            acc += x[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    float val = acc + add_input[tid];
    output[tid] = val * hardswish(val);
}

void fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_input, torch::Tensor output,
              int stride, int padding, int output_padding) {
    const int B = x.size(0); const int Ci = x.size(1);
    const int D = x.size(2); const int H = x.size(3); const int W = x.size(4);
    const int Co = weight.size(1);
    const int KD = weight.size(2); const int KH = weight.size(3); const int KW = weight.size(4);
    const int OD = output.size(2); const int OH = output.size(3); const int OW = output.size(4);

    int elements = B * Co * OD * OH * OW;
    int threads = 256;
    int blocks = (elements + threads - 1) / threads;
    
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        add_input.data_ptr<float>(), output.data_ptr<float>(),
        B, Ci, Co, D, H, W, OD, OH, OW, KD, KH, KW, stride, padding, output_padding
    );
}
'''

cpp_source = r'''
void fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor add_input, torch::Tensor output,
              int stride, int padding, int output_padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused ConvTranspose3D + Add + Hardswish");
}
'''

fused_ext = load_inline(name='fused_ops', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride=2, conv_transpose_padding=1, 
                     conv_transpose_output_padding=1, **kwargs):
    B, Co = x.size(0), conv_transpose_weight.size(1)
    OD, OH, OW = add_input.shape[2:]
    out = torch.empty((B, Co, OD, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, add_input, out, 
                       conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding)
    return out
