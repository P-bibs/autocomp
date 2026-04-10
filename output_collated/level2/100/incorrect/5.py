# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_113742/code_5.py
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

# CUDA source: Implements a fused convolution transpose kernel with post-processing (clamp + divide)
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int IC, int ID, int IH, int IW,
    int OC, int KD, int KH, int KW,
    int OD, int OH, int OW,
    int stride, int padding, float min_val, float inv_divisor) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * OC * OD * OH * OW;
    if (tid >= total_elements) return;

    // Map linear index to output coordinates
    int ow = tid % OW;
    int oh = (tid / OW) % OH;
    int od = (tid / (OW * OH)) % OD;
    int oc = (tid / (OW * OH * OD)) % OC;
    int n = tid / (OW * OH * OD * OC);

    float sum = bias[oc];
    // Implicit GEMM transpose convolution logic
    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            int id = (od + padding - kd);
            if (id % stride == 0) {
                id /= stride;
                for (int kh = 0; kh < KH; ++kh) {
                    int ih = (oh + padding - kh);
                    if (ih % stride == 0) {
                        ih /= stride;
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = (ow + padding - kw);
                            if (iw % stride == 0) {
                                iw /= stride;
                                if (id >= 0 && id < ID && ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                    sum += input[(((n * IC + ic) * ID + id) * IH + ih) * IW + iw] * 
                                           weight[(((oc * IC + ic) * KD + kd) * KH + kh) * KW + kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Fused post-processing
    sum = fmaxf(sum, min_val);
    output[tid] = sum * inv_divisor;
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, int stride, int padding, float min_val, float divisor) {
    int N = input.size(0); int IC = input.size(1);
    int ID = input.size(2); int IH = input.size(3); int IW = input.size(4);
    int OC = weight.size(0); int KD = weight.size(2); int KH = weight.size(3); int KW = weight.size(4);
    int OD = output.size(2); int OH = output.size(3); int OW = output.size(4);

    int total_elements = output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
        stride, padding, min_val, 1.0f/divisor);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, int stride, int padding, float min_val, float divisor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose3d + Clamp + Div");
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
    # Calculate output shape manually to avoid relying on conv_transpose3d
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = conv_transpose_weight.shape
    OD = (ID - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KD + conv_transpose_output_padding[0]
    OH = (IH - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KH + conv_transpose_output_padding[1]
    OW = (IW - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KW + conv_transpose_output_padding[2]
    
    output = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, 
                       conv_transpose_stride, conv_transpose_padding, min_value, divisor)
    return output
