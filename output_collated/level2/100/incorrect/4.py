# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_113742/code_4.py
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

# CUDA kernel: Transposed 3D Conv implementation (Direct sum-of-products)
# This avoids intermediate explicit upsampling/im2col.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int ic, int oc,
    int id, int ih, int iw,
    int od, int oh, int ow,
    int k, int s, int p,
    float min_val, float div) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * oc * od * oh * ow;
    if (tid >= total_elements) return;

    // Decode linear index to output coordinates
    int ow_idx = tid % ow;
    int oh_idx = (tid / ow) % oh;
    int od_idx = (tid / (ow * oh)) % od;
    int oc_idx = (tid / (ow * oh * od)) % oc;
    int b_idx  = tid / (ow * oh * od * oc);

    float val = bias[oc_idx];

    // Compute Transposed Convolution: Output(oc, od, oh, ow) = sum(input * weight)
    for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
        for (int kd = 0; kd < k; ++kd) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int id_in = od_idx + p - kd;
                    int ih_in = oh_idx + p - kh;
                    int iw_in = ow_idx + p - kw;

                    if (id_in % s == 0 && ih_in % s == 0 && iw_in % s == 0) {
                        id_in /= s; ih_in /= s; iw_in /= s;

                        if (id_in >= 0 && id_in < id && ih_in >= 0 && ih_in < ih && iw_in >= 0 && iw_in < iw) {
                            int in_offset = ((b_idx * ic + ic_idx) * id * ih * iw) +
                                            (id_in * ih * iw) + (ih_in * iw) + iw_in;
                            int wt_offset = ((oc_idx * ic + ic_idx) * k * k * k) +
                                            (kd * k * k) + (kh * k) + kw;
                            val += input[in_offset] * weight[wt_offset];
                        }
                    }
                }
            }
        }
    }

    // Fused post-processing
    val = fmaxf(val, min_val);
    output[tid] = val / div;
}

void fused_op_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, int k, int s, int p, float min_val, float div) {
    
    int batch = input.size(0);
    int ic = input.size(1);
    int id = input.size(2);
    int ih = input.size(3);
    int iw = input.size(4);
    int oc = weight.size(0);
    int od = output.size(2);
    int oh = output.size(3);
    int ow = output.size(4);

    int total_elements = batch * oc * od * oh * ow;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, ic, oc, id, ih, iw, od, oh, ow, k, s, p, min_val, div);
}
"""

cpp_source = r"""
void fused_op_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                      torch::Tensor& output, int k, int s, int p, float min_val, float div);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv + Clamp + Div");
}
"""

fused_ext = load_inline(
    name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, min_value, divisor):
    s = conv_transpose_stride
    p = conv_transpose_padding
    k = conv_transpose_weight.shape[2]
    
    od = (x.shape[2] - 1) * s - 2 * p + k
    oh = (x.shape[3] - 1) * s - 2 * p + k
    ow = (x.shape[4] - 1) * s - 2 * p + k
    
    output = torch.empty((x.shape[0], conv_transpose_weight.shape[0], od, oh, ow), 
                         device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, k, s, p, min_value, divisor)
    return output
