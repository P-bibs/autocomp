# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_042434/code_12.py
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

# CUDA Kernel for ConvTranspose3d + Fused Add + Hardswish
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float hardswish_impl(float x) {
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val * 0.16666667f;
}

// Optimized ConvTranspose3d logic as a direct kernel
__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int N, int IC, int OC, int D, int H, int W,
    int kD, int kH, int kW, int stride, int padding) 
{
    int OD = (D - 1) * stride + kD - 2 * padding;
    int OH = (H - 1) * stride + kH - 2 * padding;
    int OW = (W - 1) * stride + kW - 2 * padding;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * OC * OD * OH * OW) return;

    int tmp = index;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int od = tmp % OD; tmp /= OD;
    int oc = tmp % OC; tmp /= OC;
    int n = tmp;

    float val = bias[oc];
    
    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < kD; ++kd) {
            int id = (od + padding - kd);
            if (id % stride == 0) {
                int id_idx = id / stride;
                if (id_idx >= 0 && id_idx < D) {
                    for (int kh = 0; kh < kH; ++kh) {
                        int ih = (oh + padding - kh);
                        if (ih % stride == 0) {
                            int ih_idx = ih / stride;
                            if (ih_idx >= 0 && ih_idx < H) {
                                for (int kw = 0; kw < kW; ++kw) {
                                    int iw = (ow + padding - kw);
                                    if (iw % stride == 0) {
                                        int iw_idx = iw / stride;
                                        if (iw_idx >= 0 && iw_idx < W) {
                                            val += x[(((n * IC + ic) * D + id_idx) * H + ih_idx) * W + iw_idx] * 
                                                   weight[(((oc * IC + ic) * kD + kd) * kH + kh) * kW + kw];
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
    float final_x = val + add_input[index];
    output[index] = final_x * hardswish_impl(final_x);
}

void launch_optimized_op(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias, 
                         const at::Tensor& add_input, at::Tensor& output, int stride, int padding) {
    int N = x.size(0); int IC = x.size(1); int D = x.size(2); int H = x.size(3); int W = x.size(4);
    int OC = weight.size(1);
    int kD = weight.size(2); int kH = weight.size(3); int kW = weight.size(4);
    int OD = (D - 1) * stride + kD - 2 * padding;
    int OH = (H - 1) * stride + kH - 2 * padding;
    int OW = (W - 1) * stride + kW - 2 * padding;
    
    int numel = N * OC * OD * OH * OW;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    conv_transpose_fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        add_input.data_ptr<float>(), output.data_ptr<float>(),
        N, IC, OC, D, H, W, kD, kH, kW, stride, padding
    );
}
"""

cpp_source = r"""
void launch_optimized_op(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias, 
                         const at::Tensor& add_input, at::Tensor& output, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_optimized_op, "Fused ConvTranspose3D Add Hardswish");
}
"""

module = load_inline(name='opt_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                     extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, 
                     conv_transpose_groups, conv_transpose_dilation, bias):
    # Weight shape for ConvTranspose3d is (In, Out, k, k, k). 
    # Adjust logic to map weight dims accordingly.
    N, IC, D, H, W = x.shape
    OD = (D - 1) * conv_transpose_stride + 3 - 2 * conv_transpose_padding
    output = torch.empty((N, conv_transpose_weight.size(1), OD, OD, OD), device='cuda')
    module.fused_op(x, conv_transpose_weight, conv_transpose_bias, add_input, output, conv_transpose_stride, conv_transpose_padding)
    return output
