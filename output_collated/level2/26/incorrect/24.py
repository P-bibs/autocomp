# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_042434/code_8.py
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

# --- Optimized CUDA Implementation ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel: Perform Transpose-Conv (Simple accumulation logic) + Add + HardSwish
// Note: This implements the spatial mapping logic for ConvTranspose3d
__global__ void fused_conv_tr_add_hswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int B, int C_in, int C_out, 
    int D, int H, int W, 
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int outD, int outH, int outW) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C_out * outD * outH * outW;

    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int tmp = idx;
        int ow = tmp % outW; tmp /= outW;
        int oh = tmp % outH; tmp /= outH;
        int od = tmp % outD; tmp /= outD;
        int oc = tmp % C_out; tmp /= C_out;
        int b  = tmp;

        float val = bias[oc];
        
        // Naive transpose conv logic (kernel projection)
        for (int ic = 0; ic < C_in; ++ic) {
            for (int kd = 0; kd < kD; ++kd) {
                int id = (od + pD - kd);
                if (id % sD != 0) continue;
                id /= sD;
                if (id < 0 || id >= D) continue;

                for (int kh = 0; kh < kH; ++kh) {
                    int ih = (oh + pH - kh);
                    if (ih % sH != 0) continue;
                    ih /= sH;
                    if (ih < 0 || ih >= H) continue;

                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = (ow + pW - kw);
                        if (iw % sW != 0) continue;
                        iw /= sW;
                        if (iw < 0 || iw >= W) continue;

                        int weight_idx = oc * (C_in * kD * kH * kW) + ic * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw;
                        int input_idx = b * (C_in * D * H * W) + ic * (D * H * W) + id * (H * W) + ih * W + iw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add + HardSwish
        float x = val + add_input[idx];
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[idx] = x * (x * relu6 * 0.16666667f);
    }
}

void launch_fused(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, 
                  const at::Tensor& add_input, at::Tensor& output,
                  int stride, int padding, int outD, int outH, int outW) {
    int B = input.size(0); int C_in = input.size(1);
    int D = input.size(2); int H = input.size(3); int W = input.size(4);
    int C_out = weight.size(1); // Transposed layout for simple indexing
    int kD = weight.size(2); int kH = weight.size(3); int kW = weight.size(4);
    
    int numel = B * C_out * outD * outH * outW;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    fused_conv_tr_add_hswish_kernel<<<min(blocks, 65535), threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        add_input.data_ptr<float>(), output.data_ptr<float>(),
        B, C_in, C_out, D, H, W, kD, kH, kW, 
        stride, stride, stride, padding, padding, padding, 
        outD, outH, outW
    );
}
"""

cpp_source = r"""
void launch_fused(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, 
                  const at::Tensor& add_input, at::Tensor& output,
                  int stride, int padding, int outD, int outH, int outW);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused);
}
"""

fused_ext = load_inline(name='fused_op_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, 
                     conv_transpose_groups, conv_transpose_dilation, bias):
    B, C_in, D, H, W = x.shape
    C_out = conv_transpose_weight.shape[1]
    outD = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[2] + conv_transpose_output_padding
    outH = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[3] + conv_transpose_output_padding
    outW = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[4] + conv_transpose_output_padding
    
    output = torch.empty((B, C_out, outD, outH, outW), device='cuda')
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, add_input, output, 
                       conv_transpose_stride, conv_transpose_padding, outD, outH, outW)
    return output
