# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_8.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    # State for conv_transpose (nn.ConvTranspose2d)
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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# The CUDA source implements a custom ConvTranspose2d fused with 
# element-wise post-processing (add, min, gelu, multiply).
# This replaces standard PyTorch convolution paths with raw device compute.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int ic, int ih, int iw,
    int oc, int kh, int kw,
    int stride, int pad,
    float add_val, float mul_val,
    int oh, int ow
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * oc * oh * ow;
    
    if (out_idx >= total_elements) return;
    
    // Map linear index to 4D tensor coordinates
    int temp = out_idx;
    int w_out = temp % ow; temp /= ow;
    int h_out = temp % oh; temp /= oh;
    int c_out = temp % oc; temp /= oc;
    int n = temp;
    
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    // Perform custom ConvTranspose2d math (gradient-style accumulation)
    // For each output pixel, determine the receptive input field
    for (int c_in = 0; c_in < ic; c_in++) {
        for (int row = 0; row < kh; row++) {
            for (int col = 0; col < kw; col++) {
                int h_in_f = h_out + pad - row;
                int w_in_f = w_out + pad - col;
                
                if (h_in_f >= 0 && h_in_f % stride == 0 && w_in_f >= 0 && w_in_f % stride == 0) {
                    int h_in = h_in_f / stride;
                    int w_in = w_in_f / stride;
                    
                    if (h_in < ih && w_in < iw) {
                        int in_idx = ((n * ic + c_in) * ih + h_in) * iw + w_in;
                        int w_idx = ((c_out * ic + c_in) * kh + row) * kw + col;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    
    // Fused operations
    float val = fminf(sum + add_val, 0.0f);
    output[out_idx] = fast_gelu(val) * mul_val;
}

void launch_fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride, int padding, 
    float add_val, float mul_val
) {
    int batch = input.size(0);
    int ic = input.size(1);
    int ih = input.size(2);
    int iw = input.size(3);
    
    int oc = weight.size(0);
    int kh = weight.size(2);
    int kw = weight.size(3);
    
    int oh = output.size(2);
    int ow = output.size(3);
    
    int total_elements = batch * oc * oh * ow;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), output.data_ptr<float>(),
        batch, ic, ih, iw, oc, kh, kw, stride, padding,
        add_val, mul_val, oh, ow
    );
}
"""

cpp_source = r"""
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                       torch::Tensor output, int stride, int padding, 
                       float add_val, float mul_val);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_conv, "Fused ConvTranspose2d + Ops");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *,
    conv_transpose_weight, conv_transpose_bias,
    conv_transpose_stride, conv_transpose_padding,
    conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, add_value, multiply_value,
):
    # Determine output shapes manually
    n, ic, ih, iw = x.shape
    oc, _, kh, kw = conv_transpose_weight.shape
    oh = (ih - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kh + conv_transpose_output_padding
    ow = (iw - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kw + conv_transpose_output_padding
    
    out = torch.empty((n, oc, oh, ow), device=x.device, dtype=torch.float32)
    
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, out,
        conv_transpose_stride, conv_transpose_padding,
        float(add_value), float(multiply_value)
    )
    return out
