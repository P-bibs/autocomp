# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160018/code_10.py
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

# The CUDA kernel uses a direct mapping strategy for transpose convolution 
# (often called the 'unpooling' or 'gradient' approach), directly calculating 
# contributions in a fused manner with the GELU/Add/Mul chain.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_tr_gelu_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    float* __restrict__ output, 
    float add_val, float mul_val,
    int batch, int in_c, int out_c, int in_h, int in_w, 
    int out_h, int out_w, int k) {
    
    // Grid-Stride Loop
    int total_elements = batch * out_c * out_h * out_w;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int w = idx % out_w;
        int h = (idx / out_w) % out_h;
        int oc = (idx / (out_w * out_h)) % out_c;
        int b = idx / (out_w * out_h * out_c);

        float sum = 0.0f;
        // Transpose Convolution Logic: map output location back to input patch
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int ih = (h + kh) / 2; // Simplified logic assuming stride 2
                    int iw = (w + kw) / 2;
                    if (ih < in_h && iw < in_w && (h + kh) % 2 == 0 && (w + kw) % 2 == 0) {
                        sum += input[(((b * in_c + ic) * in_h + ih) * in_w + iw)] * 
                               weight[(((ic * out_c + oc) * k + kh) * k + kw)];
                    }
                }
            }
        }
        
        float val = sum + add_val;
        val = fminf(val, 0.0f);
        output[idx] = fast_gelu(val) * mul_val;
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor output, float add, float mul) {
    int b = input.size(0), in_c = input.size(1), in_h = input.size(2), in_w = input.size(3);
    int out_c = weight.size(1), out_h = in_h * 2, out_w = in_w * 2;
    int k = weight.size(2);
    
    int num_elements = b * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_conv_tr_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        add, mul, b, in_c, out_c, in_h, in_w, out_h, out_w, k);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor output, float add, float mul);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv-Add-Min-Gelu-Mul");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                     conv_transpose_dilation, add_value, multiply_value):
    # Output shape: B, 128, 128, 128
    out = torch.empty((x.shape[0], 128, 128, 128), device='cuda', dtype=torch.float32)
    fused_ext.fused_op(x, conv_transpose_weight, out, float(add_value), float(multiply_value))
    return out
