# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_8.py
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

# The custom CUDA kernel implements Direct Convolution Transpose (im2col-free)
# to avoid atomic operations and uses a fused element-wise epilogue.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void conv_transpose2d_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int out_h, int out_w,
    int k_size, int stride, int padding,
    float add_val, float mul_val
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;
    
    if (tid >= total_elements) return;
    
    int w_out = tid % out_w;
    int h_out = (tid / out_w) % out_h;
    int c_out = (tid / (out_w * out_h)) % out_channels;
    int n = tid / (out_w * out_h * out_channels);
    
    float acc = 0.0f;
    
    // Direct accumulation pattern: O[b, c_out, y, x] = sum(I[b, c_in, i, j] * W[c_out, c_in, ky, kx])
    // where input maps to output at: h_out = i*stride + ky - padding, w_out = j*stride + kx - padding
    // We reverse this to find contributing input elements
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int ky = 0; ky < k_size; ++ky) {
            for (int kx = 0; k_size < k_size; ++kx) {
                int h_in_idx = h_out + padding - ky;
                int w_in_idx = w_out + padding - kx;
                
                if (h_in_idx >= 0 && h_in_idx < in_h * stride && w_in_idx >= 0 && w_in_idx < in_w * stride) {
                    if (h_in_idx % stride == 0 && w_in_idx % stride == 0) {
                        int h_in = h_in_idx / stride;
                        int w_in = w_in_idx / stride;
                        
                        int in_off = ((n * in_channels + c_in) * in_h + h_in) * in_w + w_in;
                        int wt_off = ((c_out * in_channels + c_in) * k_size + ky) * k_size + kx;
                        acc += input[in_off] * weight[wt_off];
                    }
                }
            }
        }
    }
    
    acc += bias[c_out];
    
    // Fused epilogue
    float val = fminf(acc + add_val, 0.0f);
    output[tid] = fast_gelu(val) * mul_val;
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_conv_transpose(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                           int stride, int padding, float add_val, float mul_val);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv_transpose", &launch_conv_transpose, "Fused ConvTranspose2D + ops");
}
"""

cuda_logic = r"""
void launch_conv_transpose(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                           int stride, int padding, float add_val, float mul_val) {
    const auto batch = input.size(0); const auto in_c = input.size(1);
    const auto in_h = input.size(2); const auto in_w = input.size(3);
    const auto out_c = weight.size(0); const auto k_size = weight.size(2);
    const auto out_h = output.size(2); const auto out_w = output.size(3);
    
    int total = batch * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv_transpose2d_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_c, in_h, in_w, out_c, out_h, out_w, k_size, stride, padding, add_val, mul_val
    );
}
"""

fused_ext = load_inline(
    name='fused_conv_op',
    cpp_sources=cpp_source,
    cuda_sources=[cuda_kernel, cuda_logic],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    b, c, h, w = x.shape
    out_c, _, k, _ = conv_transpose_weight.shape
    out_h = (h - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k + conv_transpose_output_padding
    out_w = (w - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k + conv_transpose_output_padding
    
    out = torch.empty((b, out_c, out_h, out_w), device=x.device, dtype=x.dtype)
    fused_ext.launch_conv_transpose(x, conv_transpose_weight, conv_transpose_bias, out, 
                                    conv_transpose_stride, conv_transpose_padding, 
                                    float(add_value), float(multiply_value))
    return out
