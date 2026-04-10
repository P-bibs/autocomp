# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_12.py
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

# The custom implementation focuses on a high-performance implicit GEMM-style 
# convolution transpose and a vectorized fused element-wise kernel.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Vectorized fused kernel
__global__ void fused_op_kernel_vec4(const float* __restrict__ input, float* __restrict__ output, 
                                     float add_val, float mul_val, int num_elements) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < num_elements) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;
        
        #pragma unroll
        for(int i=0; i<4; ++i) {
            float val = ((float*)&in_vec)[i] + add_val;
            val = fminf(val, 0.0f);
            val = fast_gelu(val);
            ((float*)&out_vec)[i] = val * mul_val;
        }
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } else {
        for (int i = idx; i < num_elements; ++i) {
            float val = input[i] + add_val;
            val = fminf(val, 0.0f);
            val = fast_gelu(val);
            output[i] = val * mul_val;
        }
    }
}

// Optimized ConvTranspose2d (Implicit GEMM approach)
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int KH, int KW, int stride, int padding) {
    
    int n = blockIdx.z;
    int co = blockIdx.y;
    int h_out = blockIdx.x / ((W_out + 15) / 16) * 16 + threadIdx.y;
    int w_out = (blockIdx.x % ((W_out + 15) / 16)) * 16 + threadIdx.x;

    if (h_out >= H_out || w_out >= W_out) return;

    float acc = (bias != nullptr) ? bias[co] : 0.0f;
    
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int h_in = h_out + padding - kh;
                int w_in = w_out + padding - kw;
                
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        float in_v = input[((n * C_in + ci) * H_in + h_in) * W_in + w_in];
                        float wt_v = weight[((ci * C_out + co) * KH + kh) * KW + kw];
                        acc += in_v * wt_v;
                    }
                }
            }
        }
    }
    output[((n * C_out + co) * H_out + h_out) * W_out + w_out] = acc;
}

void conv_transpose2d_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                         int stride, int padding) {
    int N = input.size(0), C_in = input.size(1), H_in = input.size(2), W_in = input.size(3);
    int C_out = weight.size(1), KH = weight.size(2), KW = weight.size(3);
    int H_out = output.size(2), W_out = output.size(3);
    
    dim3 threads(16, 16);
    dim3 blocks(((W_out + 15) / 16 * (H_out + 15) / 16), C_out, N);
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out, KH, KW, stride, padding);
}

void fused_op(torch::Tensor input, torch::Tensor output, float add_val, float mul_val) {
    int n = input.numel();
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    fused_op_kernel_vec4<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), add_val, mul_val, n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void conv_transpose2d_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding);
void fused_op(torch::Tensor input, torch::Tensor output, float add_val, float mul_val);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose", &conv_transpose2d_op, "Custom ConvTranspose2D");
    m.def("fused_op", &fused_op, "Fused Op");
}
"""

ext = load_inline(name='ops', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    N, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = conv_transpose_weight.shape
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KH + conv_transpose_output_padding
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KW + conv_transpose_output_padding
    
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    ext.conv_transpose(x, conv_transpose_weight, conv_transpose_bias, out, conv_transpose_stride, conv_transpose_padding)
    out_fused = torch.empty_like(out)
    ext.fused_op(out, out_fused, float(add_value), float(multiply_value))
    return out_fused
