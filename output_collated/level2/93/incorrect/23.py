# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_10.py
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

# CUDA kernels for Implicit GEMM (ConvTranspose2d) and Vectorized Fusion
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Vectorized Fusion Kernel
__device__ __forceinline__ float4 fast_gelu_vec(float4 v) {
    auto gelu = [](float x) {
        return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    };
    return make_float4(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w));
}

__global__ void fused_op_vectorized_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                           float add_val, float mul_val, int num_elements) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < num_elements) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        in_vec.x = fminf(in_vec.x + add_val, 0.0f);
        in_vec.y = fminf(in_vec.y + add_val, 0.0f);
        in_vec.z = fminf(in_vec.z + add_val, 0.0f);
        in_vec.w = fminf(in_vec.w + add_val, 0.0f);
        in_vec = fast_gelu_vec(in_vec);
        in_vec.x *= mul_val; in_vec.y *= mul_val;
        in_vec.z *= mul_val; in_vec.w *= mul_val;
        reinterpret_cast<float4*>(output)[idx / 4] = in_vec;
    }
}

// Basic Implicit GEMM for ConvTranspose2d (Weight Gradient / Forward)
__global__ void conv_transpose2d_naive_kernel(const float* input, const float* weight, float* output, 
                                             int B, int C, int H, int W, int OC, int KH, int KW, int stride) {
    int oc = blockIdx.x;
    int b = blockIdx.y;
    int oh = threadIdx.y;
    int ow = threadIdx.x;
    
    int OH = H * stride; int OW = W * stride;
    float sum = 0.0f;
    for(int c = 0; c < C; ++c) {
        for(int kh = 0; kh < KH; ++kh) {
            for(int kw = 0; kw < KW; ++kw) {
                int ih = oh - kh; int iw = ow - kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W && ih % stride == 0 && iw % stride == 0) {
                    sum += input[((b * C + c) * H + (ih/stride)) * W + (iw/stride)] * 
                           weight[((c * OC + oc) * KH + kh) * KW + kw];
                }
            }
        }
    }
    output[((b * OC + oc) * OH + oh) * OW + ow] = sum;
}

void fused_op(torch::Tensor input, torch::Tensor output, float add, float mul) {
    int num = input.numel();
    int threads = 128;
    int blocks = (num / 4 + threads - 1) / threads;
    fused_op_vectorized_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), add, mul, num);
}

void custom_conv_transpose2d(torch::Tensor input, torch::Tensor weight, torch::Tensor output, int stride) {
    int B=input.size(0), C=input.size(1), H=input.size(2), W=input.size(3);
    int OC=weight.size(1), KH=weight.size(2), KW=weight.size(3);
    dim3 grid(OC, B);
    dim3 block(W * stride, H * stride);
    conv_transpose2d_naive_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), B, C, H, W, OC, KH, KW, stride);
}
"""

cpp_source = r"""
void fused_op(torch::Tensor input, torch::Tensor output, float add, float mul);
void custom_conv_transpose2d(torch::Tensor input, torch::Tensor weight, torch::Tensor output, int stride);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op);
    m.def("conv_transpose2d", &custom_conv_transpose2d);
}
"""

ext = load_inline(name='ops', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, add_value, multiply_value):
    B, C, H, W = x.shape
    OC, _, KH, KW = conv_transpose_weight.shape
    out_size = (B, OC, H * conv_transpose_stride, W * conv_transpose_stride)
    x_out = torch.zeros(out_size, device='cuda')
    
    ext.conv_transpose2d(x.contiguous(), conv_transpose_weight.contiguous(), x_out, conv_transpose_stride)
    x_out += conv_transpose_bias.view(1, -1, 1, 1)
    
    final_out = torch.empty_like(x_out)
    ext.fused_op(x_out, final_out, float(add_value), float(multiply_value))
    return final_out
