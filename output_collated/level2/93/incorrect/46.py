# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155151/code_10.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized GELU
__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// 1. Fused Post-Processing Kernel (Grid-Stride)
__global__ void fused_op_kernel(const float* input, float* output, float add_val, float mul_val, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_elements; i += stride) {
        float val = input[i] + add_val;
        val = fminf(val, 0.0f);
        val = fast_gelu(val);
        output[i] = val * mul_val;
    }
}

// 2. Custom Implicit GEMM Transposed Convolution
// Simplified for performance: Performs the accumulation for the spatial output
__global__ void conv_transpose_kernel(const float* input, const float* weight, const float* bias, float* output, 
                                     int N, int C, int H, int W, int OC, int K, int stride, int padding, int num_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_idx = blockDim.x * gridDim.x;

    for (int tid = idx; tid < num_out; tid += stride_idx) {
        int n = tid / (OC * (H * stride) * (W * stride));
        int oc = (tid / (H * stride * W * stride)) % OC;
        int oh = (tid / (W * stride)) % (H * stride);
        int ow = tid % (W * stride);
        
        float sum = (bias != nullptr) ? bias[oc] : 0.0f;
        
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int ih = (oh + padding - kh) / stride;
                    int iw = (ow + padding - kw) / stride;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W && (oh + padding - kh) % stride == 0 && (ow + padding - kw) % stride == 0) {
                        sum += input[((n * C + c) * H + ih) * W + iw] * weight[((c * OC + oc) * K + kh) * K + kw];
                    }
                }
            }
        }
        output[tid] = sum;
    }
}

void launch_fused_op(torch::Tensor input, torch::Tensor output, float add, float mul) {
    int n = input.numel();
    int threads = 256;
    int blocks = 1024;
    fused_op_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), add, mul, n);
}

void launch_conv_transpose(torch::Tensor in, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, int s, int p) {
    int N = in.size(0), C = in.size(1), H = in.size(2), W = in.size(3);
    int OC = weight.size(1), K = weight.size(2);
    int num_out = out.numel();
    int threads = 256;
    int blocks = 1024;
    conv_transpose_kernel<<<blocks, threads>>>(in.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, OC, K, s, p, num_out);
}
"""

cpp_source = r"""
void launch_fused_op(torch::Tensor input, torch::Tensor output, float add, float mul);
void launch_conv_transpose(torch::Tensor in, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, int s, int p);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op);
    m.def("conv_transpose", &launch_conv_transpose);
}
"""

module = load_inline(name='custom_ops', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    
    # Calculate output shape: H_out = (H_in - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    N, C, H, W = x.shape
    OC, _, K, _ = conv_transpose_weight.shape
    out_h = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (K - 1) + conv_transpose_output_padding + 1
    out_w = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (K - 1) + conv_transpose_output_padding + 1
    out = torch.empty((N, OC, out_h, out_w), device=x.device)
    
    module.conv_transpose(x, conv_transpose_weight, conv_transpose_bias, out, conv_transpose_stride, conv_transpose_padding)
    final_out = torch.empty_like(out)
    module.fused_op(out, final_out, float(add_value), float(multiply_value))
    return final_out
