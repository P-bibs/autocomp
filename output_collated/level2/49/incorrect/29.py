# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_095400/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose3D + Softmax + Sigmoid
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256

__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_conv_transpose3d_soft_max_sigmoid_kernel(
    const float* input,              // [N, Ci, Di, Hi, Wi]
    const float* weight,             // [Ci, Co/group, Kd, Kh, Kw]
    const float* bias,               // [Co]
    float* output,                   // [N, Co, Do, Ho, Wo]
    int N, int Ci, int Di, int Hi, int Wi,
    int Co, int Do, int Ho, int Wo,
    int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups, int dilation_d, int dilation_h, int dilation_w,
    int softmax_dim
) {
    // Get global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * Co * Do * Ho * Wo;
    
    if (idx >= total_elements) return;
    
    // Decode output tensor indices from linear index
    int tmp = idx;
    int wo = tmp % Wo; tmp /= Wo;
    int ho = tmp % Ho; tmp /= Ho;
    int doo = tmp % Do; tmp /= Do;
    int co = tmp % Co; tmp /= Co;
    int n = tmp;
    
    // Group handling
    int group_id = co * groups / Co;
    int weight_co_offset = co * (Ci / groups) * Kd * Kh * Kw;
    
    // ConvTranspose3D computation
    float conv_result = 0.0f;
    if (bias) {
        conv_result = bias[co];
    }
    
    // Iterate over input channels in this group
    for (int ci_group = 0; ci_group < Ci / groups; ++ci_group) {
        int ci = group_id * (Ci / groups) + ci_group;
        int weight_idx_base = weight_co_offset + ci_group * Kd * Kh * Kw;
        
        // Iterate over kernel dimensions
        for (int kd = 0; kd < Kd; ++kd) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    // Calculate corresponding input position
                    int id = doo * stride_d - padding_d + kd * dilation_d;
                    int ih = ho * stride_h - padding_h + kh * dilation_h;
                    int iw = wo * stride_w - padding_w + kw * dilation_w;
                    
                    // Check bounds for input
                    if (id >= 0 && id < Di && ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                        int input_idx = (((n * Ci + ci) * Di + id) * Hi + ih) * Wi + iw;
                        int weight_idx = weight_idx_base + (kd * Kh + kh) * Kw + kw;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // If softmax_dim is the channel dimension (dim=1), we need to compute softmax
    if (softmax_dim == 1) {
        // Use shared memory for softmax reduction
        extern __shared__ float shared_data[];
        int tid = threadIdx.x;
        
        // Each thread computes one element of the softmax
        float val = conv_result;
        float max_val = val;
        __syncthreads();
        
        // Reduction to find maximum in the channel dimension
        for (int stride = 1; stride < Co; stride *= 2) {
            if ((co + stride) < Co) {
                int other_idx = ((((n * Co + (co + stride)) * Do + doo) * Ho + ho) * Wo + wo);
                float other_val = (other_idx < total_elements) ? output[other_idx] : -INFINITY;
                max_val = fmaxf(max_val, other_val);
            }
            __syncthreads();
        }
        
        // Broadcast max back to all threads
        float shared_max = max_val;
        __syncthreads();
        
        // Compute exponential
        float exp_val = expf(val - shared_max);
        
        // Reduction to compute sum
        float sum_val = exp_val;
        __syncthreads();
        
        for (int stride = 1; stride < Co; stride *= 2) {
            if ((co + stride) < Co) {
                int other_idx = ((((n * Co + (co + stride)) * Do + doo) * Ho + ho) * Wo + wo);
                float other_exp = (other_idx < total_elements) ? expf(output[other_idx] - shared_max) : 0.0f;
                sum_val += other_exp;
            }
            __syncthreads();
        }
        
        // Normalize and apply sigmoid
        float softmax_val = exp_val / sum_val;
        output[idx] = sigmoid(softmax_val);
    } else {
        // For other dimensions or no softmax, just apply sigmoid directly
        output[idx] = sigmoid(conv_result);
    }
}

void fused_conv_transpose3d_soft_max_sigmoid(
    const at::Tensor input,
    const at::Tensor weight,
    const c10::optional<at::Tensor> bias,
    at::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups, int dilation_d, int dilation_h, int dilation_w,
    int softmax_dim
) {
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);
    
    int Co = weight.size(1) * groups;
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);
    
    int Do = (Di - 1) * stride_d - 2 * padding_d + dilation_d * (Kd - 1) + output_padding_d + 1;
    int Ho = (Hi - 1) * stride_h - 2 * padding_h + dilation_h * (Kh - 1) + output_padding_h + 1;
    int Wo = (Wi - 1) * stride_w - 2 * padding_w + dilation_w * (Kw - 1) + output_padding_w + 1;
    
    int total_elements = N * Co * Do * Ho * Wo;
    int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    fused_conv_transpose3d_soft_max_sigmoid_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, Ci, Di, Hi, Wi,
        Co, Do, Ho, Wo,
        Kd, Kh, Kw,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups, dilation_d, dilation_h, dilation_w,
        softmax_dim
    );
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_soft_max_sigmoid(
    const at::Tensor input,
    const at::Tensor weight,
    const c10::optional<at::Tensor> bias,
    at::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups, int dilation_d, int dilation_h, int dilation_w,
    int softmax_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_soft_max_sigmoid", &fused_conv_transpose3d_soft_max_sigmoid, "Fused ConvTranspose3D + Softmax + Sigmoid");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_soft_max_sigmoid_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    softmax_dim,
):
    # Calculate output dimensions
    N, Ci, Di, Hi, Wi = x.shape
    Co = conv_transpose_weight.size(1) * conv_transpose_groups
    Kd, Kh, Kw = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    
    stride_d, stride_h, stride_w = conv_transpose_stride, conv_transpose_stride, conv_transpose_stride
    padding_d, padding_h, padding_w = conv_transpose_padding, conv_transpose_padding, conv_transpose_padding
    output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding, conv_transpose_output_padding, conv_transpose_output_padding
    dilation_d, dilation_h, dilation_w = conv_transpose_dilation, conv_transpose_dilation, conv_transpose_dilation
    
    Do = (Di - 1) * stride_d - 2 * padding_d + dilation_d * (Kd - 1) + output_padding_d + 1
    Ho = (Hi - 1) * stride_h - 2 * padding_h + dilation_h * (Kh - 1) + output_padding_h + 1
    Wo = (Wi - 1) * stride_w - 2 * padding_w + dilation_w * (Kw - 1) + output_padding_w + 1
    
    # Create output tensor
    output = torch.empty((N, Co, Do, Ho, Wo), dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose3d_soft_max_sigmoid(
        x, conv_transpose_weight, conv_transpose_bias,
        output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        conv_transpose_groups,
        dilation_d, dilation_h, dilation_w,
        softmax_dim
    )
    
    return output

# Model parameters
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
