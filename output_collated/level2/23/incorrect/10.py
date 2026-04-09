# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_000325/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'num_groups']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'group_norm_eps']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'group_norm_weight', 'group_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, applies Group Normalization, computes the mean
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

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
    # State for conv (nn.Conv3d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    # State for group_norm (nn.GroupNorm)
    if 'group_norm_weight' in flat_state:
        state_kwargs['group_norm_weight'] = flat_state['group_norm_weight']
    else:
        state_kwargs['group_norm_weight'] = getattr(model.group_norm, 'weight', None)
    if 'group_norm_bias' in flat_state:
        state_kwargs['group_norm_bias'] = flat_state['group_norm_bias']
    else:
        state_kwargs['group_norm_bias'] = getattr(model.group_norm, 'bias', None)
    state_kwargs['group_norm_num_groups'] = model.group_norm.num_groups
    state_kwargs['group_norm_eps'] = model.group_norm.eps
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

# CUDA Kernel: Fuses Conv3D, GroupNorm and Global Reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define EPSILON 1e-5f

__global__ void fused_conv3d_groupnorm_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C_in, int C_out, int D, int H, int W, 
    int K, int pad, int stride, int groups, int num_groups, float eps) {
    
    // Calculate output dimensions
    int out_D = (D + 2 * pad - K) / stride + 1;
    int out_H = (H + 2 * pad - K) / stride + 1;
    int out_W = (W + 2 * pad - K) / stride + 1;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= N) return;
    
    // Shared memory for reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sum_sq[BLOCK_SIZE];
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int total_elements = 0;
    
    // Process all output channels for this batch
    for (int out_c = 0; out_c < C_out; out_c++) {
        int group_id = out_c * num_groups / C_out;
        int channels_per_group = C_in / num_groups;
        int start_channel = group_id * channels_per_group;
        int end_channel = start_channel + channels_per_group;
        
        // Process spatial locations
        for (int d = 0; d < out_D; d++) {
            for (int h = 0; h < out_H; h++) {
                for (int w = 0; w < out_W; w++) {
                    float conv_result = conv_bias[out_c];
                    
                    // Convolution computation
                    for (int kd = 0; kd < K; kd++) {
                        for (int kh = 0; kh < K; kh++) {
                            for (int kw = 0; kw < K; kw++) {
                                for (int c = 0; c < C_in; c++) {
                                    if (c / (C_in / groups) == out_c / (C_out / groups)) {
                                        int in_d = d * stride + kd - pad;
                                        int in_h = h * stride + kh - pad;
                                        int in_w = w * stride + kw - pad;
                                        
                                        if (in_d >= 0 && in_d < D && 
                                            in_h >= 0 && in_h < H && 
                                            in_w >= 0 && in_w < W) {
                                            int input_idx = bid * (C_in * D * H * W) + 
                                                           c * (D * H * W) + 
                                                           in_d * (H * W) + 
                                                           in_h * W + in_w;
                                            int weight_idx = out_c * (C_in / groups * K * K * K) + 
                                                            (c % (C_in / groups)) * (K * K * K) + 
                                                            kd * (K * K) + 
                                                            kh * K + kw;
                                            conv_result += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Accumulate stats for global reduction
                    local_sum += conv_result;
                    local_sum_sq += conv_result * conv_result;
                    total_elements++;
                }
            }
        }
    }
    
    // Reduction in shared memory
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Reduction tree
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    // Compute final result
    if (tid == 0) {
        float mean = shared_sum[0] / total_elements;
        float variance = shared_sum_sq[0] / total_elements - mean * mean;
        output[bid] = mean / sqrtf(variance + eps);
    }
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor output,
    int N, int C_in, int C_out, int D, int H, int W, 
    int K, int pad, int stride, int groups, int num_groups, float eps) {
    
    const int threads = BLOCK_SIZE;
    const int blocks = N;
    
    fused_conv3d_groupnorm_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, D, H, W, K, pad, stride, groups, num_groups, eps
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor output,
    int N, int C_in, int C_out, int D, int H, int W, 
    int K, int pad, int stride, int groups, int num_groups, float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv3D + GroupNorm + Reduction");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    group_norm_weight, group_norm_bias, group_norm_num_groups, group_norm_eps,
):
    # Ensure tensors are on GPU
    x = x.cuda()
    conv_weight = conv_weight.cuda()
    conv_bias = conv_bias.cuda()
    group_norm_weight = group_norm_weight.cuda()
    group_norm_bias = group_norm_bias.cuda()
    
    # Pre-allocate output tensor
    batch_size = x.size(0)
    out = torch.empty(batch_size, device=x.device, dtype=torch.float32)
    
    # Extract dimensions
    N, C_in, D, H, W = x.shape
    C_out = conv_weight.size(0)
    K = conv_weight.size(2)  # Assuming cubic kernel
    
    # Call fused CUDA kernel
    fused_ext.fused_op(
        x, conv_weight, conv_bias, group_norm_weight, group_norm_bias, out,
        N, C_in, C_out, D, H, W, 
        K, conv_padding[0], conv_stride[0], conv_groups, group_norm_num_groups, group_norm_eps
    )
    return out

# Initializers provided in the prompt
batch_size, in_channels, out_channels, D, H, W = 128, 3, 24, 24, 32, 32
kernel_size, num_groups = 3, 8

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]
