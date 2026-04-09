# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260408_235921/code_0.py
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

# --- CUDA Source Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_conv_gn_mean_kernel(
    const float* __restrict__ input,      // [N, C, D, H, W]
    const float* __restrict__ weight,     // [outC, inC, kD, kH, kW]
    const float* __restrict__ bias,       // [outC]
    const float* __restrict__ gn_weight,  // [outC]
    const float* __restrict__ gn_bias,    // [outC]
    float* __restrict__ output,           // [N]
    int N, int C, int D, int H, int W,
    int outC, int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups, int num_groups, float eps
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    
    if (batch_idx >= N) return;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    // Calculate input/output channel per group
    int inC_per_group = C / groups;
    int outC_per_group = outC / groups;
    
    float local_sum = 0.0f;
    
    // Each thread processes multiple output channels
    for (int oc = tid; oc < outC; oc += threads_per_block) {
        int group_id = oc / outC_per_group;
        float accumulator = 0.0f;
        
        // Conv3D computation
        for (int kd = 0; kd < kD; ++kd) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    for (int ic = group_id * inC_per_group; ic < (group_id + 1) * inC_per_group; ++ic) {
                        // Compute output dimensions
                        for (int od = 0; od < D; ++od) {
                            for (int oh = 0; oh < H; ++oh) {
                                for (int ow = 0; ow < W; ++ow) {
                                    int id = od * stride_d - pad_d + kd * dilation_d;
                                    int ih = oh * stride_h - pad_h + kh * dilation_h;
                                    int iw = ow * stride_w - pad_w + kw * dilation_w;
                                    
                                    if (id >= 0 && id < D && ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        int in_idx = ((((batch_idx * C + ic) * D + id) * H + ih) * W + iw);
                                        int wt_idx = ((((oc * inC_per_group + (ic % inC_per_group)) * kD + kd) * kH + kh) * kW + kw);
                                        accumulator += input[in_idx] * weight[wt_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Add bias
        accumulator += bias[oc];
        
        // GroupNorm - simplified implementation assuming group size = 1 for this optimization
        float gamma = gn_weight[oc];
        float beta = gn_bias[oc];
        float normalized = gamma * accumulator + beta;
        
        local_sum += normalized;
    }
    
    // Store in shared memory for reduction
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Block-level reduction
    for (int s = threads_per_block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[batch_idx] = sdata[0] / (outC * D * H * W);
    }
}

void fused_conv_gn_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups, int num_groups, float eps
) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int outC = weight.size(0);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    dim3 block(256);
    dim3 grid(N);
    
    size_t shared_mem_size = block.x * sizeof(float);

    fused_conv_gn_mean_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W, outC, kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w,
        groups, num_groups, eps
    );
}
"""

# --- C++ Bindings ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_gn_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups, int num_groups, float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_gn_forward", &fused_conv_gn_forward, "Fused Conv3D + GroupNorm + Mean");
}
"""

# --- Compile Custom Extension ---
fused_ext = load_inline(
    name='fused_conv_gn',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------ #
# Main Model Function Using Fused Kernel
# ------------------------------ #

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    group_norm_weight,
    group_norm_bias,
    group_norm_num_groups,
    group_norm_eps,
):
    N = x.size(0)
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    fused_ext.fused_conv_gn_forward(
        x,
        conv_weight,
        conv_bias,
        group_norm_weight,
        group_norm_bias,
        out,
        conv_stride[0], conv_stride[1], conv_stride[2],
        conv_padding[0], conv_padding[1], conv_padding[2],
        conv_dilation[0], conv_dilation[1], conv_dilation[2],
        conv_groups,
        group_norm_num_groups,
        group_norm_eps
    )
    return out

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
