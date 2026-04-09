# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101623/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# --- CUDA Kernel for Fused Conv3D + Min + Softmax ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void fused_conv_min_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int D, const int H, const int W,
    const int kD, const int kH, const int kW,
    const int padD, const int padH, const int padW,
    const int strideD, const int strideH, const int strideW,
    const int dilationD, const int dilationH, const int dilationW,
    const int outD, const int outH, const int outW
) {
    // Shared memory for softmax reduction
    extern __shared__ float smem[];

    int od = threadIdx.z;
    int oh = threadIdx.y;
    int ow = threadIdx.x;

    int total_threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;

    for (int batch = blockIdx.x; batch < batch_size; batch += gridDim.x) {
        for (int oc = blockIdx.y * blockDim.z + threadIdx.z; oc < out_channels; oc += gridDim.y * blockDim.z) {
            // Step 1: Conv3D + Min Reduction along D dimension
            float min_val = 1e30f; // Large value as initial minimum

            for (int d = 0; d < outD; d++) {
                float conv_sum = 0.0f;
                bool in_bounds = true;

                // Conv3D computation for one output point
                for (int kd = 0; kd < kD; kd++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int in_d = d * strideD - padD + kd * dilationD;
                            int in_h = oh * strideH - padH + kh * dilationH;
                            int in_w = ow * strideW - padW + kw * dilationW;

                            if (in_d >= 0 && in_d < D && 
                                in_h >= 0 && in_h < H && 
                                in_w >= 0 && in_w < W) {
                                for (int ic = 0; ic < in_channels; ic++) {
                                    int in_idx = batch * (in_channels * D * H * W) +
                                                 ic * (D * H * W) +
                                                 in_d * (H * W) +
                                                 in_h * W +
                                                 in_w;
                                    int w_idx = oc * (in_channels * kD * kH * kW) +
                                                ic * (kD * kH * kW) +
                                                kd * (kH * kW) +
                                                kh * kW +
                                                kw;
                                    conv_sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                }

                // Add bias
                conv_sum += bias[oc];

                // Update minimum
                if (conv_sum < min_val) {
                    min_val = conv_sum;
                }
            }

            // Step 2: Softmax (across channels)
            smem[tid] = min_val;
            __syncthreads();

            // Reduction to find max in shared memory
            for (int stride = total_threads_per_block / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
                }
                __syncthreads();
            }

            float max_val = smem[0];
            __syncthreads();

            // Compute exponentials and sum
            float exp_val = expf(min_val - max_val);
            smem[tid] = exp_val;
            __syncthreads();

            // Reduction to compute sum in shared memory
            for (int stride = total_threads_per_block / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }

            float sum_exp = smem[0];
            __syncthreads();

            // Final softmax result
            float softmax_result = exp_val / sum_exp;

            // Write output
            int out_idx = batch * (out_channels * outH * outW) +
                          oc * (outH * outW) +
                          oh * outW +
                          ow;
            output[out_idx] = softmax_result;
        }
    }
}

void fused_conv_min_softmax_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilationD, int dilationH, int dilationW
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    const int out_channels = weight.size(0);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int outD = (D + 2 * padD - dilationD * (kD - 1) - 1) / strideD + 1;
    const int outH = (H + 2 * padH - dilationH * (kH - 1) - 1) / strideH + 1;
    const int outW = (W + 2 * padW - dilationW * (kW - 1) - 1) / strideW + 1;

    dim3 block(16, 16, 4); // OW, OH, OC
    dim3 grid((batch_size + 1 - 1) / 1, (out_channels + block.z - 1) / block.z);

    const int shared_mem_size = block.x * block.y * block.z * sizeof(float);

    fused_conv_min_softmax_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        D, H, W,
        kD, kH, kW,
        padD, padH, padW,
        strideD, strideH, strideW,
        dilationD, dilationH, dilationW,
        outD, outH, outW
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_softmax_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilationD, int dilationH, int dilationW
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_softmax", &fused_conv_min_softmax_op, "Fused Conv3D + Min + Softmax");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_softmax_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    dim,
):
    # Extract convolution parameters
    strideD, strideH, strideW = conv_stride
    padD, padH, padW = conv_padding
    dilationD, dilationH, dilationW = conv_dilation
    
    # Calculate output dimensions
    batch_size, in_channels, D, H, W = x.shape
    out_channels, _, kD, kH, kW = conv_weight.shape
    
    outD = (D + 2 * padD - dilationD * (kD - 1) - 1) // strideD + 1
    outH = (H + 2 * padH - dilationH * (kH - 1) - 1) // strideH + 1
    outW = (W + 2 * padW - dilationW * (kW - 1) - 1) // strideW + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, outH, outW, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_min_softmax(
        x, conv_weight, conv_bias, output,
        strideD, strideH, strideW,
        padD, padH, padW,
        dilationD, dilationH, dilationW
    )
    
    return output

# Parameters
batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
dim = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, W, H)]
