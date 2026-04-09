# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_064410/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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
#include <cub/cub.cuh>

// Helper macro for computing 3D index from 5D tensor (N, C, D, H, W)
#define INDEX_5D(N, C, D, H, W, n, c, d, h, w) ((((n)*(C) + (c))*(D) + (d))*(H)*(W) + (h)*(W) + (w))

// Fused kernel performing the entire pipeline:
// Conv3D (implicit GEMM) -> MaxPool3D -> AdaptiveAvgPool3D -> Channel Summation -> Bias/Div
// This avoids writing out huge intermediate tensors to global memory.
__global__ void fused_pipeline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ final_bias,
    float divisor,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out,
    int KD, int KH, int KW,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int maxpool_kd, int maxpool_kh, int maxpool_kw,
    int maxpool_stride_d, int maxpool_stride_h, int maxpool_stride_w,
    int maxpool_pad_d, int maxpool_pad_h, int maxpool_pad_w,
    int global_d, int global_h, int global_w,
    float* __restrict__ output)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int block_id = blockIdx.x;

    int od = block_id / (global_h * global_w);
    int oh = (block_id / global_w) % global_h;
    int ow = block_id % global_w;

    if (od >= global_d || oh >= global_h || ow >= global_w) return;

    float sum_val = 0.0f;

    // Iterate over output channels
    for (int oc = 0; oc < C_out; ++oc) {
        float conv_result = 0.0f;

        // Compute starting coordinates in conv input space
        int base_d = od * maxpool_stride_d - maxpool_pad_d;
        int base_h = oh * maxpool_stride_h - maxpool_pad_h;
        int base_w = ow * maxpool_stride_w - maxpool_pad_w;

        float pooled_max = -1e30f;
        
        // Sliding window for maxpool region
        for (int pd = 0; pd < maxpool_kd; ++pd) {
            int curr_d = base_d + pd;
            for (int ph = 0; ph < maxpool_kh; ++ph) {
                int curr_h = base_h + ph;
                for (int pw = 0; pw < maxpool_kw; ++pw) {
                    int curr_w = base_w + pw;

                    // Check pooling bounds and apply avg pool sampling logic
                    if (curr_d >= 0 && curr_h >= 0 && curr_w >= 0 &&
                        curr_d < (D_in - KD + 1 + 2*conv_pad_d)/conv_stride_d &&
                        curr_h < (H_in - KH + 1 + 2*conv_pad_h)/conv_stride_h &&
                        curr_w < (W_in - KW + 1 + 2*conv_pad_w)/conv_stride_w) {

                        float reduced_val = 0.0f;
                        int conv_d = curr_d * conv_stride_d - conv_pad_d;
                        int conv_h = curr_h * conv_stride_h - conv_pad_h;
                        int conv_w = curr_w * conv_stride_w - conv_pad_w;

                        // Convolution per spatial location
                        for (int ic = 0; ic < C_in; ++ic) {
                            for (int kd = 0; kd < KD; ++kd) {
                                for (int kh = 0; kh < KH; ++kh) {
                                    for (int kw = 0; kw < KW; ++kw) {
                                        int in_d = conv_d + kd;
                                        int in_h = conv_h + kh;
                                        int in_w = conv_w + kw;

                                        if (in_d >= 0 && in_d < D_in &&
                                            in_h >= 0 && in_h < H_in &&
                                            in_w >= 0 && in_w < W_in) {
                                            float in_val = input[INDEX_5D(N, C_in, D_in, H_in, W_in, tid % N, ic, in_d, in_h, in_w)];
                                            float wt_val = weight[INDEX_5D(C_out, C_in, KD, KH, KW, oc, ic, kd, kh, kw)];
                                            reduced_val += in_val * wt_val;
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Add bias
                        reduced_val += conv_bias[oc];
                        
                        // Take max across the pooling window
                        if (reduced_val > pooled_max) {
                            pooled_max = reduced_val;
                        }
                    }
                }
            }
        }

        // Use pooled_max for this spatial position and channel
        sum_val += pooled_max;
    }

    // Shared memory reduction for final bias addition and division
    sdata[tid] = sum_val;
    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[block_id] = (sdata[0] / divisor) + final_bias[0]; // Final scalar bias assumed uniform
    }
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor final_bias, float divisor,
    int KD, int KH, int KW,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int maxpool_kd, int maxpool_kh, int maxpool_kw,
    int maxpool_stride_d, int maxpool_stride_h, int maxpool_stride_w,
    int maxpool_pad_d, int maxpool_pad_h, int maxpool_pad_w,
    int global_d, int global_h, int global_w,
    torch::Tensor output)
{
    int N = input.size(0);
    int threads = 256;
    int blocks = global_d * global_h * global_w;
    size_t shared_mem_size = threads * sizeof(float);

    fused_pipeline_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        final_bias.data_ptr<float>(),
        divisor,
        N, weight.size(1), input.size(2), input.size(3), input.size(4),
        weight.size(0),
        KD, KH, KW,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        maxpool_kd, maxpool_kh, maxpool_kw,
        maxpool_stride_d, maxpool_stride_h, maxpool_stride_w,
        maxpool_pad_d, maxpool_pad_h, maxpool_pad_w,
        global_d, global_h, global_w,
        output.data_ptr<float>()
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor final_bias, float divisor,
    int KD, int KH, int KW,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int maxpool_kd, int maxpool_kh, int maxpool_kw,
    int maxpool_stride_d, int maxpool_stride_h, int maxpool_stride_w,
    int maxpool_pad_d, int maxpool_pad_h, int maxpool_pad_w,
    int global_d, int global_h, int global_w,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv+Pool+Reduction Kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation,
                     conv_groups, max_pool_kernel_size, max_pool_stride, max_pool_padding,
                     max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices,
                     global_avg_pool_output_size, divisor, bias, sum_dim):
    
    assert conv_groups == 1, "Only conv_groups=1 is supported"
    assert isinstance(conv_stride, tuple) and len(conv_stride) == 3
    assert isinstance(conv_padding, tuple) and len(conv_padding) == 3
    assert isinstance(max_pool_kernel_size, tuple) and len(max_pool_kernel_size) == 3
    assert isinstance(max_pool_stride, tuple) and len(max_pool_stride) == 3
    assert isinstance(max_pool_padding, tuple) and len(max_pool_padding) == 3
    
    total_bias = bias.sum()

    out_d, out_h, out_w = global_avg_pool_output_size
    
    output = torch.empty([x.size(0), out_d * out_h * out_w], device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x, conv_weight, conv_bias, total_bias.unsqueeze(0),
        float(divisor),
        conv_weight.size(2), conv_weight.size(3), conv_weight.size(4), # KD, KH, KW
        conv_stride[0], conv_stride[1], conv_stride[2],
        conv_padding[0], conv_padding[1], conv_padding[2],
        max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2],
        max_pool_stride[0], max_pool_stride[1], max_pool_stride[2],
        max_pool_padding[0], max_pool_padding[1], max_pool_padding[2],
        out_d, out_h, out_w,
        output
    )
    
    return output.view(x.size(0), out_d, out_h, out_w)
