# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_0.py
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

# --- CUDA Kernels ---
cuda_kernels = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline int divUp(int a, int b) { return (a + b - 1) / b; }

__global__ void conv3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int Ci, int Di, int Hi, int Wi,
    int Co, int Dk, int Hk, int Wk,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    int dil_d, int dil_h, int dil_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = N * Co * 
        ((Di + 2 * pad_d - dil_d * (Dk - 1) - 1) / stride_d + 1) *
        ((Hi + 2 * pad_h - dil_h * (Hk - 1) - 1) / stride_h + 1) *
        ((Wi + 2 * pad_w - dil_w * (Wk - 1) - 1) / stride_w + 1);

    if (idx >= total_output_elements) return;

    int tmp = idx;
    int w_out = tmp % ((Wi + 2 * pad_w - dil_w * (Wk - 1) - 1) / stride_w + 1); tmp /= ((Wi + 2 * pad_w - dil_w * (Wk - 1) - 1) / stride_w + 1);
    int h_out = tmp % ((Hi + 2 * pad_h - dil_h * (Hk - 1) - 1) / stride_h + 1); tmp /= ((Hi + 2 * pad_h - dil_h * (Hk - 1) - 1) / stride_h + 1);
    int d_out = tmp % ((Di + 2 * pad_d - dil_d * (Dk - 1) - 1) / stride_d + 1); tmp /= ((Di + 2 * pad_d - dil_d * (Dk - 1) - 1) / stride_d + 1);
    int co = tmp % Co; tmp /= Co;
    int n = tmp;

    float sum = 0.0f;
    for (int ci = 0; ci < Ci; ++ci) {
        for (int dk = 0; dk < Dk; ++dk) {
            for (int hk = 0; hk < Hk; ++hk) {
                for (int wk = 0; wk < Wk; ++wk) {
                    int d_in = d_out * stride_d - pad_d + dk * dil_d;
                    int h_in = h_out * stride_h - pad_h + hk * dil_h;
                    int w_in = w_out * stride_w - pad_w + wk * dil_w;

                    if (d_in >= 0 && d_in < Di &&
                        h_in >= 0 && h_in < Hi &&
                        w_in >= 0 && w_in < Wi) {
                        int input_idx = n * (Ci * Di * Hi * Wi) +
                                        ci * (Di * Hi * Wi) +
                                        d_in * (Hi * Wi) +
                                        h_in * Wi + w_in;
                        int weight_idx = co * (Ci * Dk * Hk * Wk) +
                                         ci * (Dk * Hk * Wk) +
                                         dk * (Hk * Wk) +
                                         hk * Wk + wk;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    output[idx] = sum + bias[co];
}

__global__ void max_pool3d_kernel(
    const float* input, float* output,
    int N, int C, int Di, int Hi, int Wi,
    int Dk, int Hk, int Wk,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Do = (Di + 2 * pad_d - Dk) / stride_d + 1;
    int Ho = (Hi + 2 * pad_h - Hk) / stride_h + 1;
    int Wo = (Wi + 2 * pad_w - Wk) / stride_w + 1;
    int total = N * C * Do * Ho * Wo;

    if (idx >= total) return;

    int tmp = idx;
    int wo = tmp % Wo; tmp /= Wo;
    int ho = tmp % Ho; tmp /= Ho;
    int doo = tmp % Do; tmp /= Do;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    float max_val = -1e30f;
    for (int dk = 0; dk < Dk; ++dk) {
        for (int hk = 0; hk < Hk; ++hk) {
            for (int wk = 0; wk < Wk; ++wk) {
                int d_in = doo * stride_d - pad_d + dk;
                int h_in = ho * stride_h - pad_h + hk;
                int w_in = wo * stride_w - pad_w + wk;

                if (d_in >= 0 && d_in < Di &&
                    h_in >= 0 && h_in < Hi &&
                    w_in >= 0 && w_in < Wi) {
                    int input_idx = n * (C * Di * Hi * Wi) +
                                    c * (Di * Hi * Wi) +
                                    d_in * (Hi * Wi) +
                                    h_in * Wi + w_in;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
    }
    output[idx] = max_val;
}

__global__ void adaptive_avg_pool3d_kernel(
    const float* input, float* output,
    int N, int C, int Di, int Hi, int Wi,
    int Do, int Ho, int Wo
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Do * Ho * Wo;

    if (idx >= total) return;

    int tmp = idx;
    int wo = tmp % Wo; tmp /= Wo;
    int ho = tmp % Ho; tmp /= Ho;
    int doo = tmp % Do; tmp /= Do;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    int d_start = (doo * Di) / Do;
    int d_end = ((doo + 1) * Di + Do - 1) / Do;
    int h_start = (ho * Hi) / Ho;
    int h_end = ((ho + 1) * Hi + Ho - 1) / Ho;
    int w_start = (wo * Wi) / Wo;
    int w_end = ((wo + 1) * Wi + Wo - 1) / Wo;

    float sum = 0.0f;
    int count = 0;

    for (int d = d_start; d < d_end; ++d) {
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_idx = n * (C * Di * Hi * Wi) +
                                c * (Di * Hi * Wi) +
                                d * (Hi * Wi) + h * Wi + w;
                sum += input[input_idx];
                count++;
            }
        }
    }
    output[idx] = sum / count;
}

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float divisor,
    int N, int C, int D, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D * H * W;

    if (idx < total_elements) {
        int n = idx / (D * H * W);
        int d = (idx / (H * W)) % D;
        int h = (idx / W) % H;
        int w = idx % W;

        float sum_val = 0.0f;
        for (int c = 0; c < C; ++c) {
            int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            sum_val += (input[input_idx] / divisor) + bias[c];
        }
        output[idx] = sum_val;
    }
}

// Exposed Functions to Python
void launch_conv3d(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    int dil_d, int dil_h, int dil_w
) {
    int N = input.size(0);
    int Ci = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);

    int Co = weight.size(0);
    int Dk = weight.size(2);
    int Hk = weight.size(3);
    int Wk = weight.size(4);

    int Do = (Di + 2 * pad_d - dil_d * (Dk - 1) - 1) / stride_d + 1;
    int Ho = (Hi + 2 * pad_h - dil_h * (Hk - 1) - 1) / stride_h + 1;
    int Wo = (Wi + 2 * pad_w - dil_w * (Wk - 1) - 1) / stride_w + 1;

    int threads = 256;
    int total_threads = N * Co * Do * Ho * Wo;
    int blocks = divUp(total_threads, threads);

    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, Ci, Di, Hi, Wi,
        Co, Dk, Hk, Wk,
        pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, dil_d, dil_h, dil_w
    );
}

void launch_max_pool3d(
    torch::Tensor input, torch::Tensor output,
    int Dk, int Hk, int Wk,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w
) {
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);

    int threads = 256;
    int total_threads = N * C * 
        ((Di + 2 * pad_d - Dk) / stride_d + 1) *
        ((Hi + 2 * pad_h - Hk) / stride_h + 1) *
        ((Wi + 2 * pad_w - Wk) / stride_w + 1);
    int blocks = divUp(total_threads, threads);

    max_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi,
        Dk, Hk, Wk, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w
    );
}

void launch_adaptive_avg_pool3d(
    torch::Tensor input, torch::Tensor output,
    int Do, int Ho, int Wo
) {
    int N = input.size(0);
    int C = input.size(1);
    int Di = input.size(2);
    int Hi = input.size(3);
    int Wi = input.size(4);

    int threads = 256;
    int total_threads = N * C * Do * Ho * Wo;
    int blocks = divUp(total_threads, threads);

    adaptive_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, Di, Hi, Wi, Do, Ho, Wo
    );
}

void launch_fused_op(
    torch::Tensor input, torch::Tensor bias, torch::Tensor output, float divisor
) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int threads = 256;
    int blocks = divUp(N * D * H * W, threads);

    fused_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        divisor, N, C, D, H, W
    );
}
"""

# C++ Interface
cpp_source = r"""
#include <torch/extension.h>

void launch_conv3d(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int, int, int, int, int);
void launch_max_pool3d(torch::Tensor, torch::Tensor, int, int, int, int, int, int, int, int, int);
void launch_adaptive_avg_pool3d(torch::Tensor, torch::Tensor, int, int, int);
void launch_fused_op(torch::Tensor, torch::Tensor, torch::Tensor, float);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv3d", &launch_conv3d, "Conv3d CUDA kernel");
    m.def("launch_max_pool3d", &launch_max_pool3d, "MaxPool3d CUDA kernel");
    m.def("launch_adaptive_avg_pool3d", &launch_adaptive_avg_pool3d, "AdaptiveAvgPool3d CUDA kernel");
    m.def("launch_fused_op", &launch_fused_op, "Fused op kernel");
}
"""

# --- Compile CUDA Extension ---
fused_ext = load_inline(
    name='fused_custom_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernels,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
    max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation,
    max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size,
    divisor, bias, sum_dim,
):
    # Conv3d
    conv_out = torch.empty(
        (x.size(0), conv_weight.size(0),
         (x.size(2) + 2 * conv_padding[0] - conv_dilation[0] * (conv_weight.size(2) - 1) - 1) // conv_stride[0] + 1,
         (x.size(3) + 2 * conv_padding[1] - conv_dilation[1] * (conv_weight.size(3) - 1) - 1) // conv_stride[1] + 1,
         (x.size(4) + 2 * conv_padding[2] - conv_dilation[2] * (conv_weight.size(4) - 1) - 1) // conv_stride[2] + 1),
        device=x.device, dtype=x.dtype
    )
    fused_ext.launch_conv3d(x, conv_weight, conv_bias, conv_out,
                            conv_padding[0], conv_padding[1], conv_padding[2],
                            conv_stride[0], conv_stride[1], conv_stride[2],
                            conv_dilation[0], conv_dilation[1], conv_dilation[2])

    # MaxPool3d
    maxpool_out = torch.empty(
        (conv_out.size(0), conv_out.size(1),
         (conv_out.size(2) + 2 * max_pool_padding[0] - max_pool_kernel_size[0]) // max_pool_stride[0] + 1,
         (conv_out.size(3) + 2 * max_pool_padding[1] - max_pool_kernel_size[1]) // max_pool_stride[1] + 1,
         (conv_out.size(4) + 2 * max_pool_padding[2] - max_pool_kernel_size[2]) // max_pool_stride[2] + 1),
        device=x.device, dtype=x.dtype
    )
    fused_ext.launch_max_pool3d(conv_out, maxpool_out,
                                max_pool_kernel_size[0], max_pool_kernel_size[1], max_pool_kernel_size[2],
                                max_pool_stride[0], max_pool_stride[1], max_pool_stride[2],
                                max_pool_padding[0], max_pool_padding[1], max_pool_padding[2])

    # AdaptiveAvgPool3d
    out_pool = torch.empty((maxpool_out.size(0), maxpool_out.size(1),
                            global_avg_pool_output_size[0], global_avg_pool_output_size[1], global_avg_pool_output_size[2]),
                           device=x.device, dtype=x.dtype)
    fused_ext.launch_adaptive_avg_pool3d(maxpool_out, out_pool,
                                         global_avg_pool_output_size[0],
                                         global_avg_pool_output_size[1],
                                         global_avg_pool_output_size[2])

    # Fused Sum
    N, C, D, H, W = out_pool.shape
    output = torch.zeros(N, D, H, W, device=out_pool.device, dtype=out_pool.dtype)
    fused_ext.launch_fused_op(out_pool, bias.view(-1), output, divisor)

    return output

# Test parameters
batch_size = 128
in_channels = 8
out_channels = 16
depth = 16
height = 64
width = 64
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs(): return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
def get_inputs(): return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
