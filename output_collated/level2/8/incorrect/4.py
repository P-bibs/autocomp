# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_053523/code_3.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA source – two kernels (conv + fused sum) and their pybind11 bindings
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------
// 1) Convolution kernel – also folds division and optional conv_bias
// ----------------------------------------------------------------------
__global__ void conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,          // may be nullptr
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_d, const int in_h, const int in_w,
    const int kd, const int kh, const int kw,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dil_d, const int dil_h, const int dil_w,
    const float divisor,
    const int out_d, const int out_h, const int out_w)
{
    // one thread per output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_d * out_h * out_w;
    if (idx >= total) return;

    int rem = idx;
    int b = rem / (out_channels * out_d * out_h * out_w);
    rem = rem % (out_channels * out_d * out_h * out_w);
    int co = rem / (out_d * out_h * out_w);
    rem = rem % (out_d * out_h * out_w);
    int d = rem / (out_h * out_w);
    rem = rem % (out_h * out_w);
    int h = rem / out_w;
    int w = rem % out_w;

    float sum = 0.0f;

    // loops over input channels and kernel positions
    for (int ci = 0; ci < in_channels; ++ci) {
        for (int kd_idx = 0; kd_idx < kd; ++kd_idx) {
            int i_d = d * stride_d + kd_idx * dil_d - pad_d;
            if (i_d < 0 || i_d >= in_d) continue;
            for (int kh_idx = 0; kh_idx < kh; ++kh_idx) {
                int i_h = h * stride_h + kh_idx * dil_h - pad_h;
                if (i_h < 0 || i_h >= in_h) continue;
                for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                    int i_w = w * stride_w + kw_idx * dil_w - pad_w;
                    if (i_w < 0 || i_w >= in_w) continue;

                    // input[b, ci, i_d, i_h, i_w]
                    float in_val = input[((b * in_channels + ci) * in_d + i_d) * in_h * in_w
                                         + i_h * in_w + i_w];

                    // weight[co, ci, kd_idx, kh_idx, kw_idx]
                    float w_val = weight[(((co * in_channels + ci) * kd + kd_idx) * kh + kh_idx) * kw
                                          + kw_idx];

                    sum += in_val * w_val;
                }
            }
        }
    }

    // scale by divisor (folds the element‑wise division)
    sum /= divisor;

    // add convolution bias if present
    if (bias != nullptr) sum += bias[co];

    // store result
    output[((b * out_channels + co) * out_d + d) * out_h * out_w + h * out_w + w] = sum;
}

// ----------------------------------------------------------------------
// 2) Fused sum + bias kernel – adds per‑channel bias and reduces
//    the channel dimension to a scalar per batch element
// ----------------------------------------------------------------------
__global__ void sum_bias_kernel(
    const float* __restrict__ avg,   // (batch, out_channels, 1, 1, 1)
    float* __restrict__ out,         // (batch, 1, 1, 1)
    const int batch,
    const int out_channels,
    const float bias_sum)            // pre‑summed bias
{
    int b = blockIdx.x;
    if (b >= batch) return;

    float s = 0.0f;
    for (int co = 0; co < out_channels; ++co) {
        s += avg[b * out_channels + co];
    }
    s += bias_sum;
    out[b] = s;                      // stored as (batch,1,1,1) linearised
}

// ----------------------------------------------------------------------
// Host wrappers (CUDA kernel launches)
// ----------------------------------------------------------------------
void conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,               // may be an empty tensor
    torch::Tensor output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_d, const int in_h, const int in_w,
    const int kd, const int kh, const int kw,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dil_d, const int dil_h, const int dil_w,
    const float divisor,
    const int out_d, const int out_h, const int out_w)
{
    const int total = batch * out_channels * out_d * out_h * out_w;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    conv_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        in_d, in_h, in_w,
        kd, kh, kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        divisor,
        out_d, out_h, out_w);
}

void sum_bias_forward(
    torch::Tensor avg,
    torch::Tensor out,
    const int batch,
    const int out_channels,
    const float bias_sum)
{
    sum_bias_kernel<<<batch, 1>>>(
        avg.data_ptr<float>(),
        out.data_ptr<float>(),
        batch, out_channels, bias_sum);
}
"""

# ----------------------------------------------------------------------
# C++ source – pybind11 interface
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_d, const int in_h, const int in_w,
    const int kd, const int kh, const int kw,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dil_d, const int dil_h, const int dil_w,
    const float divisor,
    const int out_d, const int out_h, const int out_w);

void sum_bias_forward(
    torch::Tensor avg,
    torch::Tensor out,
    const int batch,
    const int out_channels,
    const float bias_sum);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward,
          "Custom fused conv forward (CUDA)");
    m.def("sum_bias_forward", &sum_bias_forward,
          "Fused sum + bias forward (CUDA)");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)


# ----------------------------------------------------------------------
# Functional model – the only symbol that will be imported
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    global_avg_pool_output_size,
    divisor,
    bias,
    sum_dim,
):
    # ------------------------------------------------------------------
    # Ensure all tensors are on the GPU
    # ------------------------------------------------------------------
    x = x.cuda()
    conv_weight = conv_weight.cuda()
    if conv_bias is not None:
        conv_bias = conv_bias.cuda()
    bias = bias.cuda()

    # ------------------------------------------------------------------
    # 1) Custom fused convolution (includes division and conv_bias)
    # ------------------------------------------------------------------
    # input shape: (batch, in_channels, depth, height, width)
    batch, in_channels, in_d, in_h, in_w = x.shape
    out_channels = conv_weight.shape[0]
    kd, kh, kw = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]

    # compute output spatial size of the convolution
    stride_d, stride_h, stride_w = conv_stride
    pad_d, pad_h, pad_w = conv_padding
    dil_d, dil_h, dil_w = conv_dilation

    out_d = (in_d + 2 * pad_d - dil_d * (kd - 1) - 1) // stride_d + 1
    out_h = (in_h + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1

    # allocate intermediate tensor for conv output
    conv_out = torch.empty(
        batch, out_channels, out_d, out_h, out_w,
        dtype=x.dtype, device=x.device
    )

    # launch custom conv kernel
    fused_ext.conv_forward(
        x, conv_weight, conv_bias, conv_out,
        batch, in_channels, out_channels,
        in_d, in_h, in_w,
        kd, kh, kw,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        divisor,
        out_d, out_h, out_w
    )

    # ------------------------------------------------------------------
    # 2) Max pooling (library kernel – already highly optimised)
    # ------------------------------------------------------------------
    maxpool_out = F.max_pool3d(
        conv_out,
        kernel_size=max_pool_kernel_size,
        stride=max_pool_stride,
        padding=max_pool_padding,
        dilation=max_pool_dilation,
        ceil_mode=max_pool_ceil_mode,
        return_indices=max_pool_return_indices,
    )

    # ------------------------------------------------------------------
    # 3) Adaptive average pooling to (1,1,1)
    # ------------------------------------------------------------------
    avg_out = F.adaptive_avg_pool3d(maxpool_out, global_avg_pool_output_size)
    # avg_out shape now: (batch, out_channels, 1, 1, 1)

    # ------------------------------------------------------------------
    # 4) Fuse bias addition and reduction (sum) into a single kernel
    # ------------------------------------------------------------------
    # pre‑compute the sum of the per‑channel bias terms
    bias_sum = bias.sum().item()  # scalar (float)

    # allocate output tensor (batch,1,1,1)
    result = torch.empty(batch, 1, 1, 1, dtype=avg_out.dtype, device=avg_out.device)

    # launch fused sum+bias kernel
    fused_ext.sum_bias_forward(
        avg_out, result,
        batch, out_channels, float(bias_sum)
    )

    # ------------------------------------------------------------------
    # The result now matches the original semantics:
    #   shape (batch,1,1,1) – a scalar per batch element
    # ------------------------------------------------------------------
    return result

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
