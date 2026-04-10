# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_115141/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# ----------------------------------------------------------------------
# CUDA source – the fused conv‑transpose + clamp + scale kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fusion kernel: conv_transpose3d + clamp + scale
// block.x : spatial threads (multiple of 32)
// block.y : number of output channels handled simultaneously (here 4)
__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,      // (N, C_in, D_in, H_in, W_in)
    const float* __restrict__ weight,     // (C_in, C_out, K, K, K)
    const float* __restrict__ bias,       // (C_out)  or nullptr
    float* __restrict__ output,           // (N, C_out, D_out, H_out, W_out)
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int stride, const int padding,
    const int output_padding, const int dilation,
    const int K, const float min_val, const float divisor)
{
    // ----- shared memory for a weight slice (block.y output channels) -----
    extern __shared__ float s_weight[];               // size = block.y * C_in * K^3
    const int block_y = blockDim.y;
    const int oc_start = blockIdx.y * block_y;        // first output channel for this block

    // ---- load weight slice into shared memory ----
    const int weight_per_oc = C_in * K * K * K;       // 64 * 27 = 1728
    const int weight_slice_sz = block_y * weight_per_oc;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int tot_thr = blockDim.x * blockDim.y;
    for (int idx = tid; idx < weight_slice_sz; idx += tot_thr) {
        int oc_inner = idx / weight_per_oc;               // 0 … block_y-1
        int idx_remain = idx % weight_per_oc;
        int ic = idx_remain / (K * K * K);                // input channel
        int k_idx = idx_remain % (K * K * K);             // flattened kernel index
        int oc = oc_start + oc_inner;                     // global output channel
        // global weight index: ((ic * C_out + oc) * K^3 + k_idx)
        int w_idx = ((ic * C_out + oc) * (K * K * K) + k_idx);
        s_weight[idx] = weight[w_idx];
    }
    __syncthreads();

    // ---- which output element does this thread compute? ----
    const long long total_spatial = (long long)N * D_out * H_out * W_out;
    const long long linear_out = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_out >= total_spatial) return;

    // decode (n, od, oh, ow)
    int n = linear_out / (D_out * H_out * W_out);
    int rem = linear_out % (D_out * H_out * W_out);
    int od = rem / (H_out * W_out);
    rem = rem % (H_out * W_out);
    int oh = rem / W_out;
    int ow = rem % W_out;

    // which output channel (within the block)
    int oc = oc_start + threadIdx.y;
    if (oc >= C_out) return;

    // ----- convolution accumulation -----
    float sum = 0.0f;
    for (int kd = 0; kd < K; ++kd) {
        int offset_d = od + padding - kd * dilation;
        if (offset_d % stride != 0) continue;
        int id = offset_d / stride;
        if (id < 0 || id >= D_in) continue;

        for (int kh = 0; kh < K; ++kh) {
            int offset_h = oh + padding - kh * dilation;
            if (offset_h % stride != 0) continue;
            int ih = offset_h / stride;
            if (ih < 0 || ih >= H_in) continue;

            for (int kw = 0; kw < K; ++kw) {
                int offset_w = ow + padding - kw * dilation;
                if (offset_w % stride != 0) continue;
                int iw = offset_w / stride;
                if (iw < 0 || iw >= W_in) continue;

                // input offset for every input channel
                const long long in_base = ((long long)n * C_in + 0) * (long long)D_in * H_in * W_in; // placeholder
                // we will compute the exact offset inside the ic‑loop
                for (int ic = 0; ic < C_in; ++ic) {
                    long long in_offset = ((long long)n * C_in + ic) * (long long)D_in * H_in * W_in
                                         + (long long)id * H_in * W_in + (long long)ih * W_in + iw;
                    float in_val = input[in_offset];

                    // weight index inside shared‑memory slice
                    int w_shared = ((ic * block_y + threadIdx.y) * (K * K * K)
                                   + (kd * K * K + kh * K + kw));
                    float w_val = s_weight[w_shared];
                    sum += in_val * w_val;
                }
            }
        }
    }

    // ----- bias, clamp & scale -----
    if (bias != nullptr) {
        sum += bias[oc];
    }
    if (sum < min_val) sum = min_val;
    sum = sum / divisor;

    // ---- store result ----
    long long out_offset = ((long long)n * C_out + oc) * (long long)D_out * H_out * W_out
                         + (long long)od * H_out * W_out + (long long)oh * W_out + ow;
    output[out_offset] = sum;
}

// Host wrapper that launches the kernel with the correct grid/block size
void fused_conv_transpose(
    const torch::Tensor& input,   // (N, C_in, D_in, H_in, W_in)
    const torch::Tensor& weight, // (C_in, C_out, K, K, K)
    const torch::Tensor& bias,   // (C_out) or empty
    torch::Tensor& output,       // (N, C_out, D_out, H_out, W_out)
    const int stride,
    const int padding,
    const int output_padding,
    const int dilation,
    const int K,
    const float min_val,
    const float divisor)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(1);
    // output spatial size (standard transposed‑conv formula)
    const int D_out = (D_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    const int H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;
    const int W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1;

    const int block_x = 256;                 // multiple of 32, good occupancy
    const int block_y = 4;                   // number of output channels per block
    const long long total_spatial = (long long)N * D_out * H_out * W_out;
    const int grid_x = (total_spatial + block_x - 1) / block_x;
    const int grid_y = (C_out + block_y - 1) / block_y;

    const dim3 grid(grid_x, grid_y);
    const dim3 block(block_x, block_y);

    // shared memory needed for the weight slice
    const int smem_size = block_y * C_in * K * K * K * sizeof(float);

    const float* bias_ptr = bias.numel() ? bias.data_ptr<float>() : nullptr;

    conv_transpose_fused_kernel<<<grid, block, smem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, output_padding, dilation,
        K, min_val, divisor);
}
"""

# ----------------------------------------------------------------------
# C++ binding – PyBind11 interface
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int stride,
    const int padding,
    const int output_padding,
    const int dilation,
    const int K,
    const float min_val,
    const float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &fused_conv_transpose,
          "Fused transposed 3‑D convolution with clamp and scale");
}
"""

# ----------------------------------------------------------------------
# Compile the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# The functional model that will be imported / evaluated
# ----------------------------------------------------------------------
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
    min_value,
    divisor,
):
    # Determine kernel size from weight shape (assumes cubic kernel)
    K = conv_transpose_weight.size(2)   # kernel size in depth (same for height & width)

    # Output spatial size (standard formula for transposed conv)
    D_out = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    H_out = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    W_out = (x.size(4) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + \
            conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1

    # Allocate output tensor
    out = torch.empty((x.size(0), conv_transpose_weight.size(1), D_out, H_out, W_out),
                      dtype=x.dtype, device=x.device)

    # Launch the fused CUDA kernel
    fused_ext.fused_conv_transpose(
        x,
        conv_transpose_weight,
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device=x.device),
        out,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation,
        K,
        min_value,
        divisor
    )
    return out


# ----------------------------------------------------------------------
# Helpers required by the evaluation harness
# ----------------------------------------------------------------------
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
