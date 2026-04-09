# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_084802/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
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

# --------------------------------------------------------------------------
# CUDA source – Fused: Convolution → Channel-Min → Double Tanh
# --------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N, const int C_in, const int C_out,
    const int H, const int W,
    const int K, const int stride, const int padding)
{
    const int OH = (H + 2 * padding - K) / stride + 1;
    const int OW = (W + 2 * padding - K) / stride + 1;
    const int pixel_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (pixel_idx >= N * OH * OW) return;

    const int n = pixel_idx / (OH * OW);
    const int residual = pixel_idx % (OH * OW);
    const int oh = residual / OW;
    const int ow = residual % OW;

    // Allocate shared memory for the input patch (Max: 16 channels * 3 * 3 = 144)
    extern __shared__ float patch[];

    // 1. Load input patch into shared memory
    for (int i = tid; i < C_in * K * K; i += blockDim.x) {
        int ci = i / (K * K);
        int kh = (i / K) % K;
        int kw = i % K;
        int ih = oh * stride + kh - padding;
        int iw = ow * stride + kw - padding;

        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            patch[i] = x[((n * C_in + ci) * H + ih) * W + iw];
        } else {
            patch[i] = 0.0f;
        }
    }
    __syncthreads();

    // 2. Convolution dot product for the current output channel (tid)
    float sum = bias[tid];
    for (int i = 0; i < C_in * K * K; ++i) {
        // Weight index: (co * C_in * K * K) + (ci * K * K) + (kh * K) + kw
        sum += patch[i] * weight[tid * (C_in * K * K) + i];
    }

    // 3. Block-wide reduction for Min(C_out)
    extern __shared__ float sdata[]; // Reuse or separate based on size
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 4. Double Tanh Activation
    if (tid == 0) {
        float val = sdata[0];
        out[pixel_idx] = tanhf(tanhf(val));
    }
}

void launch_fused_op(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor out, int stride, int padding) 
{
    const int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    const int C_out = weight.size(0), K = weight.size(2);
    const int OH = (H + 2 * padding - K) / stride + 1;
    const int OW = (W + 2 * padding - K) / stride + 1;

    // Shared memory: patch(C_in*K*K) + reduction_buffer(C_out)
    size_t shared_size = (C_in * K * K + C_out) * sizeof(float);
    fused_conv_min_tanh_kernel<<<N * OH * OW, C_out, shared_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), N, C_in, C_out, H, W, K, stride, padding);
}
"""

cpp_source = r"""
void launch_fused_op(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_op", &launch_fused_op, "Fused Conv-Min-Tanh Kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = conv_weight.contiguous()
    b = conv_bias.contiguous()
    
    N, C_in, H, W = x.shape
    C_out, _, K, _ = w.shape
    OH = (H + 2 * conv_padding - K) // conv_stride + 1
    OW = (W + 2 * conv_padding - K) // conv_stride + 1
    
    out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.launch_fused_op(x, w, b, out, conv_stride, conv_padding)
    return out
