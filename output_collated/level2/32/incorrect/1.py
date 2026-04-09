# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_010153/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

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
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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

# -------------------------------------------------------------------------
# CUDA source: kernel + host wrapper
# The kernel maps each output spatial pixel (N, H_out, W_out) to a thread block
# of size (C_out). Shared memory is used to store the input patch once 
# per tile and the resulting channel list before a parallel reduction.
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N, const int C_in, const int C_out,
    const int H, const int W,
    const int kH, const int kW,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int out_h, const int out_w,
    const float scale,
    float* __restrict__ output)
{
    // Each block processes one spatial output location (N, oh, ow)
    const int b = blockIdx.x;
    const int oh = blockIdx.y;
    const int ow = blockIdx.z;
    const int c = threadIdx.x;

    const int patch_size = C_in * kH * kW;
    extern __shared__ float shmem[];
    float* patch = shmem; 
    float* channel_vals = shmem + patch_size;

    // Load input patch into shared memory cooperatively
    for (int i = threadIdx.x; i < patch_size; i += blockDim.x) {
        int ic = i / (kH * kW);
        int rem = i % (kH * kW);
        int kh = rem / kW;
        int kw = rem % kW;
        int ih = oh * stride_h + kh * dilation_h - pad_h;
        int iw = ow * stride_w + kw * dilation_w - pad_w;

        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            patch[i] = input[((b * C_in + ic) * H + ih) * W + iw];
        } else {
            patch[i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute convolution for thread's channel
    if (c < C_out) {
        float acc = 0.0f;
        const float* w_ptr = weight + c * patch_size;
        for (int i = 0; i < patch_size; ++i) {
            acc += patch[i] * w_ptr[i];
        }
        if (bias != nullptr) acc += bias[c];
        channel_vals[c] = acc * scale;
    }
    __syncthreads();

    // Parallel reduction to find min across C_out
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (c < offset && (c + offset) < C_out) {
            float a = channel_vals[c];
            float b = channel_vals[c + offset];
            if (b < a) channel_vals[c] = b;
        }
        __syncthreads();
    }

    if (c == 0) {
        output[((b * out_h + oh) * out_w + ow)] = channel_vals[0];
    }
}

void fused_op_forward(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    int stride, int padding, int dilation, float scale_factor, torch::Tensor& output)
{
    int N = input.size(0), C_in = input.size(1), H = input.size(2), W = input.size(3);
    int C_out = weight.size(0), kH = weight.size(2), kW = weight.size(3);
    int out_h = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int out_w = (W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    // Use C_out threads. Shared memory holds patch + C_out results
    int shm_size = (C_in * kH * kW + C_out) * sizeof(float);
    dim3 grid(N, out_h, out_w);
    
    fused_op_kernel<<<grid, C_out, shm_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        (bias.numel() > 0 ? bias.data_ptr<float>() : nullptr),
        N, C_in, C_out, H, W, kH, kW, stride, stride, padding, padding, 
        dilation, dilation, out_h, out_w, scale_factor, output.data_ptr<float>()
    );
}
"""

cpp_source = r"""
void fused_op_forward(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int, int, int, float, torch::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_op", &fused_op_forward); }
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, scale_factor):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = conv_weight.shape
    out_h = (H + 2 * conv_padding - conv_dilation * (kH - 1) - 1) // conv_stride + 1
    out_w = (W + 2 * conv_padding - conv_dilation * (kW - 1) - 1) // conv_stride + 1
    out = torch.empty((N, 1, out_h, out_w), device=x.device, dtype=x.dtype)
    bias = conv_bias if conv_bias is not None else torch.tensor([], device=x.device)
    fused_ext.fused_op(x.contiguous(), conv_weight.contiguous(), bias.contiguous(), conv_stride, conv_padding, conv_dilation, scale_factor, out)
    return out
