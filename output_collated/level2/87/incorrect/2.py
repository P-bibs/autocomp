# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140617/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

extern "C" __global__
void fused_op_forward_kernel(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N,
    const int C_in,
    const int C_out,
    const int H_in,
    const int W_in,
    const int K_h,
    const int K_w,
    const int stride,
    const int pad,
    const float sub1,
    const float sub2)
{
    // Tile size
    const int TILE_H = 8;
    const int TILE_W = 8;

    // Decode global block index
    int idx = blockIdx.x;
    const int W_out = (W_in + 2 * pad - K_w) / stride + 1;
    const int H_out = (H_in + 2 * pad - K_h) / stride + 1;
    const int tiles_w = (W_out + TILE_W - 1) / TILE_W;
    const int tiles_h = (H_out + TILE_H - 1) / TILE_H;

    const int w_tile = idx % tiles_w;
    const int h_tile = (idx / tiles_w) % tiles_h;
    const int c_out = (idx / (tiles_w * tiles_h)) % C_out;
    const int n = idx / (tiles_w * tiles_h * C_out);

    // Thread coordinates inside the tile
    const int tx = threadIdx.x; // [0,TILE_W)
    const int ty = threadIdx.y; // [0,TILE_H)

    // Global output coordinates this thread will compute
    const int out_h = h_tile * TILE_H + ty;
    const int out_w = w_tile * TILE_W + tx;

    // Bounds check
    if (out_h >= H_out || out_w >= W_out) return;

    // Convolution computation
    float acc = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Loop over input channels and kernel window
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                // Calculate input coordinates
                int in_h = out_h * stride - pad + kh;
                int in_w = out_w * stride - pad + kw;

                // Load input value with padding check
                float in_val = 0.0f;
                if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                    in_val = in[(((n * C_in + c_in) * H_in + in_h) * W_in + in_w)];
                }

                // Load weight value
                float w_val = weight[(((c_out * C_in + c_in) * K_h + kh) * K_w + kw)];

                // Accumulate
                acc += in_val * w_val;
            }
        }
    }

    // Fuse subtractions
    acc = acc - sub1 - sub2;

    // Mish activation: x * tanh(softplus(x))
    // Using fast math intrinsics
    float sp = log1pf(__expf(acc)); // softplus
    float mish = acc * tanhf(sp);   // mish

    // Write result
    out[(((n * C_out + c_out) * H_out + out_h) * W_out + out_w)] = mish;
}

void fused_op_forward(
    torch::Tensor in,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    int stride,
    int pad,
    float sub1,
    float sub2)
{
    const int N = in.size(0);
    const int C_in = in.size(1);
    const int H_in = in.size(2);
    const int W_in = in.size(3);
    const int C_out = weight.size(0);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);

    const int H_out = (H_in + 2 * pad - K_h) / stride + 1;
    const int W_out = (W_in + 2 * pad - K_w) / stride + 1;

    const int tiles_w = (W_out + 7) / 8; // (W_out + TILE_W - 1) / TILE_W
    const int tiles_h = (H_out + 7) / 8; // (H_out + TILE_H - 1) / TILE_H
    const int total_blocks = N * C_out * tiles_h * tiles_w;

    const dim3 block(8, 8);
    const dim3 grid(total_blocks);

    fused_op_forward_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in,
        K_h, K_w,
        stride, pad,
        sub1, sub2
    );
    
    // It's good practice to check for errors, but for performance we rely on CUDA_LAUNCH_BLOCKING=1 for debugging
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    // }
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor in,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor out,
    int stride,
    int pad,
    float sub1,
    float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv2d + Bias + 2 Subtractions + Mish (CUDA)");
}
"""

# Compile the extension at import time
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False # Set to True for compilation output
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
    subtract_value_1,
    subtract_value_2,
):
    """
    Performs a fused Conv2d + Bias + 2 Subtractions + Mish operation.
    """
    # --- Sanity Checks (as per plan) ---
    if conv_groups != 1:
        raise ValueError("fused kernel only supports groups == 1")
    if conv_dilation != 1:
        raise ValueError("fused kernel only supports dilation == 1")
    if conv_weight.shape[2] != 3 or conv_weight.shape[3] != 3:
        raise ValueError("fused kernel currently only supports 3x3 kernel size")

    # --- Calculate Output Shape ---
    batch_size = x.shape[0]
    out_channels = conv_weight.shape[0]
    in_height = x.shape[2]
    in_width = x.shape[3]
    kernel_height = conv_weight.shape[2]
    kernel_width = conv_weight.shape[3]
    
    out_height = (in_height + 2 * conv_padding - kernel_height) // conv_stride + 1
    out_width = (in_width + 2 * conv_padding - kernel_width) // conv_stride + 1

    # --- Allocate Output Tensor ---
    out = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        dtype=x.dtype,
        device=x.device,
    )

    # --- Launch Fused Kernel ---
    fused_ext.fused_op(
        x, conv_weight, conv_bias,
        out,
        conv_stride, conv_padding,
        subtract_value_1, subtract_value_2
    )
    
    return out

# --- Helper functions required by the original code structure ---
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

# Ensure inputs are on the correct device for the benchmark script
get_inputs = lambda: [torch.rand(batch_size, in_channels, height, width, device='cuda')]

