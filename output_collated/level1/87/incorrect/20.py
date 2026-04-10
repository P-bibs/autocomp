# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071251/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# -------------------------------------------------------------------------
# Inline CUDA source – tiled GEMM‑style 1×1 convolution with weight in shared memory
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Maximum dimensions allowed by the kernel (must match the Python side)
#define MAX_M 128   // max output channels
#define MAX_K 64    // max input channels (or kernel size product)

__global__ void gemm_conv_kernel(
    const float* __restrict__ A,   // input: N × K
    const float* __restrict__ B,   // weight: M × K
    float* __restrict__ C,         // output: N × M
    const float* __restrict__ bias,// bias: M (may be nullptr)
    const int N,
    const int M,
    const int K)
{
    // -----------------------------------------------------------------
    // Shared memory for the whole weight matrix (M × K <= 8192 floats)
    // -----------------------------------------------------------------
    __shared__ float B_shared[MAX_M * MAX_K];

    // -----------------------------------------------------------------
    // 1. Load weight matrix into shared memory (one‑time per block)
    // -----------------------------------------------------------------
    const int weight_elems = M * K;
    for (int idx = threadIdx.x; idx < weight_elems; idx += blockDim.x) {
        B_shared[idx] = B[idx];
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // 2. Identify the pixel (row) this thread is responsible for
    // -----------------------------------------------------------------
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // -----------------------------------------------------------------
    // 3. Load the pixel's input vector (K values) into registers
    // -----------------------------------------------------------------
    float a[MAX_K];
    const float* A_row = A + row * K;
    for (int k = 0; k < K; ++k) {
        a[k] = A_row[k];
    }

    // -----------------------------------------------------------------
    // 4. Compute all output channels for this pixel
    // -----------------------------------------------------------------
    for (int oc = 0; oc < M; ++oc) {
        float sum = 0.0f;
        const float* w_row = B_shared + oc * K;
        // Unroll the inner loop for the common case (K <= 64)
        #pragma unroll
        for (int k = 0; k < MAX_K; ++k) {
            if (k < K) sum += a[k] * w_row[k];
        }
        if (bias) sum += bias[oc];
        C[row * M + oc] = sum;
    }
}

void gemm_conv_launcher(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& bias,
    torch::Tensor& C)
{
    const int N = A.size(0);   // number of pixels (batch * out_h * out_w)
    const int M = B.size(0);   // output channels
    const int K = B.size(1);   // input channels (or kernel product)

    const int block_size = 256;
    const int grid_x = (N + block_size - 1) / block_size;

    dim3 grid(grid_x);
    dim3 block(block_size);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();
    const float* bias_ptr = bias.numel() ? bias.data_ptr<float>() : nullptr;

    gemm_conv_kernel<<<grid, block>>>(
        A_ptr, B_ptr, C_ptr, bias_ptr, N, M, K);

    // Ensure the kernel finishes before the host continues
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the kernel to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void gemm_conv_launcher(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& bias,
    torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_conv", &gemm_conv_launcher,
          "Custom GEMM‑based 1×1 convolution kernel");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Helper functions (identical to the original file, only for completeness)
# -------------------------------------------------------------------------
batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

# -------------------------------------------------------------------------
# Optimized functional_model
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    """
    Performs a 1×1 convolution using a custom CUDA kernel that caches the
    weight matrix in shared memory (optimisation #1).  The semantics match the
    original F.conv2d call within a small numerical tolerance.
    """

    # ------------------------------------------------------------
    # 1. Basic sanity checks – we only support the common case:
    #    groups == 1, stride == 1, padding == 0, dilation == 1,
    #    and a 1×1 kernel.  This matches the benchmark parameters.
    # ------------------------------------------------------------
    if conv1d_groups != 1:
        raise NotImplementedError("Only groups=1 is supported in this optimisation.")
    if conv1d_stride != 1:
        raise NotImplementedError("Only stride=1 is supported in this optimisation.")
    if conv1d_padding != 0:
        raise NotImplementedError("Only padding=0 is supported in this optimisation.")
    if conv1d_dilation != 1:
        raise NotImplementedError("Only dilation=1 is supported in this optimisation.")
    # Accept both 4‑D (out_ch, in_ch, kH, kW) and already‑squeezed weights
    if conv1d_weight.dim() == 4:
        if conv1d_weight.shape[2] != 1 or conv1d_weight.shape[3] != 1:
            raise NotImplementedError("Only 1×1 kernels are supported.")
        weight = conv1d_weight.squeeze().view(conv1d_weight.size(0), -1)  # (out_ch, in_ch)
    elif conv1d_weight.dim() == 3:
        weight = conv1d_weight.squeeze().view(conv1d_weight.size(0), -1)
    else:
        weight = conv1d_weight.view(conv1d_weight.size(0), -1)

    out_ch = weight.size(0)
    k = weight.size(1)           # input channel count (here = in_channels)

    # ------------------------------------------------------------
    # 2. Compute output spatial size (always equal to input size for the
    #    1×1, stride‑1, pad‑0 case)
    # ------------------------------------------------------------
    out_h = x.size(2)
    out_w = x.size(3)

    # ------------------------------------------------------------
    # 3. Flatten the input: (batch, in_ch, H, W) -> (N, in_ch)
    #    The layout (batch, H, W, in_ch) after permute gives a
    #    row‑major matrix where each pixel's feature vector is contiguous.
    # ------------------------------------------------------------
    N = x.size(0) * out_h * out_w   # total number of pixels
    x_flat = x.permute(0, 2, 3, 1).reshape(N, k)

    # ------------------------------------------------------------
    # 4. Prepare bias tensor (may be None)
    # ------------------------------------------------------------
    if conv1d_bias is None:
        bias = torch.empty(out_ch, dtype=x.dtype, device=x.device)
    else:
        bias = conv1d_bias

    # ------------------------------------------------------------
    # 5. Allocate output matrix
    # ------------------------------------------------------------
    out_flat = torch.empty((N, out_ch), dtype=x.dtype, device=x.device)

    # ------------------------------------------------------------
    # 6. Call the custom kernel – all the work is done here
    # ------------------------------------------------------------
    fused_ext.gemm_conv(x_flat, weight, bias, out_flat)

    # ------------------------------------------------------------
    # 7. Reshape back to (batch, out_ch, H, W)
    # ------------------------------------------------------------
    out = out_flat.view(x.size(0), out_h, out_w, out_ch)
    out = out.permute(0, 3, 1, 2).contiguous()

    return out
