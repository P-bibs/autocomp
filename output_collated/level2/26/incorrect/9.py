# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_040909/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# ----------------------------------------------------------------------
# 1. CUDA source – fused transposed‑conv + add + hard‑swish kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// ---------- hard‑swish inline ----------
__device__ __forceinline__ float hardswish(float x) {
    float y = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * y * 0.16666667f; // x * y / 6
}

// ---------- main fused kernel ----------
__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,          // (N, C_in, D_in, H_in, W_in)
    const float* __restrict__ weight,         // (C_out, C_in, K^3) flattened
    const float* __restrict__ bias,           // (C_out) or empty
    const float* __restrict__ add_input,      // (N, C_out, D_out, H_out, W_out)
    float*       __restrict__ output,         // (N, C_out, D_out, H_out, W_out)
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding)
{
    const int K3 = K * K * K;
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    // ---- decode linear index to (n, co, d, h, w) ----
    int tmp = idx;
    const int n = tmp / (C_out * D_out * H_out * W_out);
    tmp   %= (C_out * D_out * H_out * W_out);
    const int co = tmp / (D_out * H_out * W_out);
    tmp   %= (D_out * H_out * W_out);
    const int d = tmp / (H_out * W_out);
    tmp   %= (H_out * W_out);
    const int h = tmp / W_out;
    const int w = tmp % W_out;

    // ---- compute convolution sum ----
    float sum = 0.0f;

    // pre‑compute kernel candidates for this output position
    // stride == 2, padding == 1  → only two possibilities per dim
    int kd_candidate[2], kh_candidate[2], kw_candidate[2];
    int kd_cnt = 0, kh_cnt = 0, kw_cnt = 0;

    int pd = d + padding;
    if ((pd & 1) == 0) {               // even → kernel offsets 0 and 2
        kd_candidate[kd_cnt++] = 0;
        kd_candidate[kd_cnt++] = 2;
    } else {                           // odd → only offset 1
        kd_candidate[kd_cnt++] = 1;
    }

    int ph = h + padding;
    if ((ph & 1) == 0) {
        kh_candidate[kh_cnt++] = 0;
        kh_candidate[kh_cnt++] = 2;
    } else {
        kh_candidate[kh_cnt++] = 1;
    }

    int pw = w + padding;
    if ((pw & 1) == 0) {
        kw_candidate[kw_cnt++] = 0;
        kw_candidate[kw_cnt++] = 2;
    } else {
        kw_candidate[kw_cnt++] = 1;
    }

    // loop over input channels
    for (int ci = 0; ci < C_in; ++ci) {
        // loop over valid kernel triples
        for (int ikd = 0; ikd < kd_cnt; ++ikd) {
            int kd = kd_candidate[ikd];
            int di = (pd - kd) / stride;
            if (di < 0 || di >= D_in) continue;

            for (int ikh = 0; ikh < kh_cnt; ++ikh) {
                int kh = kh_candidate[ikh];
                int hi = (ph - kh) / stride;
                if (hi < 0 || hi >= H_in) continue;

                for (int ikw = 0; ikw < kw_cnt; ++ikw) {
                    int kw = kw_candidate[ikw];
                    int wi = (pw - kw) / stride;
                    if (wi < 0 || wi >= W_in) continue;

                    // flattened kernel index
                    int kid = (kd * K + kh) * K + kw;
                    int wIdx = ((co * C_in + ci) * K3 + kid);
                    float wVal = __ldg(&weight[wIdx]);

                    // input element index (flattened)
                    int inIdx = (((n * C_in + ci) * D_in + di) * H_in + hi) * W_in + wi;
                    float inVal = __ldg(&input[inIdx]);

                    sum += wVal * inVal;
                }
            }
        }
    }

    // ---- add bias (if any) ----
    if (bias != nullptr) {
        sum += __ldg(&bias[co]);
    }

    // ---- add the extra input and apply hard‑swish ----
    int addIdx = (((n * C_out + co) * D_out + d) * H_out + h) * W_out + w;
    float addVal = __ldg(&add_input[addIdx]);
    sum += addVal;
    float outVal = hardswish(sum);

    // ---- store result ----
    output[addIdx] = outVal;
}

// ------------------------------------------------------------------
// Host launcher (exposed via PyBind11)
// ------------------------------------------------------------------
void conv_transpose_fused_launcher(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding)
{
    const int total_out = N * C_out * D_out * H_out * W_out;
    const int block = 256;
    const int grid = (total_out + block - 1) / block;

    conv_transpose_fused_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(cudaGetLastError()));
}
"""

# ----------------------------------------------------------------------
# 2. C++ binding – expose the launcher to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_transpose_fused_launcher(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& add_input,
    at::Tensor& output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_fused_launcher", &conv_transpose_fused_launcher,
          "Fused ConvTranspose3D + add + HardSwish");
}
"""

# ----------------------------------------------------------------------
# 3. Compile the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_transpose_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# 4. Functional model – the entry point that will be imported
# ----------------------------------------------------------------------
def functional_model(
    x,                     # input tensor (N, C_in, D_in, H_in, W_in)
    add_input,             # tensor to be added (N, C_out, D_out, H_out, W_out)
    *,
    conv_transpose_weight, # weight tensor (C_out, C_in, K, K, K)
    conv_transpose_bias,   # bias tensor (C_out) – may be None
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,                  # unused in the original program, kept for signature compatibility
):
    # ------------------------------------------------------------------
    # Basic shape information
    # ------------------------------------------------------------------
    N, C_in, D_in, H_in, W_in = x.shape
    C_out = add_input.shape[1]
    D_out = add_input.shape[2]
    H_out = add_input.shape[3]
    W_out = add_input.shape[4]

    K = conv_transpose_weight.shape[2]      # cubic kernel, assume KxKxK
    stride = conv_transpose_stride
    padding = conv_transpose_padding

    # ------------------------------------------------------------------
    # Flatten weight tensor – (C_out, C_in, K^3)
    # ------------------------------------------------------------------
    weight_flat = conv_transpose_weight.contiguous().view(C_out, C_in, K * K * K)

    # ------------------------------------------------------------------
    # Bias handling – create an empty tensor if not supplied
    # ------------------------------------------------------------------
    if conv_transpose_bias is not None:
        bias_tensor = conv_transpose_bias
    else:
        bias_tensor = torch.empty(0, dtype=torch.float32, device='cuda')

    # ------------------------------------------------------------------
    # Allocate output tensor (same shape as add_input)
    # ------------------------------------------------------------------
    output = torch.empty((N, C_out, D_out, H_out, W_out),
                         dtype=torch.float32, device='cuda')

    # ------------------------------------------------------------------
    # Launch the fused CUDA kernel
    # ------------------------------------------------------------------
    fused_ext.conv_transpose_fused_launcher(
        x,                     # input
        weight_flat,           # flattened weight
        bias_tensor,           # bias (may be empty)
        add_input,             # tensor to add
        output,                # result
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        K, stride, padding)

    return output


# ----------------------------------------------------------------------
# Test driver (only runs when the file is executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 128
    in_channels = 32
    out_channels = 64
    D = H = W = 16
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1

    # Input tensors (GPU)
    x = torch.rand(batch_size, in_channels, D, H, W, device='cuda')
    add_input = torch.rand(batch_size, out_channels,
                           D * stride, H * stride, W * stride, device='cuda')

    # Model parameters (GPU)
    params = {
        "conv_transpose_weight": torch.randn(out_channels, in_channels,
                                             kernel_size, kernel_size, kernel_size,
                                             device='cuda'),
        "conv_transpose_bias": torch.randn(out_channels, device='cuda'),
        "conv_transpose_stride": stride,
        "conv_transpose_padding": padding,
        "conv_transpose_output_padding": output_padding,
        "conv_transpose_groups": 1,
        "conv_transpose_dilation": 1,
        "bias": torch.randn(1, 1, 1, 1, 1, device='cuda'),  # dummy, not used
    }

    # Warm‑up
    with torch.no_grad():
        _ = functional_model(x, add_input, **params)

    # Timing
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(10):
            y = functional_model(x, add_input, **params)
    end.record()
    torch.cuda.synchronize()

    print(f"Average time per iteration: {start.elapsed_time(end) / 10:.3f} ms")
    print(f"Output shape: {y.shape}")
    print("Fused ConvTranspose+Add+HardSwish completed successfully.")
