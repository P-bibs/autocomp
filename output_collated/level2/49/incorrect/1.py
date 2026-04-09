# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092423/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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
#  CUDA source – kernels and host wrappers
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------------------------------
// Parameters for the transposed convolution
// -------------------------------------------------------------------
struct ConvParams {
    int N, C_in, C_out;
    int D_in, H_in, W_in;
    int D_out, H_out, W_out;
    int kD, kH, kW;
    int stride, padding, output_padding, dilation, groups;
    const float* __restrict__ input;
    const float* __restrict__ weight;
    const float* __restrict__ bias;          // may be nullptr
    float* __restrict__ output;
};

// -------------------------------------------------------------------
// Transposed convolution – gather‑based (output‑centric) kernel
// -------------------------------------------------------------------
__global__ void conv_transpose_kernel(const ConvParams p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total_out = (long long)p.N * p.C_out * p.D_out * p.H_out * p.W_out;
    if (idx >= total_out) return;

    // decode linear index to (n, oc, d, h, w)
    int rem = idx;
    int n = rem / (p.C_out * p.D_out * p.H_out * p.W_out);
    rem %= (p.C_out * p.D_out * p.H_out * p.W_out);
    int oc = rem / (p.D_out * p.H_out * p.W_out);
    rem %= (p.D_out * p.H_out * p.W_out);
    int d = rem / (p.H_out * p.W_out);
    rem %= (p.H_out * p.W_out);
    int h = rem / p.W_out;
    int w = rem % p.W_out;

    float sum = (p.bias != nullptr) ? p.bias[oc] : 0.0f;

    // loop over input channels and kernel positions
    for (int ic = 0; ic < p.C_in; ++ic) {
        for (int kd = 0; kd < p.kD; ++kd) {
            int off_d = d + p.padding - kd * p.dilation;
            if (off_d < 0 || off_d % p.stride != 0) continue;
            int di = off_d / p.stride;
            if (di < 0 || di >= p.D_in) continue;

            for (int kh = 0; kh < p.kH; ++kh) {
                int off_h = h + p.padding - kh * p.dilation;
                if (off_h < 0 || off_h % p.stride != 0) continue;
                int hi = off_h / p.stride;
                if (hi < 0 || hi >= p.H_in) continue;

                for (int kw = 0; kw < p.kW; ++kw) {
                    int off_w = w + p.padding - kw * p.dilation;
                    if (off_w < 0 || off_w % p.stride != 0) continue;
                    int wi = off_w / p.stride;
                    if (wi < 0 || wi >= p.W_in) continue;

                    // weight index: (ic, oc, kd, kh, kw)
                    int wIdx = (((ic * p.C_out + oc) * p.kD + kd) * p.kH + kh) * p.kW + kw;
                    // input index: (n, ic, di, hi, wi)
                    int iIdx = ((((n * p.C_in + ic) * p.D_in + di) * p.H_in + hi) * p.W_in + wi);
                    sum += p.weight[wIdx] * p.input[iIdx];
                }
            }
        }
    }

    // store result
    int outIdx = ((((n * p.C_out + oc) * p.D_out + d) * p.H_out + h) * p.W_out + w);
    p.output[outIdx] = sum;
}

// -------------------------------------------------------------------
// Host wrapper for the transposed convolution
// -------------------------------------------------------------------
void conv_transpose_launch(
    int N, int C_in, int C_out,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride, int padding, int output_padding, int dilation, int groups,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output)
{
    const int threads = 256;
    long long total = (long long)N * C_out * D_out * H_out * W_out;
    int blocks = (total + threads - 1) / threads;

    ConvParams p;
    p.N = N; p.C_in = C_in; p.C_out = C_out;
    p.D_in = D_in; p.H_in = H_in; p.W_in = W_in;
    p.D_out = D_out; p.H_out = H_out; p.W_out = W_out;
    p.kD = kD; p.kH = kH; p.kW = kW;
    p.stride = stride; p.padding = padding;
    p.output_padding = output_padding; p.dilation = dilation;
    p.groups = groups;
    p.input = input; p.weight = weight; p.bias = bias; p.output = output;

    conv_transpose_kernel<<<blocks, threads>>>(p);
    cudaDeviceSynchronize();
}

// -------------------------------------------------------------------
// Fused softmax (dim = 1, i.e. channel) + sigmoid kernel
// -------------------------------------------------------------------
__global__ void fused_softmax_sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W)
{
    // each block processes one (n,d,h,w) location
    const int blockId = blockIdx.x;
    const int total_positions = N * D * H * W;
    if (blockId >= total_positions) return;

    int n = blockId / (D * H * W);
    int rem = blockId % (D * H * W);
    int d = rem / (H * W);
    rem %= (H * W);
    int h = rem / W;
    int w = rem % W;

    extern __shared__ float s_data[];
    const int tid = threadIdx.x;               // 0 … C‑1  (C == blockDim.x)
    // load exp(x) for the channel handled by this thread
    int in_idx = ((((n * C + tid) * D + d) * H + h) * W + w);
    float expx = __expf(input[in_idx]);
    s_data[tid] = expx;
    __syncthreads();

    // parallel reduction – sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    float sum_exp = s_data[0];

    // softmax + sigmoid
    float softmax = expx / sum_exp;
    float sig = 1.0f / (1.0f + __expf(-softmax));

    int out_idx = ((((n * C + tid) * D + d) * H + h) * W + w);
    output[out_idx] = sig;
}

// -------------------------------------------------------------------
// Host wrapper for fused kernel
// -------------------------------------------------------------------
void fused_softmax_sigmoid_launch(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W)
{
    int num_blocks = N * D * H * W;
    int block_size = C;                 // C is guaranteed <= 1024 (here C == 64)
    int shared_mem = block_size * sizeof(float);
    fused_softmax_sigmoid_kernel<<<num_blocks, block_size, shared_mem>>>(
        input, output, N, C, D, H, W);
    cudaDeviceSynchronize();
}

// -------------------------------------------------------------------
// PyBind11 bindings
// -------------------------------------------------------------------
torch::Tensor conv_transpose(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups)
{
    // Ensure contiguous memory (the kernels expect plain float pointers)
    input = input.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) bias = bias.contiguous();

    int N = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int C_out = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int D_out = (D_in - 1) * stride - 2 * padding + dilation * (kD - 1) + output_padding + 1;
    int H_out = (H_in - 1) * stride - 2 * padding + dilation * (kH - 1) + output_padding + 1;
    int W_out = (W_in - 1) * stride - 2 * padding + dilation * (kW - 1) + output_padding + 1;

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    const float* in_ptr = input.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    conv_transpose_launch(N, C_in, C_out,
                          D_in, H_in, W_in,
                          D_out, H_out, W_out,
                          kD, kH, kW,
                          stride, padding, output_padding, dilation, groups,
                          in_ptr, w_ptr, b_ptr, out_ptr);
    return output;
}

torch::Tensor fused_softmax_sigmoid(torch::Tensor input) {
    input = input.contiguous();
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty_like(input);
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    fused_softmax_sigmoid_launch(in_ptr, out_ptr, N, C, D, H, W);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose", &conv_transpose,
          "Transposed 3‑D convolution (CUDA implementation)");
    m.def("fused_softmax_sigmoid", &fused_softmax_sigmoid,
          "Fused softmax (dim=1) + sigmoid");
}
"""

# -------------------------------------------------------------------------
#  Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_op",
    cpp_sources="",                     # no separate C++ source – everything is in cuda_source
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

# -------------------------------------------------------------------------
#  Helper functions required by the harness
# -------------------------------------------------------------------------
def get_init_inputs():
    # Parameters used to create the model in the benchmark harness
    return [32, 64, 3, 2, 1, 1]   # in_channels, out_channels, kernel, stride, padding, output_padding

def get_inputs():
    # Sample input tensor – will be moved to GPU by the harness
    batch_size = 16
    in_channels = 32
    D, H, W = 16, 32, 32
    return [torch.rand(batch_size, in_channels, D, H, W)]

# -------------------------------------------------------------------------
#  The functional model that will be evaluated
# -------------------------------------------------------------------------
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
    softmax_dim,
):
    # --------------------------------------------------------------
    # 1. Custom transposed convolution (replaces F.conv_transpose3d)
    # --------------------------------------------------------------
    conv_out = fused_ext.conv_transpose(
        x,
        conv_transpose_weight,
        conv_transpose_bias,               # may be None
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation,
        conv_transpose_groups,
    )

    # --------------------------------------------------------------
    # 2. Fused softmax + sigmoid (optimisation #12)
    #    – only for the typical case softmax over channels (dim=1)
    # --------------------------------------------------------------
    if softmax_dim == 1:
        out = fused_ext.fused_softmax_sigmoid(conv_out)
    else:
        # generic fallback (softmax over an arbitrary dimension)
        out = torch.sigmoid(torch.softmax(conv_out, dim=softmax_dim))

    return out
