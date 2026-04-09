# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_070814/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'subtract_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['linear_weight', 'linear_bias', 'subtract_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['linear_weight', 'linear_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

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
    # State for linear (nn.Linear)
    if 'linear_weight' in flat_state:
        state_kwargs['linear_weight'] = flat_state['linear_weight']
    else:
        state_kwargs['linear_weight'] = getattr(model.linear, 'weight', None)
    if 'linear_bias' in flat_state:
        state_kwargs['linear_bias'] = flat_state['linear_bias']
    else:
        state_kwargs['linear_bias'] = getattr(model.linear, 'bias', None)
    if 'subtract_value' in flat_state:
        state_kwargs['subtract_value'] = flat_state['subtract_value']
    else:
        state_kwargs['subtract_value'] = getattr(model, 'subtract_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
# 1.  CUDA source – fused tiled GEMM + element‑wise ops
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BLOCK_M = 16;   // output‑feature tile height
constexpr int BLOCK_N = 16;   // batch‑tile width
constexpr int BLOCK_K = 16;   // size of the K‑dimension tile

__global__ void fused_gemm_relu_kernel(
    const float* __restrict__ x,   // (N, K)  – input (batch, in_features)
    const float* __restrict__ w,   // (M, K)  – weight (out_features, in_features)
    const float* __restrict__ b,   // (M)     – bias, may be nullptr
    float* __restrict__ y,         // (N, M)  – output (batch, out_features)
    const float subtract_value,
    const float multiply_value,
    const int M,   // out_features
    const int N,   // batch size
    const int K    // in_features
)
{
    // ----- shared memory for the two tiles -----
    __shared__ float weight_tile[BLOCK_M][BLOCK_K];
    __shared__ float input_tile[BLOCK_N][BLOCK_K];

    // ----- thread indices -----
    int row = blockIdx.y * BLOCK_M + threadIdx.y;   // output feature
    int col = blockIdx.x * BLOCK_N + threadIdx.x;   // batch index

    // ----- accumulate the dot product -----
    float acc = 0.0f;

    // bias value (loaded once)
    float bias_val = 0.0f;
    if (b != nullptr && row < M && col < N) bias_val = b[row];

    // ----- loop over the K dimension in BLOCK_K tiles -----
    for (int k = 0; k < K; k += BLOCK_K) {
        // ---- load weight tile (row‑major) ----
        if (row < M) {
            int wcol = k + threadIdx.x;
            weight_tile[threadIdx.y][threadIdx.x] = (wcol < K) ? w[row * K + wcol] : 0.0f;
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ---- load input tile (row‑major) ----
        if (col < N) {
            int icol = k + threadIdx.y;
            input_tile[threadIdx.x][threadIdx.y] = (icol < K) ? x[col * K + icol] : 0.0f;
        } else {
            input_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product for this tile ----
        #pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
            acc += weight_tile[threadIdx.y][i] * input_tile[threadIdx.x][i];
        }

        __syncthreads();
    }

    // ----- apply bias, subtract, multiply, ReLU -----
    if (row < M && col < N) {
        float val = acc + bias_val;
        val = (val - subtract_value) * multiply_value;
        val = (val > 0.0f) ? val : 0.0f;
        y[col * M + row] = val;          // row‑major output
    }
}

// ----------------------------------------------------------------------
// Host‑side wrapper callable from Python
// ----------------------------------------------------------------------
void fused_op(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& b,
    torch::Tensor& y,
    float subtract_value,
    float multiply_value,
    int M, int N, int K)
{
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.numel() ? b.data_ptr<float>() : nullptr;
    float*       y_ptr = y.data_ptr<float>();

    dim3 block(BLOCK_N, BLOCK_M);                // 256 threads per block
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    fused_gemm_relu_kernel<<<grid, block>>>(
        x_ptr, w_ptr, b_ptr, y_ptr,
        subtract_value, multiply_value,
        M, N, K);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# 2.  C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& b,
    torch::Tensor& y,
    float subtract_value,
    float multiply_value,
    int M, int N, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Fused GEMM + (subtract, multiply, ReLU) kernel");
}
"""

# ----------------------------------------------------------------------
# 3.  Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# 4.  Functional model – replaces the original implementation
# ----------------------------------------------------------------------
def functional_model(
    x,                      # Tensor (batch, in_features)
    *,
    linear_weight,          # Tensor (out_features, in_features)
    linear_bias,            # Tensor (out_features) or None
    subtract_value,         # float
    multiply_value          # float
):
    # Ensure inputs reside on the same device and are contiguous
    if not x.is_cuda:
        x = x.cuda()
    if not linear_weight.is_cuda:
        linear_weight = linear_weight.cuda()
    if linear_bias is not None and not linear_bias.is_cuda:
        linear_bias = linear_bias.cuda()

    x = x.contiguous()
    linear_weight = linear_weight.contiguous()
    if linear_bias is not None:
        linear_bias = linear_bias.contiguous()
    else:
        # create an empty tensor (size 0) to avoid a null pointer in the kernel
        linear_bias = torch.empty(0, device='cuda', dtype=torch.float32)

    batch = x.shape[0]
    in_features = x.shape[1]
    out_features = linear_weight.shape[0]

    # Allocate output tensor
    y = torch.empty((batch, out_features), dtype=x.dtype, device=x.device)

    # Launch fused kernel
    fused_ext.fused_op(
        x, linear_weight, linear_bias, y,
        subtract_value, multiply_value,
        out_features,   # M
        batch,          # N
        in_features     # K
    )
    return y

# ----------------------------------------------------------------------
# 5.  Helpers required by the benchmark harness
# ----------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    # The harness will provide the remaining tensors (weight, bias) externally.
    # We only return the input tensor x.
    x = torch.rand(batch_size, in_features, device='cuda', dtype=torch.float32)
    return [x]
