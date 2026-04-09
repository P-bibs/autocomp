# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_073421/code_1.py
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

# CUDA kernel with tiling, vectorization, and shared memory
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define THREADS_PER_BLOCK 256

__global__ void fused_op_forward_kernel(
    const float4* __restrict__ x,
    const float4* __restrict__ w,
    const float* __restrict__ b,
    float4* __restrict__ out,
    const float subtract_value,
    const float multiply_value,
    const int M,
    const int N,
    const int K
) {
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    // Shared memory for tiles
    __shared__ float4 shmem_x[TILE_M][TILE_K / 4];
    __shared__ float4 shmem_w[TILE_N][TILE_K / 4];

    // Registers for accumulation
    float c_reg[8][8] = {0.0f};

    // Each thread computes 8x8 outputs
    for (int k = 0; k < K; k += TILE_K) {
        // Load x tile to shared memory
        int x_row = block_row * TILE_M + warp_id * 8 + (lane_id / 4);
        int x_col = k + (lane_id % 4) * 4;
        if (x_row < M && x_col + 3 < K) {
            shmem_x[warp_id * 8 + (lane_id / 4)][lane_id % 4] = x[(x_row * K + x_col) / 4];
        } else if (x_row < M) {
            float4 temp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (x_col < K) temp.x = *((float*)&x[(x_row * K + x_col) / 4]);
            if (x_col + 1 < K) temp.y = *((float*)&x[(x_row * K + x_col + 1) / 4]);
            if (x_col + 2 < K) temp.z = *((float*)&x[(x_row * K + x_col + 2) / 4]);
            if (x_col + 3 < K) temp.w = *((float*)&x[(x_row * K + x_col + 3) / 4]);
            shmem_x[warp_id * 8 + (lane_id / 4)][lane_id % 4] = temp;
        }
        
        // Load w tile to shared memory
        int w_row = block_col * TILE_N + warp_id * 8 + (lane_id / 4);
        int w_col = k + (lane_id % 4) * 4;
        if (w_row < N && w_col + 3 < K) {
            shmem_w[warp_id * 8 + (lane_id / 4)][lane_id % 4] = w[(w_row * K + w_col) / 4];
        } else if (w_row < N) {
            float4 temp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (w_col < K) temp.x = *((float*)&w[(w_row * K + w_col) / 4]);
            if (w_col + 1 < K) temp.y = *((float*)&w[(w_row * K + w_col + 1) / 4]);
            if (w_col + 2 < K) temp.z = *((float*)&w[(w_row * K + w_col + 2) / 4]);
            if (w_col + 3 < K) temp.w = *((float*)&w[(w_row * K + w_col + 3) / 4]);
            shmem_w[warp_id * 8 + (lane_id / 4)][lane_id % 4] = temp;
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int kk = 0; kk < TILE_K / 4; kk++) {
            float4 a_frag = shmem_x[lane_id / 16][(kk + lane_id / 16) % (TILE_K / 4)];
            float4 b_frag = shmem_w[lane_id % 16][(kk + lane_id % 16) % (TILE_K / 4)];

            float a_reg[4] = {a_frag.x, a_frag.y, a_frag.z, a_frag.w};
            float b_reg[4] = {b_frag.x, b_frag.y, b_frag.z, b_frag.w};

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int m = 0; m < 8; m++) {
                    #pragma unroll
                    for (int n = 0; n < 8; n++) {
                        c_reg[m][n] += a_reg[i] * b_reg[i];
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write results with fused ops
    int out_row = block_row * TILE_M + warp_id * 8 + (lane_id / 16);
    int out_col = block_col * TILE_N + (lane_id % 16) * 8;

    if (out_row < M && out_col < N) {
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                if ((out_row + m) < M && (out_col + n) < N) {
                    float val = c_reg[m][n] + b[out_col + n];
                    val = (val - subtract_value) * multiply_value;
                    val = fmaxf(val, 0.0f); // ReLU

                    int idx = ((out_row + m) * N + (out_col + n)) / 4;
                    int elem = ((out_row + m) * N + (out_col + n)) % 4;

                    float4 out_val = out[idx];
                    if (elem == 0) out_val.x = val;
                    else if (elem == 1) out_val.y = val;
                    else if (elem == 2) out_val.z = val;
                    else out_val.w = val;
                    out[idx] = out_val;
                }
            }
        }
    }
}

void fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor out,
    float subtract_value,
    float multiply_value
) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w.size(0);

    dim3 threads(32, 8); // 256 threads per block
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    fused_op_forward_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<const float4*>(w.data_ptr<float>()),
        b.data_ptr<float>(),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        subtract_value,
        multiply_value,
        M,
        N,
        K
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, float subtract_value, float multiply_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused Linear-ReLU forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    linear_weight,
    linear_bias,
    subtract_value,
    multiply_value,
):
    # Ensure tensors are contiguous and on the correct device
    x = x.contiguous()
    linear_weight = linear_weight.contiguous()
    linear_bias = linear_bias.contiguous()
    
    # Create output tensor
    out = torch.empty((x.size(0), linear_weight.size(0)), device=x.device, dtype=x.dtype)
    
    # Launch custom kernel
    fused_ext.fused_op_forward(x, linear_weight, linear_bias, out, subtract_value, multiply_value)
    
    return out

# Verification variables
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
