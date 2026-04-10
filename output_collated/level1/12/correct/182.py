# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_25.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
        super(ModelNew, self).__init__()

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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ O,
    int N,
    int M)
{
    // Shared memory one float per block to store the broadcast scalar for the current row
    extern __shared__ float sh_A[];

    const int vec_len = 4;
    const int thread_id = threadIdx.x;
    const int threads_per_block = blockDim.x;
    
    // Grid-stride loop index
    long long total_elems = (long long)N * M;
    long long base_idx = ((long long)blockIdx.x * threads_per_block + thread_id) * vec_len;
    long long stride = (long long)gridDim.x * threads_per_block * vec_len;

    for (long long idx = base_idx; idx < total_elems; idx += stride) {
        // Compute row index for this vector
        int n = idx / M;
        
        // Single thread loads A[n] per block
        if (thread_id == 0) {
            sh_A[0] = A[n];
        }
        __syncthreads();

        float a_val = sh_A[0];

        // Ensure we don't read/write past the end
        if (idx + 3 < total_elems) {
            float4 b_vec = reinterpret_cast<const float4*>(B)[idx / vec_len];
            
            float4 out_vec;
            out_vec.x = a_val * b_vec.x;
            out_vec.y = a_val * b_vec.y;
            out_vec.z = a_val * b_vec.z;
            out_vec.w = a_val * b_vec.w;
            
            reinterpret_cast<float4*>(O)[idx / vec_len] = out_vec;
        } else {
            // Scalar handling for tail end
            for (int i = 0; i < 4 && idx + i < total_elems; ++i) {
                O[idx + i] = a_val * B[idx + i];
            }
        }
        __syncthreads(); // Ensure shared memory is consumed before next row load
    }
}

void broadcast_mul_launch(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int threads_per_block = 256;
    const int total_elements = N * M;
    // Keep reasonable occupancy
    const int blocks = std::min(1024, (total_elements + 4 * threads_per_block - 1) / (4 * threads_per_block));
    
    broadcast_mul_shared_kernel<<<blocks, threads_per_block, sizeof(float)>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_launch(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_launch, "Vectorized broadcast multiplication with shared memory");
}
"""

fused_ext = load_inline(
    name='broadcast_mul_shared_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
