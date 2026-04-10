# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_25.py
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

template <typename scalar_t>
__global__ void broadcast_mul_shared_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N,
    int M)
{
    // Grid-stride over rows: A block handles one or more rows of the matrix.
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        
        // Single thread in block loads the scalar A[row] into shared memory.
        __shared__ scalar_t a_shared;
        if (threadIdx.x == 0) {
            a_shared = A[row];
        }
        __syncthreads();

        const scalar_t a_val = a_shared;

        // Grid-stride over columns: Threads process the row elements in coalesced fashion.
        for (int col = threadIdx.x; col < M; col += blockDim.x) {
            int idx = row * M + col;
            output[idx] = a_val * B[idx];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Use 256 threads per block for optimal warp occupancy
    const int threads = 256;
    // Limit blocks to N so we don't have empty blocks, capped at a reasonable grid size
    const int blocks = std::min(N, 1024);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "broadcast_mul_forward", ([&] {
        broadcast_mul_shared_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N,
            M
        );
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication with shared memory");
}
"""

fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Computes A * B, where A is [N] and B is [N, M].
    Broadcasts A across the rows of B.
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
