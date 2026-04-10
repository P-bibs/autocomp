# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_22.py
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

// Optimized kernel using 1D grid launch.
// We handle global indexing linearly to ensure coalesced access.
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    int total_elements = N * M;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized processing
    for (int i = idx * 4; i < total_elements; i += stride * 4) {
        if (i + 3 < total_elements) {
            // Calculate row index once per 4 elements. 
            // Since we operate on contiguous memory, i/M is constant for the block
            // but changes across the grid.
            int n_start = i / M;
            int n_end = (i + 3) / M;
            
            if (n_start == n_end) {
                float a_val = A[n_start];
                float4 b_val = reinterpret_cast<const float4*>(&B[i])[0];
                float4 out_val = {a_val * b_val.x, a_val * b_val.y, a_val * b_val.z, a_val * b_val.w};
                reinterpret_cast<float4*>(&output[i])[0] = out_val;
            } else {
                // Rare case: element 4-tuple crosses a row boundary
                for(int j = i; j < i + 4 && j < total_elements; ++j) {
                    output[j] = A[j / M] * B[j];
                }
            }
        } else {
            // Handling the tail of the total_elements if not multiple of 4
            for(int j = i; j < i + 4 && j < total_elements; ++j) {
                output[j] = A[j / M] * B[j];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    int total_elements = N * M;
    
    const int threads = 256;
    const int blocks = (total_elements / 4 + threads - 1) / threads;
    
    // We cap blocks to avoid excessive overhead, 1024-2048 is usually sufficient for 2080Ti
    broadcast_mul_kernel<<<min(blocks, 1024), threads>>>(
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
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication");
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
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    # Ensure contiguous for coalesced access
    A = A.contiguous()
    B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
