# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_17.py
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

// Using float4 for memory coalescing and bandwidth maximization
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    int total_elements = N * M;
    // Map threads to float4 (4 elements at a time)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;

    for (int i = idx; i < total_elements; i += stride) {
        if (i + 3 < total_elements) {
            float4 b_vec = reinterpret_cast<const float4*>(B + i)[0];
            
            float4 out_vec;
            // A is broadcasted: A[i / M]
            out_vec.x = A[(i) / M] * b_vec.x;
            out_vec.y = A[(i + 1) / M] * b_vec.y;
            out_vec.z = A[(i + 2) / M] * b_vec.z;
            out_vec.w = A[(i + 3) / M] * b_vec.w;
            
            reinterpret_cast<float4*>(output + i)[0] = out_vec;
        } else {
            // Handle tail elements
            for (int k = i; k < total_elements; ++k) {
                output[k] = A[k / M] * B[k];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    // Config: 256 threads, dynamic grid size for occupancy
    const int threads = 256;
    const int blocks = std::min((total_elements / 4 + threads - 1) / threads, 1024);
    
    broadcast_mul_kernel<<<blocks, threads>>>(
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
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous float32 tensors
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    A = A.contiguous()
    B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

# Verification placeholders
def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
