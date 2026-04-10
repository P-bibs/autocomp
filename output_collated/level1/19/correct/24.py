# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
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

# Optimized CUDA Kernel using warp-cooperative vectorized ReLU
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#define WARP_SIZE 32
#define ELEMENTS_PER_THREAD 4
#define THREADS_PER_BLOCK 256

__global__ void relu_coop_vec_kernel(const float* __restrict__ input, float* __restrict__ output, size_t num_elements) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const size_t gid = (blockIdx.x * blockDim.x + tid);

    const size_t total_vec4s = (num_elements + 3) / 4;
    const size_t vec4_per_warp = WARP_SIZE;
    const size_t vec4_per_block = THREADS_PER_BLOCK;
    const size_t block_start_vec4 = blockIdx.x * vec4_per_block;
    
    // Each thread processes one float4 per iteration
    for (size_t i = 0; i < vec4_per_warp; i += WARP_SIZE) {
        size_t idx_vec4 = block_start_vec4 + warp_id * WARP_SIZE + i + lane_id;

        if (idx_vec4 < total_vec4s) {
            float4 val = reinterpret_cast<const float4*>(input)[idx_vec4];
            
            val.x = fmaxf(0.0f, val.x);
            val.y = fmaxf(0.0f, val.y);
            val.z = fmaxf(0.0f, val.z);
            val.w = fmaxf(0.0f, val.w);

            reinterpret_cast<float4*>(output)[idx_vec4] = val;
        }
    }
}

void relu_coop_vec_launch(torch::Tensor input, torch::Tensor output) {
    size_t num_elements = input.numel();
    const size_t vec4_count = (num_elements + 3) / 4;
    
    const int threads_per_block = THREADS_PER_BLOCK;
    const int vec4s_per_block = threads_per_block;
    const int num_blocks = (vec4_count + vec4s_per_block - 1) / vec4s_per_block;

    relu_coop_vec_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void relu_coop_vec_launch(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_coop_vec", &relu_coop_vec_launch, "Cooperative Vectorized ReLU kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized ReLU using cooperative warp-level vectorized CUDA kernel.
    """
    output = torch.empty_like(x)
    fused_ext.relu_coop_vec(x, output)
    return output

# Evaluation setup constants
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda')]
