# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_20.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
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

__global__ void fused_normalize_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int dim
) {
    extern __shared__ float sdata[];
    const int batch_idx = blockIdx.x;
    
    // Pointer arithmetic for current row
    const float* x_row = input + (size_t)batch_idx * dim;
    float* out_row = output + (size_t)batch_idx * dim;

    float local_sum = 0.0f;
    // Each thread loops over dim to calculate absolute sum
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += fabsf(x_row[i]);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    // Write warp results to shared memory
    int warp_id = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        sdata[warp_id] = local_sum;
    }
    __syncthreads();

    // Reduce shared memory results inside the block
    float final_sum = 0.0f;
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < (blockDim.x + 31) / 32) ? sdata[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        final_sum = val;
    }

    // Broadcast normalized multiplier
    float inv_mean = (final_sum == 0.0f) ? 0.0f : (float)dim / final_sum;
    
    // Apply normalization
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_mean;
    }
}

void fused_normalize_forward(const torch::Tensor input, torch::Tensor output, int batch_size, int dim) {
    const int threads = 256; // Balanced occupancy for 2080Ti
    const int shared_mem = (threads / 32) * sizeof(float);
    
    fused_normalize_kernel<<<batch_size, threads, shared_mem>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim
    );
}
"""

cpp_source = r"""
void fused_normalize_forward(const torch::Tensor input, torch::Tensor output, int batch_size, int dim);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_forward, "Fused normalize");
}
"""

fused_ext = load_inline(
    name='fused_normalize_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    # Ensure inputs are contiguous and correct shape
    x = x.contiguous()
    output = torch.empty_like(x)
    
    # Kernel assumes x is on CUDA
    fused_ext.fused_normalize(x, output, x.size(0), x.size(1))
    return output
