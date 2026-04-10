# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134650/code_28.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# --------------------------------------------------------------
# Optimized CUDA kernel – direct register accumulation
# The original code's shared memory implementation was suboptimal 
# due to excessive thread synchronization. By eliminating shared 
# memory and accumulating in registers, we drastically reduce latency.
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B, const int D1, const int D2) {
    // Each thread block processes a specific batch index
    const int b = blockIdx.x;
    if (b >= B) return;

    // Tile index within the D2 dimension
    const int tile_id = blockIdx.y;
    const int tid = threadIdx.x;

    // Calculate column range for this thread
    const int j_start = (tile_id * TILE_SIZE + tid) * ELEMENTS_PER_THREAD;

    // Accumulate in registers to avoid global memory round-trips
    float local_sum[ELEMENTS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Reduction over D1: stride is D2
    const float* b_ptr = input + (long)b * D1 * D2;
    for (int i = 0; i < D1; ++i) {
        const float* row_ptr = b_ptr + (long)i * D2;
        #pragma unroll
        for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
            int j = j_start + v;
            if (j < D2) {
                local_sum[v] += row_ptr[j];
            }
        }
    }

    // Write result back to global output buffer
    float* out_ptr = output + (long)b * D2;
    #pragma unroll
    for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
        int j = j_start + v;
        if (j < D2) {
            out_ptr[j] = local_sum[v];
        }
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // Launch config: block.x=256, grid.x=B, grid.y=ceil(D2 / (256*4))
    dim3 threads(TILE_SIZE);
    dim3 blocks(B, (D2 + TILE_SIZE * ELEMENTS_PER_THREAD - 1) / (TILE_SIZE * ELEMENTS_PER_THREAD));

    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Optimized sum along dim 1");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Computes sum along dimension 1.
    Input: (B, D1, D2)
    Output: (B, 1, D2)
    """
    assert dim == 1
    # Create output buffer of size (B, D2)
    output = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    # Launch specialized CUDA kernel
    sum_ext.sum_dim1(x, output)
    # Result matches original shape (B, 1, D2)
    return output.unsqueeze(1)
