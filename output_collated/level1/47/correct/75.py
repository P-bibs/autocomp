# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141159/code_8.py
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
#  CUDA kernel with grid-stride loop optimization
# --------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                int B, int D1, int D2) {
    // Thread mapping
    const int batch = blockIdx.x;                      // 0 .. B-1
    const int thread_id = threadIdx.x;                 // 0 .. blockDim.x-1
    const int block_y   = blockIdx.y;                  // column-grid index

    // Global column index this thread starts with
    const int cols_per_grid = blockDim.x * gridDim.y;  // stride over columns
    int col = block_y * blockDim.x + thread_id;        // first column handled

    // Guard against out-of-bounds batch
    if (batch >= B) return;

    // Grid-stride loop over the D2 dimension
    while (col < D2) {
        float sum = 0.0f;

        // Reduction over dimension 1 (D1)
        #pragma unroll 4
        for (int i = 0; i < D1; ++i) {
            // layout: (batch, i, col)
            const int idx = ((batch * D1) + i) * D2 + col;
            sum += input[idx];
        }

        // Write the result
        output[batch * D2 + col] = sum;

        // Advance to next column handled by this thread
        col += cols_per_grid;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    const int threads = 256;
    const dim3 blockDim(threads);
    const dim3 gridDim(B, (D2 + threads - 1) / threads);

    sum_dim1_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
        
    // Propagate possible launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1 (grid-stride version)");
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
    # Only support dim=1 as per problem definition
    assert dim == 1
    # Output shape: (batch_size, 1, dim2)
    output = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, output)
    return output.unsqueeze(1)

# --- Evaluation setup ---
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
