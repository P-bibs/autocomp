# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134650/code_8.py
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
#  CUDA kernel – reduction without shared memory (fewer syncs)
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256               // must be a multiple of 32
#define ELEMENTS_PER_THREAD 8      // how many contiguous D2 elements each thread handles

// -----------------------------------------------------------------
__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int B, int D1, int D2) {
    // block (B, tile_id) and thread (tid) identify a contiguous chunk of D2
    const int b        = blockIdx.x;                     // batch index
    const int tile_id  = blockIdx.y;                     // tile along D2
    const int tid      = threadIdx.x;                    // thread inside the tile

    // starting column (along D2) handled by this thread
    const int j_start = (tile_id * TILE_SIZE + tid) * ELEMENTS_PER_THREAD;

    // early‑exit if the batch index is out of range (should never happen)
    if (b >= B) return;

    // -----------------------------------------------------------------
    //  Register accumulation
    // -----------------------------------------------------------------
    float local_sum[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
        local_sum[v] = 0.0f;
    }

    // Loop over the reduction dimension (D1)
    for (int i = 0; i < D1; ++i) {
        // Pointer to the beginning of the row we are reducing
        const float* row = input + b * D1 * D2 + i * D2 + j_start;

        // Load up to ELEMENTS_PER_THREAD contiguous elements (guarded by bounds)
        #pragma unroll
        for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
            int j = j_start + v;
            float val = (j < D2) ? row[v] : 0.0f;   // out‑of‑bounds elements are treated as 0
            local_sum[v] += val;
        }
    }

    // -----------------------------------------------------------------
    //  Write the final sums back to global memory
    // -----------------------------------------------------------------
    float* out_ptr = output + b * D2 + j_start;

    #pragma unroll
    for (int v = 0; v < ELEMENTS_PER_THREAD; ++v) {
        int j = j_start + v;
        if (j < D2) {
            out_ptr[v] = local_sum[v];
        }
    }
}

// -----------------------------------------------------------------
void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    dim3 threads(TILE_SIZE);
    dim3 blocks(B,
                (D2 + TILE_SIZE * ELEMENTS_PER_THREAD - 1) /
                (TILE_SIZE * ELEMENTS_PER_THREAD));

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
    m.def("sum_dim1", &sum_dim1,
          "Sum along dimension 1 (no‑shared‑memory, reduced syncs)");
}
"""

# --------------------------------------------------------------
#  Build the inline extension
# --------------------------------------------------------------
sum_ext = load_inline(
    name='sum_dim1_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# --------------------------------------------------------------
#  Functional model – unchanged API
# --------------------------------------------------------------
def functional_model(x, *, dim):
    """Reduce `x` along dimension 1 (the middle dimension).

    Returns a tensor of shape (batch, 1, dim2) to match the original API.
    """
    assert dim == 1, "Only dim=1 is supported by this custom kernel."
    # allocate output (batch, dim2)
    out = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, out)
    # restore the singleton dimension expected by the original code
    return out.unsqueeze(1)

# --------------------------------------------------------------
#  Evaluation helpers (unchanged from the original script)
# --------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]

# If this file is imported, only `functional_model` will be used.
