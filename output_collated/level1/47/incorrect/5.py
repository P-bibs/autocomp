# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_9.py
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
#  CUDA kernel with shared memory tiling to minimize global memory accesses
# --------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

 constexpr int VEC = 4;
 constexpr int tile_D1 = 16;
 constexpr int tile_D2_threads = 128;
 constexpr int tile_D2 = tile_D2_threads * VEC; // 512

__global__ void sum_dim1_tile_kernel(
        const float* __restrict__  input,
        float*       __restrict__  output,
        int B, int D1, int D2) {

    extern __shared__ float smem[];
    float* tile = smem;

    int b = blockIdx.x;
    int tile_start_j = blockIdx.y * tile_D2;
    int lane = threadIdx.x; // 0 to 127

    // Each thread handles VEC=4 columns
    int j_base = tile_start_j + lane * VEC;

    // Phase 1: Load tile from global memory to shared memory
    for (int i = 0; i < tile_D1; ++i) {
        int global_i = i;
        if (global_i < D1) {
            for (int v = 0; v < VEC; ++v) {
                int j = j_base + v;
                if (j < D2) {
                    tile[i * tile_D2 + lane * VEC + v] =
                        input[b * D1 * D2 + global_i * D2 + j];
                }
            }
        }
    }
    __syncthreads();

    // Phase 2: Reduce along tile_D1 dimension in shared memory
    float sum[VEC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < tile_D1; ++i) {
        if (i < D1) {
            for (int v = 0; v < VEC; ++v) {
                int j = j_base + v;
                if (j < D2) {
                    sum[v] += tile[i * tile_D2 + lane * VEC + v];
                }
            }
        }
    }

    // Phase 3: Write result to global memory
    for (int v = 0; v < VEC; ++v) {
        int j = j_base + v;
        if (j < D2) {
            output[b * D2 + j] = sum[v];
        }
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    dim3 threads(tile_D2_threads); // 128 threads
    dim3 blocks(B, (D2 + tile_D2 - 1) / tile_D2);
    size_t shmem_bytes = tile_D1 * tile_D2 * sizeof(float);

    sum_dim1_tile_kernel<<<blocks, threads, shmem_bytes>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1 with shared memory tiling");
}
"""

# Compile the extension
sum_ext = load_inline(
    name='sum_dim1',
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
