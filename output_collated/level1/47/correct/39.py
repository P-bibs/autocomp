# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_0.py
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
#  CUDA kernel with memory coalescing optimization
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM1 32
#define TILE_DIM2 32

__global__ void sum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                int B, int D1, int D2) {
    // Shared memory to hold a tile of input
    __shared__ float shared_input[TILE_DIM1][TILE_DIM2 + 1];

    int b = blockIdx.x;
    int tile_d1 = blockIdx.y;
    int tile_d2 = blockIdx.z;
    
    int tid_d1 = threadIdx.y;
    int tid_d2 = threadIdx.x;

    if (b >= B) return;

    float sum = 0.0f;
    
    // Process multiple elements along D1
    for (int i = tile_d1 * TILE_DIM1 + tid_d1; i < D1; i += gridDim.y * TILE_DIM1) {
        // Load data coalesced into shared memory
        int j_global = tile_d2 * TILE_DIM2 + tid_d2;
        if (j_global < D2) {
            shared_input[tid_d1][tid_d2] = input[b * D1 * D2 + i * D2 + j_global];
        } else {
            shared_input[tid_d1][tid_d2] = 0.0f;
        }
        
        __syncthreads();
        
        // Sum along D1 dimension within this tile
        if (tid_d1 == 0 && j_global < D2) {
            for (int k = 0; k < min(TILE_DIM1, D1 - tile_d1 * TILE_DIM1); ++k) {
                sum += shared_input[k][tid_d2];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (tid_d1 == 0) {
        int j_global = tile_d2 * TILE_DIM2 + tid_d2;
        if (j_global < D2) {
            atomicAdd(&output[b * D2 + j_global], sum);
        }
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    
    // Initialize output to zero
    cudaMemset(output.data_ptr<float>(), 0, B * D2 * sizeof(float));
    
    // Launch configuration: 2D thread blocks for coalesced access
    dim3 threads(TILE_DIM2, TILE_DIM1);
    dim3 blocks(B, min(65535, (D1 + TILE_DIM1 - 1) / TILE_DIM1), (D2 + TILE_DIM2 - 1) / TILE_DIM2);
    
    sum_dim1_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1 with coalesced memory access");
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
