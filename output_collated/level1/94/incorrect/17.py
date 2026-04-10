# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_151305/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['predictions', 'targets']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
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

__global__ void mse_kernel(const float* __restrict__ preds, const float* __restrict__ targets, float* result, size_t n) {
    extern __shared__ float sdata[];
    float sum = 0.0f;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better memory coalescing and handling large arrays
    for (size_t i = idx; i < n; i += stride) {
        float diff = preds[i] - targets[i];
        sum += diff * diff;
    }

    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store warp results in shared memory
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction within the first warp
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + 31) / 32) ? sdata[lane_id] : 0.0f;
        
        // Reduce within first warp
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Thread 0 writes the final result
        if (lane_id == 0) {
            atomicAdd(result, sum);
        }
    }
}

void fused_op_forward(torch::Tensor preds, torch::Tensor targets, torch::Tensor result) {
    size_t n = preds.numel();
    const int threads_per_block = 256;
    const int blocks = min(65535, (int)((n + threads_per_block - 1) / threads_per_block));
    const int warps_per_block = (threads_per_block + 31) / 32;
    
    mse_kernel<<<blocks, threads_per_block, warps_per_block * sizeof(float)>>>(
        preds.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        result.data_ptr<float>(), 
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor preds, torch::Tensor targets, torch::Tensor result);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "MSE Forward with Block-Level Reduction");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(predictions, targets):
    n = predictions.numel()
    result = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    fused_ext.fused_op(predictions.contiguous(), targets.contiguous(), result)
    return result / n

def get_init_inputs(): 
    return []

batch_size = 32768
input_shape = (32768,)

def get_inputs():
    scale = torch.rand(()).cuda()
    return [torch.rand(batch_size, *input_shape, device='cuda')*scale, 
            torch.rand(batch_size, *input_shape, device='cuda')]
