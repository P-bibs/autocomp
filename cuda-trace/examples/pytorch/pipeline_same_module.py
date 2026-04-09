import torch
from torch import nn
from torch.utils.cpp_extension import load_inline


cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void scale_into_kernel(T* output, const T* input, const T scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale;
    }
}

template <typename T>
__global__ void shift_kernel(const T* input, T* output, const T shift, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + shift;
    }
}

void scale_into_forward(torch::Tensor output, torch::Tensor input, double scale) {
    const size_t size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "scale_into_kernel", [&] {
        scale_into_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            static_cast<scalar_t>(scale),
            size
        );
    });
    cudaDeviceSynchronize();
}

torch::Tensor shift_forward(torch::Tensor input, double shift) {
    auto output = torch::empty_like(input);
    const size_t size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "shift_kernel", [&] {
        shift_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(shift),
            size
        );
    });
    cudaDeviceSynchronize();
    return output;
}
"""


cpp_source = r"""
#include <torch/extension.h>

void scale_into_forward(torch::Tensor output, torch::Tensor input, double scale);
torch::Tensor shift_forward(torch::Tensor input, double shift);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_forward", &scale_into_forward, "Scale into output");
    m.def("shift_forward", &shift_forward, "Shift output");
}
"""


fused_mod = load_inline(
    name="pipeline_same_module_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)


def get_init_inputs():
    in_dim = 2048
    out_dim = 1024
    scale = 3.0
    shift = -0.5
    return (in_dim, out_dim, scale, shift)


def get_inputs():
    batch_size = 128
    in_dim = 2048
    return (torch.rand(batch_size, in_dim).cuda(),)


class ModelNew(nn.Module):
    def __init__(self, in_dim, out_dim, scale, shift):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.scale = float(scale)
        self.shift = float(shift)

    def forward(self, x):
        a = self.linear(x)
        b = torch.zeros_like(a)
        fused_mod.scale_forward(b, a, self.scale)
        c = fused_mod.shift_forward(b, self.shift)
        return c
