import torch
from torch import nn
from torch.utils.cpp_extension import load_inline


scale_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void scale_kernel(const T* input, T* output, const T scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale;
    }
}

torch::Tensor scale_forward(torch::Tensor input, double scale) {
    auto output = torch::empty_like(input);
    const size_t size = input.numel();
    const int threads = 128;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "scale_kernel", [&] {
        scale_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(scale),
            size
        );
    });
    cudaDeviceSynchronize();
    return output;
}
"""


scale_cpp_source = r"""
#include <torch/extension.h>

torch::Tensor scale_forward(torch::Tensor input, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &scale_forward, "Scale op");
}
"""


shift_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void shift_kernel(const T* input, T* output, const T shift, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + shift;
    }
}

torch::Tensor shift_forward(torch::Tensor input, double shift) {
    auto output = torch::empty_like(input);
    const size_t size = input.numel();
    const int threads = 512;
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


shift_cpp_source = r"""
#include <torch/extension.h>

torch::Tensor shift_forward(torch::Tensor input, double shift);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shift_forward, "Shift op");
}
"""


scale_mod = load_inline(
    name="pipeline_scale_multi_module",
    cpp_sources=scale_cpp_source,
    cuda_sources=scale_cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

shift_mod = load_inline(
    name="pipeline_shift_multi_module",
    cpp_sources=shift_cpp_source,
    cuda_sources=shift_cuda_source,
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
        b = torch.relu(a)
        c = scale_mod.forward(b, self.scale)
        d = shift_mod.forward(c, self.shift)
        return d
