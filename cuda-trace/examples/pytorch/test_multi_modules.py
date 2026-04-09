import torch
from torch import nn
from torch.utils.cpp_extension import load_inline


cuda_source_one = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void scale_kernel(const T* inp, T* out, const T scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = inp[idx] * scale;
    }
}

torch::Tensor scale_cuda(torch::Tensor input, double scale) {
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


cpp_source_one = r"""
#include <torch/extension.h>

torch::Tensor scale_cuda(torch::Tensor input, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &scale_cuda, "Scale CUDA op");
}
"""


cuda_source_two = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename T>
__global__ void shift_kernel(const T* inp, T* out, const T shift, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = inp[idx] + shift;
    }
}

torch::Tensor shift_cuda(torch::Tensor input, double shift) {
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


cpp_source_two = r"""
#include <torch/extension.h>

torch::Tensor shift_cuda(torch::Tensor input, double shift);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shift_cuda, "Shift CUDA op");
}
"""


scale_mod = load_inline(
    name="fused_module_scale",
    cpp_sources=cpp_source_one,
    cuda_sources=cuda_source_one,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

shift_mod = load_inline(
    name="fused_module_shift",
    cpp_sources=cpp_source_two,
    cuda_sources=cuda_source_two,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_dim, out_dim, scale, shift):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.scale = float(scale)
        self.shift = float(shift)

    def forward(self, x):
        out = self.linear(x)
        out = scale_mod.forward(out, self.scale)
        return shift_mod.forward(out, self.shift)


def main():
    batch_size = 128
    in_dim = 2048
    out_dim = 1024
    scale = 3.0
    shift = -0.5

    model = ModelNew(in_dim, out_dim, scale, shift).cuda()
    input_tensor = torch.rand(batch_size, in_dim).cuda()
    output = model(input_tensor)
    print(output[:4, :8])


main()
