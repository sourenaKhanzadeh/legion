#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// CUDA Kernel for squaring numbers
__global__ void squareKernel(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = data[idx] * data[idx];
    }
}

// Function to run CUDA kernel
torch::Tensor squareTensor(torch::Tensor input) {
    input = input.to(torch::kCUDA);

    float* data = input.data_ptr<float>();
    int size = input.numel();
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(data, size);
    cudaDeviceSynchronize();

    return input;
}

// Pybind11 binding
PYBIND11_MODULE(gpu_compute, m) {
    m.def("square_tensor", &squareTensor, "Square a tensor using CUDA");
}
