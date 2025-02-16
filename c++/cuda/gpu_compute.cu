#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for squaring elements
__global__ void squareKernel(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = data[idx] * data[idx];
    }
}

// Function to run CUDA kernel
torch::Tensor squareTensor(torch::Tensor input) {
    input = input.to(torch::kCUDA);  // Move tensor to GPU

    float* data = input.data_ptr<float>();
    int size = input.numel();
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(data, size);
    cudaDeviceSynchronize();  // Ensure execution completes

    return input;
}

// Bind to Python
PYBIND11_MODULE(gpu_compute, m) {
    m.def("square_tensor", &squareTensor, "Square a tensor using CUDA");
}
