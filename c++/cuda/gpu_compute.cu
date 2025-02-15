#include <torch/extension.h>
#include <vector>

// CUDA kernel function to square elements
__global__ void squareKernel(float* d_data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_data[idx] = d_data[idx] * d_data[idx];
    }
}

// CUDA function to execute the kernel
torch::Tensor squareTensor(torch::Tensor input) {
    // Ensure input is on CUDA
    input = input.to(torch::kCUDA);
    
    // Get raw pointer to data
    float* d_data = input.data_ptr<float>();
    
    // Define CUDA kernel execution parameters
    int size = input.numel();
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);

    // Ensure kernel execution is completed
    cudaDeviceSynchronize();

    return input;  // Return the modified tensor
}

// Binding C++ function to Python
PYBIND11_MODULE(gpu_compute, m) {
    m.def("square_tensor", &squareTensor, "Square a tensor using CUDA");
}
