#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA Kernel for Matrix Multiplication
__global__ void matmulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Function to Run CUDA Matrix Multiplication on a Specific GPU
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int device_id) {
    int N = A.size(0);

    // Move tensors to the correct GPU
    A = A.to(torch::kCUDA, device_id);
    B = B.to(torch::kCUDA, device_id);

    // Allocate memory for the result
    torch::Tensor C = torch::zeros({N, N}, A.options().device(device_id));

    // Get raw pointers
    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Define grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the CUDA kernel on the selected GPU
    cudaSetDevice(device_id);
    matmulKernel<<<numBlocks, threadsPerBlock>>>(A_ptr, B_ptr, C_ptr, N);
    cudaDeviceSynchronize(); // Wait for computation to finish

    return C;
}

// Python Binding for Multi-GPU Matrix Multiplication
PYBIND11_MODULE(gpu_compute, m) {
    m.def("matmul_cuda", &matmul_cuda, "Matrix Multiplication on a Specific GPU");
}
