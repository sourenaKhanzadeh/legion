from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Define the path to the C++ CUDA files
cuda_path = os.path.abspath("../C++/cuda")

setup(
    name="gpu_compute",
    ext_modules=[
        CUDAExtension(
            "gpu_compute",
            [os.path.join(cuda_path, "gpu_compute.cu")],  # Locate CUDA file correctly
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
