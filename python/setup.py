from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Define the path to the C++ CUDA files
compute_reg_path = os.path.abspath("../c++/cuda")

setup(
    name="gpu_compute",
    ext_modules=[
        CUDAExtension(
            "gpu_compute",
            [os.path.join(compute_reg_path, "gpu_compute.cu")],  # Locate CUDA file correctly
        ),
        CUDAExtension(
            "mat_mult",
            [os.path.join(compute_reg_path, "mat_mult.cu")],  # Locate CUDA file correctly
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
