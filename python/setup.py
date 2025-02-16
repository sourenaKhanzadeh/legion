from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import shutil

# Define the path to the C++ CUDA files
compute_reg_path = os.path.abspath("../c++/cuda")

# Force delete old build directory (prevents permission issues)
build_dir = os.path.abspath("./build")
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)

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
