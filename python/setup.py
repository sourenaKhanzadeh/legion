from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="gpu_compute",
    ext_modules=[
        CUDAExtension("gpu_compute", ["gpu_compute.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
