from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import pathlib

setup(
    name="gpu_compute",
    ext_modules=[
        CUDAExtension("gpu_compute", [str(pathlib.Path(__file__).parent.parent / "c++/cuda/gpu_compute.cu")]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
