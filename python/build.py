import os
import subprocess
import sys
import pathlib
import pybind11
import dotenv
from torch.utils.cpp_extension import include_paths, library_paths

dotenv.load_dotenv()


PYTHON_INCLUDE = pathlib.Path(sys.base_prefix) / "include"
PYTHON_LIB = pathlib.Path(sys.base_prefix) / "libs"

# Detect PyTorch & Pybind11 paths
TORCH_INCLUDE = include_paths()
TORCH_LIB = library_paths()[0]
PYBIND_INCLUDE = pybind11.get_include()

# Detect CUDA paths
CUDA_PATH = pathlib.Path(os.getenv("CUDA_PATH"))
CUDA_INCLUDE = CUDA_PATH / "include"
CUDA_LIB = CUDA_PATH / "lib" / "x64"
CUDA_BIN = CUDA_PATH / "bin"

ALL_CUDA_SOURCES = pathlib.Path(__file__).parent.parent / "c++/cuda"

# Set filenames
CUDA_SOURCES = [ALL_CUDA_SOURCES / pth for pth in os.listdir(ALL_CUDA_SOURCES)]
CUDA_OUTPUT = [source.stem + ".dll" for source in CUDA_SOURCES]

# Compile CUDA PyTorch extension using nvcc
print("🚀 Compiling CUDA PyTorch Extension...")
for source, output in zip(CUDA_SOURCES, CUDA_OUTPUT):
    print(f"🔨 Compiling {source} -> {output}")
    cuda_compile_cmd = [
        "nvcc",
        "-o", str(output), "--shared", str(source),
        "-std=c++17",  # Use C++17
        "-Xcompiler", "/MD /W0",
        f"-I{CUDA_INCLUDE}",
        f"-I{PYTHON_INCLUDE}",  # Fix missing Python.h
        f"-I{pathlib.Path(PYBIND_INCLUDE)}",
        f"-I{pathlib.Path(TORCH_INCLUDE[0])}",
        f"-I{pathlib.Path(TORCH_INCLUDE[1])}",
        f'-L{pathlib.Path(TORCH_LIB)}',
        f'-L{CUDA_LIB}',
        f"-L{PYTHON_LIB}",  # Fix missing Python library
        "-ltorch", "-lc10", "-lcudart", "-ltorch_cpu",  "-ltorch_python", "-lATen"
        # "--compiler-options", "'/MD'",
    ]
    subprocess.run(cuda_compile_cmd, shell=True, check=True)

print("✅ CUDA PyTorch Extension built successfully!")
