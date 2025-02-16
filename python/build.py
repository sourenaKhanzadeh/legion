import os
import subprocess
import sys
import torch
import pathlib
import dotenv

dotenv.load_dotenv()

# Detect PyTorch paths
TORCH_INCLUDE = torch.utils.cpp_extension.include_paths()
TORCH_LIB = torch.utils.cpp_extension.library_paths()[0]

# Detect CUDA paths
CUDA_PATH = os.getenv("CUDA_PATH")
CUDA_INCLUDE = f"{CUDA_PATH}/include"
CUDA_LIB = f"{CUDA_PATH}/lib/x64"
CUDA_BIN = f"{CUDA_PATH}/bin"

CUDA_DIR = pathlib.Path(__file__).parent.parent / "c++/cuda"
# Set filenames
CUDA_SOURCES = os.listdir(CUDA_DIR)
CUDA_OUTPUT = [f"{CUDA_DIR}/{source.replace('.cu', '.dll')}" for source in CUDA_SOURCES]

# Compile CUDA code using nvcc
print("🚀 Compiling CUDA kernel...")
for source, output in zip(CUDA_SOURCES, CUDA_OUTPUT):
    print(f"Compiling {source} -> {output}")
    cuda_compile_cmd = [
        f'"{CUDA_BIN}/nvcc.exe"',
        "-shared", "-o", output, source,
    f"-I{CUDA_INCLUDE}",
    f"-I{TORCH_INCLUDE[0]}",
    f"-I{TORCH_INCLUDE[1]}",
    f"-L{TORCH_LIB}",
    f"-L{CUDA_LIB}",
    "-ltorch", "-lc10", "-lcudart"
]
subprocess.run(" ".join(cuda_compile_cmd), shell=True, check=True)


print("✅ Build complete! Run `main.exe` to test.")

