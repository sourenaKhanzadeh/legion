from fastapi import FastAPI
import torch
import gpu_compute  # Import compiled CUDA extension
from pydantic import BaseModel
import socket

app = FastAPI()
HOSTNAME = socket.gethostname()

class ComputeRequest(BaseModel):
    data: list
    operation: str

@app.post("/compute")
async def compute(request: ComputeRequest):
    try:
        tensor = torch.tensor(request.data, dtype=torch.float32).cuda()

        if request.operation == "square":
            result = gpu_compute.square_tensor(tensor)  # Call C++ CUDA function
        else:
            return {"error": "Unsupported operation"}

        return {"gpu": HOSTNAME, "result": result.cpu().tolist()}

    except Exception as e:
        return {"error": str(e)}



@app.get("/nvidia-smi")
async def get_nvidia_smi():
    # get device name
    device_name = torch.cuda.get_device_name(0)
    return {"nvidia-smi": device_name}

# Run the worker
# Start using: uvicorn gpu_worker:app --host 0.0.0.0 --port 8001
