from fastapi import FastAPI
import torch
from pydantic import BaseModel
import socket

app = FastAPI()

# Get the hostname (for identification)
HOSTNAME = socket.gethostname()

# Define request model
class ComputeRequest(BaseModel):
    data: list
    operation: str

@app.post("/compute")
async def compute(request: ComputeRequest):
    try:
        tensor = torch.tensor(request.data, dtype=torch.float32).cuda()
        
        if request.operation == "square":
            result = tensor ** 2
        elif request.operation == "cube":
            result = tensor ** 3
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
