from fastapi import FastAPI
import torch
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Define a request model
class ComputeRequest(BaseModel):
    data: list  # List of numbers
    operation: str  # Either "square" or "cube"

@app.post("/compute")
async def compute(request: ComputeRequest):
    try:
        # Convert input data to a PyTorch tensor
        tensor = torch.tensor(request.data, dtype=torch.float32).cuda()

        # Perform computation based on the operation
        if request.operation == "square":
            result = tensor ** 2
        elif request.operation == "cube":
            result = tensor ** 3
        else:
            return {"error": "Unsupported operation"}

        return {"result": result.cpu().tolist()}
    
    except Exception as e:
        return {"error": str(e)}

# Run the server using: uvicorn gpu_server:app --host 0.0.0.0 --port 8000
