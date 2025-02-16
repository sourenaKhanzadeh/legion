import json
from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel
import random

app = FastAPI()

# List of available GPU workers (Modify with actual IPs)
GPU_WORKERS = [
    "http://192.168.2.48:8000/compute",
    "http://192.168.2.59:8002/compute",
]

class ComputeRequest(BaseModel):
    data: list
    operation: str

@app.post("/compute")
async def distribute_compute(request: ComputeRequest):
    if not GPU_WORKERS:
        raise HTTPException(status_code=503, detail="No available GPU workers")

    # Pick a GPU worker randomly (Round-robin can be added)
    selected_worker = random.choice(GPU_WORKERS)

    async with httpx.AsyncClient() as client:
        response = await client.post(selected_worker, json=request.dict())
    
    return response.json()


@app.get("/nvidia-smi")
async def get_nvidia_smi():
    if not GPU_WORKERS:
        raise HTTPException(status_code=503, detail="No available GPU workers")
    gpus = []
    for worker in GPU_WORKERS:
        async with httpx.AsyncClient() as client:
            response = await client.get(worker.replace("compute", "nvidia-smi"))
            if response.status_code == 200:
                gpus.append(response.json())
    return json.dumps(gpus) if gpus else json.dumps({"error": "No available GPU workers"})

# Run with: uvicorn master_server:app --host 0.0.0.0 --port 8000