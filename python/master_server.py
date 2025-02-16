import json
from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel
import random
import torch 
import numpy as np
import asyncio
app = FastAPI()

# List of available GPU workers (Modify with actual IPs)
GPU_WORKERS = [
    "http://192.168.2.48:8000/compute",
    "http://192.168.2.59:8002/compute",
]

class ComputeRequest(BaseModel):
    data: list
    operation: str

class MatMulRequest(BaseModel):
    A: list
    B: list


@app.post("/compute")
async def distribute_compute(request: ComputeRequest):
    if not GPU_WORKERS:
        raise HTTPException(status_code=503, detail="No available GPU workers")

    # Pick a GPU worker randomly (Round-robin can be added)
    selected_worker = random.choice(GPU_WORKERS)

    async with httpx.AsyncClient() as client:
        response = await client.post(selected_worker, json=request.dict())
    
    return response.json()

@app.post("/matmul")
async def distribute_matmul(request: MatMulRequest):
    A = torch.tensor(request.A, dtype=torch.float32)
    B = torch.tensor(request.B, dtype=torch.float32)

    num_gpus = len(GPU_WORKERS)
    chunk_size = A.size(0) // num_gpus

    # Split matrix A row-wise
    sub_matrices = torch.chunk(A, num_gpus, dim=0)

    tasks = []
    async with httpx.AsyncClient() as client:
        for i, sub_matrix in enumerate(sub_matrices):
            payload = {
                "A": sub_matrix.tolist(),
                "B": B.tolist()
            }
            tasks.append(client.post(GPU_WORKERS[i].replace("compute", "matmul"), json=payload))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    results = []
    gpus = {}
    for response in responses:
        if isinstance(response, Exception):
            raise HTTPException(status_code=500, detail=f"GPU Worker Error: {str(response)}")
        try:
            res = response.json()
            if "result" in res:
                results.append(res["result"])
                gpus[len(results) - 1] = res["GPU"]
            elif "error" in res:
                raise HTTPException(status_code=500, detail=res["error"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse GPU response: {str(e)}")

    # Concatenate results
    if not results:
        raise HTTPException(status_code=500, detail="No valid results from GPU workers")
    
    final_result = np.vstack(results).tolist()

    return {"result": final_result, **gpus}

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