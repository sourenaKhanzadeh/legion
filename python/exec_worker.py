from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.post("/execute_script")
async def execute_script(script_path: str):
    try:
        result = subprocess.run(["python3", script_path], capture_output=True, text=True, check=True)
        return {"output": result.stdout, "error": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}

# Run: uvicorn gpu_worker:app --host 0.0.0.0 --port 8001
