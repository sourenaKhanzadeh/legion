from fastapi import APIRouter, HTTPException
import subprocess
import tempfile
import os
import sys
router = APIRouter()

@router.post("/execute_script")
async def execute_script(script_bytes: bytes):
    try:
        # Create a temporary file for the script
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_script:
            temp_script.write(script_bytes)
            temp_script_path = temp_script.name  # Store the path
        
        python_path = sys.executable
        # Execute the script
        result = subprocess.run([python_path, temp_script_path], capture_output=True, text=True)

        # Read the output
        output = result.stdout
        error = result.stderr

        # Delete the temporary script after execution
        os.remove(temp_script_path)

        return {"output": output, "error": error}

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

# Run: uvicorn gpu_worker:app --host 0.0.0.0 --port 8001
