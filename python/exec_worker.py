from fastapi import APIRouter, HTTPException, Body
import subprocess
import tempfile
import os
import sys
import zipfile
import yaml
router = APIRouter()

@router.post("/execute_script")
async def execute_script(script_bytes: bytes = Body(...)):
    try:
        # Create a temporary file to store the script
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_script:
            temp_script.write(script_bytes)
            temp_script_path = temp_script.name  # Save path for execution
        
        # Execute the script
        result = subprocess.run([sys.executable, temp_script_path], capture_output=True, text=True)

        # Read the execution results
        output = result.stdout
        error = result.stderr

        # Delete the temporary script
        os.remove(temp_script_path)

        return {"output": output, "error": error}

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


@router.post("/execute_project")
async def execute_project(project_zip: bytes = Body(...)):
    try:
        # Create a temporary directory for the project
        with tempfile.TemporaryDirectory() as temp_dir:
            # store the zip file in the temp directory
            with open(os.path.join(temp_dir, "project.zip"), "wb") as f:
                f.write(project_zip)
            # unzip the project.zip file
            with zipfile.ZipFile(os.path.join(temp_dir, "project.zip"), 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            # get main.yaml file
            main_yaml = os.path.join(temp_dir, "main.yaml")
            if not os.path.exists(main_yaml):
                return {"output": "", "error": "main.yaml not found"}
            
            # get the command from the main.yaml file
            with open(main_yaml, 'r') as f:
                main_yaml_content = yaml.safe_load(f)
            main_root = main_yaml_content.get("main", "")
            if not main_root:
                return {"output": "", "error": "main not found in main.yaml"}
            
            # Execute the project
            result = subprocess.run([sys.executable, os.path.join(temp_dir, main_root)], capture_output=True, text=True)

            # Read the execution results
            output = result.stdout
            error = result.stderr

            return {"output": output, "error": error}

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    

# Run: uvicorn gpu_worker:app --host 0.0.0.0 --port 8001
