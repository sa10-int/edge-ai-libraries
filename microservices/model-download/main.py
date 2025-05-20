from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from logger import logger
from huggingface_hub import snapshot_download
import os
import gc
import subprocess

app = FastAPI()

# Define the input model
class ModelRequest(BaseModel):
    model_name: str
    hf_token: str
    model_type: str = None  # Optional by default, but will be validated if is_ovms is True
    is_ovms: bool = False

@app.post("/download-model/")
async def download_model(request: ModelRequest):
    """
    Endpoint to download a model from Hugging Face.
    """
    try:
        # Validate that model_type is provided if is_ovms is True
        if request.is_ovms and not request.model_type:
            raise HTTPException(
                status_code=400,
                detail="model_type is required when is_ovms is True"
            )

        # Set the Hugging Face token as an environment variable
        os.environ["HF_TOKEN"] = request.hf_token
        logger.info(f"Initiating model download with parameters: model_name={request.model_name}, model_type={request.model_type}, is_ovms={request.is_ovms}")
        # Download the entire model repository
        model_path = snapshot_download(repo_id=request.model_name, use_auth_token=request.hf_token)

        # If is_ovms is True, apply model conversion
        if request.is_ovms:
            convert_model_to_ovms(
                model_name=request.model_name,
                weight_format="int8",  # Specify the desired weight format
                huggingface_token=request.hf_token,
                model_type=request.model_type
            )

        # Explicitly clean up resources to avoid semaphore warnings
        gc.collect()

        return {"message": "Model downloaded successfully", "model_path": model_path}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading or converting model: {str(e)}")

def convert_model_to_ovms(model_name, weight_format, huggingface_token, model_type):
    """
    Downloads a model from Hugging Face, converts it to OVMS format, and prepares it for deployment.

    Args:
        model_name (str): The name of the Hugging Face model to download.
        weight_format (str): The weight format for the exported model (e.g., "int4", "fp16").
        huggingface_token (str): The Hugging Face API token for authentication.
        model_type (str): The type of the model (e.g., "llm", "embeddings", "rerank").

    Returns:
        dict: A success message if the process completes successfully.

    Raises:
        HTTPException: If any step in the process fails.
    """
    try:
        # Map model_type to export type
        export_type_map = {
            "llm": "text_generation",
            "embeddings": "embeddings",
            "rerank": "rerank"
        }

        # Validate model_type
        if model_type not in export_type_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {model_type}. Must be one of {list(export_type_map.keys())}."
            )

        export_type = export_type_map[model_type]

        # Step 1: Install required Python packages
        requirements_url = "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt"
        logger.info("Installing required Python packages...")
        subprocess.run(["pip3", "install", "-r", requirements_url], check=True)

        # Step 2: Log in to Hugging Face
        logger.info("Logging in to Hugging Face...")
        result = subprocess.run(["huggingface-cli", "login", "--token", huggingface_token])
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail="Failed to log in to Hugging Face. Please check your token.")

        # Step 3: Download the export_model.py script
        logger.info("Downloading export_model.py script...")
        export_script_url = "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py"
        subprocess.run(["curl", "-o", "export_model.py", export_script_url], check=True)

        # Step 4: Export the model
        logger.info(f"Exporting model: {model_name} with weight format: {weight_format} and export type: {export_type}...")
        os.makedirs("models", exist_ok=True)
        command = [
            "python3", "export_model.py", export_type,
            "--source_model", model_name,
            "--weight-format", weight_format,
            "--config_file_path", "models/config.json",
            "--model_repository_path", "models",
            "--target_device", "CPU"
        ]
        subprocess.run(command, check=True)

        return {"message": f"Model successfully downloaded, converted, and prepared for OVMS deployment as {export_type}."}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during the process: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")