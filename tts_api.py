import os
import uuid
import pathlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from indextts.infer import IndexTTS
import uvicorn

# --- Configuration ---
MODEL_DIR = "checkpoints"
CONFIG_PATH = "checkpoints/config.yaml"
OUTPUT_DIR = "api_outputs"
HOST = "127.0.0.1"
PORT = 8001

# --- Pydantic Models ---
class TTSRequest(BaseModel):
    text: str
    reference_voice_path: str

class TTSResponse(BaseModel):
    output_path: str

# --- Global Variables & Setup ---
app = FastAPI(title="IndexTTS API")
tts_model = None
output_path_obj = pathlib.Path(OUTPUT_DIR)

@app.on_event("startup")
async def load_model():
    """Load the TTS model on server startup."""
    global tts_model
    print("Loading IndexTTS model...")
    try:
        # Determine if FP16 should be used (basic check for CUDA availability)
        # A more robust check might involve checking torch.cuda.is_available()
        # and potentially the GPU capability, but keeping it simple based on webui.py
        try:
            import torch
            is_fp16 = torch.cuda.is_available()
            print(f"CUDA available: {is_fp16}. Using FP16: {is_fp16}")
        except ImportError:
            is_fp16 = False
            print("PyTorch not found or CUDA not available. Using FP32.")

        tts_model = IndexTTS(model_dir=MODEL_DIR, cfg_path=CONFIG_PATH, is_fp16=is_fp16)
        print("IndexTTS model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL Error loading IndexTTS model: {e}")
        # Depending on desired behavior, you might want the app to exit or handle this differently
        raise RuntimeError(f"Failed to load TTS model: {e}") from e

    # Create output directory if it doesn't exist
    output_path_obj.mkdir(parents=True, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")

# --- API Endpoint ---
@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    """
    Synthesizes audio from text using a reference voice audio file.
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model is not loaded or failed to load.")

    # Validate reference voice path
    if not os.path.exists(request.reference_voice_path):
        raise HTTPException(
            status_code=400,
            detail=f"Reference voice path does not exist: {request.reference_voice_path}"
        )

    # Generate unique output filename
    output_filename = f"{uuid.uuid4()}.wav"
    absolute_output_path = output_path_obj.resolve() / output_filename

    print(f"Synthesizing audio for text: '{request.text[:50]}...'")
    print(f"Reference voice: {request.reference_voice_path}")
    print(f"Output path: {absolute_output_path}")

    try:
        # Run TTS inference
        tts_model.infer(
            reference_voice_path=request.reference_voice_path,
            text=request.text,
            output_path=str(absolute_output_path) # Ensure path is string
        )
        print("Synthesis successful.")
        return TTSResponse(output_path=str(absolute_output_path))
    except Exception as e:
        print(f"Error during TTS inference: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to synthesize audio: {e}"
        )

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting IndexTTS API server on {HOST}:{PORT}")
    # Note: Running directly like this is mainly for debugging.
    # Production deployment should use a process manager like Gunicorn with Uvicorn workers.
    uvicorn.run(app, host=HOST, port=PORT)

# How to run from terminal (assuming in index-tts directory with environment active):
# uvicorn tts_api:app --reload --host 127.0.0.1 --port 8001
