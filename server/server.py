from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from main import process_video
from fastapi.concurrency import run_in_threadpool
import shutil
import os
from pathlib import Path

# Get paths relative to this file's location
SERVER_DIR = Path(__file__).parent
PROJECT_DIR = SERVER_DIR.parent
MODEL_DIR = PROJECT_DIR / "model"

# Use relative paths for uploads and outputs
UPLOAD_DIR = SERVER_DIR / "uploads"
OUTPUT_DIR = SERVER_DIR / "outputs"

# Get model path
MODEL_PATH = MODEL_DIR / "full_best_dynamic.onnx"

app=FastAPI()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    input_path = UPLOAD_DIR / file.filename
    
    with open(input_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "File Uploaded",
        "filename": file.filename
    }

@app.post("/process/")
async def process_file(filename: str = Form(...), sport: str = Form(...)):
    input_path = UPLOAD_DIR / filename

    if not input_path.exists():
        return {"error": "file not found"}

    output_path = await run_in_threadpool(
        process_video,
        video_path=str(input_path),
        model_path=str(MODEL_PATH),
        sport=sport
    )

    return {
        "message":"Processing done",
        "output_filename": os.path.basename(output_path)
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = OUTPUT_DIR / f"processed_{filename}"
    if not file_path.exists():
        return {"error": "File not found"}
    
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)