import shutil
import subprocess
import os
from contextlib import asynccontextmanager
from pathlib import Path

import modal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

# ── Paths ──────────────────────────────────────────────────────────
SERVER_DIR = Path(__file__).parent
PROJECT_DIR = SERVER_DIR.parent

UPLOAD_DIR = SERVER_DIR / "uploads"
OUTPUT_DIR = SERVER_DIR / "outputs"
FRONTEND_DIST_DIR = SERVER_DIR / "frontend" / "dist"

MODAL_MAIN_PATH = PROJECT_DIR / "modal_main.py"
MODAL_APP_NAME = "thesis-tracker-pro"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Modal Lifecycle (deploy on startup, stop on shutdown) ──────────
SKIP_MODAL = os.getenv("SKIP_MODAL", "false").lower() == "true"

@asynccontextmanager
async def lifespan(app):
    if SKIP_MODAL:
        print("⏭️ Skipping Modal deployment (SKIP_MODAL=true) for faster frontend dev...")
        yield
        return

    # STARTUP: deploy Modal app
    print(f"🚀 Deploying Modal app '{MODAL_APP_NAME}'...")
    try:
        subprocess.run(
            ["modal", "deploy", str(MODAL_MAIN_PATH)],
            check=True,
        )
        print(f"✅ Modal app '{MODAL_APP_NAME}' deployed and ready.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to deploy Modal app: {e}")
        print("   The server will start, but processing requests will fail.")
    
    yield  # ← server is running and handling requests here

    # SHUTDOWN: stop Modal app
    print(f"🛑 Stopping Modal app '{MODAL_APP_NAME}'...")
    try:
        subprocess.run(
            ["modal", "app", "stop", MODAL_APP_NAME],
            check=True,
        )
        print(f"✅ Modal app '{MODAL_APP_NAME}' stopped.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to stop Modal app: {e}")

# ── FastAPI App ────────────────────────────────────────────────────
app = FastAPI(title="Thesis Tracker Pro Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────
@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...)):
    input_path = UPLOAD_DIR / file.filename
    
    with open(input_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "File Uploaded",
        "filename": file.filename
    }

def run_modal_task(sport: str, filename: str, input_path: str):
    # 1. Upload to Modal Volume
    print(f"Uploading {filename} to Modal Volume...")
    try:
        subprocess.run(
            ["modal", "volume", "put", "grad_proj_vol", input_path, f"input_folder/{filename}", "--force"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to upload to Modal Volume: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file to remote storage.")

    # 2. Call the deployed function directly via Modal SDK
    print(f"Calling deployed Modal function for {sport} - {filename}...")
    try:
        process_fn = modal.Function.from_name(MODAL_APP_NAME, "process_tracker_remote")
        video_bytes = process_fn.remote(
            sport=sport,
            input_filename=filename,
            enable_team_assignment=True,
        )
    except Exception as e:
        print(f"Modal remote call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process video remotely: {e}")

    if not video_bytes:
        raise HTTPException(status_code=500, detail="Remote tracking failed. No video returned.")

    # 3. Save the returned bytes
    output_filename = f"{filename}_processed.mp4"
    output_path = OUTPUT_DIR / output_filename
    with open(output_path, "wb") as f:
        f.write(video_bytes)

    # 4. Re-encode to H.264 so browsers can play it inline
    #    (OpenCV's mp4v codec is not browser-compatible)
    browser_filename = f"{filename}_browser.mp4"
    browser_path = OUTPUT_DIR / browser_filename
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(output_path),
                "-c:v", "libx264", "-preset", "fast",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
                str(browser_path),
            ],
            check=True,
            capture_output=True,
        )
        print(f"Re-encoded to browser-friendly H.264: {browser_path}")
    except FileNotFoundError:
        print("ffmpeg not found — serving original file (may not play in browser)")
        browser_filename = output_filename
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg re-encode failed: {e.stderr}")
        browser_filename = output_filename

    return browser_filename

@app.post("/api/process/")
async def process_file(filename: str = Form(...), sport: str = Form(...)):
    input_path = UPLOAD_DIR / filename

    if not input_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        output_filename = await run_in_threadpool(
            run_modal_task,
            sport=sport,
            filename=filename,
            input_path=str(input_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Processing done",
        "output_filename": output_filename
    }

@app.get("/api/download/{filename}")
def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(str(file_path), media_type="video/mp4", filename=filename)

# ── Serve Frontend Static Files ───────────────────────────────────
if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="assets")

    # Catch-all route to serve the SPA index.html
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        index_file = FRONTEND_DIST_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(), status_code=200)
        return {"error": "Frontend not built. Please run 'npm run build' inside server/frontend"}