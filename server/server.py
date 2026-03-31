from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os

app=FastAPI()
UPLOAD_DIR="uploads"
OUTPUT_DIR="outputs"

os.makedirs(UPLOAD_DIR,exist_ok=True)
os.makedirs(OUTPUT_DIR,exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    input_path=os.path.join(UPLOAD_DIR, file.filename)
    
    with open(input_path,'wb') as buffer:
        shutil.copyfileobj(file.file,buffer)
        
    output_path = os.path.join(OUTPUT_DIR, f"processed_{file.filename}")
    shutil.copy(input_path,output_path)
    
    return {"filename": file.filename}

@app.get("/download/{filename}")
def download_file(filename:str):
    file_path=os.path.join(OUTPUT_DIR,f"processed_{filename}")
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    return FileResponse(file_path,media_type="application/octet_steam",filename=file_path)