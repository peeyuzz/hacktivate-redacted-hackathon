from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import shutil
import os
from app.redactor import Redactor
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

if os.path.exists("static"):
    shutil.rmtree("static")

os.makedirs(os.path.join("static", "redacted_files"))


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

Path("static/redacted_files").mkdir(parents=True, exist_ok=True)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# ... (rest of the imports and setup remain the same)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"static/uploaded_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.debug(f"File saved to {file_path}")
        
        # Perform redaction
        redactor = Redactor(file_path, plan_type="pro")
        redacted_path = redactor.redact()
        logger.debug(f"redacted_path: {redacted_path}")

        if redacted_path is None or not os.path.exists(redacted_path):
            raise ValueError(f"Redaction failed or file not found: {redacted_path}")
        
        logger.debug(f"File redacted, new path: {redacted_path}")
        
        # Move redacted file to static/redacted_files
        new_path = f"static/redacted_files/{os.path.basename(redacted_path)}"
        shutil.move(redacted_path, new_path)
        
        logger.debug(f"Redacted file moved to {new_path}")
        
        # Clean up original uploaded file
        os.remove(file_path)
        
        return {"filename": os.path.basename(new_path)}
    except Exception as e:
        logger.exception("An error occurred during file processing")
        raise HTTPException(status_code=500, detail=str(e))

# ... (rest of the code remains the same)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"static/redacted_files/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)