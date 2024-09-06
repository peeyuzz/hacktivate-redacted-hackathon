# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pathlib import Path
from dotenv import load_dotenv
import shutil
import os
import logging


# Import the auth routes
from app.routers.auth_routes import router as auth_router
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# MongoDB connection setup
MONGODB_URL = os.environ["MONGODB_URL"]
client = AsyncIOMotorClient(MONGODB_URL)
db = client["file_management"]
files_collection = db["files"]

# JWT Config
SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = os.environ["ALGORITHM"]

# FastAPI security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

if os.path.exists("static"):
    shutil.rmtree("static")

os.makedirs(os.path.join("static", "redacted_files"))
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
Path("static/redacted_files").mkdir(parents=True, exist_ok=True)

# Include the authentication router
app.include_router(auth_router)

# Middleware to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user = await db["users"].find_one({"username": username})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    try:
        file_path = f"static/uploaded_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.debug(f"File saved to {file_path}")
        
        # Perform processing on file
        redacted_path = f"static/redacted_files/redacted_{file.filename}"
        shutil.copyfile(file_path, redacted_path)
        
        if not os.path.exists(redacted_path):
            raise ValueError(f"Processing failed or file not found: {redacted_path}")
        
        logger.debug(f"File processed, new path: {redacted_path}")
        
        # Store file metadata in MongoDB
        file_data = {
            "user_id": str(current_user["_id"]),
            "original_filename": file.filename,
            "stored_filename": f"redacted_{file.filename}",
            "file_type": file.content_type,
            "created_at": datetime.utcnow()
        }
        await files_collection.insert_one(file_data)
        
        # Clean up original uploaded file
        os.remove(file_path)
        
        return {"filename": file_data["stored_filename"]}
    except Exception as e:
        logger.exception("An error occurred during file processing")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"static/redacted_files/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/user/files")
async def get_user_files(current_user: dict = Depends(get_current_user)):
    user_files = await files_collection.find({"user_id": str(current_user["_id"])}).to_list(100)
    return [{"filename": file["stored_filename"], "type": file["file_type"], "created_at": file["created_at"]} for file in user_files]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
