# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import shutil
import os
import logging
from app.redactor import Redactor
from pathlib import Path
from uuid import uuid4
from fastapi.responses import FileResponse
# Import the auth routes
from app.routers.auth_routes import router as auth_router
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from app.models.user_models import FileResponseModel
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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


@app.post("/upload", response_model=FileResponseModel)
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    try:
        # Generate a unique identifier for the uploaded file
        unique_id = uuid4().hex
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"uploaded_{unique_id}{file_extension}"
        file_path = f"static/{unique_filename}"

        # Save the uploaded file to a temporary path
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.debug(f"File saved to {file_path}")

        # Perform redaction
        redactor = Redactor(file_path, plan_type="pro")
        redacted_path = redactor.redact()
        logger.debug(f"Redacted path: {redacted_path}")

        if redacted_path is None or not os.path.exists(redacted_path):
            raise ValueError(
                f"Redaction failed or file not found: {redacted_path}")

        # Generate a unique name for the redacted file
        redacted_filename = f"redacted_{unique_id}{file_extension}"
        new_path = f"static/redacted_files/{redacted_filename}"
        shutil.move(redacted_path, new_path)
        logger.debug(f"Redacted file moved to {new_path}")

        # Store file metadata in MongoDB
        file_data = {
            "user_id": str(current_user["_id"]),
            "original_filename": file.filename,
            "stored_filename": redacted_filename,
            "file_type": file.content_type,
            "created_at": datetime.utcnow()
        }
        await files_collection.insert_one(file_data)

        # Return the file metadata for downloading
        return FileResponseModel(
            filename=redacted_filename,
            type=file.content_type,
            created_at=datetime.utcnow()
        )

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
