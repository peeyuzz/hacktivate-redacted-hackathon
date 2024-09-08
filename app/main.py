import os
import shutil
import logging
from uuid import uuid4
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

from jose import JWTError, jwt
from dotenv import load_dotenv

import cloudinary
import cloudinary.uploader
import cloudinary.api

from PIL import Image
import fitz  # PyMuPDF
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from redactor import Redactor
from routers.auth_routes import router as auth_router
from models.user_models import FileResponseModel

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
# Cloud variables
CLOUD_NAME = os.environ['CLOUD_NAME']
CLOUD_API_KEY = os.environ['CLOUD_API_KEY']
CLOUD_API_SECRET = os.environ['CLOUD_API_SECRET']

cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=CLOUD_API_KEY,
    api_secret=CLOUD_API_SECRET
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


def compress_image(input_path: str, output_path: str):
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img.save(output_path, "JPEG", optimize=True, quality=75)


def compress_pdf(input_path: str, output_path: str):
    doc = fitz.open(input_path)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()


def compress_video(input_path: str, output_path: str):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libx264", bitrate="500k")


def compress_audio(input_path: str, output_path: str):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="mp3", bitrate="128k")


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

        # logger.debug(f"File saved to {file_path}")

        # Perform redaction
        redactor = Redactor(file_path, plan_type="pro")
        redacted_path = redactor.redact()
        # logger.debug(f"Redacted path: {redacted_path}")

        if redacted_path is None or not os.path.exists(redacted_path):
            raise ValueError(
                f"Redaction failed or file not found: {redacted_path}")

        # Delete the original uploaded file after redaction
        if os.path.exists(file_path):
            os.remove(file_path)
            # logger.debug(f"Original file deleted: {file_path}")

        # Generate a unique name for the redacted and compressed file
        compressed_filename = f"{unique_id}_compressed{file_extension}"
        compressed_path = f"static/redacted_files/{compressed_filename}"

        # Compress the redacted file based on its type
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            compress_image(redacted_path, compressed_path)
        elif file_extension.lower() == '.pdf':
            compress_pdf(redacted_path, compressed_path)
        elif file_extension.lower() in ['.mp4', '.mov', '.avi']:
            compress_video(redacted_path, compressed_path)
        elif file_extension.lower() in ['.mp3', '.wav', '.ogg']:
            compress_audio(redacted_path, compressed_path)
        else:
            # If file type is unsupported, simply move the redacted file
            shutil.move(redacted_path, compressed_path)

        # logger.debug(f"Compressed file moved to {compressed_path}")

        # Delete the original redacted file after compression
        if os.path.exists(redacted_path):
            os.remove(redacted_path)
            # logger.debug(f"Redacted file deleted: {redacted_path}")

        # Determine the resource type for Cloudinary upload
        resource_type = 'raw'
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            resource_type = 'image'
        elif file_extension.lower() in ['.mp4', '.mov', '.avi']:
            resource_type = 'video'

        # Upload the compressed file to Cloudinary
        cloudinary_response = cloudinary.uploader.upload(
            compressed_path,
            resource_type=resource_type,
            access_mode='public',
            public_id=f"{compressed_filename}" 
        )
        # logger.debug(f"Compressed file uploaded to Cloudinary: {
                    #  cloudinary_response['secure_url']}")

        # Delete the local compressed file after uploading
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
            # logger.debug(f"Compressed file deleted from local storage: {
                        #  compressed_path}")

        # Store file metadata in MongoDB
        file_data = {
            "user_id": str(current_user["_id"]),
            "original_filename": file.filename,  # The original file name uploaded
            "stored_filename": compressed_filename,  # The file name after compression
            "file_type": file.content_type,
            # URL of the uploaded file on Cloudinary
            "cloudinary_url": cloudinary_response['secure_url'],
            "created_at": datetime.utcnow()
        }
        await files_collection.insert_one(file_data)

        # Return the file metadata for downloading
        return FileResponseModel(
            filename=compressed_filename,  # Send back the compressed file name
            type=file.content_type,
            original_filename=file.filename,
            cloudinary_url=cloudinary_response['secure_url'],  # Cloudinary URL
            created_at=datetime.utcnow()
        )

    except Exception as e:
        logger.exception("An error occurred during file processing")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        # Generate the URL for the file from Cloudinary
        file_url = cloudinary.CloudinaryImage(filename).build_url()
        
        # Use RedirectResponse to redirect to the Cloudinary URL
        return RedirectResponse(url=file_url)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail="File not found on Cloudinary")

@app.get("/user/files")
async def get_user_files(current_user: dict = Depends(get_current_user)):
    user_files = await files_collection.find({"user_id": str(current_user["_id"])}).to_list(100)
    return [{
        "original_filename": file["original_filename"],
        "stored_filename": file["stored_filename"],
        "type": file["file_type"],
        "cloudinary_url": file["cloudinary_url"],
        "created_at": file["created_at"],
    } for file in user_files]
       
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
