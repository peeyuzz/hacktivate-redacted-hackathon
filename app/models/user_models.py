from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


# User model
class User(BaseModel):
    email: EmailStr
    username: str
    password: str

# Token model
class Token(BaseModel):
    access_token: str
    token_type: str

# Utility class for token data
class TokenData(BaseModel):
    username: Optional[str] = None

# Response model for file metadata
class FileResponseModel(BaseModel):
    filename: str
    type: str
    created_at: datetime