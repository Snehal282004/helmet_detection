from pydantic import BaseModel
from typing import Optional


class UserCreate(BaseModel):

    name: str

    email: str

    password: str


class UserResponse(BaseModel):

    id: int

    name: str

    email: str

    profile_image: Optional[str]

    class Config:
        orm_mode = True