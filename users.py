from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.dependencies import get_db

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)


@router.get("/")
def get_users(
    db: Session = Depends(get_db)
):

    return {
        "message": "Users API Working"
    }


@router.post("/register")
def register_user():

    return {
        "message": "User Registered Successfully"
    }