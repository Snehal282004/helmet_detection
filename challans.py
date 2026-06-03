from fastapi import APIRouter
from sqlalchemy.orm import Session
from fastapi import Depends

from app.dependencies import get_db

router = APIRouter(
    prefix="/challans",
    tags=["Challans"]
)


@router.get("/")
def get_challans(
    db: Session = Depends(get_db)
):

    return {
        "message": "Challans API Working"
    }


@router.post("/generate")
def generate_challan():

    return {
        "message": "Challan Generated Successfully"
    }