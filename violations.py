from fastapi import APIRouter
from sqlalchemy.orm import Session
from fastapi import Depends

from app.dependencies import get_db

router = APIRouter(
    prefix="/violations",
    tags=["Violations"]
)


@router.get("/")
def get_violations(
    db: Session = Depends(get_db)
):

    return {
        "message": "Violations API Working"
    }


@router.post("/report")
def report_violation():

    return {
        "message": "Violation Reported"
    }