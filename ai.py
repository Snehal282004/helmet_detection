from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.services.ai.helmet_detector import detect_helmet

router = APIRouter(
    prefix="/ai",
    tags=["AI"]
)


@router.post("/detect")
def detect_violation(file: UploadFile = File(...)):

    # CREATE FOLDER IF NOT EXISTS

    os.makedirs("uploads/violations", exist_ok=True)

    # FILE PATH

    file_path = f"uploads/violations/{file.filename}"

    # SAVE IMAGE

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # CHECK FILE EXISTS

    if not os.path.exists(file_path):

        return {
            "error": "Image not saved properly"
        }

    # CHECK FILE SIZE

    if os.path.getsize(file_path) == 0:

        return {
            "error": "Uploaded image is empty"
        }

    # AI DETECTION

    result = detect_helmet(file_path)

    return result