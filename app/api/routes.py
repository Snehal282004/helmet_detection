from fastapi import APIRouter, UploadFile, File
import shutil
import os
import cv2

from app.services.processor import process_frame
from app.db.queries import get_all_plates

router = APIRouter()

UPLOAD_DIR = "input/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================
#  POST API → Upload Image / Video & Detect Plates
# =====================================================
@router.post("/detect")
async def detect_vehicle(file: UploadFile = File(...)):

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    plates_found = []

    # =========================
    # IMAGE PROCESSING
    # =========================
    if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):

        frame = cv2.imread(file_path)

        if frame is None:
            return {"error": "Invalid image"}

     
        plates_found, plate_boxes = process_frame(frame, file_path)

        return {
            "type": "image",
            "plates": plates_found,
            "boxes": plate_boxes,
            "count": len(plates_found)
        }

    # =========================
    # VIDEO PROCESSING
    # =========================
    elif file.filename.lower().endswith((".mp4", ".avi", ".mov")):

        cap = cv2.VideoCapture(file_path)

        all_plates = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # FIX: tuple unpacking
            plates, _ = process_frame(frame)

            all_plates.extend(plates)

        cap.release()

        # remove duplicates
        unique_plates = list(set(all_plates))

        return {
            "type": "video",
            "plates": unique_plates,
            "count": len(unique_plates)
        }

    else:
        return {"error": "Unsupported file format"}


# =====================================================
#  GET API → Fetch All Stored Plates from DB
# =====================================================
@router.get("/plates")
def get_plates():

    data = get_all_plates()

    if not data:
        return {"plates": []}

    # FIXED according to NEW DB structure
    return {
        "plates": [
            {
                "plate_number": row[0],
                "confidence": row[1],
                "camera_id": row[2],
                "location": row[3],
                "media_path": row[4],
                "timestamp": row[5]
            }
            for row in data
        ]
    }