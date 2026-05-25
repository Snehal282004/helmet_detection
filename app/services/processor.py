import re
import cv2
from paddleocr import PaddleOCR

from app.services.detection import detect_plates
from app.db.queries import insert_plate

# =========================
# INIT OCR (load once only)
# =========================
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


# =========================
# CLEAN OCR TEXT
# =========================
def clean_plate(text):
    text = text.upper()
    text = text.replace(" ", "")
    text = text.replace(".", "")
    text = text.replace("-", "")
    return text


# =========================
# VALIDATE NUMBER PLATE
# =========================
def is_valid_plate(text):
    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$"
    return bool(re.match(pattern, text))


# =========================
# MAIN PROCESS FUNCTION
# =========================
def process_frame(frame, media_path=None):

    plates = []

    # detect plate boxes from YOLO
    plate_boxes = detect_plates(frame)

    h, w = frame.shape[:2]

    print("\n===== ALL DETECTED PLATES =====")

    for idx, box in enumerate(plate_boxes):

        x1, y1, x2, y2 = box

        # safe boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # crop image
        crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        # OCR
        result = ocr.ocr(crop, cls=True)

        if not result or len(result) == 0 or result[0] is None:
            continue

        for line in result[0]:

            text = line[1][0]
            conf = float(line[1][1])

            text = clean_plate(text)

            if conf < 0.5:
                continue

            if len(text) < 6:
                continue

            print(f"Vehicle {idx+1}: {text} ({conf:.2f})")

            plates.append(text)

            # =========================
            # INSERT INTO DATABASE
            # =========================
            insert_plate(
                plate_number=text,
                confidence=conf,
                camera_id=None,
                location=None,
                media_path=media_path
            )

    # remove duplicates only for API response
    return plates , plate_boxes
   