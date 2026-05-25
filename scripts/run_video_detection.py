import cv2
import os

from app.services.detection import detect_bikes, detect_plates
from app.services.ocr import read_plate

# =====================================================
# INPUT VIDEO PATH
# =====================================================
VIDEO_PATH = r"D:\PLATE_DETECTION\input\videos\sample.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# =====================================================
# OUTPUT VIDEO
# =====================================================
OUTPUT_DIR = "output/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "output_sample.mp4"),
    fourcc,
    fps,
    (w, h)
)

frame_count = 0

# =====================================================
# PROCESS VIDEO FRAME BY FRAME
# =====================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # OPTIONAL: skip frames for speed
    if frame_count % 2 != 0:
        continue

    # =====================================================
    # STEP 1: DETECT BIKES
    # =====================================================
    bike_boxes = detect_bikes(frame)

    for (x1, y1, x2, y2) in bike_boxes:

        bike_crop = frame[y1:y2, x1:x2]

        if bike_crop.size == 0:
            continue

        # =================================================
        # STEP 2: DETECT PLATES
        # =================================================
        plate_boxes = detect_plates(bike_crop)

        for (px1, py1, px2, py2) in plate_boxes:

            fx1 = px1 + x1
            fy1 = py1 + y1
            fx2 = px2 + x1
            fy2 = py2 + y1

            plate_img = frame[fy1:fy2, fx1:fx2]

            if plate_img.size == 0:
                continue

            # =================================================
            # STEP 3: OCR
            # =================================================
            text = read_plate(plate_img)

            if text:

                print(f"Frame {frame_count} → Plate: {text}")

                # DRAW BOX
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                cv2.putText(frame, text, (fx1, fy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

    # =====================================================
    # WRITE FRAME
    # =====================================================
    out.write(frame)

# =====================================================
# CLEANUP
# =====================================================
cap.release()
out.release()

print("Video processing completed. Saved to output/results/output_sample.mp4")