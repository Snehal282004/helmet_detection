import cv2
import os
from app.services.detection import detect_bikes, detect_plates
from app.services.processor import process_frame


os.environ["FLAGS_logtostderr"] = "0"
os.environ["GLOG_minloglevel"] = "3"

OUTPUT_DIR = "output/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load image
frame = cv2.imread(r"D:\PLATE_DETECTION\input\images\img10.jpg")

if frame is None:
    print("Error: Image not found or wrong path")
    exit()


# =========================
# OCR RESULT (FINAL TEXT)
# =========================

plates, plate_boxes = process_frame(frame)

print("\n===== ALL DETECTED PLATES =====")
for idx, plate in enumerate(plates):
    print(f"Vehicle {idx+1}: {plate}")

# =========================
# DETECT BIKES (BOXES)
# =========================
bike_boxes = detect_bikes(frame)

for (x1, y1, x2, y2) in bike_boxes:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Bike", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# =========================
# DETECT PLATES (BOXES)
# =========================

for (x1, y1, x2, y2) in plate_boxes:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, "Plate", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# =========================
# SHOW OUTPUT
# =========================
output_path = os.path.join(OUTPUT_DIR, "output_result.jpg")

cv2.imwrite(output_path, frame)

print(f"Saved result at: {output_path}")
