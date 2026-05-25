import cv2
import re
from paddleocr import PaddleOCR
import os
os.environ["FLAGS_logtostderr"] = "0"
os.environ["GLOG_minloglevel"] = "3"

# =========================
# INIT OCR MODEL
# =========================
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
# =========================
# IMAGE PATH
# =========================
image_path = "input/images/img10.jpg"

print("Using image:", image_path)
print("Exists:", os.path.exists(image_path))

# =========================
# READ IMAGE
# =========================
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not loaded. Check path!")
    exit()

# =========================
# RUN OCR
# =========================
result = ocr.ocr(image, cls=True)

# =========================
# CLEAN OUTPUT
# =========================
print("\n===== DETECTED TEXT =====")

plates = []

for line in result:
    for word in line:
        text = word[1][0]
        confidence = word[1][1]

        print(f"{text} ({confidence:.2f})")

        # Optional: filter possible number plates
        if re.search(r'[A-Z]{2}\s?\d{1,2}', text):
            plates.append(text)

# =========================
# FINAL RESULT (ONLY PLATE)
# =========================
print("\n===== FINAL PLATE OUTPUT =====")

if plates:
    # pick best match (first one)
    print("Detected Plate:", plates[0])
else:
    print("No valid number plate detected")