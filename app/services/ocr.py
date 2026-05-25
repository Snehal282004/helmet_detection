from paddleocr import PaddleOCR
import cv2
import re

ocr = PaddleOCR(use_angle_cls=True, lang='en')


def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def read_plate(img):

    if img is None:
        return ""

    try:
        img = cv2.resize(img, (320, 80))

        result = ocr.ocr(img)

        if not result or not result[0]:
            return ""

        text = ""

        for line in result[0]:

            # SAFE CHECK (IMPORTANT FIX)
            if line is None or len(line) < 2:
                continue

            if line[1] is None:
                continue

            detected_text = line[1][0]
            confidence = line[1][1]

            if confidence > 0.5:
                text += detected_text

        return clean_plate(text)

    except Exception as e:
        print("OCR Error:", e)
        return ""







