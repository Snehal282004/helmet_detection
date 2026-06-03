import easyocr


reader = easyocr.Reader(['en'], gpu=False)

def extract_number_plate(image_path: str):

    result = reader.readtext(image_path)

    detected_text = ""

    confidence = 0.0

    for detection in result:

        text = detection[1]

        score = detection[2]

        detected_text = text

        confidence = score * 100

        break

    return {
        "plate_number": detected_text,
        "confidence_score": round(confidence, 2)
    }