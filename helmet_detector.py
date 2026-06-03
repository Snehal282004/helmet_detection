from ultralytics import YOLO
import easyocr
import uuid

model = YOLO("app/ml/best.pt")

model.to("cpu")

reader = easyocr.Reader(['en'])


def detect_helmet(image_path: str):

    results = model(image_path)

    total_riders = 0
    helmet_wearing_count = 0
    without_helmet_count = 0
    confidence_score = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0]) * 100

            confidence_score = round(conf, 2)

            # YOUR MODEL CLASS IDS

            # 0 = helmet
            # 1 = no_helmet

            if cls == 0:

                helmet_wearing_count += 1
                total_riders += 1

            elif cls == 1:

                without_helmet_count += 1
                total_riders += 1

    violation_detected = without_helmet_count > 0

    violation_type = "NO_HELMET" if violation_detected else "NONE"

    # OCR DETECTION

    ocr_result = reader.readtext(image_path)

    vehicle_number = "NOT_DETECTED"

    if len(ocr_result) > 0:

        vehicle_number = ocr_result[0][1]

    # DYNAMIC FINE CALCULATION

    fine_amount = 0

    if violation_type == "NO_HELMET":

        fine_amount = 500

    # AUTO CHALLAN

    challan_generated = violation_detected

    challan_number = f"CHL-{uuid.uuid4().hex[:10]}"

    return {

        "vehicle_number": vehicle_number,

        "total_riders": total_riders,

        "helmet_wearing_count": helmet_wearing_count,

        "without_helmet_count": without_helmet_count,

        "violation_detected": violation_detected,

        "violation_type": violation_type,

        "fine_amount": fine_amount,

        "challan_generated": challan_generated,

        "challan_number": challan_number,

        "confidence_score": confidence_score,

        "evidence_image": image_path,

        "review_status": "PENDING"
    }