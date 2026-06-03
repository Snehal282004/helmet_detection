from app.services.ai.helmet_detector import detect_helmet

from app.services.ai.plate_ocr import extract_number_plate

from app.services.ai.challan_service import generate_challan

from app.models.ai_detection import (
    HelmetDetection,
    NumberPlateDetection,
    AIViolationEvent
)


def process_violation(
    db,
    image_path,
    session_id
):

    # HELMET DETECTION

    helmet_result = detect_helmet(image_path)

    # OCR

    plate_result = extract_number_plate(image_path)

    # SAVE HELMET DETECTION

    helmet_detection = HelmetDetection(

        session_id=session_id,

        vehicle_number=plate_result["plate_number"],

        rider_count=helmet_result["rider_count"],

        helmet_detected=helmet_result["helmet_detected"],

        violation_detected=helmet_result["violation_detected"],

        confidence_score=helmet_result["confidence_score"],

        cropped_rider_image=image_path
    )

    db.add(helmet_detection)

    # SAVE NUMBER PLATE DETECTION

    plate_detection = NumberPlateDetection(

        session_id=session_id,

        detected_number=plate_result["plate_number"],

        confidence_score=plate_result["confidence_score"],

        plate_image_url=image_path,

        full_image_url=image_path,

        detection_status="detected"
    )

    db.add(plate_detection)

    # CREATE VIOLATION EVENT

    violation_event = AIViolationEvent(

        vehicle_number=plate_result["plate_number"],

        violation_type_id=1,

        detection_session_id=session_id,

        ai_confidence=helmet_result["confidence_score"],

        evidence_image_url=image_path,

        review_status="pending"
    )

    db.add(violation_event)

    db.commit()

    # AUTO GENERATE CHALLAN

    challan = generate_challan(

        db=db,

        vehicle_number=plate_result["plate_number"],

        violation_id=1,

        confidence_score=helmet_result["confidence_score"]
    )

    return {
        "helmet_result": helmet_result,
        "plate_result": plate_result,
        "challan_generated": challan is not None
    }