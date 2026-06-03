from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Numeric,
    ForeignKey,
    DateTime
)

from sqlalchemy.sql import func

from app.database import Base


class AIDetectionSession(Base):

    __tablename__ = "ai_detection_sessions"

    id = Column(Integer, primary_key=True)

    uploaded_by = Column(Integer, ForeignKey("users.id"))

    media_upload_id = Column(Integer)

    detection_type = Column(String)

    processing_status = Column(String, default="pending")

    processed_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True),
                        server_default=func.now())
    
class NumberPlateDetection(Base):

    __tablename__ = "number_plate_detections"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        Integer,
        ForeignKey("ai_detection_sessions.id")
    )

    detected_number = Column(String, index=True)

    confidence_score = Column(Numeric(5,2))

    plate_image_url = Column(String)

    full_image_url = Column(String)

    detection_status = Column(String)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

class HelmetDetection(Base):

    __tablename__ = "helmet_detections"

    id = Column(Integer, primary_key=True)

    session_id = Column(
        Integer,
        ForeignKey("ai_detection_sessions.id")
    )

    vehicle_number = Column(String)

    rider_count = Column(Integer)

    helmet_detected = Column(Boolean)

    violation_detected = Column(Boolean)

    confidence_score = Column(Numeric(5,2))

    cropped_rider_image = Column(String)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

class AIViolationEvent(Base):

    __tablename__ = "ai_violation_events"

    id = Column(Integer, primary_key=True)

    vehicle_number = Column(String, index=True)

    violation_type_id = Column(Integer)

    detection_session_id = Column(
        Integer,
        ForeignKey("ai_detection_sessions.id")
    )

    detected_by_ai = Column(Boolean, default=True)

    ai_confidence = Column(Numeric(5,2))

    evidence_image_url = Column(String)

    evidence_video_url = Column(String)

    review_status = Column(
        String,
        default="pending"
    )

    reviewed_by = Column(Integer)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )