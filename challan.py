from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Numeric,
    ForeignKey
)

from app.database import Base


class Challan(Base):

    __tablename__ = "challans"

    id = Column(Integer, primary_key=True, index=True)

    vehicle_id = Column(Integer, ForeignKey("vehicles.id"))

    violation_id = Column(Integer, ForeignKey("violations.id"))

    amount = Column(Integer)

    challan_status = Column(String, default="pending")

    ai_detected = Column(Boolean, default=False)

    confidence_score = Column(Numeric(5,2))