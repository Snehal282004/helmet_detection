from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Numeric,
    DateTime
)

from sqlalchemy.sql import func

from app.database import Base


class AIModel(Base):

    __tablename__ = "ai_models"

    id = Column(Integer, primary_key=True)

    model_name = Column(String)

    model_version = Column(String)

    model_type = Column(String)

    # helmet
    # ocr
    # mobile_detection

    accuracy = Column(Numeric(5,2))

    is_active = Column(Boolean, default=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )