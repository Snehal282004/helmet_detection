from app.models.vehicle import Vehicle
from app.models.challan import Challan


def generate_challan(
    db,
    vehicle_number,
    violation_id,
    confidence_score
):

    vehicle = db.query(Vehicle).filter(
        Vehicle.vehicle_number == vehicle_number
    ).first()

    if not vehicle:

        return None

    challan = Challan(
        vehicle_id=vehicle.id,
        violation_id=violation_id,
        amount=500,
        challan_status="pending",
        ai_detected=True,
        confidence_score=confidence_score
    )

    db.add(challan)

    db.commit()

    db.refresh(challan)

    return challan