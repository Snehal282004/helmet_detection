from app.db.connection import get_connection


# =====================================================
# INSERT plate (SAFE FOR IMAGE + VIDEO)
# =====================================================
def insert_plate(plate_number, confidence=0.0, camera_id=None, location=None, media_path=None):

    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        # clean OCR output
        plate_number = plate_number.upper().replace(" ", "").replace("\n", "")

        # =================================================
        # SAFE INSERT (handles optional fields)
        # =================================================
        cur.execute("""
            INSERT INTO bike_plates 
            (plate_number, confidence, camera_id, location, media_path)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            plate_number,
            confidence,
            camera_id,
            location,
            media_path
            
        ))

        conn.commit()

    except Exception as e:
        print("DB Insert Error:", e)

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


# =====================================================
# FETCH all plates (FIXED for NEW DB STRUCTURE)
# =====================================================
def get_all_plates():

    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT 
                plate_number, 
                confidence, 
                camera_id, 
                location, 
                media_path, 
                timestamp
            FROM bike_plates
            ORDER BY id DESC
        """)

        return cur.fetchall()

    except Exception as e:
        print("DB Fetch Error:", e)
        return []

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()