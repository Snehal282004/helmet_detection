from ultralytics import YOLO

# =====================================================
# LOAD MODELS
# =====================================================
bike_model = YOLO("yolov8s.pt")
plate_model = YOLO("models/plate_v11.pt")

print("Plate model loaded successfully")


# =====================================================
# BIKE DETECTION
# =====================================================
def detect_bikes(frame):

    results = bike_model(frame)

    bike_boxes = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            if result.names[class_id] == "motorcycle":

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                bike_boxes.append((x1, y1, x2, y2))

    return bike_boxes


# =====================================================
# PLATE DETECTION
# =====================================================
def detect_plates(frame):

    # LOWER CONFIDENCE
    results = plate_model(frame, conf=0.15)

    plate_boxes = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print("CONF:", conf, "BOX:", (x1, y1, x2, y2))

            # VERY LOW FILTER
            if conf < 0.15:
                continue

            width = x2 - x1
            height = y2 - y1

            # MINIMUM SIZE ONLY
            if width < 15 or height < 10:
                continue

            # KEEP ALL VALID BOXES
            plate_boxes.append((x1, y1, x2, y2))

    print("Detected plates:", len(plate_boxes))

    return plate_boxes 