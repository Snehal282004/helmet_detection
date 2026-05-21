from ultralytics import YOLO
import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from number_detector import extract_plate_text

# =========================================
# LOAD MODELS
# =========================================

vehicle_model = YOLO("yolov8n.pt")
helmet_model = YOLO("helmet.pt")

# =========================================
# FILE PICKER
# =========================================

Tk().withdraw()

source = askopenfilename(
    title="Select Image or Video"
)

if not source or not os.path.exists(source):
    print("File not found!")
    exit()

# =========================================
# FILE TYPES
# =========================================

image_extensions = [
    '.jpg', '.jpeg', '.png',
    '.bmp', '.webp', '.avif'
]

video_extensions = [
    '.mp4', '.avi', '.mov', '.mkv'
]

ext = os.path.splitext(source)[1].lower()

# =========================================
# GLOBAL COUNTS
# =========================================

counted_motorcycles = set()
counted_helmet = set()
counted_no_helmet = set()


total_motorcycles = 0
total_helmet = 0
total_no_helmet = 0
plate_cache = {}
# =========================================
# IOU FUNCTION
# =========================================

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea

    if union == 0:
        return 0

    return interArea / union

# =========================================
# REMOVE DUPLICATES
# =========================================

def remove_duplicates(boxes, threshold=0.6):

    filtered = []

    for box in boxes:

        duplicate = False

        for f in filtered:

            if iou(box, f) > threshold:
                duplicate = True
                break

        if not duplicate:
            filtered.append(box)

    return filtered

# =========================================
# DRIVER CHECK
# =========================================

def is_driver(person_box, motorcycle_box):

    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = motorcycle_box

    # Person center
    pcx = (px1 + px2) // 2
    pcy = (py1 + py2) // 2

    # Driver should overlap bike center
    bike_center_x = (mx1 + mx2) // 2

    # Better driver logic
    inside_x = abs(pcx - bike_center_x) < ((mx2 - mx1) * 0.6)

    inside_y = (
        py2 > my1 and
        py1 < my2
    )

    return inside_x and inside_y

# =========================================
# HELMET DETECTION
# =========================================

def detect_helmet(head_crop):

    if head_crop.size == 0:
        return False

    # Improve small object visibility
    head_crop = cv2.resize(
        head_crop,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Slight sharpening
    kernel = [
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ]

    kernel = cv2.UMat(
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )
    )

    results = helmet_model.predict(
        head_crop,
        conf=0.10,
        imgsz=640,
        verbose=False
    )[0]

    if results.boxes is None:
        return False

    if len(results.boxes) == 0:
        return False

    for cls, conf in zip(
        results.boxes.cls,
        results.boxes.conf
    ):

        label = helmet_model.names[int(cls)].lower()

        if "helmet" in label and float(conf) > 0.10:
            return True

    return False

# =========================================
# MAIN PROCESS
# =========================================

def process_frame(frame, use_tracking=False):

    global total_motorcycles
    global total_helmet
    global total_no_helmet

    h, w = frame.shape[:2]

    # Resize huge frames
    if w > 1400:

        scale = 1400 / w

        frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale))
        )

    # =========================================
    # DETECTION
    # =========================================

    results = vehicle_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        imgsz=960,
        verbose=False
    )[0]

    if results.boxes is None:
        return frame

    names = vehicle_model.names

    motorcycles = []
    persons = []

    # =========================================
    # EXTRACT OBJECTS
    # =========================================

    for box, cls, tid in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.id
    ):

        if tid is None:
            continue

        track_id = int(tid)

        label = names[int(cls)].lower()

        x1, y1, x2, y2 = map(int, box)

        if label in ["motorcycle", "motorbike"]:

            motorcycles.append(
                (x1, y1, x2, y2, track_id)
            )

        elif label == "person":

            persons.append(
                (x1, y1, x2, y2, track_id)
            )

    # =========================================
    # REMOVE DUPLICATES
    # =========================================

    if not use_tracking:

        clean = remove_duplicates([
            (p[0], p[1], p[2], p[3])
            for p in persons
        ])

        persons_clean = []

        for c in clean:
            persons_clean.append(c)

    else:
        persons_clean = persons

    # =========================================
    # DRAW MOTORCYCLES
    # =========================================

    for m in motorcycles:

        mx1, my1, mx2, my2, bike_id = m

        if bike_id not in counted_motorcycles:

            counted_motorcycles.add(bike_id)
            total_motorcycles += 1

        cv2.rectangle(
            frame,
            (mx1, my1),
            (mx2, my2),
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Bike {bike_id}",
            (mx1, my1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    # =========================================
    # DRIVER HEAD + HELMET
    # =========================================

    for m in motorcycles:

        mx1, my1, mx2, my2, bike_id = m

        best_person = None
        best_score = 0

        # =========================================
        # FIND DRIVER
        # =========================================

        for p in persons_clean:

            if len(p) == 4:

                px1, py1, px2, py2 = p
                person_id = f"{px1}-{py1}"

            else:

                px1, py1, px2, py2, person_id = p

            if is_driver(
                (px1, py1, px2, py2),
                (mx1, my1, mx2, my2)
            ):

                # Choose closest person to bike center
                person_center_x = (px1 + px2) // 2
                bike_center_x = (mx1 + mx2) // 2

                distance = abs(person_center_x - bike_center_x)

                score = 1000 - distance

                if score > best_score:

                    best_score = score

                    best_person = (
                        px1, py1,
                        px2, py2,
                        person_id
                    )

        if best_person is None:
            continue

        px1, py1, px2, py2, track_id = best_person

        # =========================================
        # ONLY DRIVER HEAD REGION
        # =========================================

        person_width = px2 - px1
        person_height = py2 - py1

        # Head region only
        hx1 = px1 + int(person_width * 0.20)
        hx2 = px2 - int(person_width * 0.20)

        hy1 = py1
        hy2 = py1 + int(person_height * 0.30)

        # Safety bounds
        hx1 = max(0, hx1)
        hy1 = max(0, hy1)

        hx2 = min(frame.shape[1], hx2)
        hy2 = min(frame.shape[0], hy2)

        head_crop = frame[
            hy1:hy2,
            hx1:hx2
        ]

        if head_crop.size == 0:
            continue

        # =========================================
        # HELMET CHECK
        # =========================================

        has_helmet = detect_helmet(head_crop)

        if has_helmet:

            if track_id not in counted_helmet:

                counted_helmet.add(track_id)
                total_helmet += 1

            color = (0, 255, 0)
            text = "Driver Helmet"

        else:

            if track_id not in counted_no_helmet:

                counted_no_helmet.add(track_id)
                total_no_helmet += 1

            color = (0, 0, 255)
            text = "Driver No Helmet"

            # =========================================
            # NUMBER PLATE (ONLY NO HELMET)
            # =========================================

            if track_id not in plate_cache:

                plate_text = extract_plate_text(frame, (mx1, my1, mx2, my2))
                plate_cache[track_id] = plate_text

            else:
                plate_text = plate_cache[track_id]

            if plate_text:
                cv2.putText(
                    frame,
                    f"Plate: {plate_text}",
                    (mx1, my2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


        # =========================================
        # DRAW ONLY HEAD BOX
        # =========================================

        cv2.rectangle(
            frame,
            (hx1, hy1),
            (hx2, hy2),
            color,
            3
        )

        cv2.putText(
            frame,
            text,
            (hx1, hy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # =========================================
    # LIVE COUNTS
    # =========================================

    cv2.putText(
        frame,
        f"Motorcycles: {total_motorcycles}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Helmet: {total_helmet}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"No Helmet: {total_no_helmet}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return frame

# =========================================
# IMAGE MODE
# =========================================

if ext in image_extensions:

    frame = cv2.imread(source)

    output = process_frame(
        frame,
        use_tracking=False
    )

    cv2.imshow(
        "Helmet Detection",
        output
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    print("\n========== FINAL COUNTS ==========")

    print("Total Motorcycles :", total_motorcycles)
    print("Total Helmet      :", total_helmet)
    print("Total No Helmet   :", total_no_helmet)

# =========================================
# VIDEO MODE
# =========================================

elif ext in video_extensions:

    cap = cv2.VideoCapture(source)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        output = process_frame(
            frame,
            use_tracking=True
        )

        cv2.imshow(
            "Helmet Detection",
            output
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

    print("\n========== FINAL COUNTS ==========")

    print("Total Motorcycles :", total_motorcycles)
    print("Total Helmet      :", total_helmet)
    print("Total No Helmet   :", total_no_helmet)

else:
    print("Unsupported file format!")