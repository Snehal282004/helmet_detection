from ultralytics import YOLO
import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

try:
    from number_detector import extract_plate_text
except ImportError:
    def extract_plate_text(frame, box): return None

# =========================================
# LOAD MODELS & DYNAMICALLY INDEX CLASSES
# =========================================
vehicle_model = YOLO("yolov8n.pt")
helmet_model = YOLO("helmet.pt")

HELMET_CLASSES = []
for idx, name in helmet_model.names.items():
    clean_name = name.lower().replace("-", "").replace("_", "").replace(" ", "").strip()
    if clean_name in ["helmet", "withhelmet", "wearinghelmet", "helmeted"]:
        HELMET_CLASSES.append(idx)

if not HELMET_CLASSES:
    HELMET_CLASSES.append(0)

# =========================================
# FILE PICKER
# =========================================
Tk().withdraw()
source = askopenfilename(title="Select Image or Video")

if not source or not os.path.exists(source):
    print("File not found!")
    exit()

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif']
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
ext = os.path.splitext(source)[1].lower()

# Persistent System Registries
counted_motorcycles = set()
plate_cache = {}

# CRITICAL STATE MANAGEMENT SYSTEM
motorcycle_states = {}       # Remembers confirmed state: "none", "helmet", "no_helmet"
flicker_history = {}         # Track frame-by-frame stability: { bike_id: {"state": "xxx", "count": 0} }

total_motorcycles = 0
total_helmet = 0
total_no_helmet = 0

def iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea)

# =========================================
# MAIN PROCESS STREAM
# =========================================
def process_frame(frame, use_tracking=False):
    global total_motorcycles, total_helmet, total_no_helmet
    h, w = frame.shape[:2]

    if w > 1400:
        scale = 1400 / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        h, w = frame.shape[:2]

    # Run native dual-stream inference layers
    if use_tracking:
        bike_results = vehicle_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.30, imgsz=960, verbose=False)[0]
    else:
        bike_results = vehicle_model.predict(frame, conf=0.30, imgsz=960, verbose=False)[0]

    helmet_results = helmet_model.predict(frame, conf=0.15, imgsz=960, verbose=False)[0]

    if bike_results.boxes is None: 
        return frame

    raw_motorcycles = []
    bike_names = vehicle_model.names
    for i, (box, cls) in enumerate(zip(bike_results.boxes.xyxy, bike_results.boxes.cls)):
        tid = bike_results.boxes.id[i].item() if (bike_results.boxes.id is not None) else i
        track_id = int(tid)
        if bike_names[int(cls)].lower() in ["motorcycle", "motorbike"]:
            raw_motorcycles.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), track_id))

    # Vehicle Non-Maximum Suppression Filter
    motorcycles = []
    for b in sorted(raw_motorcycles, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True):
        if not any(iou(b[:4], m[:4]) > 0.40 for m in motorcycles):
            motorcycles.append(b)

    # Collect head targets
    detected_heads = []
    if helmet_results.boxes is not None:
        for box, cls in zip(helmet_results.boxes.xyxy, helmet_results.boxes.cls):
            class_idx = int(cls)
            is_helmet = (class_idx in HELMET_CLASSES)
            detected_heads.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), is_helmet))

    for m in motorcycles:
        mx1, my1, mx2, my2, bike_id = m
        bike_cx = (mx1 + mx2) // 2
        bike_w = mx2 - mx1
        bike_h = my2 - my1

        # Initialize tracking state maps for completely new vehicles
        if bike_id not in counted_motorcycles:
            counted_motorcycles.add(bike_id)
            total_motorcycles += 1
            motorcycle_states[bike_id] = "none"
            flicker_history[bike_id] = {"state": "none", "count": 0}

        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 255, 0), 2)
        cv2.putText(frame, f"Bike {bike_id}", (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        best_head = None
        min_distance = float('inf')

        for h_box in detected_heads:
            hx1, hy1, hx2, hy2, is_helmet = h_box
            hcx = (hx1 + hx2) // 2
            
            inside_x = abs(hcx - bike_cx) < (bike_w * 0.45)
            vertical_valid = (hy2 > my1 - int(bike_h * 0.35)) and (hy1 < my1 + int(bike_h * 0.30))

            if inside_x and vertical_valid:
                spatial_dist = abs(hcx - bike_cx) + abs(hy2 - my1)
                if spatial_dist < min_distance:
                    min_distance = spatial_dist
                    best_head = h_box

        # Evaluate what the model sees in THIS specific frame
        frame_raw_prediction = "none"
        if best_head is not None:
            frame_raw_prediction = "helmet" if best_head[4] else "no_helmet"
        else:
            frame_raw_prediction = "no_helmet"

        # =========================================================
        # THE DYNAMIC RECONCILIATION ENGINE (WITH DE-FLICKER FILTER)
        # =========================================================
        # Step 1: Track if the raw prediction matches our current observation buffer
        if frame_raw_prediction == flicker_history[bike_id]["state"]:
            flicker_history[bike_id]["count"] += 1
        else:
            # If state changed, reset the stabilization buffer threshold counter
            flicker_history[bike_id]["state"] = frame_raw_prediction
            flicker_history[bike_id]["count"] = 1

        # Step 2: Only change the official state if it remains stable for 3 consecutive frames
        if flicker_history[bike_id]["count"] >= 3:
            stable_new_state = flicker_history[bike_id]["state"]
            previous_official_state = motorcycle_states[bike_id]

            # If the confirmed stable state is different from history, apply corrections
            if stable_new_state != previous_official_state:
                
                # Case A: Transitioning out of uninitialized state
                if previous_official_state == "none":
                    if stable_new_state == "helmet": total_helmet += 1
                    elif stable_new_state == "no_helmet": total_no_helmet += 1

                # Case B: DYNAMIC CORRECTION (Switched from No Helmet back to Helmet)
                elif previous_official_state == "no_helmet" and stable_new_state == "helmet":
                    total_no_helmet = max(0, total_no_helmet - 1)  # SUBTRACT FROM NO-HELMET
                    total_helmet += 1                               # ADD TO HELMET

                # Case C: DYNAMIC CORRECTION (Switched from Helmet down to No Helmet)
                elif previous_official_state == "helmet" and stable_new_state == "no_helmet":
                    total_helmet = max(0, total_helmet - 1)        # SUBTRACT FROM HELMET
                    total_no_helmet += 1                            # ADD TO NO-HELMET

                # Lock the newly verified state into history
                motorcycle_states[bike_id] = stable_new_state

        # =========================================================
        # INTERFACE VISUALIZATION RENDERING LAYER
        # =========================================================
        # Use our officially validated state to decide the bounding box colors
        current_locked_state = motorcycle_states[bike_id]
        
        if current_locked_state == "helmet" and best_head is not None:
            hx1, hy1, hx2, hy2, _ = best_head
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
            cv2.putText(frame, "Driver Helmet", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Render a red violation box if state is confirmed as no_helmet, or still calibrating
            if best_head is not None:
                hx1, hy1, hx2, hy2, _ = best_head
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                cv2.putText(frame, "Driver No Helmet", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (mx1, my1), (mx2, my1 + int(bike_h * 0.20)), (0, 0, 255), 2)
                cv2.putText(frame, "Driver No Helmet", (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Trigger Number Plate Recognition only if the vehicle is locked in as a violation
        if current_locked_state == "no_helmet":
            if bike_id not in plate_cache:
                plate_text = extract_plate_text(frame, (mx1, my1, mx2, my2))
                plate_cache[bike_id] = plate_text
            else:
                plate_text = plate_cache[bike_id]

            if plate_text:
                cv2.putText(frame, f"Plate: {plate_text}", (mx1, my2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Core panel display system counters
    cv2.putText(frame, f"Motorcycles: {total_motorcycles}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Helmet: {total_helmet}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"No Helmet: {total_no_helmet}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# =========================================
# RUNTIME SELECTION MANAGERS
# =========================================
if ext in image_extensions:
    frame = cv2.imread(source)
    output = process_frame(frame, use_tracking=False)
    cv2.imshow("Helmet Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif ext in video_extensions:
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret: break
        output = process_frame(frame, use_tracking=True)
        cv2.imshow("Helmet Detection", output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Unsupported format asset input!")

print("\n========== FINAL COUNTS ==========")
print("Total Motorcycles :", total_motorcycles)
print("Total Driver Helmet :", total_helmet)
print("Total Driver No Helmet :", total_no_helmet)
