import cv2
import math
import cvzone
from ultralytics import YOLO

model = YOLO("Weights/best.pt")  
cap = cv2.VideoCapture("Media/sample.mp4")

classNames = ['With Helmet', 'Without Helmet']

frame_skip = 4
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    img = cv2.resize(img, (416, 320))

    results = model(img, stream=False, imgsz=416)

    # 🔥 COUNT VARIABLES
    helmet_count = 0
    no_helmet_count = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # 🔥 COUNT LOGIC
            if cls == 0:
                helmet_count += 1
            elif cls == 1:
                no_helmet_count += 1

            # Draw
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(
                img,
                f'{classNames[cls]} {conf:.2f}',
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1
            )

    # 🔥 SHOW COUNT ON SCREEN
    cv2.putText(img, f'Helmet: {helmet_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(img, f'No Helmet: {no_helmet_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # 🔥 PRINT IN TERMINAL
    print(f"Helmet: {helmet_count} | No Helmet: {no_helmet_count}")

    cv2.imshow("Helmet Detection FAST", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()