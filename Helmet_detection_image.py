import cv2
from ultralytics import YOLO

# Load model
model = YOLO("Weights/best.pt")

# 🔥 GIVE CORRECT IMAGE PATH
image_path = r"C:\Users\Shree\Desktop\Helmet-Detection-project\Media\riders_10.jpg"

# Read image
img = cv2.imread(image_path)

#  CHECK (VERY IMPORTANT)
if img is None:
    print("❌ Error: Image not found. Check path!")
    exit()

# Run detection
results = model(img)

# Get output image with boxes
annotated_img = results[0].plot()

# Show result
cv2.imshow("Result", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()