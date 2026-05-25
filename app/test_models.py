from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("models/plate_v11.pt")

# Print class names
print("Classes:", model.names)

# Test image path
image_path = "input/images/img5.jpeg"


# Run prediction on image
results = model(image_path)

# First result
result = results[0]

# Draw detections
annotated = result.plot()

# Save output
cv2.imwrite("test_output.jpg", annotated)

print("Saved result as test_output.jpg")


