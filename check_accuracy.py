from ultralytics import YOLO

model = YOLO("Weights/best.pt")

metrics = model.val(
    data="C:/Users/Shree/Desktop/Helmet-Detection-project/Datasets/data.yaml"
)

print(metrics)