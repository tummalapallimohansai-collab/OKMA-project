from ultralytics import YOLO

# Load trained model
model = YOLO("../training/runs/detect/fall_detection-5/weights/best.pt")

# Test image
results = model("../test_images/fall.jpg", show=True)

results[0].save("result.jpg")