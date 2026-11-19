from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("../models/helmet_detection_best.pt")

# Image path
image_path = "../testing/test.jpg"

# Run prediction
results = model(image_path)

# Display result
annotated = results[0].plot()

cv2.imshow("Helmet Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()