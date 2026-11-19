
from pathlib import Path
from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 Small model
    model = YOLO("yolov8s.pt")

    # Get project root directory
    root = Path(__file__).resolve().parents[1]

    # Dataset configuration path
    data_yaml = root / "datasets" / "helmet_dataset" / "data.yaml"

    # Train the model
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="../models",
        name="helmet_detection"
    )

    print("Training Completed Successfully!")

if __name__ == "__main__":
    main()
