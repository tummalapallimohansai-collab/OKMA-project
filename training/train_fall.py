from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data="../datasets/fall_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="fall_detection",
        workers=0
    )

if __name__ == "__main__":
    main()