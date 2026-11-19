from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8s.pt")

    # Train the model
    model.train(
        data=r"C:\Users\Mohan Sai\OneDrive\Desktop\OKMA\datasets\helmet_dataset\data.yaml",
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