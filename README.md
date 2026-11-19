# OKMA - Helmet and Fall Detection System

## Project Overview
OKMA is a computer vision-based safety monitoring project designed to detect helmet usage and fall incidents using deep learning.

The system uses YOLOv8 models to identify safety-related events and supports real-time monitoring through a camera.

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- NumPy

## Project Structure
```text
OKMA/
  app.py
  requirements.txt
  training/
    train.py
    results/
      helmet_detection-7/
        args.yaml
        results.csv
  testing/
  utils/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tummalapallimohansai-collab/OKMA-project.git
cd OKMA-project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Training

The helmet detection model uses pretrained YOLOv8 Small (`yolov8s.pt`).

Training configuration:
- Epochs: 50
- Image size: 640 x 640
- Batch size: 16
- Device: GPU 0

Run training:
```bash
python training/train.py
```

**Dataset requirement:** Place the helmet dataset under `datasets/helmet_dataset/` and provide its `data.yaml` configuration file before training. The dataset is not included in this repository.

## Training Results

The recorded results from `helmet_detection-7` include:

| Metric | Epoch 37 | Epoch 50 |
|---|---:|---:|
| Precision | 88.56% | 83.39% |
| Recall | 64.61% | 63.30% |
| mAP@50 | 68.69% | 68.26% |
| mAP@50-95 | 45.35% | 45.52% |

The complete epoch-wise metrics are available in:
`training/results/helmet_detection-7/results.csv`

## Confidence Thresholds

The current inference code uses different confidence thresholds depending on the detection mode:

| File | Detection | Confidence |
|---|---|---:|
| `testing/live_detection.py` | Helmet | 0.50 |
| `testing/live_detection.py` | Fall | 0.40 |
| `testing/live_monitoring.py` | Helmet | 0.70 |
| `testing/live_monitoring.py` | Fall | 0.50 |

These are prediction-time confidence settings from the source code.

## Notes
- A compatible NVIDIA GPU is needed for the configured `device=0` training setting. Update the device setting if training on CPU.
- Dataset files and trained model weights are not included in this repository.
- Training metrics represent the recorded run and may vary with different datasets or environments.