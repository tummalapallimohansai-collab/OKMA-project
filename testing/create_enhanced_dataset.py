import cv2
from pathlib import Path
from enhancement.low_light import enhance_low_light

source = Path("../datasets/helmet_dataset")
destination = Path("../datasets/helmet_dataset_enhanced")

for split in ["train", "valid", "test"]:

    source_images = source / split / "images"
    source_labels = source / split / "labels"

    destination_images = destination / split / "images"
    destination_labels = destination / split / "labels"

    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    for image_path in source_images.glob("*"):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        enhanced = enhance_low_light(image)

        output_path = destination_images / image_path.name
        cv2.imwrite(str(output_path), enhanced)

    for label_path in source_labels.glob("*"):
        output_path = destination_labels / label_path.name
        output_path.write_text(
            label_path.read_text(),
            encoding="utf-8"
        )

print("Enhanced dataset created successfully.")