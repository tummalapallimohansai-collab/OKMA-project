import cv2
from pathlib import Path
from enhancement.low_light import enhance_low_light

source = Path("../datasets/helmet_dataset/valid/images")
output = Path("../outputs/low_light_comparison")
output.mkdir(parents=True, exist_ok=True)

images = list(source.glob("*"))

for image_path in images:

    frame = cv2.imread(str(image_path))

    if frame is None:
        continue

    enhanced = enhance_low_light(frame)

    comparison = cv2.hconcat([frame, enhanced])

    cv2.putText(
        comparison,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        comparison,
        "CLAHE Enhanced",
        (frame.shape[1] + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    output_path = output / f"comparison_{image_path.name}"
    cv2.imwrite(str(output_path), comparison)

    print("Created:", output_path)

    break

print("Low-light comparison image created.")