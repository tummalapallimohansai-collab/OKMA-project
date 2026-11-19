import cv2


def enhance_low_light(frame):
    # Convert BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Improve brightness/contrast of L channel
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR
    enhanced_frame = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_frame