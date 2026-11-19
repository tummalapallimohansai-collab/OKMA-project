from ultralytics import YOLO
import cv2
import pygame
import time
from enhancement.low_light import enhance_low_light

helmet_model = YOLO("../models/helmet_detection_best.pt")
fall_model = YOLO("../training/runs/detect/fall_detection-5/weights/best.pt")

pygame.mixer.init()

helmet_alert = pygame.mixer.Sound("assets/alert.mp3")
fall_alert = pygame.mixer.Sound("assets/fall_alert.mp3")

helmet_channel = pygame.mixer.Channel(0)
fall_channel = pygame.mixer.Channel(1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

window_name = "Coal Mine Safety Monitoring"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    960,
    720
)

helmet_last_alert = 0
HELMET_COOLDOWN = 5

fall_start_time = None
fall_last_alert = 0
FALL_DELAY = 10
FALL_COOLDOWN = 10

print("Camera connected successfully.")
print("Starting live monitoring...")
print("GPU inference enabled.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Frame not received.")
        break

    enhanced_frame = enhance_low_light(frame)

    helmet_results = helmet_model(
        enhanced_frame,
        conf=0.50,
        imgsz=640,
        device=0,
        verbose=False
    )

    fall_results = fall_model(
        enhanced_frame,
        conf=0.40,
        imgsz=640,
        device=0,
        verbose=False
    )

    helmet_detected = False
    head_detected = False
    fall_detected = False

    annotated_frame = enhanced_frame.copy()

    for result in helmet_results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = helmet_model.names[class_id].lower()
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if class_name == "helmet":

                helmet_detected = True

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"Helmet {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            elif class_name == "head":

                head_detected = True

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"No Helmet {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    for result in fall_results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = fall_model.names[class_id].lower()
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if class_name == "fall":

                fall_detected = True

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    annotated_frame,
                    f"FALL {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    current_time = time.time()

    if head_detected and not helmet_detected:

        cv2.putText(
            annotated_frame,
            "NO HELMET!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        if current_time - helmet_last_alert >= HELMET_COOLDOWN:

            if not helmet_channel.get_busy():

                helmet_channel.play(helmet_alert)

                helmet_last_alert = current_time

                print("ALERT: Worker without helmet!")

    elif fall_detected:

        cv2.putText(
            annotated_frame,
            "FALL DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            annotated_frame,
            "SAFE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    if fall_detected:

        if fall_start_time is None:

            fall_start_time = current_time

            print("Possible fall detected. Timer started.")

        elapsed_time = current_time - fall_start_time

        remaining_time = max(
            0,
            FALL_DELAY - elapsed_time
        )

        cv2.putText(
            annotated_frame,
            f"Fall Confirmation: {remaining_time:.1f}s",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        if (
            elapsed_time >= FALL_DELAY
            and
            current_time - fall_last_alert >= FALL_COOLDOWN
        ):

            if not fall_channel.get_busy():

                fall_channel.play(fall_alert)

                fall_last_alert = current_time

                print("EMERGENCY: Fall confirmed!")

    else:

        fall_start_time = None

    cv2.putText(
        annotated_frame,
        "AI-Based Coal Mine Safety Monitoring",
        (20, annotated_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        window_name,
        annotated_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("Monitoring stopped.")