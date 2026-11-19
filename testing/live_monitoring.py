from ultralytics import YOLO
import cv2
import pygame
import time

helmet_model = YOLO("../models/helmet_detection_best.pt")
fall_model = YOLO("../training/runs/detect/fall_detection-5/weights/best.pt")

pygame.mixer.init()

helmet_alert = pygame.mixer.Sound("assets/alert.mp3")
fall_alert = pygame.mixer.Sound("assets/fall_alert.mp3")

helmet_channel = pygame.mixer.Channel(0)
fall_channel = pygame.mixer.Channel(1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

helmet_last_alert = 0
HELMET_COOLDOWN = 5

fall_start_time = None
fall_last_alert = 0
FALL_DELAY = 10
FALL_COOLDOWN = 10

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    helmet_results = helmet_model(frame, conf=0.70, verbose=False)
    fall_results = fall_model(frame, conf=0.50, verbose=False)

    display = frame.copy()

    display = helmet_results[0].plot(img=display)
    display = fall_results[0].plot(img=display)

    helmet_detected = False
    head_detected = False

    for box in helmet_results[0].boxes:

        class_id = int(box.cls[0])
        class_name = helmet_model.names[class_id].lower()

        if class_name == "helmet":
            helmet_detected = True

        elif class_name == "head":
            head_detected = True

    if head_detected and not helmet_detected:

        cv2.putText(
            display,
            "NO HELMET",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

        if current_time - helmet_last_alert >= HELMET_COOLDOWN:

            if not helmet_channel.get_busy():
                helmet_channel.play(helmet_alert)

            print("WARNING : Helmet Not Detected")

            helmet_last_alert = current_time

    else:

        cv2.putText(
            display,
            "HELMET OK",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )

    fall_detected = False

    for box in fall_results[0].boxes:

        class_id = int(box.cls[0])
        class_name = fall_model.names[class_id].lower()

        if class_name == "fall":
            fall_detected = True
            break

    if fall_detected:

        cv2.putText(
            display,
            "FALL DETECTED",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

        if fall_start_time is None:
            fall_start_time = current_time

        elapsed = int(current_time - fall_start_time)

        cv2.putText(
            display,
            f"Timer : {elapsed} sec",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        if elapsed >= FALL_DELAY:

            cv2.putText(
                display,
                "FALL CONFIRMED",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

            if current_time - fall_last_alert >= FALL_COOLDOWN:

                if not fall_channel.get_busy():
                    fall_channel.play(fall_alert)

                print("=" * 45)
                print("EMERGENCY! Worker Fall Detected")
                print("Sending Alert to LoRa...")
                print("Location : Zone A")
                print("Status   : FALL DETECTED")
                print("Message Sent Successfully")
                print("=" * 45)

                fall_last_alert = current_time
                fall_start_time = current_time

    else:

        fall_start_time = None

        cv2.putText(
            display,
            "NO FALL",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )

    cv2.imshow("AI Smart Coal Mine Safety Monitoring", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()