import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Camera started")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    cv2.namedWindow(
        "Coal Mine Safety Monitoring",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Coal Mine Safety Monitoring",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()