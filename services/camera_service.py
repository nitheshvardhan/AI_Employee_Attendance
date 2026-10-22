import cv2

# Initialize the detector only once.
hog_detector = cv2.HOGDescriptor()
hog_detector.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)


def open_camera(camera_index=0):
    """Open the default webcam."""
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open the camera. Check camera permissions "
            "or try another camera index."
        )

    return camera


def read_camera_frame(camera):
    """Read a frame and convert it to RGB for Streamlit."""
    success, frame = camera.read()

    if not success:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



def detect_people(frame):
    # Frame received from read_camera_frame() is RGB.
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    boxes, weights = hog_detector.detectMultiScale(
        bgr_frame,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.05,
    )

    result = frame.copy()

    for x, y, width, height in boxes:
        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            4,
        )

        cv2.putText(
            result,
            "PERSON DETECTED",
            (x, max(y - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    return result, len(boxes)
