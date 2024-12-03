import cv2
from lib.video_processor import frame_writer


def detect_faces(frame):
    """Detect faces in a frame."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in faces:
        frame_writer(frame, x, y, w, h, text="Face")
