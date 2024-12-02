import cv2
from deepface import DeepFace

def detect_emotion(frame, detector_backend="mtcnn", enforce_detection=True):
    """Detect emotions in a video frame using DeepFace."""
    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
        for face in result:
            region = face["region"]
            x, y, width, height = region["x"], region["y"], region["w"], region["h"]
            dominant_emotion = face["dominant_emotion"]

            # Draw rectangle around the face and label emotion
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Emotion: {dominant_emotion}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
    except Exception as e:
        print(f"Error detecting emotion: {e}")
    return frame
