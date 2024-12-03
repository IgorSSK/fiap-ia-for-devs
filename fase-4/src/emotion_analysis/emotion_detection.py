from deepface import DeepFace
from lib.video_processor import frame_writer


def detect_emotion(frame, detector_backend="mtcnn", enforce_detection=True):
    """Detect emotions in a video frame using DeepFace."""
    emotion = None
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
            frame_writer(frame, x, y, width, height, text=dominant_emotion)
            emotion = f"Emotion: {dominant_emotion}"
    except Exception as e:
        print(f"Error detecting emotion: {e}")
    return frame, emotion
