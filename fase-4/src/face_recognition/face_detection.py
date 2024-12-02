import cv2
from tqdm import tqdm
from emotion_analysis.emotion_detection import detect_emotion
from activity_detection.activity_classification import detect_activity

def detect_faces(video_path, output_path, frame_skip=2):
    """Process video frames for face and activity detection."""
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    frame_number = 0

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            # Detect emotions
            frame = detect_emotion(frame)

            # Detect activity
            frame = detect_activity(frame)

        frame_number += 1
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")
