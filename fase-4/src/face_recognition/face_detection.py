import cv2
import numpy as np
from tqdm import tqdm
from emotion_analysis.emotion_detection import detect_emotion
from activity_detection.activity_classification import detect_activity

def detect_faces(video_path, output_path, report_output, frame_skip=2):
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
    processed_frames = 0
    anomaly_count = 0
    last_landmarks = None

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            frame = detect_emotion(frame)

            frame, anomalies, last_landmarks = detect_activity(frame, last_landmarks)
            anomaly_count += anomalies

            processed_frames += 1

            if frame is None:
                print(f"Warning: Frame {frame_number} is None. Skipping...")
                continue

            if not isinstance(frame, (np.ndarray)) or frame.dtype != np.uint8:
                print(f"Error: Frame {frame_number} is not a valid uint8 numpy array. Skipping...")
                continue

            out.write(frame)

        frame_number += 1

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total anomalies detected: {anomaly_count}")

    with open(report_output, "w") as report_file:
        report_file.write(f"Total frames processed: {processed_frames}\n")
        report_file.write(f"Total anomalies detected: {anomaly_count}\n")
    print(f"Report saved to {report_output}")
