import os

import cv2
import numpy as np
from tqdm import tqdm
from emotion_analysis.emotion_detection import detect_emotion
from activity_detection.activity_classification import detect_activity
from summary_generation.summary_creator import create_summary
from face_recognition.face_detection import detect_faces


def process_video(video_input, video_output, report_dir, frame_skip=2):
    """Process video frames for face and activity detection."""
    # Load video
    cap = cv2.VideoCapture(video_input)

    # Configure output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    frame_number = 0
    processed_frames = 0
    anomaly_count = 0
    log_detections = []

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            print(f"Warning: Frame {frame_number} is None. Skipping...")
            continue

        if not isinstance(frame, (np.ndarray)) or frame.dtype != np.uint8:
            print(
                f"Error: Frame {frame_number} is not a valid uint8 numpy array. Skipping..."
            )
            continue

        if frame_number % frame_skip == 0:
            # Detect faces
            # frame = detect_faces(frame)
            # Detect emotion
            frame, emotion = detect_emotion(frame)
            # Detect activity
            frame, activity = detect_activity(frame, frame_number, anomaly_count)
            activity = "Unknown activity"
            log_detections.append(
                f"Current person emotion is {emotion} and current activity is {activity}"
            )
            processed_frames += 1

        out.write(frame)
        frame_number += 1

    complete_analysis = ".".join(log_detections)
    with open(
        os.path.join(report_dir, "complete_analysis.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(complete_analysis)

    create_summary(complete_analysis, os.path.join(report_dir, "summary.txt"))

    cap.release()
    out.release()

    print(f"Processed video saved to {video_output}")
    print(
        f"Complete analysis saved to {os.path.join(report_dir, 'complete_analysis.txt')}"
    )
    print(f"Summary saved to {os.path.join(report_dir, 'summary.txt')}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Total anomalies detected: {anomaly_count}")

    with open(os.path.join(report_dir, "report.txt"), "w") as report_file:
        report_file.write(f"Total frames processed: {processed_frames}\n")
        report_file.write(f"Total anomalies detected: {anomaly_count}\n")
    print(f"Report saved to {report_output}")


if __name__ == "__main__":

    video_input = os.path.abspath("assets/data/inputs/short_version.mp4")
    video_output = os.path.abspath("assets/data/outputs/processed_video.mp4")
    report_output = os.path.abspath("assets/data/outputs")

    print(f"Input video: {video_input}")
    process_video(video_input, video_output, report_output)
