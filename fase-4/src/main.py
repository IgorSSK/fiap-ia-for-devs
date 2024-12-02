import os
import cv2
from tqdm import tqdm
from lib.face_detection import detect_faces
from lib.emotion_detection import analyze_emotion
from lib.activity_classification import detect_activity
from lib.summary_creator import generate_summary


def process_video(
    video_input, video_output, report_output, frame_skip=5, stop_frame=-1
):
    prev_frame = None
    activities = []
    emotions = []
    anomalies = 0
    analyzed_frames = 0

    cap = cv2.VideoCapture(video_input)

    # Prepare the video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")

    # Process video
    for _ in tqdm(range(total_frames)):
        try:
            ret, frame = cap.read()
            if not ret or (stop_frame > 0 and analyzed_frames >= stop_frame):
                break

            # Skip frames
            if analyzed_frames % frame_skip == 0:
                faces = detect_faces(frame)

                # Análise de rostos e emoções
                for x, y, w, h in faces:
                    face_img = frame[y : y + h, x : x + w]
                    emotion = analyze_emotion(face_img)
                    emotions.append(emotion)

                    # Desenhar o rosto no vídeo
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        emotion,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )

                # Detecção de atividade (simples movimento)
                if prev_frame is not None:
                    activity_detected = detect_activity(frame)
                    if activity_detected:
                        activities.append(activity_detected)
                    elif (
                        activity_detected == "Unknown activity"
                        or activity_detected == "No pose detected"
                    ):
                        anomalies += 1
        except Exception as e:
            print(f"Error processing frame {analyzed_frames}: {e}")

        out.write(frame)
        prev_frame = frame
        analyzed_frames += 1

    cap.release()
    cv2.destroyAllWindows()

    summary = generate_summary(activities, emotions)

    with open(report_output, "w") as f:
        f.write(f"Summary: {summary}\n")
        f.write(f"Anomalies detected: {anomalies}\n")
        f.write(f"Total frames analyzed: {analyzed_frames}\n")


if __name__ == "__main__":

    video_input = os.path.abspath("assets/data/inputs/challenge.mp4")
    video_output = os.path.abspath("assets/data/outputs/processed_video.mp4")
    report_output = os.path.abspath("assets/data/outputs/report.txt")

    process_video(video_input, video_output, report_output, stop_frame=1500)
