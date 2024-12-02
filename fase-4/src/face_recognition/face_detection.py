import cv2
import mediapipe as mp
from tqdm import tqdm
from deepface import DeepFace


def detect_faces(video_path, output_path):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Draw bounding box
                mp_drawing.draw_detection(frame, detection)

                # Extract face bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = (
                    bboxC.xmin * w,
                    bboxC.ymin * h,
                    bboxC.width * w,
                    bboxC.height * h,
                )

                # Crop the detected face
                face_crop = frame[int(y) : int(y + height), int(x) : int(x + width)]
                try:
                    # Analyze face using DeepFace
                    analysis = DeepFace.analyze(
                        face_crop,
                        actions=["age", "gender", "emotion"],
                        enforce_detection=False,
                    )

                    # Overlay analysis on frame
                    cv2.putText(
                        frame,
                        f"Emotion: {analysis['dominant_emotion']}",
                        (int(x), int(y - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"Age: {analysis['age']}, Gender: {analysis['gender']}",
                        (int(x), int(y - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                except Exception as e:
                    print(f"Error analyzing face: {e}")

        # Write frame to output
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


if __name__ == "__main__":
    video_input = "assets/data/input/challenge.mp4"
    video_output = "assets/data/output/processed_video.mp4"
    detect_faces(video_input, video_output)
