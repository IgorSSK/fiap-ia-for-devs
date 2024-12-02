import cv2
import mediapipe as mp
from tqdm import tqdm
from deepface import DeepFace


def detect_faces(video_path, output_path):
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Define parameters
    frame_skip = 2  # Process every 5th frame to optimize performance
    detector_backend = "mtcnn"  # Options: 'mtcnn', 'opencv', 'ssd', etc.
    enforce_detection = True  # Ensures only detected faces are analyzed

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

    frame_number = 0

    for _ in tqdm(range(frame_count)):
        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_number % frame_skip == 0:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                )

                # Overlay analysis on frame
                for face in result:
                    region = face["region"]
                    x, y, width, height = (
                        region["x"],
                        region["y"],
                        region["w"],
                        region["h"],
                    )
                    dominant_emotion = face["dominant_emotion"]

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + width, y + height),
                        (0, 255, 0),
                        2,
                    )
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
                print(f"Error processing frame: {e}")

            # Activity detection
            # try:

            #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     mp_pose = mp.solutions.pose
            #     pose = mp_pose.Pose()
            #     mp_drawing = mp.solutions.drawing_utils

            #     results = pose.process(rgb_frame)
            #     if results.pose_landmarks:
            #         mp_drawing.draw_landmarks(
            #             frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            #         )
            # except Exception as e:
            #     print(f"Error processing frame: {e}")

            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as pose:
                # Convert the BGR image to RGB and process it with MediaPipe Pose.'
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                # Draw the pose annotation on the frame.
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                )

        frame_number += 1

        # Write frame to output
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")
