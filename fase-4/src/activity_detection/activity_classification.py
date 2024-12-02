import cv2
import mediapipe as mp

def detect_activity(frame, last_landmarks=None):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
            )

            if last_landmarks:
                anomalies = 0
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    dx = abs(landmark.x - last_landmarks[i].x)
                    dy = abs(landmark.y - last_landmarks[i].y)
                    if dx > 0.1 or dy > 0.1:
                        anomalies += 1
                return frame, anomalies, results.pose_landmarks.landmark
        return frame, 0, results.pose_landmarks.landmark if results.pose_landmarks else None

