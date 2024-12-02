import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
previous_wrist_y = None
previous_wrist_x = None


def detect_activity(frame, frame_count, anomalies):
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        activity = ""
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
            )
            landmarks = results.pose_landmarks.landmark

            # Categoriza a atividade
            activity = categorize_activity(landmarks, frame_count)

            # Exibe a atividade na tela
            cv2.putText(
                frame,
                f"Activity: {activity}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            if activity == "Unknown activity":
                anomalies += 1
        return (frame, activity)


def categorize_activity(landmarks, frame_count):
    global previous_wrist_y, previous_wrist_x

    if not landmarks:
        return "Unknown activity"

    # Pontos importantes
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Média das posições
    hip_y = (left_hip.y + right_hip.y) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    knee_y = (left_knee.y + right_knee.y) / 2
    wrist_y = (left_wrist.y + right_wrist.y) / 2

    # Levantar os braços
    if wrist_y < shoulder_y:
        return "Raising arms"

    # Sentado
    if hip_y < shoulder_y and hip_y > knee_y:
        return "Sitting"

    # Em pé
    if abs(hip_y - knee_y) > 0.2 and hip_y < shoulder_y:
        return "Standing"

    # Agachado
    if hip_y > knee_y:
        return "Squatting"

    # Andando (baseado em movimento cíclico das pernas e braços)
    if frame_count % 10 == 0:  # Verifica a cada 10 quadros
        # Diferença vertical dos joelhos
        knee_difference = abs(left_knee.y - right_knee.y)
        ankle_difference = abs(left_ankle.y - right_ankle.y)

        # Diferença horizontal dos braços (movimento pendular)
        wrist_difference = abs(left_wrist.x - right_wrist.x)

        # Condições para detectar "Walking"
        if knee_difference > 0.1 and ankle_difference > 0.1 and wrist_difference > 0.15:
            return "Walking"

    # Acenar (movimento repetitivo da mão)
    if (
        previous_wrist_y is not None and frame_count % 5 == 0
    ):  # Verifica a cada 5 quadros
        wrist_movement = abs(previous_wrist_y - wrist_y)
        if wrist_movement > 0.05:  # Limite de deslocamento
            return "Waving"

    # Pegar (mão se aproximando do quadril ou chão)
    if wrist_y > hip_y and wrist_y > knee_y:
        return "Picking up"

    # Atualiza posições anteriores para detecção de movimento
    previous_wrist_y = wrist_y
    previous_wrist_x = (left_wrist.x + right_wrist.x) / 2

    return "Unknown activity"
