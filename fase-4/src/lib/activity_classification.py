import math
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Função principal para detectar atividades
def detect_activity(frame):
    """
    Detecta atividade em um frame e desenha landmarks da pose.
    """
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.7,  # Aumentando a precisão
        min_tracking_confidence=0.7,
    ) as pose:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
            )
            landmarks = results.pose_landmarks.landmark
            activity = categorize_activity(landmarks)
            cv2.putText(
                frame,
                f"Activity: {activity}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
        else:
            activity = "No pose detected"

        return activity


def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos: A, B e C.
    """
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    )
    return math.degrees(math.acos(max(min(cosine_angle, 1.0), -1.0)))


def categorize_activity(landmarks):
    """
    Categoriza a atividade com base nos landmarks detectados.
    """

    def avg_y(*points):
        return sum(point.y for point in points) / len(points)

    # Pontos relevantes
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
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    # Cálculo médio das posições verticais
    shoulder_y = avg_y(left_shoulder, right_shoulder)
    hip_y = avg_y(left_hip, right_hip)
    knee_y = avg_y(left_knee, right_knee)
    ankle_y = avg_y(left_ankle, right_ankle)
    wrist_y = avg_y(left_wrist, right_wrist)
    elbow_y = avg_y(left_elbow, right_elbow)

    # Ângulos importantes
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    elbow_angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    elbow_angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
    torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)

    # Inclinação da cabeça
    head_tilt = nose.y - shoulder_y  # Diferença vertical entre nariz e linha dos ombros

    # Classificação de leitura e escrita
    if (
        wrist_y > elbow_y
        and abs(elbow_angle_left - 90) < 25
        and abs(elbow_angle_right - 90) < 25
    ):
        if head_tilt < -0.05:  # Cabeça inclinada levemente para baixo
            return "Reading"
        return "Writing"  # Cabeça reta ou levemente inclinada para cima

    if wrist_y < shoulder_y:
        if wrist_y < hip_y:
            return "Jumping" if abs(hip_y - knee_y) > 0.2 else "Raising arms"
        return "Raising arms"

    if hip_y < shoulder_y and abs(hip_y - knee_y) > 0.2:
        if abs(knee_angle - 90) < 15:
            return "Sitting"
        if abs(knee_angle - 180) < 15:
            return "Standing"
        return "Lunging"

    if hip_y > knee_y and abs(knee_angle - 90) < 15:
        return "Squatting"

    if wrist_y > hip_y and wrist_y > knee_y and abs(torso_angle - 45) < 10:
        return "Picking up"

    if abs(hip_y - knee_y) < 0.1 and hip_y > knee_y:  # Quadril próximo dos joelhos
        return "Sitting"  # Sentado

    if (
        hip_y < shoulder_y and abs(hip_y - knee_y) > 0.2
    ):  # Quadril bem acima dos joelhos
        return "Standing"  # De pé

    if abs(left_knee.y - right_knee.y) > 0.1 and knee_angle > 15:
        return "Walking"

    if ankle_y - knee_y < 0.1 and abs(knee_angle - 90) < 15:
        return "Running"

    return "Unknown activity"
