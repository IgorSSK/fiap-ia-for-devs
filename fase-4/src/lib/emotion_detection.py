from deepface import DeepFace

# Função para analisar expressões emocionais
def analyze_emotion(face_image):
    result = DeepFace.analyze(face_image, actions=["emotion"], enforce_detection=False)
    return result[0]["dominant_emotion"] if result else "Unknown"