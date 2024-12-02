import os
from face_recognition.face_detection import detect_faces

if __name__ == "__main__":

    video_input = os.path.abspath("assets/data/inputs/short_version.mp4")
    video_output = os.path.abspath("assets/data/outputs/processed_video.mp4")
    report_output = os.path.abspath("assets/data/outputs/report.txt")

    print(f"Input video: {video_input}")
    detect_faces(video_input, video_output, report_output)
