import cv2


def frame_writer(frame, x, y, width, height, text=None):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    if text:
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
