import cv2

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Video:", video_path)
    print("Frames:", frame_count)
    print("FPS:", fps)

    cap.release()