from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-face.pt")

def extract_faces(video_path):
    cap = cv2.VideoCapture(video_path)

    faces = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes.xyxy:

                x1,y1,x2,y2 = map(int, box)

                face = frame[y1:y2,x1:x2]

                if face.size != 0:
                    faces.append(face)

    cap.release()

    return faces