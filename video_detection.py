import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from cnn_model import DeepfakeCNN

# Load model
model = DeepfakeCNN()
model.load_state_dict(
    torch.load(
        "deepfake_model.pth",
        map_location="cpu"
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

video_path = input("Enter video path: ")

cap = cv2.VideoCapture(video_path)

fake_count = 0
real_count = 0

frame_num = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_num += 1

    # Use every 10th frame
    if frame_num % 10 != 0:
        continue

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(image)

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        pred = torch.argmax(
            output,
            dim=1
        ).item()

    if pred == 0:
        fake_count += 1
    else:
        real_count += 1

cap.release()

print("\nFrames predicted")

print("Fake:", fake_count)
print("Real:", real_count)

if fake_count > real_count:
    print("\nFINAL RESULT: FAKE VIDEO")
else:
    print("\nFINAL RESULT: REAL VIDEO")