import cv2
import torch
from PIL import Image
from torchvision import transforms

from cnn_model import DeepfakeCNN

# Load Model
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

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

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

    label = "FAKE" if pred == 0 else "REAL"

    cv2.putText(
        frame,
        label,
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "Live Deepfake Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()