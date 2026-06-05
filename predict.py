import torch
from PIL import Image
from torchvision import transforms

from cnn_model import DeepfakeCNN

device = torch.device("cpu")

model = DeepfakeCNN()
model.load_state_dict(torch.load("deepfake_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image_path = "dataset/frames/fake/fake1/frame_0.jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(output, dim=1)

if prediction.item() == 0:
    print("Prediction: FAKE")
else:
    print("Prediction: REAL")