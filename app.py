import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from cnn_model import DeepfakeCNN

model = DeepfakeCNN()
model.load_state_dict(
    torch.load("deepfake_model.pth")
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

st.title("Deepfake Detection System")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        prediction = torch.argmax(
            output,
            dim=1
        )

    if prediction.item() == 0:
        st.error("FAKE")
    else:
        st.success("REAL")