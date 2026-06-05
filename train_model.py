import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model import DeepfakeCNN
from dataset_loader import loader

device = torch.device("cpu")

model = DeepfakeCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "deepfake_model.pth"
)

print("Model Saved")