import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from cnn_model import DeepfakeCNN
from dataset_loader import loader

# Load Model
model = DeepfakeCNN()
model.load_state_dict(
    torch.load(
        "deepfake_model.pth",
        map_location="cpu"
    )
)

model.eval()

all_preds = []
all_labels = []

# Evaluation
with torch.no_grad():

    for images, labels in loader:

        outputs = model(images)

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# Metrics
accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average="binary"
)

recall = recall_score(
    all_labels,
    all_preds,
    average="binary"
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="binary"
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\n===== MODEL EVALUATION =====\n")

print(
    f"Accuracy : {accuracy*100:.2f}%"
)

print(
    f"Precision: {precision*100:.2f}%"
)

print(
    f"Recall   : {recall*100:.2f}%"
)

print(
    f"F1 Score : {f1*100:.2f}%"
)

print("\nConfusion Matrix:")

print(cm)