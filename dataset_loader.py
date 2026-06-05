from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "dataset/frames",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

print("Classes:", dataset.classes)
print("Images:", len(dataset))