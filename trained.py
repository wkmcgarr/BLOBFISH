import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import os

selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
epochs = 120
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def filter_dataset(dataset):
    idxs = [i for i, (_, label) in enumerate(dataset) if dataset.classes[label] in selected_classes]
    return Subset(dataset, idxs)

train_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_raw = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainset = filter_dataset(train_raw)
testset = filter_dataset(test_raw)

def remap_labels(subset):
    new_data = []
    for i in range(len(subset)):
        img, label = subset.dataset[subset.indices[i]]
        class_name = subset.dataset.classes[label]
        new_label = class_to_idx[class_name]
        new_data.append((img, new_label))
    return new_data

train_data = remap_labels(trainset)
test_data = remap_labels(testset)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

trainloader = DataLoader(SimpleDataset(train_data), batch_size=batch_size, shuffle=True)
testloader = DataLoader(SimpleDataset(test_data), batch_size=100, shuffle=False)

class FPGAReadyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # → 32×16×16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # → 64×8×8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)                  # → 128×4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * 4 * 4, 5)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = FPGAReadyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.2f} - Accuracy: {acc:.2f}%")
    scheduler.step()

os.makedirs('output', exist_ok=True)
torch.save(model.state_dict(), 'output/fpga_ready_cifar5_90acc.pth')
print("✅ Saved to output/fpga_ready_cifar5_90acc.pth")
