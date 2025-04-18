import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import os
import random

# ==== Output setup ====
out_dir = "conv1_io"
os.makedirs(out_dir, exist_ok=True)

# ==== Model ====
class FPGAReadyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),     # Layer 0
            nn.BatchNorm2d(32),                # Layer 1
            nn.ReLU(),                         # Layer 2
            nn.MaxPool2d(2),                   # Layer 3
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * 4 * 4, 5)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ==== Load model ====
model = FPGAReadyCNN()
model.load_state_dict(torch.load('output/fpga_ready_cifar5_90acc.pth', map_location='cpu'))
model.eval()

# ==== Load 1 CIFAR-10 test image ====
selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
indices = [i for i, (_, lbl) in enumerate(dataset) if dataset.classes[lbl] in selected_classes]
img, label = dataset[random.choice(indices)]

# ==== Save RGB image matrix (unnormalized) ====
img_np = img.numpy().transpose(1, 2, 0)  # C×H×W → H×W×C
img_rgb = (img_np * 0.5 + 0.5)  # Undo normalization
np.savetxt(f"{out_dir}/input_rgb.txt", img_rgb.reshape(-1, 3), fmt='%.6f', header='R\tG\tB')

# ==== Hook first conv layer ====
conv1_input = None
conv1_output = None

def save_io(module, input, output):
    global conv1_input, conv1_output
    conv1_input = input[0].detach().cpu().numpy()
    conv1_output = output.detach().cpu().numpy()

model.features[0].register_forward_hook(save_io)

# ==== Run inference ====
with torch.no_grad():
    model(img.unsqueeze(0))  # Add batch dim

# ==== Save inputs and outputs of conv1 ====
np.save(f"{out_dir}/conv1_input.npy", conv1_input)
np.save(f"{out_dir}/conv1_output.npy", conv1_output)

np.savetxt(f"{out_dir}/conv1_input_flat.txt", conv1_input.flatten(), fmt="%.6f")
np.savetxt(f"{out_dir}/conv1_output_flat.txt", conv1_output.flatten(), fmt="%.6f")

print("✅ Done! Saved RGB matrix, conv1 input/output to:", out_dir)
