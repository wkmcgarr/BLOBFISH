import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Subset
import random

# ========= Config =========

selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========= Model Definition =========

class FPGAReadyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

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
        x = self.features(x)
        return self.classifier(x)

# ========= Load Model =========

model = FPGAReadyCNN().to(device)
model.load_state_dict(torch.load('output/fpga_ready_cifar5_90acc.pth', map_location=device))
model.eval()

# ========= Hook Setup =========

layer_outputs = {}

def get_hook(name):
    def hook(module, input, output):
        layer_outputs[name] = output.detach().cpu()
    return hook

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Linear)):
        module.register_forward_hook(get_hook(name))

# ========= Load CIFAR Test Image =========

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Filter to selected classes
selected_indices = [i for i, (_, label) in enumerate(full_testset) if full_testset.classes[label] in selected_classes]
filtered_subset = Subset(full_testset, selected_indices)

# Pick a random image
img_tensor, label_idx = random.choice(filtered_subset)
img_tensor = img_tensor.unsqueeze(0).to(device)

# Get class name
class_name = full_testset.classes[label_idx]

print(f"\n🎯 Running model on a random '{class_name}' image...\n")


from torchvision.utils import save_image

# Save the image before normalization if needed
unnorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))  # reverse normalization (0.5, 0.5, 0.5)
img_to_save = unnorm(img_tensor.squeeze(0).cpu())  # remove batch dim

save_image(img_to_save, "random_cifar_sample.png")

# ========= Run Inference =========

with torch.no_grad():
    _ = model(img_tensor)

# ========= Print Layer Outputs =========

for name, output in layer_outputs.items():
    print(f"\n🔹 Layer: {name}")
    print(output.numpy())
    print('-' * 60)
