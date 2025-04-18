import torch
from torchvision import datasets, transforms
import random
import numpy as np

# ====== Setup ======
selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# ====== Select a sample from our class subset ======
indices = [i for i, (_, lbl) in enumerate(dataset) if dataset.classes[lbl] in selected_classes]
img_tensor, label = dataset[random.choice(indices)]

class_name = dataset.classes[label]
print(f"\n🎯 Selected Class: {class_name} (label: {label})")

# ====== Normalized Image Tensor ======
print("\n📉 Normalized Pixel Values (shape: C×H×W):")
print(img_tensor)

# ====== Unnormalize to RGB 0–1 ======
unnormalized = img_tensor * 0.5 + 0.5  # Since you used Normalize((0.5,), (0.5,))
rgb_image = unnormalized.numpy().transpose(1, 2, 0)  # H×W×C

print("\n🎨 Unnormalized RGB values (pixel [0, 0]):")
print(rgb_image[0, 0])  # R, G, B values for top-left pixel

# ====== Optional: Save RGB matrix to text ======
np.savetxt("image_rgb_matrix.txt", rgb_image.reshape(-1, 3), fmt="%.6f", header="R\tG\tB")
print("\n✅ Saved RGB matrix to image_rgb_matrix.txt")
