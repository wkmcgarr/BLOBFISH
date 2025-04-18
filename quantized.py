import torch
import torch.nn as nn
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import os

# ======= Config =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
dump_dir = 'quant_dump'
os.makedirs(dump_dir, exist_ok=True)

# ======= Quantization-Ready Model =======
class FPGAReadyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
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
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self.features, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.features, ['4', '5', '6'], inplace=True)
        torch.quantization.fuse_modules(self.features, ['8', '9', '10'], inplace=True)

# ======= Dataset =======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
selected_indices = [i for i, (_, label) in enumerate(full_testset) if full_testset.classes[label] in selected_classes]
test_subset = Subset(full_testset, selected_indices)

def remap_labels(subset):
    new_data = []
    for i in range(len(subset)):
        img, label = subset.dataset[subset.indices[i]]
        new_label = class_to_idx[subset.dataset.classes[label]]
        new_data.append((img, new_label))
    return new_data

test_data = remap_labels(test_subset)
testloader = DataLoader(test_data, batch_size=100, shuffle=False)

# ======= Evaluation =======
def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# ======= Load & Evaluate Float Model =======
float_model = FPGAReadyCNN()
float_model.load_state_dict(torch.load('output/fpga_ready_cifar5_90acc.pth', map_location='cpu'))
float_model.eval()
float_acc = evaluate(float_model, testloader)
print(f"🎯 Float32 Model Accuracy: {float_acc:.2f}%")

# ======= Quantization Setup =======
quant_model = FPGAReadyCNN()
quant_model.load_state_dict(torch.load('output/fpga_ready_cifar5_90acc.pth', map_location='cpu'))
quant_model.eval()
quant_model.fuse_model()

torch.backends.quantized.engine = 'qnnpack'
quant_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(quant_model, inplace=True)

# 🔁 Calibration pass
with torch.no_grad():
    for i, (images, _) in enumerate(testloader):
        quant_model(images)
        if i > 20: break  # Calibrate with ~2000 images

torch.quantization.convert(quant_model, inplace=True)

# ======= Evaluate Quantized Model =======
quant_acc = evaluate(quant_model, testloader)
print(f"📦 Quantized INT8 Model Accuracy: {quant_acc:.2f}%")

# ======= Dump Quantized Weights =======
def dump_quantized_weights(model, out_dir):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'int_repr'):
            print("AAA")
            w = module.weight
            w_int = w.int_repr().cpu().numpy()
            print(w) 
            print(w_int) 
            scale = w.q_scale()
            zp = w.q_zero_point()
            weight_path = os.path.join(out_dir, f"{name.replace('.', '_')}_weight.txt")
            with open(weight_path, 'w') as f:
                f.write(f"# scale: {scale}, zero_point: {zp}\n")
                for val in w_int.flatten():
                    f.write(f"{val:4d}  {format(val & 0xFF, '08b')}\n")

        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            b = module.bias.detach().cpu().numpy()
            if b is not None:
                bias_path = os.path.join(out_dir, f"{name.replace('.', '_')}_bias.txt")
                with open(bias_path, 'w') as f:
                    for val in b.flatten():
                        f.write(f"{val:.6f}\n")
dump_quantized_weights(quant_model, dump_dir)
print(f"✅ Quantized weights dumped to `{dump_dir}/`")

