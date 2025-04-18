import torch
import torch.nn as nn
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torchvision.utils import save_image
import os
import random
import numpy as np

# ========= Config =========
selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}
output_dir = 'quant_int_outputs'
os.makedirs(output_dir, exist_ok=True)
device = torch.device('cpu')  
# quantized models run on CPU

# ========= Model =========
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

# ========= Load & Quantize Model =========
torch.backends.quantized.engine = 'qnnpack'
model = FPGAReadyCNN()
model.load_state_dict(torch.load('output/fpga_ready_cifar5_90acc.pth', map_location='cpu'))
model.eval()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)
model(torch.randn(1, 3, 32, 32))  # dummy calibration
torch.quantization.convert(model, inplace=True)
print("\n✅ Verifying quantized layers:")
for name, module in model.named_modules():
    if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
        print(f"{name} → {type(module)} ✅")
    elif isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"{name} → {type(module)} ⚠️ Still float")

model.eval()

# ========= Hook Setup =========
layer_io = {}

def get_hook(name):
    def hook(module, input, output):
        layer_io[f"{name}_input"] = input[0].detach().cpu()
        layer_io[f"{name}_output"] = output.detach().cpu()
    return hook

for name, module in model.named_modules():
    if isinstance(module, (nn.quantized.Conv2d, nn.ReLU, nn.quantized.Linear, nn.MaxPool2d)):
        module.register_forward_hook(get_hook(name))

# ========= Load Specific Test Image =========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
raw_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Pick one specific class
target_class = "dog"  # change to "airplane", "ship", etc.
target_class_idx = raw_dataset.classes.index(target_class)

# Find first matching image
for i, (img, lbl) in enumerate(raw_dataset):
    if lbl == target_class_idx:
        img_tensor = img
        label = lbl
        break

# Save image for verification
unnorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
save_image(unnorm(img_tensor), os.path.join(output_dir, 'input_image.png'))

# Save float pixel values
np.savetxt(os.path.join(output_dir, 'input_image_float.txt'),
           img_tensor.numpy().reshape(3, 32, 32).transpose(1, 2, 0).reshape(-1, 3),
           fmt='%.6f', header='R\tG\tB')

# ========= Inference =========
with torch.no_grad():
    output = model(img_tensor.unsqueeze(0))

# ========= Classification Result =========
pred = torch.argmax(output, dim=1).item()
print(f"\nActual: {raw_dataset.classes[label]} | Predicted: {idx_to_class[pred]}")
print(f"Logits: {output.numpy().flatten()}")
np.savetxt(os.path.join(output_dir, 'logits.txt'), output.numpy().flatten(), fmt='%.6f')

# ========= Dump Quantized Inputs/Outputs =========
for name, tensor in layer_io.items():
    out_path = os.path.join(output_dir, f"{name.replace('.', '_')}_int.txt")
    
    if tensor.is_quantized:
        arr = tensor.int_repr().cpu().numpy().astype(np.int8).flatten()
        with open(out_path, 'w') as f:
            for val in arr:
                f.write(f"{val:4d}\n")
        print(f"Quant INT: {name} → {out_path}")
    
    else:
        arr = tensor.cpu().numpy().flatten()
        with open(out_path.replace('_int.txt', '_float.txt'), 'w') as f:
            for val in arr:
                f.write(f"{val:.6f}\n")
        print(f"Non-quantized: {name} → {out_path.replace('_int.txt', '_float.txt')}")


for name, module in model.named_modules():
    if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
        print(f"{name}: {type(module)}")

for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        print(f"{name}: weight exists")
    if hasattr(module, 'weight') and hasattr(module.weight, 'int_repr'):
        print(f"{name}: int_repr() works ✅")

weights_file = os.path.join(output_dir, 'weights.txt')
with open(weights_file, 'w') as f:
    for name, module in model.named_modules():
        if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
            f.write(f"\n=== {name} ===\n")
            
            # Get quantized weight tensor
            w = module.weight()
            w_int = w.int_repr().cpu().numpy()
            scale = w.q_scale()
            zp = w.q_zero_point()

            f.write(f"# scale: {scale:.8f}, zero_point: {zp}\n")
            f.write(f"# weight shape: {w_int.shape}\n")

            if w_int.ndim == 4:
                # Conv2d: (out_channels, in_channels, kH, kW)
                for oc in range(w_int.shape[0]):
                    f.write(f"\n[output_channel {oc}]\n")
                    for ic in range(w_int.shape[1]):
                        f.write(f" in_channel {ic}:\n")
                        for row in w_int[oc, ic]:
                            f.write("  " + "  ".join(f"{val:4d}" for val in row) + "\n")
            elif w_int.ndim == 2:
                # Linear: (out_features, in_features)
                for row in w_int:
                    f.write("  " + "  ".join(f"{val:4d}" for val in row) + "\n")
            else:
                f.write("# Unsupported weight shape\n")

            # Bias handling
            if hasattr(module, 'bias') and module.bias is not None:
                bias_fp32 = module.bias().detach().cpu().numpy()
                bias_int = (bias_fp32 / scale).round().astype(np.int32)
                f.write(f"\n# bias (int32):\n")
                f.write("  " + "  ".join(f"{val:6d}" for val in bias_int) + "\n")

print(f"\n✅ All weights and quantized biases saved to: {weights_file}")

print("\n📐 Q15 multipliers for all quantized layers:")

for name, module in model.named_modules():
    if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
        try:
            input_tensor = layer_io.get(f"{name}_input")
            output_tensor = layer_io.get(f"{name}_output")

            if input_tensor is None or output_tensor is None:
                raise ValueError("Missing input/output hook tensors")

            input_scale = input_tensor.q_scale()
            output_scale = output_tensor.q_scale()
            weight_scale = module.weight().q_scale()

            multiplier = (input_scale * weight_scale) / output_scale
            q15 = round(multiplier * (1 << 15))

            print(f"{name:20s} | in={input_scale:.6f}, w={weight_scale:.6f}, out={output_scale:.6f} "
                  f"=> mult={multiplier:.6f} → Q15: {q15}")
        except Exception as e:
            print(f"[WARN] Skipping {name} due to error: {e}")



print("\n📐 Quantization Parameters (scale and zero_point):\n")
for name, module in model.named_modules():
    if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
        inp = layer_io.get(f"{name}_input")
        out = layer_io.get(f"{name}_output")
        if inp is not None and out is not None:
            print(f"{name}")
            print(f"  Input scale:       {inp.q_scale():.6f}   zero_point: {inp.q_zero_point()}")
            print(f"  Weight scale:      {module.weight().q_scale():.6f}   zero_point: {module.weight().q_zero_point()}")
            print(f"  Output scale:      {out.q_scale():.6f}   zero_point: {out.q_zero_point()}")
            print()


# ======= Set output dir =======
output_dir = 'new_output'
os.makedirs(output_dir, exist_ok=True)

# ======= Save Tensor Function =======
def save_tensor_matrix(tensor, path, name=""):
    with open(path, 'w') as f:
        f.write(f"# shape: {list(tensor.shape)}\n")
        if tensor.dim() == 4:  # [N, C, H, W] e.g. Conv2d
            N, C, H, W = tensor.shape
            for n in range(N):
                for c in range(C):
                    f.write(f"# {name} [N={n}, C={c}]\n")
                    for h in range(H):
                        row = "  ".join(f"{int(tensor[n, c, h, w].item()):4d}" for w in range(W))
                        f.write(row + "\n")
                    f.write("\n")
        elif tensor.dim() == 2:  # [N, F] e.g. Linear
            N, F = tensor.shape
            for n in range(N):
                row = "  ".join(f"{int(tensor[n, f].item()):4d}" for f in range(F))
                f.write(f"# {name} [N={n}]\n{row}\n\n")
        elif tensor.dim() == 1:
            row = "  ".join(f"{int(val.item()):4d}" for val in tensor)
            f.write(f"# {name} [1D]\n{row}\n")
        else:
            f.write("# Unsupported shape\n")
            f.write(str(tensor))

# ======= Save all hooked tensors =======
for name, tensor in layer_io.items():
    clean_name = name.replace('.', '_')
    out_path = os.path.join(output_dir, f"{clean_name}_int.txt")

    if tensor.is_quantized:
        save_tensor_matrix(tensor.int_repr(), out_path, name)
        print(f"✅ Quantized: {name} saved as {out_path}")
    else:
        out_path = out_path.replace('_int.txt', '_float.txt')
        save_tensor_matrix(tensor, out_path, name)
        print(f"🟡 Float: {name} saved as {out_path}")



# ======= Save Quantized Weights and Biases to new_output =======
weights_dir = os.path.join(output_dir, 'weights')
os.makedirs(weights_dir, exist_ok=True)

for name, module in model.named_modules():
    if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
        weight_tensor = module.weight()
        weight_int = weight_tensor.int_repr().cpu()
        scale = weight_tensor.q_scale()
        zp = weight_tensor.q_zero_point()

        weight_path = os.path.join(weights_dir, f"{name.replace('.', '_')}_weight.txt")
        with open(weight_path, 'w') as f:
            f.write(f"# shape: {list(weight_int.shape)}\n")
            f.write(f"# scale: {scale:.8f}, zero_point: {zp}\n")
            if weight_int.dim() == 4:
                for oc in range(weight_int.shape[0]):
                    f.write(f"\n[output_channel {oc}]\n")
                    for ic in range(weight_int.shape[1]):
                        f.write(f" in_channel {ic}:\n")
                        for row in weight_int[oc, ic]:
                            f.write("  " + "  ".join(f"{int(v):4d}" for v in row) + "\n")
            elif weight_int.dim() == 2:
                for row in weight_int:
                    f.write("  " + "  ".join(f"{int(v):4d}" for v in row) + "\n")
            else:
                f.write(str(weight_int.numpy()))

        print(f"💾 Saved: {weight_path}")

        # Save bias
        if hasattr(module, 'bias') and module.bias is not None:
            bias_fp32 = module.bias().detach().cpu().numpy()
            bias_int = (bias_fp32 / scale).round().astype(np.int32)
            bias_path = os.path.join(weights_dir, f"{name.replace('.', '_')}_bias.txt")
            with open(bias_path, 'w') as f:
                f.write(f"# shape: {bias_fp32.shape}, quantized scale: {scale:.8f}\n")
                f.write("  " + "  ".join(f"{val:6d}" for val in bias_int) + "\n")
            print(f"💾 Saved: {bias_path}")
