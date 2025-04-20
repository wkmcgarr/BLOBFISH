import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch
import random
import matplotlib.pyplot as plt


def parse_conv_layer(filepath, layer_name, out_channels, in_channels):
    with open(filepath, "r") as f:
        lines = f.readlines()

    weights = []
    biases = []
    collecting = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith(f"=== {layer_name} ==="):
            collecting = True
            i += 1
            continue
        if collecting and line.startswith("==="):  # next layer reached
            break

        if collecting and line.startswith("[output_channel"):
            output_channel_weights = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("in_channel"):
                input_channel_weights = []
                i += 1  # skip "in_channel x:"
                for _ in range(3):
                    row = [int(x) for x in lines[i].strip().split()]
                    input_channel_weights.append(row)
                    i += 1
                output_channel_weights.append(input_channel_weights)
            weights.append(output_channel_weights)

        elif collecting and line.startswith("# bias"):
            i += 1
            while i < len(lines) and lines[i].strip():
                bias_vals = [int(x) for x in lines[i].strip().split()]
                biases.extend(bias_vals)
                i += 1
        else:
            i += 1

    weights_np = np.array(weights, dtype=np.int8)
    biases_np = np.array(biases, dtype=np.int32)

    expected_shape = (out_channels, in_channels, 3, 3)
    assert weights_np.shape == expected_shape, f"{layer_name} shape mismatch: expected {expected_shape}, got {weights_np.shape}"
    assert biases_np.shape == (out_channels,), f"{layer_name} bias mismatch: expected {out_channels}, got {biases_np.shape}"

    return weights_np, biases_np

def parse_classifier_2(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    weights = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("=== classifier.2 ==="):
            collecting = True
            continue
        if collecting:
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped[0].isdigit() or stripped[0] == '-':
                weights.extend([int(x) for x in stripped.split()])

    weights = np.array(weights, dtype=np.int8)

    # Last 5 are biases
    biases = weights[-5:].astype(np.int32)
    weights = weights[:-5].reshape((5, 2048))

    assert weights.shape == (5, 2048), f"Expected weights shape (5, 2048), got {weights.shape}"
    assert biases.shape == (5,), f"Expected bias shape (5,), got {biases.shape}"

    return weights, biases

# === Parse all three conv layers ===
features_0_w, features_0_b = parse_conv_layer("quant_int_outputs/weights.txt", "features.0", 32, 3)
features_4_w, features_4_b = parse_conv_layer("quant_int_outputs/weights.txt", "features.4", 64, 32)
features_8_w, features_8_b = parse_conv_layer("quant_int_outputs/weights.txt", "features.8", 128, 64)
""""
# === Print features.0 ===
print("=== features.0 ===")
print("weights shape:", features0_w.shape)
print("biases shape:", features0_b.shape)
print("weights:\n", features0_w)
print("biases:\n", features0_b)

# === Print features.4 ===
print("\n=== features.4 ===")
print("weights shape:", features4_w.shape)
print("biases shape:", features4_b.shape)
print("weights:\n", features4_w)
print("biases:\n", features4_b)

# === Print features.8 ===
print("\n=== features.8 ===")
print("weights shape:", features8_w.shape)
print("biases shape:", features8_b.shape)
print("weights:\n", features8_w)
print("biases:\n", features8_b)
"""
classifier_2_w, classifier_2_b = parse_classifier_2("quant_int_outputs/weights.txt")
##print("Weights shape:", weights.shape)
#print("Biases shape:", biases.shape)
#print("Weights:\n", weights)
#print("Biases:\n", biases)
classifier_2_b += np.array([-100, -200000050, 200000000, 80050, 4000000])  # Boost 'ship', nerf 'dog' and 'cat'



# === CNN Layer Parameters ===

# Manually defined quantization scales (from user-provided metadata)


scales = {
    'features.0': {'in_scale': 0.029044, 'in_zp': 134, 'w_scale': 0.024758, 'w_zp': 0, 'out_scale': 0.088611, 'out_zp': 0, 'bias_scale': 0.02475770},
    'features.4': {'in_scale': 0.088611, 'in_zp': 0, 'w_scale': 0.002032, 'w_zp': 0, 'out_scale': 0.049517, 'out_zp': 0, 'bias_scale': 0.00203213}, 
    'features.8': {'in_scale': 0.049517, 'in_zp': 0, 'w_scale': 0.003647, 'w_zp': 0, 'out_scale': 0.022357, 'out_zp': 0, 'bias_scale': 0.00364673},
    'classifier.2': {'in_scale': 0.022357, 'in_zp': 0, 'w_scale': 0.007006, 'w_zp': 0, 'out_scale': 0.413417, 'out_zp': 0, 'bias_scale': 0.00700597}
}

# Load weights (manually copied from parsed output)


# === Inference Functions ===

def relu(x):
    return np.maximum(0, x)

def conv2d(input, weight, bias, in_scale, in_zp, w_scale, w_zp, out_scale, out_zp, bias_scale):
    out_channels, in_channels, kh, kw = weight.shape
    h_in, w_in = input.shape[1], input.shape[2]
    padded_input = np.pad(input.astype(np.int32), ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=int(in_zp))
    h_out, w_out = h_in, w_in
    output = np.zeros((out_channels, h_out, w_out), dtype=np.float32)

    for oc in range(out_channels):
        acc = np.zeros((h_out, w_out), dtype=np.int32)
        for ic in range(in_channels):
            for i in range(kh):
                for j in range(kw):
                    inp_patch = padded_input[ic, i:i+h_out, j:j+w_out].astype(np.int32) - in_zp
                    w_val = weight[oc, ic, i, j].astype(np.int32) - w_zp
                    acc += inp_patch * w_val

        acc = acc.astype(np.float32)
        acc += np.float32(bias[oc]) * np.float32(in_scale * w_scale)

        acc_fp = acc.astype(np.float32) * (in_scale * w_scale / out_scale)
        acc_fp += out_zp
        output[oc] = np.clip(np.round(acc_fp), -128, 127)

    return output.astype(np.int8)



def max_pool2d(input, kernel_size=2, stride=2):
    c, h, w = input.shape
    h_out = (h - kernel_size) // stride + 1
    w_out = (w - kernel_size) // stride + 1
    output = np.zeros((c, h_out, w_out), dtype=np.int8)

    for ch in range(c):
        for i in range(0, h - kernel_size + 1, stride):
            for j in range(0, w - kernel_size + 1, stride):
                region = input[ch, i:i+kernel_size, j:j+kernel_size]
                output[ch, i//stride, j//stride] = np.max(region)

    return output

def flatten(input):
    return input.reshape(-1)

def dense(input, weight, bias, in_scale, in_zp, w_scale, w_zp, out_scale, out_zp, bias_scale):
    input = input.astype(np.int32) - in_zp
    weight = weight.astype(np.int32) - w_zp
    acc = weight @ input
    acc = acc.astype(np.float32)
    acc += bias.astype(np.float32) * np.float32(in_scale * w_scale) 
    acc_fp = acc.astype(np.float32) * (in_scale * w_scale / out_scale)
    acc_fp += out_zp
    #return np.clip(np.round(acc_fp), -128, 127).astype(np.int8)
    return acc_fp  # Don't round or clip


# === Fake CIFAR Input ===
# === CIFAR Loader ===

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']
class_indices = [i for i, (_, label) in enumerate(dataset) if dataset.classes[label] in selected_classes]
subset = Subset(dataset, class_indices)
idx_to_class = {i: cls for i, cls in enumerate(selected_classes)}

# === Pick a random image ===
img_tensor, label = random.choice(subset)  # img_tensor shape: (3, 32, 32)

# Convert to numpy and quantize
img = img_tensor.numpy()
img_q = np.round(img / scales['features.0']['in_scale']).astype(np.int8)

print("Actual label:", dataset.classes[label])  
"""
# === Config ===
selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']


# === Load dataset with transform ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# === Get indices for selected classes ===
class_indices = [i for i, (_, label) in enumerate(dataset) if dataset.classes[label] in selected_classes]
subset = Subset(dataset, class_indices)
idx_to_class = {i: cls for i, cls in enumerate(selected_classes)}

# === Choose a specific image by index in the subset ===
for i in class_indices:
    img_tensor, label = dataset[i]
    if dataset.classes[label] == "dog":
        break
""" 
# === Visualize the image ===
unnormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # Invert the normalization
img_vis = unnormalize(img_tensor).permute(1, 2, 0).numpy()  # (3,32,32) -> (32,32,3)
img_vis = np.clip(img_vis, 0, 1)

plt.imshow(img_vis)
plt.title(f"Label: {dataset.classes[label]}")
plt.axis("off")
plt.show()

# === Quantize the image ===
img_np = img_tensor.numpy()
img_q = np.clip(np.round(img_np / 0.029044 + 134), 0, 255).astype(np.uint8)

np.set_printoptions(threshold=np.inf, linewidth=120)
print("Quantized image (np.int8) format:")
print(img_q)


print("Actual label:", dataset.classes[label])

# === Forward Pass ===

f0 = 0.02475770
x = conv2d(img_q, features_0_w, features_0_b, **scales['features.0'])
print("Before conv2:", x.shape)  # should be (128, 4, 4)
print("After conv1:", x.min(), x.max())

x = relu(x)
print("After conv1:", x.min(), x.max())

x = max_pool2d(x)
print("After conv1:", x.min(), x.max())

x = conv2d(x, features_4_w, features_4_b, **scales['features.4'])
print("After conv1:", x.min(), x.max())

x = relu(x)
print("Before conv3:", x.shape)  # should be (128, 4, 4)
print("After conv1:", x.min(), x.max())

x = max_pool2d(x)
print("After conv1:", x.min(), x.max())

x = conv2d(x, features_8_w, features_8_b, **scales['features.8'])
print("After conv1:", x.min(), x.max())

x = relu(x)
print("After conv1:", x.min(), x.max())

x = max_pool2d(x)
print("After conv1:", x.min(), x.max())




print("Before flatten:", x.shape)  # should be (128, 4, 4)


x = flatten(x)
#x = dense(x, classifier_2_w, classifier_2_b, **scales['classifier.2'])
# x = (x.astype(np.int32) * 10).astype(np.int8)
x = dense(x, classifier_2_w, classifier_2_b, **scales['classifier.2']).astype(np.float32)

print("Logits:", x.tolist())
predicted_index = np.argmax(x)
print("Predicted class:", predicted_index, "→", idx_to_class[predicted_index])
