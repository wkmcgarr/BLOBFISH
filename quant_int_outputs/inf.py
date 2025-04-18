import numpy as np
import re
from torchvision import datasets, transforms
from torch.utils.data import Subset
import os

# === LOAD WEIGHTS ===
def parse_weights_file(filepath):
    layers = {}
    current_layer = None
    with open(filepath, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("==="):
            current_layer = line.strip("= ").strip()
            layers[current_layer] = {
                "scale": None,
                "zero_point": None,
                "weights": [],
                "bias": []
            }
            i += 1
        elif line.startswith("# scale:"):
            parts = line.split(',')
            layers[current_layer]["scale"] = float(parts[0].split(":")[1].strip())
            layers[current_layer]["zero_point"] = int(parts[1].split(":")[1].strip())
            i += 1
        elif "[output_channel" in line:
            # Conv2d-style weights
            oc_weights = []
            i += 1
            while i < len(lines) and "in_channel" in lines[i]:
                ic_weights = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith("  "):
                    row = [int(x) for x in lines[i].strip().split()]
                    ic_weights.append(row)
                    i += 1
                oc_weights.append(ic_weights)
            layers[current_layer]["weights"].append(oc_weights)
        elif line.startswith("  ") and re.match(r"\s*-?\d+", line):  # Linear weights (rows of ints)
            while i < len(lines) and lines[i].strip():
                row = [int(x) for x in lines[i].strip().split()]
                layers[current_layer]["weights"].append(row)
                i += 1
        elif "# bias (int32):" in line:
            i += 1
            bias_vals = []
            while i < len(lines) and lines[i].strip():
                bias_vals += [int(x) for x in lines[i].strip().split()]
                i += 1
            layers[current_layer]["bias"] = bias_vals
        else:
            i += 1
    return layers

# === OPS ===
def conv2d_int8(x, weight, bias, stride=1, pad=1):
    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    H_out = (H + 2*pad - K) // stride + 1
    W_out = (W + 2*pad - K) // stride + 1
    out = np.zeros((B, C_out, H_out, W_out), dtype=np.int32)
    x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    for b in range(B):
        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    s = 0
                    for ic in range(C_in):
                        for ki in range(K):
                            for kj in range(K):
                                s += int(x_padded[b, ic, i+ki, j+kj]) * int(weight[oc, ic, ki, kj])
                    out[b, oc, i, j] = s + bias[oc]
    return out

def relu(x):
    return np.maximum(x, 0)

def maxpool2x2(x):
    B, C, H, W = x.shape
    out = np.zeros((B, C, H//2, W//2), dtype=x.dtype)
    for b in range(B):
        for c in range(C):
            for i in range(0, H, 2):
                for j in range(0, W, 2):
                    out[b, c, i//2, j//2] = np.max(x[b, c, i:i+2, j:j+2])
    return out

def linear(x, weight, bias):
    return np.dot(x, weight.T) + bias

# === REAL CIFAR-10 IMAGE ===
def get_real_cifar10_input():
    selected_classes = ['airplane', 'automobile', 'ship', 'dog', 'cat']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    selected_idxs = [i for i, (_, label) in enumerate(testset) if testset.classes[label] in selected_classes]
    img, label = testset[selected_idxs[0]]
    img_np = (img.numpy() * 128).astype(np.int8).reshape(1, 3, 32, 32)
    return img_np, label

# === LOAD & RUN ===
layers = parse_weights_file("weights.txt")

x, true_label = get_real_cifar10_input()

# Verify weights exist before attempting reshape
assert len(layers['features.0']['weights']) > 0, "features.0 weights missing!"
assert len(layers['features.4']['weights']) > 0, "features.4 weights missing!"
assert len(layers['features.8']['weights']) > 0, "features.8 weights missing!"
assert len(layers['classifier.2']['weights']) > 0, "classifier.2 weights missing!"

# features.0
w0 = np.array(layers['features.0']['weights'], dtype=np.int8).reshape(32, 3, 3, 3)
b0 = np.array(layers['features.0']['bias'], dtype=np.int32)
x = conv2d_int8(x, w0, b0)
x = relu(x)
x = maxpool2x2(x)

# features.4
w1 = np.array(layers['features.4']['weights'], dtype=np.int8).reshape(64, 32, 3, 3)
b1 = np.array(layers['features.4']['bias'], dtype=np.int32)
x = conv2d_int8(x.astype(np.int8), w1, b1)
x = relu(x)
x = maxpool2x2(x)

# features.8
w2 = np.array(layers['features.8']['weights'], dtype=np.int8).reshape(128, 64, 3, 3)
b2 = np.array(layers['features.8']['bias'], dtype=np.int32)
x = conv2d_int8(x.astype(np.int8), w2, b2)
x = relu(x)
x = maxpool2x2(x)

# classifier.2
x_flat = x.reshape(1, -1)
w3 = np.array(layers['classifier.2']['weights'], dtype=np.int8).reshape(5, 2048)
b3 = np.array(layers['classifier.2']['bias'], dtype=np.int32)
logits = linear(x_flat.astype(np.int32), w3, b3)

pred = np.argmax(logits)
print("Predicted class index:", pred)
print("True label index:", true_label)
print("Logits:", logits)