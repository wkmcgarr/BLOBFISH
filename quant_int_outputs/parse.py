import numpy as np
from torchvision import datasets, transforms

def parse_features_0_weights(filepath):
    weights = []
    biases = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("[output_channel"):
            oc_weights = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("in_channel"):
                i += 1  # skip "in_channel N:"
                kernel = []
                rows_read = 0
                while i < len(lines) and rows_read < 3:
                    row = lines[i].strip()
                    if not row or row.startswith("in_channel") or row.startswith("[output_channel") or row.startswith("#"):
                        break
                    kernel.append([int(x) for x in row.split()])
                    i += 1
                    rows_read += 1
                if len(kernel) == 3:
                    oc_weights.append(kernel)
            weights.append(oc_weights)
        elif line.startswith("# bias"):
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("==="):
                biases += [int(x) for x in lines[i].strip().split()]
                i += 1
        else:
            i += 1

    # Validation
    expected_shape = (32, 3, 3, 3)
    assert len(weights) == 32, f"Expected 32 output channels, got {len(weights)}"
    for oc_idx, oc in enumerate(weights):
        if len(oc) != 3:
            raise ValueError(f"Output channel {oc_idx} has {len(oc)} input channels (expected 3)")
        for ic in oc:
            if len(ic) != 3 or any(len(row) != 3 for row in ic):
                raise ValueError(f"Kernel shape in output channel {oc_idx} is incorrect")
    weight_array = np.array(weights, dtype=np.int8)
    bias_array = np.array(biases, dtype=np.int32)
    return weight_array, bias_array

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

def get_real_cifar10_input():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    img, label = dataset[0]
    img_np = (img.numpy() * 128).astype(np.int8).reshape(1, 3, 32, 32)
    return img_np, label

# === RUN ===
weights, biases = parse_features_0_weights("weights.txt")
x, true_label = get_real_cifar10_input()
out = conv2d_int8(x, weights, biases)

print("Conv output shape:", out.shape)
print("True label:", true_label)
