import numpy as np

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

# === Parse both layers ===
features0_w, features0_b = parse_conv_layer("quant_int_outputs/weights.txt", "features.0", 32, 3)
features4_w, features4_b = parse_conv_layer("quant_int_outputs/weights.txt", "features.4", 64, 32)

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
