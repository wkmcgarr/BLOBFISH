import numpy as np

def parse_features_0(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    weights = []
    biases = []
    collecting = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("=== features.0 ==="):
            collecting = True
            i += 1
            continue
        if collecting and line.startswith("==="):
            break  # Stop if another section starts

        if collecting and line.startswith("[output_channel"):
            if len(weights) == 32:
                break  # Stop parsing after 32 output channels
            output_channel_weights = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("in_channel"):
                input_channel_weights = []
                i += 1  # skip "in_channel" line
                for _ in range(3):  # 3 rows per 3x3 kernel
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
    biases_np = np.array(biases[:32], dtype=np.int32)  # limit to first 32 if extras exist

    assert weights_np.shape == (32, 3, 3, 3), f"Expected (32, 3, 3, 3), got {weights_np.shape}"
    assert biases_np.shape == (32,), f"Expected 32 biases, got {biases_np.shape}"

    return weights_np, biases_np