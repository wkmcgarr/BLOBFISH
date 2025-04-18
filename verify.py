import numpy as np
import re

# ========== Load quantized int8 input ==========
input_vals = np.loadtxt('quant_int_outputs/features_0_input_int.txt', usecols=0).astype(np.int8)
input_q = input_vals.reshape((1, 3, 32, 32))  # [B, C_in, H, W]

# ========== Load int8 output to compare ==========
output_vals = np.loadtxt('quant_int_outputs/features_0_output_int.txt', usecols=0).astype(np.int8)
output_ref = output_vals.reshape((1, 32, 32, 32))  # [B, C_out, H, W]

# ========== Parse weights.txt ==========
def parse_weights_and_bias(filename):
    weights = np.zeros((32, 3, 3, 3), dtype=np.int8)
    biases = np.zeros((32,), dtype=np.int32)

    with open(filename, 'r') as f:
        lines = f.readlines()

    oc = -1
    ic = 0
    row_idx = 0
    parsing_weights = False
    parsing_bias = False

    for line in lines:
        if line.startswith("=== features.0 ==="):
            parsing_weights = True
            continue
        if parsing_weights and "# scale" in line:
            continue  # We ignore the float scale
        elif parsing_weights and line.startswith("[output_channel"):
            oc = int(re.findall(r'\d+', line)[0])
            ic = 0
            row_idx = 0
        elif parsing_weights and line.strip().startswith("in_channel"):
            ic = int(re.findall(r'\d+', line)[0])
            row_idx = 0
        elif parsing_weights and line.strip().startswith("# bias"):
            parsing_weights = False
            parsing_bias = True
            continue
        elif parsing_weights and len(line.strip()) > 0 and not line.strip().startswith("#"):
            vals = [int(v) for v in line.strip().split()]
            weights[oc][ic][row_idx] = vals
            row_idx += 1
        elif parsing_bias and len(line.strip()) > 0:
            biases = np.array([int(v) for v in line.strip().split()], dtype=np.int32)
            break

    return weights, biases

weights, biases = parse_weights_and_bias('quant_int_outputs/weights.txt')

# ========== Your Q15 fixed-point scale multiplier ==========
# Requant scale = (input_scale * weight_scale) / output_scale ≈ 0.00842259
# Q15: 0.00842259 × 32768 ≈ 276
scale_q15 = 276

# ========== Pure int conv + bias + ReLU ==========
def conv2d_int(input_q, weights, biases):
    B, C_in, H, W = input_q.shape
    C_out, _, kH, kW = weights.shape
    out = np.zeros((B, C_out, H, W), dtype=np.int32)

    pad = kH // 2
    input_padded = np.pad(input_q, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    for b in range(B):
        for oc in range(C_out):
            for y in range(H):
                for x in range(W):
                    acc = 0
                    for ic in range(C_in):
                        for ky in range(kH):
                            for kx in range(kW):
                                acc += int(input_padded[b, ic, y + ky, x + kx]) * int(weights[oc, ic, ky, kx])
                    acc += int(biases[oc])
                    out[b, oc, y, x] = max(0, acc)  # ReLU
    return out

output_int32 = conv2d_int(input_q, weights, biases)

# ========== Requantize with Q15 multiplier (int-only) ==========
requant = ((output_int32 * scale_q15 + (1 << 14)) >> 15).astype(np.int32)
output_clipped = np.clip(requant, -128, 127).astype(np.int8)

# ========== Compare ==========
mismatches = (output_clipped != output_ref)
error_count = np.sum(mismatches)
total = output_ref.size

print(f"\n✅ Fully integer Conv1 + ReLU + Q15 Requant done")
print(f"📐 Scale Q15 multiplier used: {scale_q15}")
print(f"🔍 Total values: {total}")
print(f"❌ Mismatches: {error_count}")
print(f"✅ Match rate: {(1 - error_count / total) * 100:.4f}%")

if error_count > 0:
    idxs = np.argwhere(mismatches)
    print("\nFirst few mismatches:")
    for b, c, y, x in idxs[:10]:
        print(f"  @[{c},{y},{x}]: expected {output_ref[b, c, y, x]}, got {output_clipped[b, c, y, x]}")
