import numpy as np

# Inputs
input_q = np.array([
    [[106, 104, 105],
     [107, 103, 106],
     [107, 105, 113]],
    
    [[ 99,  98, 101],
     [100,  97, 103],
     [100, 100, 109]],
    
    [[ 91,  91,  98],
     [ 92,  91,  98],
     [ 93,  93, 104]]
], dtype=np.int16)

weight_q = np.array([
    [[-14, 10, -5],
     [ -5, 10, -5],
     [ -9, -12, 18]],
    
    [[ 5, 23, 4],
     [-19, -9, -19],
     [ -6, -15, 12]],
    
    [[ -4, 23, -8],
     [ -9, -19, -23],
     [ 11, -2, 15]]
], dtype=np.int16)

# Quant params
input_zp = 115
weight_zp = 0
output_zp = 0
bias_q = -889  # int32 bias

# Q15 scale factor
input_scale = 0.031660
weight_scale = 0.024758
output_scale = 0.083797

# Convert scale to Q15
real_multiplier = (input_scale * weight_scale) / output_scale
q15_multiplier = int(round(real_multiplier * (1 << 15)))  # Fixed-point multiplier
shift = 15  # corresponds to Q15

# Accumulate int32
acc = 0
for c in range(3):
    for i in range(3):
        for j in range(3):
            x = input_q[c, i, j] - input_zp
            w = weight_q[c, i, j] - weight_zp
            acc += x * w

# Add pre-quantized bias (already in int32 scale)
acc += bias_q

# Apply fixed-point scaling (Q15-style)
acc_scaled = (acc * q15_multiplier + (1 << (shift - 1))) >> shift  # rounding right shift

# Add output zero point and clip to int8
output_q = max(-128, min(127, acc_scaled + output_zp))

print("Quantized Output:", output_q)